"""
Real PyTorch / Diffusers inference engine for Qwen-Image-2.1-Turbo
(`qwen/image-2.1/text-to-image` & `qwen/image-2.1/edit`).

Uses the downloaded `Comfy-Org/Qwen-Image-2.1` pre-quantized `.safetensors` weights:
  - `diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors` (or `*_bf16.safetensors`)
  - `text_encoders/qwen3vl_8b_int8_convrot.safetensors` (or `*_bf16.safetensors`)
  - `vae/qwen_image_2.1_vae_bf16.safetensors`

Loads local Hugging Face config & processor JSONs installed during setup from
`Qwen/Qwen-Image-2.1-Turbo`, remaps ComfyUI tensor
naming into `diffusers.QwenImage21Pipeline` components, preserves 8-bit `int8_convrot`
weights in RAM/VRAM via `StrataConvRotLinear`, and enforces Stage 1 VRAM eviction +
Strata Block Streaming (`StrataBlockOffloadManager`) across stages.
"""

from __future__ import annotations

import json
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from openfal.config import EngineConfig
from openfal.engine.memory_hierarchy import StrataMemoryHierarchy
from openfal.runtime.common import (
    check_inference_dependencies,
    clear_device_cache,
    compute_resident_blocks,
    ensure_model_configs,
    find_first_safetensors,
    load_safetensors_state_dict,
    resolve_compute_device_and_dtype,
)
from openfal.runtime.quant import (
    StrataBlockOffloadManager,
    quantized_module_swap_from_state_dict,
)


def _split_gate_up_tensor(value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    half = value.shape[0] // 2
    return value[:half].contiguous(), value[half:].contiguous()


def convert_qwen_image21_transformer_state_dict(raw_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Remap `Comfy-Org/Qwen-Image-2.1` transformer keys -> `diffusers.QwenImage21Transformer2DModel`.

    Handles:
    - Stripping optional `model.diffusion_model.` prefix
    - Splitting fused SwiGLU `img_mlp.gate_up.{weight,weight_scale,comfy_quant}` (gate rows first)
      into `img_mlp.gate_layer.*` and `img_mlp.proj.*`
    - Dropping attention metadata tensors (`*.comfy_attention.config`)
    """
    out: Dict[str, torch.Tensor] = {}
    keys = list(raw_sd.keys())
    # Record gate_up weight out_features so scales can be sliced or broadcast safely
    gate_up_rows: Dict[str, int] = {}
    for k in keys:
        clean = k.removeprefix("model.diffusion_model.")
        if clean.endswith("img_mlp.gate_up.weight"):
            gate_up_rows[clean.removesuffix(".weight")] = raw_sd[k].shape[0]

    for k in keys:
        val = raw_sd[k]
        new_k = k.removeprefix("model.diffusion_model.")
        if new_k.endswith(".comfy_attention.config") or new_k.endswith(".config"):
            continue

        if new_k.endswith("img_mlp.gate_up.weight"):
            g, p = _split_gate_up_tensor(val)
            out[new_k.replace("gate_up", "gate_layer")] = g
            out[new_k.replace("gate_up", "proj")] = p
        elif new_k.endswith("img_mlp.gate_up.weight_scale"):
            base = new_k.removesuffix(".weight_scale")
            expected_rows = gate_up_rows.get(base)
            if val.ndim >= 1 and expected_rows and val.shape[0] == expected_rows:
                g_s, p_s = _split_gate_up_tensor(val)
            else:
                g_s, p_s = val, val.clone()
            out[new_k.replace("gate_up", "gate_layer")] = g_s
            out[new_k.replace("gate_up", "proj")] = p_s
        elif new_k.endswith("img_mlp.gate_up.comfy_quant"):
            out[new_k.replace("gate_up", "gate_layer")] = val
            out[new_k.replace("gate_up", "proj")] = val.clone()
        else:
            out[new_k] = val

    return out


def convert_qwen3vl_text_encoder_state_dict(raw_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Remap Comfy Qwen3-VL text encoder checkpoint keys -> `transformers.Qwen3VLForConditionalGeneration`.

    `Comfy-Org` stores language backbone keys directly under `model.{embed_tokens,layers,norm}`
    and vision keys under `model.visual.*` or `visual.*`, whereas `transformers >= 5.0` expects
    `model.language_model.*` + `model.visual.*`.
    """
    out: Dict[str, torch.Tensor] = {}
    for k, v in raw_sd.items():
        if k.startswith(("model.embed_tokens.", "model.layers.", "model.norm.")):
            out["model.language_model." + k[len("model.") :]] = v
        elif k.startswith("visual."):
            out["model." + k] = v
        else:
            out[k] = v
    return out


_VAE_RESIDUAL_REPLACEMENTS = (
    (".residual.0.", ".norm1."),
    (".residual.2.", ".conv1."),
    (".residual.3.", ".norm2."),
    (".residual.6.", ".conv2."),
    (".shortcut.", ".conv_shortcut."),
)


def _rewrite_vae_residual_suffix(key: str) -> str:
    prefixed = key.startswith(("residual.", "shortcut."))
    work = f".{key}" if prefixed else key
    for old, new in _VAE_RESIDUAL_REPLACEMENTS:
        if old in work:
            work = work.replace(old, new)
    return work.lstrip(".") if prefixed else work


def convert_qwen_image21_vae_state_dict(raw_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Remap all 238 keys of `Comfy-Org/Qwen-Image-2.1 :: vae/qwen_image_2.1_vae_bf16.safetensors`
    into `diffusers.AutoencoderKLQwenImage21` (`Conv2d` / singleton temporal squeeze).
    """
    out: Dict[str, torch.Tensor] = {}
    up_re = re.compile(r"^decoder\.upsamples\.(\d+)\.upsamples\.(\d+)\.(.+)$")
    down_re = re.compile(r"^encoder\.downsamples\.(\d+)\.downsamples\.(\d+)\.(.+)$")

    for k, v in raw_sd.items():
        if k.startswith("conv1."):
            nk = "quant_conv." + k[len("conv1.") :]
        elif k.startswith("conv2."):
            nk = "post_quant_conv." + k[len("conv2.") :]
        elif k.startswith(("encoder.conv1.", "decoder.conv1.")):
            side, rest = k.split(".conv1.", 1)
            nk = f"{side}.conv_in.{rest}"
        elif k.startswith(("encoder.head.0.", "decoder.head.0.")):
            side, rest = k.split(".head.0.", 1)
            nk = f"{side}.norm_out.{rest}"
        elif k.startswith(("encoder.head.2.", "decoder.head.2.")):
            side, rest = k.split(".head.2.", 1)
            nk = f"{side}.conv_out.{rest}"
        elif k.startswith(("encoder.middle.0.", "decoder.middle.0.")):
            side, rest = k.split(".middle.0.", 1)
            nk = f"{side}.mid_block.resnets.0.{_rewrite_vae_residual_suffix(rest)}"
        elif k.startswith(("encoder.middle.1.", "decoder.middle.1.")):
            side, rest = k.split(".middle.1.", 1)
            nk = f"{side}.mid_block.attentions.0.{rest}"
        elif k.startswith(("encoder.middle.2.", "decoder.middle.2.")):
            side, rest = k.split(".middle.2.", 1)
            nk = f"{side}.mid_block.resnets.1.{_rewrite_vae_residual_suffix(rest)}"
        else:
            m_up = up_re.match(k)
            m_down = down_re.match(k)
            if m_up:
                stage, sub, rest = int(m_up.group(1)), int(m_up.group(2)), m_up.group(3)
                if sub < 3:
                    nk = f"decoder.up_blocks.{stage}.resnets.{sub}.{_rewrite_vae_residual_suffix(rest)}"
                else:
                    nk = f"decoder.up_blocks.{stage}.upsampler.{rest}"
            elif m_down:
                stage, sub, rest = int(m_down.group(1)), int(m_down.group(2)), m_down.group(3)
                if sub < 2:
                    nk = f"encoder.down_blocks.{stage}.resnets.{sub}.{_rewrite_vae_residual_suffix(rest)}"
                else:
                    nk = f"encoder.down_blocks.{stage}.downsampler.{rest}"
            else:
                nk = _rewrite_vae_residual_suffix(k)

        # Diffusers `QwenImage21CausalConv3d` subclasses `nn.Conv2d` with 4D kernels `[O, I, kH, kW]`,
        # whereas Comfy saves `[O, I, 1, kH, kW]`.
        tensor = v.squeeze(2) if (v.ndim == 5 and v.shape[2] == 1) else v
        out[nk] = tensor

    return out


def _rematerialize_meta_buffers(model: nn.Module, reference_ctor: Callable[[], nn.Module]) -> None:
    """
    Replace small non-persistent buffers / plain tensor attributes left on `meta` by
    `accelerate.init_empty_weights()` (such as RoPE `inv_freq`, `freqs`, `pos_freqs`) from a
    lightweight CPU reference construction, or verify zero `meta` tensors remain.
    """
    meta_submods: set[str] = set()
    for name, mod in model.named_modules():
        for buf in mod._buffers.values():
            if buf is not None and buf.is_meta:
                meta_submods.add(name)
        for attr_val in vars(mod).values():
            if isinstance(attr_val, torch.Tensor) and attr_val.is_meta:
                meta_submods.add(name)

    if not meta_submods:
        return

    ref = reference_ctor()
    ref_mods = dict(ref.named_modules())
    for mod_name in meta_submods:
        if mod_name not in ref_mods:
            continue
        tgt = model.get_submodule(mod_name) if mod_name else model
        src = ref_mods[mod_name]
        for bname, sbuf in src._buffers.items():
            if bname in tgt._buffers:
                tgt._buffers[bname] = sbuf.clone() if sbuf is not None else None
        for aname, aval in vars(src).items():
            if isinstance(aval, torch.Tensor):
                setattr(tgt, aname, aval.clone())


class RealQwenImage21Runner:
    """Real neural network inference runner for `Qwen-Image-2.1-Turbo`."""

    HF_CONFIG_REPO = "Qwen/Qwen-Image-2.1-Turbo"
    WEIGHT_SUBDIR = "Qwen-Image-2.1-Turbo"

    def __init__(self, config: EngineConfig, memory: StrataMemoryHierarchy):
        self.config = config
        self.memory = memory
        self.model_root = config.weights_dir / self.WEIGHT_SUBDIR
        self.cfg_dir = self.model_root / "hf_configs"

        self._pipe: Any = None
        self._sample_sigmas: Optional[List[float]] = None
        self._dit_offloader: Optional[StrataBlockOffloadManager] = None
        self._te_offloader: Optional[StrataBlockOffloadManager] = None
        self._prompt_embed_cache: Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]] = {}

    def resolve_weight_paths(self) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
        dit = find_first_safetensors(
            self.model_root,
            [
                "diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors",
                "diffusion_models/qwen_image_2.1_turbo_bf16.safetensors",
                "diffusion_models/qwen_image_2.1_int8_convrot.safetensors",
                "diffusion_models/qwen_image_2.1_bf16.safetensors",
            ],
        )
        te = find_first_safetensors(
            self.model_root,
            [
                "text_encoders/qwen3vl_8b_int8_convrot.safetensors",
                "text_encoders/qwen3vl_8b_bf16.safetensors",
            ],
        )
        vae = find_first_safetensors(
            self.model_root,
            [
                "vae/qwen_image_2.1_vae_bf16.safetensors",
                "vae/diffusion_pytorch_model.safetensors",
            ],
        )
        return dit, te, vae

    def has_weights(self) -> bool:
        dit, _, _ = self.resolve_weight_paths()
        return dit is not None

    def _ensure_configs(self, on_log: Optional[Callable[[str], None]] = None) -> Path:
        return ensure_model_configs(
            "qwen_image_21", self.config.weights_dir, on_log=on_log,
        )

    def _ensure_loaded(self, on_log: Optional[Callable[[str], None]] = None) -> None:
        if self._pipe is not None:
            return

        check_inference_dependencies(("diffusers", "transformers", "accelerate", "torchvision"))

        dit_path, te_path, vae_path = self.resolve_weight_paths()
        missing_weights = [
            label
            for label, path in (("DiT", dit_path), ("Qwen3-VL Text Encoder", te_path), ("VAE", vae_path))
            if path is None
        ]
        if missing_weights:
            raise RuntimeError(
                f"Qwen-Image-2.1 DiT weights exist in '{self.model_root}', but missing component(s): "
                f"{', '.join(missing_weights)}. Run: openfal download --component qwen_image_21"
            )

        cfg_dir = self._ensure_configs(on_log=on_log)
        device, compute_dtype = resolve_compute_device_and_dtype()

        from accelerate import init_empty_weights
        from diffusers import (
            AutoencoderKLQwenImage21,
            FlowMatchEulerDiscreteScheduler,
            QwenImage21Pipeline,
            QwenImage21Transformer2DModel,
        )
        from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

        model_index = json.loads((cfg_dir / "model_index.json").read_text(encoding="utf-8"))
        self._sample_sigmas = model_index.get("sample_sigmas")

        if on_log:
            on_log(
                f"[Qwen-Image-2.1-Turbo] Loading real checkpoints on {device} ({compute_dtype}) "
                f"with Strata 8-bit/BF16 staging..."
            )

        # 1. Text Encoder (Qwen3-VL-8B INT8 ConvRot -> ~8.3 GB RAM, 0 MB duplicate allocation)
        te_cfg = Qwen3VLConfig.from_pretrained(str(cfg_dir / "text_encoder"), local_files_only=True)
        with init_empty_weights():
            text_encoder = Qwen3VLForConditionalGeneration(te_cfg)
        te_raw = load_safetensors_state_dict(te_path, device="cpu")
        te_mapped = convert_qwen3vl_text_encoder_state_dict(te_raw)
        del te_raw
        te_missing, te_unexp = quantized_module_swap_from_state_dict(
            text_encoder,
            te_mapped,
            compute_dtype=compute_dtype,
            embedding_names=("model.language_model.embed_tokens",),
        )
        del te_mapped
        if te_missing or te_unexp:
            raise RuntimeError(f"Qwen text encoder checkpoint mismatch: missing={te_missing}, unexpected={te_unexp}")
        text_encoder.eval()
        # Re-materialize small non-persistent RoPE buffers (left on `meta` by init_empty_weights)
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLTextRotaryEmbedding,
            Qwen3VLVisionRotaryEmbedding,
        )

        text_encoder.model.language_model.rotary_emb = Qwen3VLTextRotaryEmbedding(config=te_cfg.text_config)
        try:
            text_encoder.model.visual.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(te_cfg.vision_config)
        except (TypeError, AttributeError):
            head_dim = te_cfg.vision_config.hidden_size // te_cfg.vision_config.num_heads
            text_encoder.model.visual.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)
        if on_log:
            on_log(
                f"[Qwen3-VL-8B] Loaded '{te_path.name}' "
                f"(missing={len(te_missing)}, unexpected={len(te_unexp)})"
            )

        # 2. Transformer (QwenImage21Transformer2DModel INT8 ConvRot -> ~7.5 GB RAM)
        tr_cfg = QwenImage21Transformer2DModel.load_config(str(cfg_dir / "transformer"), local_files_only=True)
        with init_empty_weights():
            transformer = QwenImage21Transformer2DModel.from_config(tr_cfg)
        tr_raw = load_safetensors_state_dict(dit_path, device="cpu")
        tr_mapped = convert_qwen_image21_transformer_state_dict(tr_raw)
        del tr_raw
        tr_missing, tr_unexp = quantized_module_swap_from_state_dict(
            transformer, tr_mapped, compute_dtype=compute_dtype
        )
        del tr_mapped
        if tr_missing or tr_unexp:
            raise RuntimeError(f"Qwen transformer checkpoint mismatch: missing={tr_missing}, unexpected={tr_unexp}")
        transformer.eval()
        from diffusers.models.transformers.transformer_qwenimage21 import (
            QwenImage21Rope,
            QwenImage21TemporalTimesteps,
        )

        # Re-init tiny non-persistent RoPE / time_proj tensors built under init_empty_weights
        transformer.pos_embed = QwenImage21Rope(
            theta=10000, axes_dim=list(transformer.config.axes_dims_rope)
        )
        transformer.time_text_embed.time_proj = QwenImage21TemporalTimesteps(timestep_dim=256)
        if on_log:
            on_log(
                f"[Qwen-Image-2.1-Turbo] Loaded '{dit_path.name}' "
                f"(32 blocks, missing={len(tr_missing)}, unexpected={len(tr_unexp)})"
            )

        # 3. VAE (AutoencoderKLQwenImage21 BF16 -> ~0.5 GB RAM)
        vae_cfg = AutoencoderKLQwenImage21.load_config(str(cfg_dir / "vae"), local_files_only=True)
        vae = AutoencoderKLQwenImage21.from_config(vae_cfg)
        vae_raw = load_safetensors_state_dict(vae_path, device="cpu")
        vae_mapped = convert_qwen_image21_vae_state_dict(vae_raw)
        del vae_raw
        vae.load_state_dict(
            {k: (v.to(dtype=compute_dtype) if v.is_floating_point() else v) for k, v in vae_mapped.items()},
            strict=True,
            assign=True,
        )
        del vae_mapped
        vae.to(dtype=compute_dtype)
        vae.eval()
        if on_log:
            on_log(f"[Qwen-Image-2.1-VAE] Loaded '{vae_path.name}' (238/238 tensors mapped cleanly)")

        # 4. Processor & Scheduler
        processor = Qwen3VLProcessor.from_pretrained(str(cfg_dir / "processor"), local_files_only=True)
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(cfg_dir / "scheduler"), local_files_only=True)

        self._pipe = QwenImage21Pipeline(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            processor=processor,
            transformer=transformer,
            sample_sigmas=self._sample_sigmas,
        )
        self._pipe.set_progress_bar_config(disable=True)

        # Configure Strata Counter/Pantry splitters (zero weight duplication)
        te_resident = compute_resident_blocks(
            self.config,
            total_blocks=len(text_encoder.model.language_model.layers),
            estimated_model_vram_gb=8.5,
        )
        if te_resident < len(text_encoder.model.language_model.layers):
            self._te_offloader = StrataBlockOffloadManager(
                blocks=text_encoder.model.language_model.layers,
                device=device,
                resident_vram_blocks=te_resident,
            )

        dit_resident = compute_resident_blocks(
            self.config,
            total_blocks=len(transformer.transformer_blocks),
            estimated_model_vram_gb=7.8,
        )
        if dit_resident < len(transformer.transformer_blocks):
            self._dit_offloader = StrataBlockOffloadManager(
                blocks=transformer.transformer_blocks,
                device=device,
                resident_vram_blocks=dit_resident,
            )

    def _move_non_block_modules(self, module: nn.Module, skip_attr: str, device: torch.device) -> None:
        for name, child in module.named_children():
            if name != skip_attr:
                child.to(device)

    def _resolve_sigmas(self, num_steps: int) -> List[float]:
        base = self._sample_sigmas or [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]
        if num_steps <= 0 or num_steps == len(base):
            return list(base)
        if num_steps == 1:
            return [base[0]]
        src_x = np.linspace(0.0, 1.0, len(base))
        dst_x = np.linspace(0.0, 1.0, num_steps)
        return [round(float(v), 6) for v in np.interp(dst_x, src_x, base)]

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        width: int,
        height: int,
        num_steps: int,
        seed: int,
        guidance_scale: float = 1.0,
        negative_prompt: Optional[str] = None,
        reference_images: Optional[List[Image.Image]] = None,
        output_path: Optional[Path] = None,
        on_log: Optional[Callable[[str], None]] = None,
    ) -> Image.Image:
        """Run full 3-stage Strata PyTorch inference for Qwen-Image-2.1-Turbo (T2I or Edit)."""
        if self.memory.sleeping:
            self.memory.wake()
        self.memory.switch_partition("Qwen-Image-2.1")
        self._ensure_loaded(on_log=on_log)

        from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import (
            calculate_dimensions,
            calculate_shift,
            retrieve_timesteps,
        )
        from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache

        pipe = self._pipe
        device, compute_dtype = resolve_compute_device_and_dtype()

        # Round dimensions to multiple of 32 (vae_scale_factor*2 = 16 -> 32 safest for 2x2 tokens)
        width = max(256, (width // 32) * 32)
        height = max(256, (height // 32) * 32)

        do_cfg = negative_prompt is not None and guidance_scale > 1.0
        cond_imgs = reference_images if reference_images else None

        te_input_images: Optional[List[Image.Image]] = None
        vae_images: Optional[List[torch.Tensor]] = None
        input_image_sizes: List[Tuple[int, int]] = []
        if cond_imgs is not None:
            output_res = max(width, height)
            te_input_images = []
            vae_images = []
            for raw_img in cond_imgs:
                img_rgba = raw_img if raw_img.mode == "RGBA" else raw_img.convert("RGBA")
                w0, h0 = img_rgba.size
                rw, rh, _ = calculate_dimensions(output_res * output_res, w0 / h0)
                input_image_sizes.append((rw, rh))
                te_input_images.append(pipe.image_processor.resize(img_rgba, height=rh, width=rw))
                vae_images.append(pipe.image_processor.preprocess(img_rgba, height=rh, width=rw).unsqueeze(2))

        # ---------------- Stage 1: Qwen3-VL-8B Conditioning ----------------
        t2i_cache_key = f"{prompt}|{do_cfg}|{negative_prompt or ''}" if cond_imgs is None else None
        cached = self._prompt_embed_cache.get(t2i_cache_key) if t2i_cache_key else None

        if cached is not None and not do_cfg:
            self.memory.begin_qwen3_vl_stage(cache_hit=True)
            prompt_embeds_cpu, mask_cpu, img_pad_mask_cpu = cached
            prompt_embeds = prompt_embeds_cpu.to(device=device, dtype=compute_dtype)
            prompt_embeds_mask = mask_cpu.to(device=device) if mask_cpu is not None else None
            image_pad_mask = img_pad_mask_cpu.to(device=device)
            if on_log:
                on_log("[Qwen3-VL-8B] Text embedding cache HIT (0 ms text encoder pass)")
            self.memory.evict_qwen3_vl_stage()
        else:
            self.memory.begin_qwen3_vl_stage(cache_hit=False)
            if self._te_offloader is not None:
                pipe.text_encoder.model.visual.to(device)
                pipe.text_encoder.model.language_model.embed_tokens.to(device)
                pipe.text_encoder.model.language_model.norm.to(device)
                pipe.text_encoder.model.language_model.rotary_emb.to(device)
                self._te_offloader.prepare()
                self._te_offloader.attach_hooks()
                if on_log:
                    on_log(
                        f"[Qwen3-VL-8B] Streaming 36 layers ({self._te_offloader.resident_vram_blocks}/36 "
                        "VRAM Counter + pinned RAM Pantry)..."
                    )
            else:
                pipe.text_encoder.to(device)
                if on_log:
                    on_log("[Qwen3-VL-8B] Encoding prompt in INT8 ConvRot on GPU Counter...")

            prompt_embeds, prompt_embeds_mask, image_pad_mask = pipe.encode_prompt(
                prompt=prompt,
                image=te_input_images,
                device=device,
                num_images_per_prompt=1,
            )
            if do_cfg:
                neg_embeds, neg_mask, neg_img_pad_mask = pipe.encode_prompt(
                    prompt=negative_prompt or "",
                    image=te_input_images,
                    device=device,
                    num_images_per_prompt=1,
                )
            else:
                neg_embeds = neg_mask = neg_img_pad_mask = None

            if t2i_cache_key and not do_cfg:
                if len(self._prompt_embed_cache) > 64:
                    self._prompt_embed_cache.clear()
                self._prompt_embed_cache[t2i_cache_key] = (
                    prompt_embeds.detach().cpu(),
                    prompt_embeds_mask.detach().cpu() if prompt_embeds_mask is not None else None,
                    image_pad_mask.detach().cpu(),
                )

            # Evict Stage 1 text encoder completely before DiT starts
            if self._te_offloader is not None:
                self._te_offloader.release_all_to_cpu()
            pipe.text_encoder.to("cpu")
            clear_device_cache(device)
            self.memory.evict_qwen3_vl_stage()
            if on_log:
                on_log("[Qwen3-VL-8B] Extracted multimodal embeddings & evicted text encoder from VRAM")

        # ---------------- Stage 2: QwenImage21Transformer2DModel Denoising ----------------
        if cond_imgs is not None:
            pipe.vae.to(device)

        gen_dev = "cpu" if device.type == "mps" else device
        generator = torch.Generator(device=gen_dev).manual_seed(int(seed))
        num_channels_latents = pipe.transformer.config.in_channels
        latents, input_images_latents = pipe.prepare_latents(
            vae_images,
            1,
            num_channels_latents,
            height,
            width,
            compute_dtype,
            device,
            generator,
        )
        if cond_imgs is not None:
            pipe.vae.to("cpu")
            clear_device_cache(device)

        img_shapes = [
            [
                *[
                    (1, vh // pipe.vae_scale_factor, vw // pipe.vae_scale_factor)
                    for vw, vh in input_image_sizes
                ],
                (1, height // pipe.vae_scale_factor, width // pipe.vae_scale_factor),
            ]
        ]

        sigmas = self._resolve_sigmas(num_steps)
        mu = calculate_shift(
            latents.shape[1],
            pipe.scheduler.config.get("base_image_seq_len", 256),
            pipe.scheduler.config.get("max_image_seq_len", 8192),
            pipe.scheduler.config.get("base_shift", 0.5),
            pipe.scheduler.config.get("max_shift", 0.9),
        )
        timesteps, actual_steps = retrieve_timesteps(pipe.scheduler, len(sigmas), device, sigmas=sigmas, mu=mu)

        def _append_target_slots(mask: torch.Tensor) -> torch.Tensor:
            return torch.cat([mask, mask.new_ones(mask.shape[0], latents.shape[1] // 4)], dim=1)

        image_pad_mask = _append_target_slots(image_pad_mask)
        if do_cfg and neg_img_pad_mask is not None:
            neg_img_pad_mask = _append_target_slots(neg_img_pad_mask)

        # Stage Transformer onto device (100% resident on RTX 5090, or async Pantry streaming)
        if self._dit_offloader is not None:
            self._move_non_block_modules(pipe.transformer, "transformer_blocks", device)
            self._dit_offloader.prepare()
            self._dit_offloader.attach_hooks()
            counter_cnt = self._dit_offloader.resident_vram_blocks
            pantry_cnt = self._dit_offloader.total_blocks - counter_cnt
        else:
            pipe.transformer.to(device)
            counter_cnt = len(pipe.transformer.transformer_blocks)
            pantry_cnt = 0

        if on_log:
            on_log(
                f"[Qwen-Image-2.1-Turbo] Denoising {actual_steps} steps @ {width}x{height} "
                f"({counter_cnt}/32 Counter, {pantry_cnt}/32 Pantry, sigmas={len(sigmas)})"
            )

        num_blocks = len(pipe.transformer.transformer_blocks)
        cache_enabled = bool(pipe.transformer.config.causal_condition)
        cond_cache = QwenImage21KVCache(num_blocks) if cache_enabled else None
        neg_cache = QwenImage21KVCache(num_blocks) if (cache_enabled and do_cfg) else None

        pipe.scheduler.set_begin_index(0)
        for i, t in enumerate(timesteps):
            kv_mode = "extract" if (cache_enabled and i == 0) else ("cached" if cache_enabled else None)
            latent_in = torch.cat([input_images_latents, latents], dim=1) if input_images_latents is not None else latents
            t_vec = t.expand(latents.shape[0]).to(latents.dtype)

            noise_pred = pipe.transformer(
                hidden_states=latent_in,
                timestep=t_vec / 1000.0,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_mask=prompt_embeds_mask,
                img_shapes=img_shapes,
                img_mask=image_pad_mask,
                kv_cache=cond_cache,
                kv_cache_mode=kv_mode,
                return_dict=False,
            )[0]
            noise_pred = noise_pred[:, -latents.size(1) :]

            if do_cfg and neg_embeds is not None:
                neg_pred = pipe.transformer(
                    hidden_states=latent_in,
                    timestep=t_vec / 1000.0,
                    encoder_hidden_states=neg_embeds,
                    encoder_hidden_states_mask=neg_mask,
                    img_shapes=img_shapes,
                    img_mask=neg_img_pad_mask,
                    kv_cache=neg_cache,
                    kv_cache_mode=kv_mode,
                    return_dict=False,
                )[0][:, -latents.size(1) :]
                noise_pred = neg_pred + guidance_scale * (noise_pred - neg_pred)

            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            self.memory.record_dit_block_step(
                step_idx=i,
                total_steps=actual_steps,
                active_block=min(num_blocks - 1, counter_cnt),
                prefetch_block=None if pantry_cnt == 0 else min(num_blocks - 1, counter_cnt + 1),
                dma_ms=0.0,
            )

        if self._dit_offloader is not None:
            self._dit_offloader.release_all_to_cpu()
        else:
            pipe.transformer.to("cpu")
        clear_device_cache(device)

        # ---------------- Stage 3: AutoencoderKLQwenImage21 Decode ----------------
        self.memory.begin_vae_stage()
        pipe.vae.to(device)
        unpacked = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor).to(
            device=device, dtype=compute_dtype
        )
        lmean = (
            torch.tensor(pipe.vae.config.latents_mean, device=device, dtype=compute_dtype)
            .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        )
        lstd = (
            torch.tensor(pipe.vae.config.latents_std, device=device, dtype=compute_dtype)
            .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        )
        unpacked = unpacked * lstd + lmean
        decoded = pipe.vae.decode(unpacked, return_dict=False)[0][:, :, 0]
        pil_img = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
        pipe.vae.to("cpu")
        clear_device_cache(device)
        self.memory.finish_generation()

        if pil_img.mode == "RGBA":
            alpha_arr = np.asarray(pil_img.getchannel("A"))
            if alpha_arr.min() >= 252:
                pil_img = pil_img.convert("RGB")

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pil_img.save(output_path, quality=95)
        return pil_img

    def evict(self) -> None:
        """Unload GPU residency cleanly when switching to a different backend partition."""
        if self._pipe is None:
            return
        if self._dit_offloader is not None:
            self._dit_offloader.release_all_to_cpu()
        if self._te_offloader is not None:
            self._te_offloader.release_all_to_cpu()
        self._pipe.transformer.to("cpu")
        self._pipe.text_encoder.to("cpu")
        self._pipe.vae.to("cpu")
        clear_device_cache()
