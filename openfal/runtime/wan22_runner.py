"""
Real PyTorch / Diffusers inference engine for Alibaba Wan 2.2 2-Expert MoE (27B Total / 14B Active):
  - `Wan2.2-T2V-A14B` (`wan/v2.2/text-to-video`)
  - `Wan2.2-I2V-A14B` (`wan/v2.2/image-to-video`)
  - `Wan2.2-Animate-14B` (`wan/v2.2/animate`, `recast`)

Uses `Comfy-Org/Wan_2.2_ComfyUI_Repackaged` FP8 checkpoints paired with `StrataFP8ScaledLinear`
and `StrataBlockOffloadManager` so:
1. `UMT5-XXL` (FP8, ~5.5 GB) encodes the prompt on GPU then immediately evicts to CPU.
2. High-Noise Expert (14B FP8, ~14.3 GB) runs `t >= boundary_timestep`, then evicts cleanly.
3. Low-Noise Expert (14B FP8, ~14.3 GB) runs `t < boundary_timestep`, then evicts cleanly.
4. `AutoencoderKLWan` (16-ch) decodes the video latents -> frames -> H.264 `.mp4`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image

from openfal.config import EngineConfig
from openfal.engine.memory_hierarchy import StrataMemoryHierarchy
from openfal.runtime.common import (
    check_inference_dependencies,
    clear_device_cache,
    compute_resident_blocks,
    encode_frames_to_mp4,
    ensure_model_configs,
    find_matching_safetensors,
    load_safetensors_state_dict,
    resolve_compute_device_and_dtype,
)
from openfal.runtime.attention import prefer_cudnn_sdpa
from openfal.runtime.quant import (
    PinnedModuleStager,
    StrataBlockOffloadManager,
    quantized_module_swap_from_state_dict,
)


# Wan's published defaults explicitly suppress overexposure, stylization and
# static frames. See Wan-Video/Wan2.2/wan/configs/shared_config.py.
WAN_DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)
WAN_FLOW_SHIFTS = {"Wan2.2-T2V-A14B": 12.0, "Wan2.2-I2V-A14B": 5.0}


def _set_wan_timesteps(scheduler: Any, steps: int, device: torch.device) -> None:
    """Match Wan's native UniPC sigma grid, including its initial noise level.

    Wan starts at float32(1 - 1 / training_steps) and ends at zero. Diffusers'
    default flow grid starts at one and ends at 1 / training_steps instead.
    Supply the unshifted grid; the scheduler applies its configured flow_shift.
    """
    import numpy as np

    sigma_max = float(np.float32(1.0 - 1.0 / scheduler.config.num_train_timesteps))
    sigmas = np.linspace(sigma_max, 0.0, steps + 1)[:-1]
    scheduler.set_timesteps(sigmas=sigmas, device=device)


def convert_wan22_transformer_state_dict(raw_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert a ComfyUI Wan 2.2 quantized single-file checkpoint (`FP8` `.scale_weight` or
    `INT8 ConvRot` `.weight_scale` + `.comfy_quant`) to Diffusers `WanTransformer3DModel`
    or `WanAnimateTransformer3DModel` naming.
    """
    check_inference_dependencies(("diffusers",))
    from diffusers.loaders.single_file_utils import convert_wan_transformer_to_diffusers

    cleaned: Dict[str, torch.Tensor] = {}
    for k, v in raw_sd.items():
        if k == "scaled_fp8" or k.endswith((".scale_input", ".input_scale", ".comfy_attention.config")):
            continue
        cleaned[k] = v
    converted = convert_wan_transformer_to_diffusers(cleaned)

    # Map WanAnimateMotionEncoder 4x4 blur_kernel buffers (`convs.1..7.{conv2,skip}.0.kernel`)
    for i in range(7):
        conv_idx = i + 1
        k_c2 = f"motion_encoder.enc.net_app.convs.{conv_idx}.conv2.0.kernel"
        k_sk = f"motion_encoder.enc.net_app.convs.{conv_idx}.skip.0.kernel"
        if k_c2 in converted:
            converted[f"motion_encoder.res_blocks.{i}.conv2.blur_kernel"] = converted.pop(k_c2)
        if k_sk in converted:
            converted[f"motion_encoder.res_blocks.{i}.conv_skip.blur_kernel"] = converted.pop(k_sk)
    return converted


def _materialize_remaining_meta(module: nn.Module, dtype: torch.dtype) -> None:
    """Reject incomplete checkpoint loads instead of inventing trained weights."""
    remaining = [
        name
        for name, tensor in list(module.named_parameters()) + list(module.named_buffers())
        if tensor.is_meta
    ]
    if remaining:
        raise RuntimeError(f"Wan checkpoint left uninitialized tensors: {remaining[:12]}")


def _validate_checkpoint_keys(missing: List[str], unexpected: List[str], label: str) -> None:
    if missing or unexpected:
        raise RuntimeError(
            f"{label} checkpoint is incompatible: missing={missing[:12]}, unexpected={unexpected[:12]}"
        )


def convert_umt5_fp8_state_dict(raw_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip Comfy metadata tensors from `umt5_xxl_fp8_e4m3fn_scaled.safetensors` for UMT5EncoderModel."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in raw_sd.items():
        if k in ("spiece_model", "scaled_fp8") or k.endswith((".scale_input", ".input_scale")):
            continue
        out[k] = v
    if "shared.weight" in out and "encoder.embed_tokens.weight" not in out:
        out["encoder.embed_tokens.weight"] = out["shared.weight"]
    return out


class RealWan22Runner:
    """Real neural network inference runner for Wan 2.2 14B 2-Expert MoE video generation."""

    REPO_MAP = {
        "Wan2.2-T2V-A14B": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "Wan2.2-I2V-A14B": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "Wan2.2-Animate-14B": "Wan-AI/Wan2.2-Animate-14B-Diffusers",
    }

    def __init__(self, config: EngineConfig, memory: StrataMemoryHierarchy):
        self.config = config
        self.memory = memory
        self.weights_dir = config.weights_dir

        self._tokenizer: Any = None
        self._text_encoder: Any = None
        self._te_stager: Optional[PinnedModuleStager] = None
        self._vae: Any = None
        self._experts: Dict[str, Tuple[Any, Optional[StrataBlockOffloadManager]]] = {}
        self._expert_stagers: Dict[str, PinnedModuleStager] = {}
        self._prompt_embed_cache: Dict[str, torch.Tensor] = {}
        self._cfg_dirs: Dict[str, Path] = {}

    def _search_roots(self) -> List[Path]:
        return [
            self.weights_dir / "Wan2.2-T2V-A14B",
            self.weights_dir / "Wan2.2-I2V-A14B",
            self.weights_dir / "Wan2.2-Animate-14B",
            self.weights_dir,
        ]

    def _find_shared_umt5(self) -> Optional[Path]:
        for root in self._search_roots():
            hit = find_matching_safetensors(root, ["umt5_xxl"])
            if hit is not None:
                return hit
        return None

    def _find_16ch_vae(self) -> Optional[Path]:
        # File names are not architecture guarantees: older installations contain
        # a 48-channel TI2V-5B VAE under a generic Wan 2.2 name.
        from safetensors import safe_open

        checked: set[Path] = set()
        for root in self._search_roots():
            for candidate in root.rglob("*.safetensors"):
                if candidate in checked:
                    continue
                checked.add(candidate)
                named_wan_vae = candidate.name in ("wan_2.1_vae.safetensors", "wan2.2_vae.safetensors")
                native_vae = candidate.name == "diffusion_pytorch_model.safetensors" and candidate.parent.name == "vae"
                if not (named_wan_vae or native_vae):
                    continue
                with safe_open(str(candidate), framework="pt", device="cpu") as checkpoint:
                    for key in ("conv2.weight", "post_quant_conv.weight"):
                        if key not in checkpoint.keys():
                            continue
                        shape = checkpoint.get_slice(key).get_shape()
                        if len(shape) == 5 and shape[:2] == [16, 16]:
                            return candidate
        return None

    def _resolve_expert_paths(self, variant: str) -> Tuple[Optional[Path], Optional[Path]]:
        if variant == "Wan2.2-I2V-A14B":
            root = self.weights_dir / "Wan2.2-I2V-A14B"
            hi = find_matching_safetensors(root, ["i2v", "high_noise"])
            lo = find_matching_safetensors(root, ["i2v", "low_noise"])
            return hi, lo
        if variant == "Wan2.2-Animate-14B":
            root = self.weights_dir / "Wan2.2-Animate-14B"
            single = find_matching_safetensors(root, ["wan2.2_animate"])
            return single, None
        root = self.weights_dir / "Wan2.2-T2V-A14B"
        hi = find_matching_safetensors(root, ["t2v", "high_noise"])
        lo = find_matching_safetensors(root, ["t2v", "low_noise"])
        return hi, lo

    def has_weights_for(self, variant: str) -> bool:
        hi, lo = self._resolve_expert_paths(variant)
        return hi is not None and (variant == "Wan2.2-Animate-14B" or lo is not None)

    def _ensure_configs(self, variant: str, on_log: Optional[Callable[[str], None]] = None) -> Path:
        if variant in self._cfg_dirs:
            return self._cfg_dirs[variant]
        component = {
            "Wan2.2-T2V-A14B": "wan22_t2v_moe",
            "Wan2.2-I2V-A14B": "wan22_i2v_moe",
            "Wan2.2-Animate-14B": "wan22_animate",
        }[variant]
        target = ensure_model_configs(component, self.weights_dir, on_log=on_log)
        self._cfg_dirs[variant] = target
        return target

    def _ensure_16ch_vae_file(self, on_log: Optional[Callable[[str], None]] = None) -> Path:
        existing = self._find_16ch_vae()
        if existing is not None:
            return existing
        raise RuntimeError(
            "Wan 2.2 A14B requires the 16-channel wan_2.1_vae.safetensors; "
            "the 48-channel TI2V-5B VAE is incompatible. "
            "Run: openfal download --component wan22_t2v_moe --runtime-assets-only"
        )

    def _load_shared_text_encoder(self, cfg_dir: Path, on_log: Optional[Callable[[str], None]] = None) -> None:
        if self._text_encoder is not None:
            return
        umt5_path = self._find_shared_umt5()
        if umt5_path is None:
            raise RuntimeError(
                f"Wan 2.2 UMT5 text encoder not found under '{self.weights_dir}'. "
                "Run: openfal download --component wan22_t2v_moe"
            )

        from accelerate import init_empty_weights
        from transformers import T5TokenizerFast, UMT5Config, UMT5EncoderModel

        _, compute_dtype = resolve_compute_device_and_dtype()
        self._tokenizer = T5TokenizerFast.from_pretrained(str(cfg_dir / "tokenizer"), local_files_only=True)
        te_cfg = UMT5Config.from_pretrained(str(cfg_dir / "text_encoder"), local_files_only=True)
        with init_empty_weights():
            te = UMT5EncoderModel(te_cfg)
        raw_sd = load_safetensors_state_dict(umt5_path, device="cpu")
        mapped_sd = convert_umt5_fp8_state_dict(raw_sd)
        del raw_sd
        missing, unexp = quantized_module_swap_from_state_dict(te, mapped_sd, compute_dtype=compute_dtype)
        del mapped_sd
        _validate_checkpoint_keys(missing, unexp, "Wan UMT5")
        _materialize_remaining_meta(te, compute_dtype)
        te.eval()
        if on_log:
            on_log(
                f"[Wan2.2 UMT5-XXL] Loaded '{umt5_path.name}' in FP8 "
                f"(missing={len(missing)}, unexpected={len(unexp)})"
            )
        self._text_encoder = te

    def _load_shared_vae(self, cfg_dir: Path, on_log: Optional[Callable[[str], None]] = None) -> None:
        if self._vae is not None:
            return
        from diffusers import AutoencoderKLWan
        from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers

        vae_path = self._ensure_16ch_vae_file(on_log=on_log)
        vae_cfg = AutoencoderKLWan.load_config(str(cfg_dir / "vae"), local_files_only=True)
        vae = AutoencoderKLWan.from_config(vae_cfg)
        vae_raw = load_safetensors_state_dict(vae_path, device="cpu")
        vae_sd = convert_wan_vae_to_diffusers(vae_raw)
        del vae_raw
        vae.load_state_dict(
            {k: (v.to(dtype=torch.float32) if v.is_floating_point() else v) for k, v in vae_sd.items()},
            strict=True,
            assign=True,
        )
        del vae_sd
        vae.eval()
        self._vae = vae
        if on_log:
            on_log(f"[Wan2.2-VAE] Loaded 16-ch AutoencoderKLWan from '{vae_path.name}'")

    def _get_expert(
        self,
        key: str,
        ckpt_path: Path,
        cfg_dir: Path,
        on_log: Optional[Callable[[str], None]] = None,
    ) -> Tuple[Any, Optional[StrataBlockOffloadManager]]:
        if key in self._experts:
            return self._experts[key]

        from accelerate import init_empty_weights
        from diffusers import WanAnimateTransformer3DModel, WanTransformer3DModel
        from diffusers.models.transformers.transformer_wan import WanRotaryPosEmbed

        device, compute_dtype = resolve_compute_device_and_dtype()
        is_animate = "Animate" in key or "animate" in ckpt_path.name.lower()
        model_cls = WanAnimateTransformer3DModel if is_animate else WanTransformer3DModel
        config_subdir = "transformer_2" if key.endswith(":low") else "transformer"
        tr_cfg = model_cls.load_config(str(cfg_dir / config_subdir), local_files_only=True)
        with init_empty_weights():
            model = model_cls.from_config(tr_cfg)

        raw_sd = load_safetensors_state_dict(ckpt_path, device="cpu")
        mapped_sd = convert_wan22_transformer_state_dict(raw_sd)
        del raw_sd
        missing, unexp = quantized_module_swap_from_state_dict(model, mapped_sd, compute_dtype=compute_dtype)

        # Populate non-persistent blur_kernel buffers on WanAnimateMotionEncoder if present
        if is_animate and hasattr(model, "motion_encoder"):
            from diffusers.models.transformers.transformer_wan_animate import WanAnimateMotionEncoder

            fresh_me = WanAnimateMotionEncoder(
                size=model.config.motion_encoder_size,
                style_dim=model.config.motion_style_dim,
                motion_dim=model.config.motion_dim,
                out_dim=model.config.motion_encoder_dim,
                channels=model.config.motion_encoder_channel_sizes,
            )
            for idx, blk in enumerate(model.motion_encoder.res_blocks):
                c2_k = f"motion_encoder.res_blocks.{idx}.conv2.blur_kernel"
                sk_k = f"motion_encoder.res_blocks.{idx}.conv_skip.blur_kernel"
                blk.conv2._buffers["blur_kernel"] = (
                    mapped_sd[c2_k].to(dtype=compute_dtype, device="cpu")
                    if c2_k in mapped_sd
                    else fresh_me.res_blocks[idx].conv2.blur_kernel.to(dtype=compute_dtype, device="cpu")
                )
                blk.conv_skip._buffers["blur_kernel"] = (
                    mapped_sd[sk_k].to(dtype=compute_dtype, device="cpu")
                    if sk_k in mapped_sd
                    else fresh_me.res_blocks[idx].conv_skip.blur_kernel.to(dtype=compute_dtype, device="cpu")
                )
            unexp = [k for k in unexp if not k.endswith(".blur_kernel")]
        del mapped_sd

        _validate_checkpoint_keys(missing, unexp, f"Wan expert {ckpt_path.name}")

        # Re-init tiny non-persistent RoPE frequency buffers
        model.rope = WanRotaryPosEmbed(
            attention_head_dim=model.config.attention_head_dim,
            patch_size=model.config.patch_size,
            max_seq_len=model.config.rope_max_seq_len,
        )
        _materialize_remaining_meta(model, compute_dtype)
        model.eval()

        num_blocks = len(model.blocks)
        resident = compute_resident_blocks(self.config, total_blocks=num_blocks, estimated_model_vram_gb=14.3)
        offloader: Optional[StrataBlockOffloadManager] = None
        if resident < num_blocks:
            offloader = StrataBlockOffloadManager(
                blocks=model.blocks,
                device=device,
                resident_vram_blocks=resident,
            )

        self._experts[key] = (model, offloader)
        if on_log:
            on_log(
                f"[Wan2.2 MoE] Prepared expert '{ckpt_path.name}' "
                f"({resident}/{num_blocks} Counter blocks, missing={len(missing)}, unexpected={len(unexp)})"
            )
        return model, offloader

    def _activate_expert(
        self,
        model: nn.Module,
        offloader: Optional[StrataBlockOffloadManager],
        device: torch.device,
        expert_key: Optional[str] = None,
    ) -> None:
        if offloader is not None:
            for p in model.parameters(recurse=False):
                p.data = p.data.to(device)
            for bname, buf in list(model._buffers.items()):
                if buf is not None:
                    model._buffers[bname] = buf.to(device)
            for name, child in model.named_children():
                if name != "blocks":
                    child.to(device)
            offloader.prepare()
            offloader.attach_hooks()
        elif expert_key is not None and device.type == "cuda":
            stager = self._expert_stagers.get(expert_key)
            if stager is None:
                stager = PinnedModuleStager(model, device)
                self._expert_stagers[expert_key] = stager
            stager.stage_to_device(device)
        else:
            model.to(device)

    def _deactivate_expert(
        self,
        model: nn.Module,
        offloader: Optional[StrataBlockOffloadManager],
        device: torch.device,
        expert_key: Optional[str] = None,
    ) -> None:
        if offloader is not None:
            offloader.release_all_to_cpu()
            model.to("cpu")
        elif expert_key is not None and expert_key in self._expert_stagers:
            self._expert_stagers[expert_key].evict_to_cpu()
        else:
            model.to("cpu")
        clear_device_cache(device)

    @torch.inference_mode()
    def generate_video(
        self,
        variant: str,
        prompt: str,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        num_steps: int,
        seed: int,
        guidance_scale: float = 4.0,
        guidance_scale_2: Optional[float] = None,
        negative_prompt: str = "",
        start_image: Optional[Image.Image] = None,
        output_path: Optional[Path] = None,
        on_log: Optional[Callable[[str], None]] = None,
    ) -> Path:
        """Execute full 3-stage Stage-wise MoE Wan 2.2 video generation -> `.mp4`."""
        if variant == "Wan2.2-Animate-14B":
            raise ValueError(
                "Wan Animate uses RealWanAnimateRunner with pose/face conditioning. "
                "Use the wan/v2.2/animate or wan/v2.2/recast API with image_url and driving_video_url."
            )
        if variant == "Wan2.2-I2V-A14B" and start_image is None:
            raise ValueError("Wan image-to-video requires a start image.")
        if self.memory.sleeping:
            self.memory.wake()
        self.memory.switch_partition(variant)

        check_inference_dependencies(("diffusers", "transformers", "accelerate"))

        hi_path, lo_path = self._resolve_expert_paths(variant)
        if hi_path is None or lo_path is None:
            raise RuntimeError(
                f"Missing Wan 2.2 high/low-noise expert weights for '{variant}' under '{self.weights_dir}'. "
                "Run: openfal download --component alibaba-only"
            )

        cfg_dir = self._ensure_configs(variant, on_log=on_log)
        device, compute_dtype = resolve_compute_device_and_dtype()

        from diffusers import UniPCMultistepScheduler
        from diffusers.video_processor import VideoProcessor

        # Align spatial dims to /16 and frames to (4*k + 1)
        width = max(256, (width // 16) * 16)
        height = max(256, (height // 16) * 16)
        num_frames = max(17, min(81, ((max(17, num_frames) - 1) // 4) * 4 + 1))

        self._load_shared_text_encoder(cfg_dir, on_log=on_log)
        self._load_shared_vae(cfg_dir, on_log=on_log)

        # A14B's native schedules differ by task, not output resolution.
        flow_shift = WAN_FLOW_SHIFTS[variant]
        scheduler = UniPCMultistepScheduler.from_pretrained(
            str(cfg_dir / "scheduler"), flow_shift=flow_shift, local_files_only=True
        )

        import json

        model_index = json.loads((cfg_dir / "model_index.json").read_text(encoding="utf-8"))
        boundary_ratio = model_index.get("boundary_ratio", 0.875 if "T2V" in variant else 0.9)
        if lo_path is None:
            boundary_ratio = None

        # -------- Stage 1: UMT5-XXL Prompt Encoding + Eviction --------
        do_cfg = guidance_scale > 1.0 or (guidance_scale_2 is not None and guidance_scale_2 > 1.0)
        neg_prompt = negative_prompt or WAN_DEFAULT_NEGATIVE_PROMPT
        use_te_cache = bool(self.config.enable_text_encoder_cache)
        pos_key = prompt.strip()
        neg_key = neg_prompt.strip()
        cached_pos = self._prompt_embed_cache.get(pos_key) if use_te_cache else None
        cached_neg = self._prompt_embed_cache.get(neg_key) if (use_te_cache and do_cfg) else None
        all_cached = cached_pos is not None and (not do_cfg or cached_neg is not None)

        if all_cached:
            self.memory.begin_qwen3_vl_stage(cache_hit=True)
            prompt_embeds = cached_pos.to(device=device, dtype=compute_dtype)
            neg_embeds = cached_neg.to(device=device, dtype=compute_dtype) if do_cfg and cached_neg is not None else None
            if on_log:
                on_log("[Wan2.2 UMT5-XXL] Text embedding cache HIT (0 ms text encoder pass)")
            self.memory.evict_qwen3_vl_stage()
        else:
            self.memory.begin_qwen3_vl_stage(cache_hit=False)
            if device.type == "cuda":
                if self._te_stager is None:
                    self._te_stager = PinnedModuleStager(self._text_encoder, device)
                self._te_stager.stage_to_device(device)
            else:
                self._text_encoder.to(device)
            if on_log:
                on_log("[Wan2.2 UMT5-XXL] Encoding prompt on GPU Counter (FP8)...")

            def _encode_t5(texts: List[str]) -> torch.Tensor:
                clean = [t.strip() for t in texts]
                text_inputs = self._tokenizer(
                    clean,
                    padding="max_length",
                    max_length=512,
                    truncation=True,
                    add_special_tokens=True,
                    return_attention_mask=True,
                    return_tensors="pt",
                )
                ids = text_inputs.input_ids.to(device)
                mask = text_inputs.attention_mask.to(device)
                seq_lens = mask.gt(0).sum(dim=1).long()
                with prefer_cudnn_sdpa(device):
                    embeds = self._text_encoder(ids, mask).last_hidden_state.to(compute_dtype)
                # Zero out padding and re-pad per official Wan convention
                trimmed = [u[:v] for u, v in zip(embeds, seq_lens)]
                padded = [torch.cat([u, u.new_zeros(512 - u.size(0), u.size(1))]) for u in trimmed]
                return torch.stack(padded, dim=0)

            prompt_embeds = cached_pos.to(device=device, dtype=compute_dtype) if cached_pos is not None else _encode_t5([prompt])
            if do_cfg:
                neg_embeds = cached_neg.to(device=device, dtype=compute_dtype) if cached_neg is not None else _encode_t5([neg_prompt])
            else:
                neg_embeds = None

            if use_te_cache:
                if len(self._prompt_embed_cache) > 64:
                    self._prompt_embed_cache.clear()
                if cached_pos is None:
                    self._prompt_embed_cache[pos_key] = prompt_embeds.detach().cpu()
                if do_cfg and neg_embeds is not None and cached_neg is None:
                    self._prompt_embed_cache[neg_key] = neg_embeds.detach().cpu()

            if self._te_stager is not None:
                self._te_stager.evict_to_cpu()
            else:
                self._text_encoder.to("cpu")
            clear_device_cache(device)
            self.memory.evict_qwen3_vl_stage()
            if on_log:
                on_log("[Wan2.2 UMT5-XXL] Evicted UMT5-XXL from VRAM (0 MB resident)")

        # -------- Prepare Latents (+ I2V Conditioning) --------
        gen_dev = "cpu" if device.type == "mps" else device
        generator = torch.Generator(device=gen_dev).manual_seed(int(seed))
        z_dim = self._vae.config.z_dim
        t_scale = self._vae.config.scale_factor_temporal if hasattr(self._vae.config, "scale_factor_temporal") else 4
        s_scale = self._vae.config.scale_factor_spatial if hasattr(self._vae.config, "scale_factor_spatial") else 8
        latent_f = (num_frames - 1) // t_scale + 1
        latent_h = height // s_scale
        latent_w = width // s_scale

        latents = torch.randn(
            (1, z_dim, latent_f, latent_h, latent_w),
            generator=generator,
            device=device if gen_dev != "cpu" else "cpu",
            dtype=torch.float32,
        ).to(device)

        # The reference VAE runs in FP32 even with a BF16/FP8 transformer.
        vae_dtype = self._vae.dtype
        lmean = torch.tensor(self._vae.config.latents_mean, device=device, dtype=vae_dtype).view(1, z_dim, 1, 1, 1)
        lstd = 1.0 / torch.tensor(self._vae.config.latents_std, device=device, dtype=vae_dtype).view(
            1, z_dim, 1, 1, 1
        )

        condition: Optional[torch.Tensor] = None

        if variant == "Wan2.2-I2V-A14B":
            video_proc = VideoProcessor(vae_scale_factor=s_scale)
            img_rgb = start_image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
            img_tensor = video_proc.preprocess(img_rgb, height=height, width=width).to(device, dtype=vae_dtype)
            vid_cond = torch.cat(
                [
                    img_tensor[:, :, None],
                    img_tensor.new_zeros(1, 3, num_frames - 1, height, width),
                ],
                dim=2,
            )
            self._vae.to(device)
            enc = self._vae.encode(vid_cond).latent_dist.mode()
            self._vae.to("cpu")
            clear_device_cache(device)

            enc = (enc - lmean.to(enc.dtype)) * lstd.to(enc.dtype)

            mask_lat = torch.ones(1, 1, num_frames, latent_h, latent_w, device=device, dtype=enc.dtype)
            mask_lat[:, :, 1:] = 0
            first_four = mask_lat[:, :, :1].repeat(1, 4, 1, 1, 1)
            rest = mask_lat[:, :, 1:].reshape(1, -1, 4, latent_h, latent_w).permute(0, 2, 1, 3, 4)
            mask_packed = torch.cat([first_four, rest], dim=2)
            condition = torch.cat([mask_packed, enc], dim=1)

        # -------- Stage 2: Two-Expert MoE Denoising --------
        steps = max(4, int(num_steps))
        _set_wan_timesteps(scheduler, steps, device)
        timesteps = scheduler.timesteps
        boundary_ts = (boundary_ratio * scheduler.config.num_train_timesteps) if boundary_ratio else None
        g2 = guidance_scale_2 if guidance_scale_2 is not None else guidance_scale

        active_key: Optional[str] = None
        active_tuple: Optional[Tuple[Any, Optional[StrataBlockOffloadManager]]] = None

        if on_log:
            on_log(
                f"[Wan2.2 MoE] Denoising {steps} steps ({num_frames} frames @ {width}x{height}, "
                f"boundary={boundary_ts})"
            )

        with prefer_cudnn_sdpa(device):
            for i, t in enumerate(timesteps):
                is_high = (boundary_ts is None) or (float(t.item()) >= boundary_ts)
                expert_key = f"{variant}:high" if is_high else f"{variant}:low"
                expert_path = hi_path if is_high else (lo_path or hi_path)

                if active_key != expert_key:
                    if active_tuple is not None:
                        self._deactivate_expert(active_tuple[0], active_tuple[1], device, expert_key=active_key)
                    active_tuple = self._get_expert(expert_key, expert_path, cfg_dir, on_log=on_log)
                    self._activate_expert(active_tuple[0], active_tuple[1], device, expert_key=expert_key)
                    active_key = expert_key

                model, offloader = active_tuple
                cur_cfg = guidance_scale if is_high else g2
                model_in = (
                    torch.cat([latents.to(compute_dtype), condition.to(compute_dtype)], dim=1)
                    if condition is not None
                    else latents.to(compute_dtype)
                )
                ts_in = t.expand(latents.shape[0])

                # Batch cond + uncond CFG in one forward pass on >=24 GB CUDA GPUs when DiT is fully resident
                batch_cfg = (
                    do_cfg
                    and neg_embeds is not None
                    and device.type == "cuda"
                    and offloader is None
                    and self.config.profile.total_vram_gb >= 24.0
                )
                if batch_cfg:
                    pred_both = model(
                        hidden_states=torch.cat([model_in, model_in], dim=0),
                        timestep=ts_in.expand(2),
                        encoder_hidden_states=torch.cat([prompt_embeds, neg_embeds], dim=0),
                        return_dict=False,
                    )[0]
                    noise_pred_cond, noise_uncond = pred_both.chunk(2, dim=0)
                    noise_pred = noise_uncond + cur_cfg * (noise_pred_cond - noise_uncond)
                else:
                    noise_pred = model(
                        hidden_states=model_in,
                        timestep=ts_in,
                        encoder_hidden_states=prompt_embeds,
                        return_dict=False,
                    )[0]
                    if do_cfg and neg_embeds is not None:
                        noise_uncond = model(
                            hidden_states=model_in,
                            timestep=ts_in,
                            encoder_hidden_states=neg_embeds,
                            return_dict=False,
                        )[0]
                        noise_pred = noise_uncond + cur_cfg * (noise_pred - noise_uncond)

                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                self.memory.record_dit_block_step(
                    step_idx=i,
                    total_steps=steps,
                    active_block=(offloader.resident_vram_blocks if offloader else len(model.blocks)) - 1,
                    prefetch_block=None,
                    dma_ms=0.0,
                )

        if active_tuple is not None:
            self._deactivate_expert(active_tuple[0], active_tuple[1], device, expert_key=active_key)

        # -------- Stage 3: AutoencoderKLWan Decode -> H.264 MP4 --------
        self.memory.begin_vae_stage()
        self._vae.to(device)
        z_lat = latents.to(device=device, dtype=vae_dtype)
        z_lat = z_lat / lstd + lmean
        with prefer_cudnn_sdpa(device):
            decoded = self._vae.decode(z_lat, return_dict=False)[0]
        video_proc = VideoProcessor(vae_scale_factor=s_scale)
        frames = video_proc.postprocess_video(decoded, output_type="pil")[0]
        self._vae.to("cpu")
        clear_device_cache(device)
        self.memory.finish_generation()

        if output_path is not None:
            encode_frames_to_mp4(frames=frames, out_path=output_path, fps=fps)
        return output_path or Path("output.mp4")

    def evict(self) -> None:
        device, _ = resolve_compute_device_and_dtype()
        for key, (model, offloader) in self._experts.items():
            self._deactivate_expert(model, offloader, device, expert_key=key)
        for stager in self._expert_stagers.values():
            stager.release()
        self._expert_stagers.clear()
        if self._te_stager is not None:
            self._te_stager.release()
            self._te_stager = None
        if self._text_encoder is not None:
            self._text_encoder.to("cpu")
        if self._vae is not None:
            self._vae.to("cpu")
        clear_device_cache(device)
