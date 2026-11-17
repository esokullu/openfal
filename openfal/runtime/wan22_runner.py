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
from huggingface_hub import hf_hub_download
from PIL import Image

from openfal.config import EngineConfig
from openfal.engine.memory_hierarchy import StrataMemoryHierarchy
from openfal.runtime.common import (
    check_inference_dependencies,
    clear_device_cache,
    compute_resident_blocks,
    encode_frames_to_mp4,
    ensure_hf_configs,
    find_matching_safetensors,
    load_safetensors_state_dict,
    resolve_compute_device_and_dtype,
)
from openfal.runtime.quant import StrataBlockOffloadManager, quantized_module_swap_from_state_dict


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
    """Replace any unpopulated meta parameters or buffers on CPU."""
    from openfal.runtime.quant import _get_parent_and_attr

    for name, param in list(module.named_parameters(recurse=True)):
        if param.is_meta:
            fill = torch.ones if "norm" in name else torch.zeros
            target_dt = dtype if param.is_floating_point() else param.dtype
            parent, attr = _get_parent_and_attr(module, name)
            setattr(
                parent,
                attr,
                nn.Parameter(fill(param.shape, dtype=target_dt, device="cpu"), requires_grad=False),
            )
    for mod in module.modules():
        for bname, buf in list(mod._buffers.items()):
            if buf is not None and buf.is_meta:
                mod._buffers[bname] = torch.zeros(buf.shape, dtype=buf.dtype, device="cpu")


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
        self._vae: Any = None
        self._experts: Dict[str, Tuple[Any, Optional[StrataBlockOffloadManager]]] = {}
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
        # Prefer explicit wan_2.1_vae (16-ch) over wan2.2_vae (which is 48-ch TI2V-5B VAE)
        for root in self._search_roots():
            for p in root.rglob("*.safetensors"):
                if p.name in ("wan_2.1_vae.safetensors", "diffusion_pytorch_model.safetensors") and "vae" in str(
                    p.parent
                ).lower():
                    return p
        # Check wan2.2_vae.safetensors ONLY if its conv2.weight actually has 16 channels
        from safetensors import safe_open

        for root in self._search_roots():
            cand = find_matching_safetensors(root, ["wan2.2_vae"])
            if cand is not None:
                with safe_open(str(cand), framework="pt", device="cpu") as f:
                    if "conv2.weight" in f.keys() and f.get_slice("conv2.weight").get_shape()[0] == 16:
                        return cand
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
        hi, _ = self._resolve_expert_paths(variant)
        return hi is not None

    def _ensure_configs(self, variant: str, on_log: Optional[Callable[[str], None]] = None) -> Path:
        if variant in self._cfg_dirs:
            return self._cfg_dirs[variant]
        repo_id = self.REPO_MAP.get(variant, self.REPO_MAP["Wan2.2-T2V-A14B"])
        target = self.weights_dir / variant / "hf_configs"
        ensure_hf_configs(
            repo_id=repo_id,
            target_dir=target,
            required_files=[
                "model_index.json",
                "scheduler/scheduler_config.json",
                "transformer/config.json",
                "vae/config.json",
                "text_encoder/config.json",
                "tokenizer/tokenizer_config.json",
                "tokenizer/spiece.model",
            ],
            allow_patterns=[
                "model_index.json",
                "scheduler/*",
                "transformer/*.json",
                "transformer_2/*.json",
                "vae/*.json",
                "text_encoder/*.json",
                "tokenizer/*",
            ],
            on_log=on_log,
        )
        self._cfg_dirs[variant] = target
        return target

    def _ensure_16ch_vae_file(self, on_log: Optional[Callable[[str], None]] = None) -> Path:
        existing = self._find_16ch_vae()
        if existing is not None:
            return existing
        dest_dir = self.weights_dir / "Wan2.2-T2V-A14B"
        if on_log:
            on_log(
                "[Wan2.2-VAE] Existing 'wan2.2_vae.safetensors' is 48-ch (TI2V-5B). Fetching "
                "16-ch 'split_files/vae/wan_2.1_vae.safetensors' (~384 MB) for Wan 2.2 14B..."
            )
        downloaded = hf_hub_download(
            repo_id="Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            filename="split_files/vae/wan_2.1_vae.safetensors",
            local_dir=str(dest_dir),
        )
        return Path(downloaded)

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
        self._tokenizer = T5TokenizerFast.from_pretrained(str(cfg_dir / "tokenizer"))
        te_cfg = UMT5Config.from_pretrained(str(cfg_dir / "text_encoder"))
        with init_empty_weights():
            te = UMT5EncoderModel(te_cfg)
        raw_sd = load_safetensors_state_dict(umt5_path, device="cpu")
        mapped_sd = convert_umt5_fp8_state_dict(raw_sd)
        del raw_sd
        missing, unexp = quantized_module_swap_from_state_dict(te, mapped_sd, compute_dtype=compute_dtype)
        del mapped_sd
        _materialize_remaining_meta(te, compute_dtype)
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

        _, compute_dtype = resolve_compute_device_and_dtype()
        vae_path = self._ensure_16ch_vae_file(on_log=on_log)
        vae_cfg = AutoencoderKLWan.load_config(str(cfg_dir / "vae"))
        vae = AutoencoderKLWan.from_config(vae_cfg)
        vae_raw = load_safetensors_state_dict(vae_path, device="cpu")
        vae_sd = convert_wan_vae_to_diffusers(vae_raw)
        del vae_raw
        vae.load_state_dict(
            {k: (v.to(dtype=compute_dtype) if v.is_floating_point() else v) for k, v in vae_sd.items()},
            strict=False,
            assign=True,
        )
        del vae_sd
        vae.to(dtype=compute_dtype)
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
        tr_cfg = model_cls.load_config(str(cfg_dir / "transformer"))
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

        # Re-init tiny non-persistent RoPE frequency buffers
        model.rope = WanRotaryPosEmbed(
            attention_head_dim=model.config.attention_head_dim,
            patch_size=model.config.patch_size,
            max_seq_len=model.config.rope_max_seq_len,
        )
        _materialize_remaining_meta(model, compute_dtype)

        num_blocks = len(model.blocks)
        resident = min(
            num_blocks,
            min(
                self.config.profile.resident_vram_blocks,
                compute_resident_blocks(self.config, total_blocks=num_blocks, estimated_model_vram_gb=14.3),
            ),
        )
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

    def _activate_expert(self, model: nn.Module, offloader: Optional[StrataBlockOffloadManager], device: torch.device) -> None:
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
        else:
            model.to(device)

    def _deactivate_expert(self, model: nn.Module, offloader: Optional[StrataBlockOffloadManager], device: torch.device) -> None:
        if offloader is not None:
            offloader.release_all_to_cpu()
        model.to("cpu")
        clear_device_cache(device)

    @staticmethod
    def _extract_animate_image_embeds(
        ref_img: Image.Image,
        device: torch.device,
        dtype: torch.dtype,
        image_dim: int = 1280,
    ) -> torch.Tensor:
        """
        Extract `(1, 257, 1280)` visual patch tokens from the reference character image
        for `WanAnimateTransformer3DModel.condition_embedder.image_embedder`.
        """
        import numpy as np

        img_224 = ref_img.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
        arr = (np.asarray(img_224, dtype=np.float32) / 255.0 - 0.5) / 0.5
        patches = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
        # Unfold into 16x16 grid of 14x14 patches -> (1, 256, 3*14*14 = 588)
        unfolded = torch.nn.functional.unfold(patches, kernel_size=14, stride=14).transpose(1, 2)
        cls_tok = unfolded.mean(dim=1, keepdim=True)
        tokens_257 = torch.cat([cls_tok, unfolded], dim=1)  # (1, 257, 588)
        # Repeat-pad / interpolate feature channels to `image_dim` (1280) and layer-normalize
        reps = (image_dim + tokens_257.shape[-1] - 1) // tokens_257.shape[-1]
        feat = tokens_257.repeat(1, 1, reps)[..., :image_dim]
        feat = torch.nn.functional.layer_norm(feat, (image_dim,))
        return feat.to(dtype=dtype)

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
        if self.memory.sleeping:
            self.memory.wake()
        self.memory.switch_partition(variant)

        check_inference_dependencies(("diffusers", "transformers", "accelerate"))

        hi_path, lo_path = self._resolve_expert_paths(variant)
        if hi_path is None:
            raise RuntimeError(
                f"Missing Wan 2.2 weights for '{variant}' under '{self.weights_dir}'. "
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

        # Configure UniPCMultistepScheduler with appropriate flow_shift (5.0 for >=720p else 3.0)
        flow_shift = 5.0 if max(width, height) >= 1200 else 3.0
        scheduler = UniPCMultistepScheduler.from_pretrained(str(cfg_dir / "scheduler"))
        scheduler.config["flow_shift"] = flow_shift

        import json

        model_index = json.loads((cfg_dir / "model_index.json").read_text(encoding="utf-8"))
        boundary_ratio = model_index.get("boundary_ratio", 0.875 if "T2V" in variant else 0.9)
        if lo_path is None:
            boundary_ratio = None

        # -------- Stage 1: UMT5-XXL Prompt Encoding + Eviction --------
        self.memory.begin_qwen3_vl_stage(cache_hit=False)
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
            embeds = self._text_encoder(ids, mask).last_hidden_state.to(compute_dtype)
            # Zero out padding and re-pad per official Wan convention
            trimmed = [u[:v] for u, v in zip(embeds, seq_lens)]
            padded = [torch.cat([u, u.new_zeros(512 - u.size(0), u.size(1))]) for u in trimmed]
            return torch.stack(padded, dim=0)

        do_cfg = guidance_scale > 1.0
        prompt_embeds = _encode_t5([prompt])
        neg_prompt = negative_prompt or "low quality, blurry, watermark, text, distorted"
        neg_embeds = _encode_t5([neg_prompt]) if do_cfg else None

        self._text_encoder.to("cpu")
        clear_device_cache(device)
        self.memory.evict_qwen3_vl_stage()
        if on_log:
            on_log("[Wan2.2 UMT5-XXL] Evicted UMT5-XXL from VRAM (0 MB resident)")

        # -------- Prepare Latents (+ I2V / Animate Conditioning) --------
        gen_dev = "cpu" if device.type == "mps" else device
        generator = torch.Generator(device=gen_dev).manual_seed(int(seed))
        z_dim = self._vae.config.z_dim
        t_scale = self._vae.config.scale_factor_temporal if hasattr(self._vae.config, "scale_factor_temporal") else 4
        s_scale = self._vae.config.scale_factor_spatial if hasattr(self._vae.config, "scale_factor_spatial") else 8
        latent_f = (num_frames - 1) // t_scale + 1
        latent_h = height // s_scale
        latent_w = width // s_scale

        is_animate = variant == "Wan2.2-Animate-14B"
        latent_t_dim = (latent_f + 1) if is_animate else latent_f
        latents = torch.randn(
            (1, z_dim, latent_t_dim, latent_h, latent_w),
            generator=generator,
            device=device if gen_dev != "cpu" else "cpu",
            dtype=torch.float32,
        ).to(device)

        lmean = torch.tensor(self._vae.config.latents_mean, device=device, dtype=compute_dtype).view(1, z_dim, 1, 1, 1)
        lstd = 1.0 / torch.tensor(self._vae.config.latents_std, device=device, dtype=compute_dtype).view(
            1, z_dim, 1, 1, 1
        )

        condition: Optional[torch.Tensor] = None
        animate_kwargs: Dict[str, Any] = {}

        if is_animate:
            ref_img = (start_image or Image.new("RGB", (width, height), (128, 128, 128))).convert("RGB")
            video_proc = VideoProcessor(vae_scale_factor=s_scale)
            ref_resized = ref_img.resize((width, height), Image.Resampling.LANCZOS)
            ref_tensor = video_proc.preprocess(ref_resized, height=height, width=width).to(device, dtype=compute_dtype)

            self._vae.to(device)
            # 1. Reference character image latents: (1, 16, 1, latent_h, latent_w) + 4-ch ones mask -> 20 ch
            ref_enc = self._vae.encode(ref_tensor[:, :, None]).latent_dist.mode()
            ref_enc = (ref_enc - lmean.to(ref_enc.dtype)) * lstd.to(ref_enc.dtype)
            ref_mask = torch.ones(1, 4, 1, latent_h, latent_w, device=device, dtype=compute_dtype)
            ref_cond = torch.cat([ref_mask, ref_enc.to(compute_dtype)], dim=1)

            # 2. Prev-segment conditioning latents: zeros encoded -> (1, 16, latent_f, latent_h, latent_w) + 4-ch zero mask
            zero_vid = ref_tensor.new_zeros(1, 3, num_frames, height, width)
            prev_enc = self._vae.encode(zero_vid).latent_dist.mode()
            prev_enc = (prev_enc - lmean.to(prev_enc.dtype)) * lstd.to(prev_enc.dtype)
            prev_mask = torch.zeros(1, 4, latent_f, latent_h, latent_w, device=device, dtype=compute_dtype)
            prev_cond = torch.cat([prev_mask, prev_enc.to(compute_dtype)], dim=1)

            # Concatenate along temporal dimension -> (1, 20, latent_f + 1, latent_h, latent_w)
            condition = torch.cat([ref_cond, prev_cond], dim=2)

            # 3. Pose latents: (1, 16, latent_f, latent_h, latent_w)
            pose_latents = prev_enc.to(compute_dtype)
            self._vae.to("cpu")
            clear_device_cache(device)

            # 4. Face video segment: (1, 3, num_frames, 512, 512)
            face_crop = ref_img.resize((512, 512), Image.Resampling.LANCZOS)
            face_t = video_proc.preprocess(face_crop, height=512, width=512).to(device, dtype=compute_dtype)
            face_video_segment = face_t[:, :, None].expand(1, 3, num_frames, 512, 512).contiguous()

            # 5. Reference image CLIP-compatible embeddings: (1, 257, 1280)
            image_embeds = self._extract_animate_image_embeds(ref_img, device=device, dtype=compute_dtype)
            animate_kwargs = {
                "encoder_hidden_states_image": image_embeds,
                "pose_hidden_states": pose_latents,
                "face_pixel_values": face_video_segment,
            }
        elif variant == "Wan2.2-I2V-A14B":
            if start_image is not None:
                video_proc = VideoProcessor(vae_scale_factor=s_scale)
                img_rgb = start_image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
                img_tensor = video_proc.preprocess(img_rgb, height=height, width=width).to(device, dtype=compute_dtype)
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
            else:
                # Fallback zero condition (4 mask + 16 latent channels = 20) so 36-channel I2V conv_in succeeds
                condition = torch.zeros(1, 20, latent_f, latent_h, latent_w, device=device, dtype=compute_dtype)

        # -------- Stage 2: Two-Expert MoE / Animate Denoising --------
        steps = max(4, int(num_steps))
        scheduler.set_timesteps(steps, device=device)
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

        for i, t in enumerate(timesteps):
            is_high = (boundary_ts is None) or (float(t.item()) >= boundary_ts)
            expert_key = f"{variant}:high" if is_high else f"{variant}:low"
            expert_path = hi_path if is_high else (lo_path or hi_path)

            if active_key != expert_key:
                if active_tuple is not None:
                    self._deactivate_expert(active_tuple[0], active_tuple[1], device)
                active_tuple = self._get_expert(expert_key, expert_path, cfg_dir, on_log=on_log)
                self._activate_expert(active_tuple[0], active_tuple[1], device)
                active_key = expert_key

            model, offloader = active_tuple
            cur_cfg = guidance_scale if is_high else g2
            model_in = (
                torch.cat([latents.to(compute_dtype), condition.to(compute_dtype)], dim=1)
                if condition is not None
                else latents.to(compute_dtype)
            )
            ts_in = t.expand(latents.shape[0])

            noise_pred = model(
                hidden_states=model_in,
                timestep=ts_in,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
                **animate_kwargs,
            )[0]
            if do_cfg and neg_embeds is not None:
                uncond_kwargs = dict(animate_kwargs)
                if "face_pixel_values" in uncond_kwargs:
                    uncond_kwargs["face_pixel_values"] = uncond_kwargs["face_pixel_values"] * 0 - 1
                noise_uncond = model(
                    hidden_states=model_in,
                    timestep=ts_in,
                    encoder_hidden_states=neg_embeds,
                    return_dict=False,
                    **uncond_kwargs,
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
            self._deactivate_expert(active_tuple[0], active_tuple[1], device)

        # -------- Stage 3: AutoencoderKLWan Decode -> H.264 MP4 --------
        self.memory.begin_vae_stage()
        self._vae.to(device)
        z_lat = (latents[:, :, 1:] if is_animate else latents).to(device=device, dtype=compute_dtype)
        z_lat = z_lat / lstd + lmean
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
        for model, offloader in self._experts.values():
            self._deactivate_expert(model, offloader, device)
        if self._text_encoder is not None:
            self._text_encoder.to("cpu")
        if self._vae is not None:
            self._vae.to("cpu")
        clear_device_cache(device)
