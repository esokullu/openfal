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
    `INT8 ConvRot` `.weight_scale` + `.comfy_quant`) to Diffusers `WanTransformer3DModel` naming.
    """
    from diffusers.loaders.single_file_utils import convert_wan_transformer_to_diffusers

    cleaned: Dict[str, torch.Tensor] = {}
    for k, v in raw_sd.items():
        if k == "scaled_fp8" or k.endswith((".scale_input", ".input_scale", ".comfy_attention.config")):
            continue
        cleaned[k] = v
    return convert_wan_transformer_to_diffusers(cleaned)


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
        from diffusers import WanTransformer3DModel
        from diffusers.models.transformers.transformer_wan import WanRotaryPosEmbed

        device, compute_dtype = resolve_compute_device_and_dtype()
        tr_cfg = WanTransformer3DModel.load_config(str(cfg_dir / "transformer"))
        with init_empty_weights():
            model = WanTransformer3DModel.from_config(tr_cfg)

        raw_sd = load_safetensors_state_dict(ckpt_path, device="cpu")
        mapped_sd = convert_wan22_transformer_state_dict(raw_sd)
        del raw_sd
        missing, unexp = quantized_module_swap_from_state_dict(model, mapped_sd, compute_dtype=compute_dtype)
        del mapped_sd

        # Re-init tiny non-persistent RoPE frequency buffers
        model.rope = WanRotaryPosEmbed(
            attention_head_dim=model.config.attention_head_dim,
            patch_size=model.config.patch_size,
            max_seq_len=model.config.rope_max_seq_len,
        )

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

    def _activate_expert(self, model: nn.Module, offloader: Optional[StrataBlockOffloadManager], device: torch.device) -> None:
        if offloader is not None:
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

        if variant == "Wan2.2-Animate-14B":
            raise RuntimeError(
                "Wan2.2-Animate-14B checkpoints require CLIP-Vision + DWPose/Face landmark extractors "
                "which are not yet bundled in the local cache; use wan/v2.2/text-to-video or wan/v2.2/image-to-video."
            )

        hi_path, lo_path = self._resolve_expert_paths(variant)
        if hi_path is None:
            raise RuntimeError(
                f"Missing Wan 2.2 weights for '{variant}' under '{self.weights_dir}'. "
                "Run: openfal download --component alibaba-only"
            )

        cfg_dir = self._ensure_configs(variant, on_log=on_log)
        device, compute_dtype = resolve_compute_device_and_dtype()

        from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler, WanImageToVideoPipeline, WanPipeline
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

        # -------- Prepare Latents (+ I2V Start Image Condition) --------
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

        condition: Optional[torch.Tensor] = None
        if variant == "Wan2.2-I2V-A14B" and start_image is not None:
            video_proc = VideoProcessor(vae_scale_factor=s_scale)
            img_rgb = start_image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
            img_tensor = video_proc.preprocess(img_rgb, height=height, width=width).to(device, dtype=compute_dtype)
            # Pad target frames with zeros matching WanImageToVideoPipeline
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

            lmean = torch.tensor(self._vae.config.latents_mean, device=device, dtype=enc.dtype).view(1, z_dim, 1, 1, 1)
            lstd = 1.0 / torch.tensor(self._vae.config.latents_std, device=device, dtype=enc.dtype).view(
                1, z_dim, 1, 1, 1
            )
            enc = (enc - lmean) * lstd

            mask_lat = torch.ones(1, 1, num_frames, latent_h, latent_w, device=device, dtype=enc.dtype)
            mask_lat[:, :, 1:] = 0
            first_four = mask_lat[:, :, :1].repeat(1, 4, 1, 1, 1)
            rest = mask_lat[:, :, 1:].reshape(1, -1, 4, latent_h, latent_w).permute(0, 2, 1, 3, 4)
            mask_packed = torch.cat([first_four, rest], dim=2)
            condition = torch.cat([mask_packed, enc], dim=1)

        # -------- Stage 2: Two-Expert MoE Denoising (High -> Low Swap) --------
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
            model_in = torch.cat([latents.to(compute_dtype), condition.to(compute_dtype)], dim=1) if condition is not None else latents.to(compute_dtype)
            ts_in = t.expand(latents.shape[0])

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
            self._deactivate_expert(active_tuple[0], active_tuple[1], device)

        # -------- Stage 3: AutoencoderKLWan Decode -> H.264 MP4 --------
        self.memory.begin_vae_stage()
        self._vae.to(device)
        z_lat = latents.to(device=device, dtype=compute_dtype)
        lmean = torch.tensor(self._vae.config.latents_mean, device=device, dtype=compute_dtype).view(1, z_dim, 1, 1, 1)
        lstd = 1.0 / torch.tensor(self._vae.config.latents_std, device=device, dtype=compute_dtype).view(1, z_dim, 1, 1, 1)
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
