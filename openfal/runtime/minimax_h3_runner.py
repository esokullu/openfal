"""
Real PyTorch / Diffusers inference engine for MiniMax-H3 (33B Dense Multimodal DiT + Native Stereo Audio):
  - `FL2VA`: Text-to-Video+Audio & First/Last Keyframe Video+Audio (`minimax_h3_fl2va_*`)
  - `Ref2VA`: Multi-Reference Omni-Video+Audio (`minimax_h3_ref2va_*`)

Supports both official unpruned HF weights AND `Comfy-Org/MiniMax-H3` pruned FP8 / INT8 ConvRot
single-file checkpoints (`minimax_h3_fl2va_pruned_fp8_scaled.safetensors` with rank-8 `adaln_t_table`),
plus Qwen3-VL-32B Layer-50 streamed text/vision conditioning (`StrataBlockOffloadManager`),
30/50 Counter + 20/50 Pinned RAM Pantry DiT offloading, MiniMax-H3 ViT Video VAE, and BigVGAN Stereo Audio VAE.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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
from openfal.runtime.qwen_image_runner import convert_qwen3vl_text_encoder_state_dict


class _PrunedAdalnTableEmbedder(nn.Module):
    """Interpolated lookup into `Comfy-Org` rank-8 `adaln_t_table` (`[1025, 8]`) for pruned MiniMax-H3."""

    def __init__(self, table: torch.Tensor):
        super().__init__()
        self.table = nn.Parameter(table.detach().to(torch.float32).contiguous(), requires_grad=False)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        max_idx = self.table.shape[0] - 1
        pos = (timestep.float() * float(max_idx)).clamp(0.0, float(max_idx))
        lo = torch.floor(pos).long()
        hi = torch.clamp(lo + 1, max=max_idx)
        frac = (pos - lo.float()).unsqueeze(-1)
        emb = self.table[lo] * (1.0 - frac) + self.table[hi] * frac
        return emb


class _PrunedAdaLayerNormModulation(nn.Module):
    """Drop-in modulation module for pruned MiniMax-H3 (temb is already post-SiLU SVD rank-8)."""

    def __init__(self, hidden_size: int, orig_linear: nn.Linear):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = orig_linear

    def forward(self, temb: torch.Tensor):
        dtype = next(self.linear.parameters()).dtype
        proj = self.linear(temb.to(dtype))
        return proj.view(-1, 6 * self.hidden_size).chunk(6, dim=-1)


class _PrunedAdaLayerNormOut(nn.Module):
    """Drop-in final AdaLN output module for pruned MiniMax-H3 (no extra SiLU on rank-8 temb)."""

    def __init__(self, orig_norm: nn.Module, orig_linear: nn.Linear):
        super().__init__()
        self.norm = orig_norm
        self.linear = orig_linear

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        timestep_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        dtype = next(self.linear.parameters()).dtype
        shift, scale = self.linear(temb.to(dtype)).chunk(2, dim=-1)
        hidden_states = self.norm(hidden_states)
        if timestep_indices is not None:
            return hidden_states * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(
                0, timestep_indices
            )
        return hidden_states * (1.0 + scale) + shift


def convert_minimax_h3_dit_state_dict(
    raw_sd: Dict[str, torch.Tensor],
    qkv_layout: str = "stacked",
    num_attention_heads: int = 56,
    attention_head_dim: int = 128,
    num_pdd_heads: int = 0,
) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
    """
    Convert `Comfy-Org/MiniMax-H3` or `Comfy-Org/MiniMax-H3-Image` single-file (pruned or unpruned)
    FP8/INT8 checkpoint to Diffusers `MiniMaxH3Transformer3DModel` / `MiniMaxH3ImageTransformer2DModel`
    layout, preserving quant scales and SwiGLU/QKV/PDD-head semantics.
    """
    rename_map = {
        "token_refiner.blocks.": "token_refiner.refiner_blocks.",
        "time_embedder.proj_in.": "time_embedder.linear_1.",
        "time_embedder.proj_out.": "time_embedder.linear_2.",
        "video_patch_proj.": "proj_in.",
        "audio_patch_proj.": "audio_proj_in.",
        "condition_proj.": "context_embedder.",
        "final_layer.norm.": "norm_out.norm.",
        "final_layer.adaln_proj.linear.": "norm_out.linear.",
        "final_layer.video_out.": "proj_out.",
        "final_layer.audio_out.": "audio_proj_out.",
        ".attn.q_norm.": ".attn.norm_q.",
        ".attn.k_norm.": ".attn.norm_k.",
        ".attn.out_proj.": ".attn.to_out.0.",
        ".mlp.fc1.": ".ff.net.0.proj.",
        ".mlp.fc2.": ".ff.net.2.",
    }

    adaln_t_table = raw_sd.get("adaln_t_table")
    out: Dict[str, torch.Tensor] = {}

    for k, v in raw_sd.items():
        if k in ("adaln_t_table", "rope.inv_freq", "scaled_fp8") or k.endswith(
            (".scale_input", ".input_scale", ".comfy_attention.config")
        ):
            continue
        nk = k.replace("blocks.", "transformer_blocks.", 1) if k.startswith("blocks.") else k
        for old, new in rename_map.items():
            nk = nk.replace(old, new)
        out[nk] = v

    # Split stacked PDD output heads (`proj_out.{weight,bias}` of shape [6*96, ...]) for MiniMax-H3-Image
    if num_pdd_heads > 0:
        for suffix in ("weight", "bias"):
            pk = f"proj_out.{suffix}"
            if pk in out and out[pk].shape[0] % num_pdd_heads == 0:
                chunks = torch.chunk(out.pop(pk), num_pdd_heads, dim=0)
                for h_idx, chunk in enumerate(chunks):
                    out[f"pdd_proj_out.{h_idx}.{suffix}"] = chunk.contiguous()

    # Track fc1 / qkv weight row counts for accurate per-row scale splitting
    fc1_rows = {k.removesuffix(".weight"): v.shape[0] for k, v in out.items() if k.endswith(".ff.net.0.proj.weight")}
    qkv_rows = {k.removesuffix(".weight"): v.shape[0] for k, v in out.items() if k.endswith(".attn.qkv_proj.weight")}

    def _swap_gate_value(tensor: torch.Tensor) -> torch.Tensor:
        gate, val = torch.chunk(tensor, 2, dim=0)
        return torch.cat([val, gate], dim=0).contiguous()

    def _split_qkv(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if qkv_layout == "interleaved":
            unflat = tensor.unflatten(0, (num_attention_heads, 3, attention_head_dim))
            q, k, v = [w.flatten(0, 1).contiguous() for w in unflat.unbind(dim=1)]
            return q, k, v
        q, k, v = torch.chunk(tensor, 3, dim=0)
        return q.contiguous(), k.contiguous(), v.contiguous()

    # Transform fused SwiGLU ([gate; value] -> [value; gate]) and QKV
    for k in list(out.keys()):
        if k.endswith(".ff.net.0.proj.weight") or k.endswith(".ff.net.0.proj.bias"):
            out[k] = _swap_gate_value(out[k])
        elif k.endswith(".ff.net.0.proj.weight_scale") or k.endswith(".ff.net.0.proj.scale_weight"):
            base = k.rsplit(".", 1)[0]
            rows = fc1_rows.get(base)
            if out[k].ndim >= 1 and rows and out[k].shape[0] == rows:
                out[k] = _swap_gate_value(out[k])
        elif ".attn.qkv_proj." in k:
            val = out.pop(k)
            base, suffix = k.split(".attn.qkv_proj.", 1)
            if suffix in ("weight", "bias"):
                q, kk, vv = _split_qkv(val)
            elif suffix in ("weight_scale", "scale_weight") and val.ndim >= 1 and val.shape[0] == qkv_rows.get(
                f"{base}.attn.qkv_proj"
            ):
                q, kk, vv = _split_qkv(val)
            else:
                q, kk, vv = val, val.clone(), val.clone()
            out[f"{base}.attn.to_q.{suffix}"] = q
            out[f"{base}.attn.to_k.{suffix}"] = kk
            out[f"{base}.attn.to_v.{suffix}"] = vv

    return out, adaln_t_table


def convert_minimax_h3_video_vae_state_dict(raw_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert `Comfy-Org/MiniMax-H3 :: minimax_h3_video_vae_*` keys -> `diffusers.AutoencoderKLMiniMaxH3`."""
    out: Dict[str, torch.Tensor] = {}
    down_blk_re = re.compile(r"^encoder\.down\.(\d+)\.block\.(\d+)\.(.+)$")
    down_samp_re = re.compile(r"^encoder\.down\.(\d+)\.downsample\.(.+)$")

    for k, v in raw_sd.items():
        if k in ("decoder.mask_token", "latents_mean", "latents_std") or k.endswith(
            (".scale_input", ".input_scale", ".comfy_attention.config")
        ):
            continue

        m_blk = down_blk_re.match(k)
        m_ds = down_samp_re.match(k)
        if m_blk:
            rest = m_blk.group(3).replace("nin_shortcut.", "conv_shortcut.")
            nk = f"encoder.down_blocks.{m_blk.group(1)}.resnets.{m_blk.group(2)}.{rest}"
        elif m_ds:
            nk = f"encoder.down_blocks.{m_ds.group(1)}.downsamplers.0.{m_ds.group(2)}"
        elif k.startswith("decoder.x_embedder."):
            nk = "decoder.proj_in." + k[len("decoder.x_embedder.") :]
        elif ".attn.to_out." in k:
            nk = k.replace(".attn.to_out.", ".attn.to_out.0.")
        elif ".ff.w1." in k:
            nk = k.replace(".ff.w1.", ".ff.net.0.proj.")
        elif ".ff.w2." in k:
            nk = k.replace(".ff.w2.", ".ff.net.2.")
        else:
            nk = k
        out[nk] = v

    # Track w1 / to_qkv weight row counts for accurate per-row scale & ConvRot metadata splitting
    w1_rows = {k.removesuffix(".weight"): v.shape[0] for k, v in out.items() if k.endswith(".ff.net.0.proj.weight")}
    qkv_rows = {k.removesuffix(".weight"): v.shape[0] for k, v in out.items() if k.endswith(".attn.to_qkv.weight")}

    def _swap_gate_value(tensor: torch.Tensor) -> torch.Tensor:
        gate, val = torch.chunk(tensor, 2, dim=0)
        return torch.cat([val, gate], dim=0).contiguous()

    # Split ViT decoder fused to_qkv -> to_q/to_k/to_v and swap gated w1 -> [value; gate]
    for k in list(out.keys()):
        if k.endswith(".ff.net.0.proj.weight") or k.endswith(".ff.net.0.proj.bias"):
            out[k] = _swap_gate_value(out[k])
        elif k.endswith(".ff.net.0.proj.weight_scale") or k.endswith(".ff.net.0.proj.scale_weight"):
            base = k.rsplit(".", 1)[0]
            rows = w1_rows.get(base)
            if out[k].ndim >= 1 and rows and out[k].shape[0] == rows:
                out[k] = _swap_gate_value(out[k])
        elif ".attn.to_qkv." in k:
            tensor = out.pop(k)
            base, suffix = k.split(".attn.to_qkv.", 1)
            if suffix in ("weight", "bias"):
                q, kk, vv = torch.chunk(tensor, 3, dim=0)
            elif suffix in ("weight_scale", "scale_weight") and tensor.ndim >= 1 and tensor.shape[0] == qkv_rows.get(
                f"{base}.attn.to_qkv"
            ):
                q, kk, vv = torch.chunk(tensor, 3, dim=0)
            else:
                q, kk, vv = tensor, tensor.clone(), tensor.clone()
            out[f"{base}.attn.to_q.{suffix}"] = q.contiguous()
            out[f"{base}.attn.to_k.{suffix}"] = kk.contiguous()
            out[f"{base}.attn.to_v.{suffix}"] = vv.contiguous()

    return out


def _strip_audio_weight_norm(audio_vae: nn.Module) -> None:
    """Remove weight_norm parametrizations on `AutoencoderKLMiniMaxH3Audio` to match flat Comfy checkpoints."""
    for mod in audio_vae.modules():
        if hasattr(mod, "parametrizations") and "weight" in mod.parametrizations:
            torch.nn.utils.parametrize.remove_parametrizations(mod, "weight", leave_parametrized=False)
        else:
            try:
                torch.nn.utils.remove_weight_norm(mod)
            except ValueError:
                pass


class _PassthroughDecoderLayer(nn.Module):
    """Lightweight no-op decoder layer preserving `hidden_states[50]` when pruned at Layer 50."""

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return hidden_states


def _materialize_remaining_meta(module: nn.Module, dtype: torch.dtype) -> None:
    """Replace any unpopulated meta parameters/buffers (e.g. pruned trailing norms) on CPU."""
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


def _build_minimax_pipeline_state(pipe: Any, init_inputs: Dict[str, Any]) -> Any:
    """Populate a `diffusers.modular_pipelines.PipelineState` from named inputs & defaults."""
    from diffusers.modular_pipelines.modular_pipeline import PipelineState

    state = PipelineState()
    passed_kwargs = dict(init_inputs)
    for expected in pipe.blocks.inputs:
        name = expected.name
        default = expected.default
        kt = expected.kwargs_type
        if name in passed_kwargs:
            state.set(name, passed_kwargs.pop(name), kt)
        elif kt is not None and kt in passed_kwargs:
            for k, v in passed_kwargs.pop(kt).items():
                state.set(k, v, kt)
        elif name is not None and name not in state.values and default is not None:
            state.set(name, default, kt)
    return state


class RealMiniMaxH3Runner:
    """Real neural network inference runner for MiniMax-H3 `FL2VA`, `Ref2VA`, and `H3-Image`."""

    HF_REPO = "MiniMaxAI/MiniMax-H3"
    HF_IMAGE_REPO = "MiniMaxAI/MiniMax-H3-Image"

    def __init__(self, config: EngineConfig, memory: StrataMemoryHierarchy):
        self.config = config
        self.memory = memory
        self.weights_dir = config.weights_dir

        self._pipe: Any = None
        self._loaded_partition: Optional[str] = None
        self._dit_offloader: Optional[StrataBlockOffloadManager] = None
        self._text_encoder: Any = None
        self._tokenizer: Any = None
        self._processor: Any = None
        self._te_offloader: Optional[StrataBlockOffloadManager] = None
        self._cfg_dir: Optional[Path] = None
        self._img_cfg_dir: Optional[Path] = None

    def _find_weights(self, partition: str) -> Dict[str, Optional[Path]]:
        dit = None
        if partition == "Ref2VA":
            dit = find_matching_safetensors(self.weights_dir / "Ref2VA", ["minimax_h3_ref2va"])
        elif partition == "FL2VA":
            dit = find_matching_safetensors(self.weights_dir / "FL2VA", ["minimax_h3_fl2va"])
        elif partition == "H3-Image":
            dit = find_matching_safetensors(self.weights_dir / "H3-Image", ["minimax_h3_image", "diffusion_models"])

        te = find_matching_safetensors(self.weights_dir / "text_encoder", ["qwen3vl_32b"])
        vvae = find_matching_safetensors(self.weights_dir / "vae", ["minimax_h3_video_vae"])
        avae = find_matching_safetensors(self.weights_dir / "vae", ["minimax_h3_audio_vae"])
        ivae = find_matching_safetensors(self.weights_dir / "H3-Image", ["minimax_h3_image_vae"])
        return {
            "dit": dit,
            "text_encoder": te,
            "video_vae": vvae,
            "audio_vae": avae,
            "image_vae": ivae,
        }

    def has_weights_for(self, partition: str) -> bool:
        paths = self._find_weights(partition)
        return paths["dit"] is not None

    def _ensure_configs(self, on_log: Optional[Callable[[str], None]] = None) -> Path:
        if self._cfg_dir is not None:
            return self._cfg_dir
        target = self.weights_dir / "minimax_h3_hf_configs"
        ensure_hf_configs(
            repo_id=self.HF_REPO,
            target_dir=target,
            required_files=[
                "modular_model_index.json",
                "scheduler/scheduler_config.json",
                "audio_scheduler/scheduler_config.json",
                "transformer/config.json",
                "vae/config.json",
                "audio_vae/config.json",
                "text_encoder/config.json",
                "tokenizer/tokenizer_config.json",
                "tokenizer/tokenizer.json",
                "processor/processor_config.json",
                "processor/tokenizer.json",
                "processor/chat_template.json",
            ],
            allow_patterns=[
                "modular_model_index.json",
                "scheduler/*.json",
                "audio_scheduler/*.json",
                "transformer/*.json",
                "transformer_ref/*.json",
                "vae/*.json",
                "audio_vae/*.json",
                "text_encoder/*.json",
                "tokenizer/*",
                "processor/*",
            ],
            on_log=on_log,
        )
        self._cfg_dir = target
        return target

    def _ensure_image_configs(self, on_log: Optional[Callable[[str], None]] = None) -> Path:
        if self._img_cfg_dir is not None:
            return self._img_cfg_dir
        target = self.weights_dir / "minimax_h3_image_hf_configs"
        ensure_hf_configs(
            repo_id=self.HF_IMAGE_REPO,
            target_dir=target,
            required_files=[
                "modular_model_index.json",
                "scheduler/scheduler_config.json",
                "transformer_diffusers/config.json",
                "vae_diffusers/config.json",
            ],
            allow_patterns=[
                "modular_model_index.json",
                "scheduler/*.json",
                "transformer_diffusers/*.json",
                "vae_diffusers/*.json",
            ],
            on_log=on_log,
        )
        self._img_cfg_dir = target
        return target

    def _ensure_shared_qwen3vl(
        self,
        cfg_dir: Path,
        te_path: Path,
        on_log: Optional[Callable[[str], None]] = None,
    ) -> None:
        if self._text_encoder is not None:
            return
        if "nvfp4" in te_path.name.lower():
            raise RuntimeError(
                f"Text encoder '{te_path.name}' uses NVFP4 quantization; please download "
                "the INT8 ConvRot encoder (`qwen3vl_32b_minimax_h3_int8_convrot.safetensors`) with --quant fp8_e4m3."
            )

        from accelerate import init_empty_weights
        from transformers import (
            Qwen2TokenizerFast,
            Qwen3VLConfig,
            Qwen3VLForConditionalGeneration,
            Qwen3VLProcessor,
        )
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLTextRotaryEmbedding,
            Qwen3VLVisionRotaryEmbedding,
        )

        device, compute_dtype = resolve_compute_device_and_dtype()
        te_cfg = Qwen3VLConfig.from_pretrained(str(cfg_dir / "text_encoder"))
        with init_empty_weights():
            text_encoder = Qwen3VLForConditionalGeneration(te_cfg)
        te_raw = load_safetensors_state_dict(te_path, device="cpu")
        te_mapped = convert_qwen3vl_text_encoder_state_dict(te_raw)
        del te_raw
        has_lm_head = "lm_head.weight" in te_mapped
        has_layer_50 = any(k.startswith("model.language_model.layers.50.") for k in te_mapped)
        if not has_lm_head:
            # Pruned Qwen3-VL-32B conditioner stops at Layer 50 and omits lm_head (~1.5 GB saved)
            text_encoder.lm_head = nn.Identity()

        # Truncate unpopulated layers >= 50 BEFORE swap so missing count reflects only active layers
        orig_layers = text_encoder.model.language_model.layers
        if len(orig_layers) > 51 and not has_layer_50:
            text_encoder.model.language_model.layers = nn.ModuleList(
                list(orig_layers[:50]) + [_PassthroughDecoderLayer()]
            )

        te_missing, _ = quantized_module_swap_from_state_dict(
            text_encoder,
            te_mapped,
            compute_dtype=compute_dtype,
            embedding_names=("model.language_model.embed_tokens",),
        )
        del te_mapped

        text_encoder.model.language_model.rotary_emb = Qwen3VLTextRotaryEmbedding(config=te_cfg.text_config)
        try:
            text_encoder.model.visual.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(te_cfg.vision_config)
        except (TypeError, AttributeError):
            head_dim = te_cfg.vision_config.hidden_size // te_cfg.vision_config.num_heads
            text_encoder.model.visual.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)
        _materialize_remaining_meta(text_encoder, compute_dtype)

        te_layers = text_encoder.model.language_model.layers
        # Stream Qwen3-VL-32B (~25.3 GB INT8) through a rolling layer window on <=48 GB GPUs
        if self.config.profile.total_vram_gb < 48.0:
            te_resident = min(len(te_layers), max(4, min(16, int((self.config.profile.total_vram_gb - 12.0) // 0.5))))
        else:
            te_resident = len(te_layers)
        self._te_offloader = (
            StrataBlockOffloadManager(te_layers, device=device, resident_vram_blocks=te_resident)
            if te_resident < len(te_layers)
            else None
        )
        self._text_encoder = text_encoder
        self._tokenizer = Qwen2TokenizerFast.from_pretrained(str(cfg_dir / "tokenizer"))
        self._processor = Qwen3VLProcessor.from_pretrained(str(cfg_dir / "processor"))
        if on_log:
            on_log(
                f"[MiniMax-H3 Qwen3-VL-32B] Loaded '{te_path.name}' "
                f"({te_resident}/{len(te_layers)} layers Counter, missing={len(te_missing)})"
            )

    @staticmethod
    def _wrap_modular_pipeline_device(pipe: Any, device: torch.device) -> Any:
        base_cls = pipe.__class__

        class _StrataMiniMaxModularPipeline(base_cls):  # type: ignore[misc,valid-type]
            @property
            def _execution_device(self):
                override = getattr(self, "_strata_device", None)
                if override is not None:
                    return torch.device(override)
                return super()._execution_device

            @property
            def min_duration(self):
                return 0.5

            @property
            def max_duration(self):
                return 16.0

        pipe.__class__ = _StrataMiniMaxModularPipeline
        pipe._strata_device = device
        return pipe

    def _ensure_loaded(self, partition: str, on_log: Optional[Callable[[str], None]] = None) -> None:
        if self._pipe is not None and self._loaded_partition == partition:
            return

        check_inference_dependencies(("diffusers", "transformers", "accelerate"))

        paths = self._find_weights(partition)
        required_keys = ("dit", "text_encoder", "image_vae") if partition == "H3-Image" else ("dit", "text_encoder", "video_vae", "audio_vae")
        missing_files = [k for k in required_keys if paths.get(k) is None]
        if missing_files:
            raise RuntimeError(
                f"MiniMax-H3 partition '{partition}' weights found partially, missing: {', '.join(missing_files)}. "
                "Run: openfal download --component all --quant fp8_e4m3"
            )

        self._evict_active_pipe_only()
        cfg_dir = self._ensure_configs(on_log=on_log)
        device, compute_dtype = resolve_compute_device_and_dtype()
        self._ensure_shared_qwen3vl(cfg_dir=cfg_dir, te_path=paths["text_encoder"], on_log=on_log)

        from accelerate import init_empty_weights
        from diffusers import (
            AutoencoderKLMiniMaxH3,
            AutoencoderKLMiniMaxH3Audio,
            AutoencoderKLMiniMaxH3Image,
            MiniMaxH3ImageBlocks,
            MiniMaxH3ImageTransformer2DModel,
            MiniMaxH3Scheduler,
            MiniMaxH3Transformer3DModel,
        )
        from diffusers.image_processor import VaeImageProcessor
        from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Blocks
        from diffusers.video_processor import VideoProcessor

        if partition == "H3-Image":
            img_cfg_dir = self._ensure_image_configs(on_log=on_log)
            tr_cfg = json.loads((img_cfg_dir / "transformer_diffusers" / "config.json").read_text(encoding="utf-8"))
            num_pdd = int(tr_cfg.get("num_pdd_heads", 6))

            dit_raw = load_safetensors_state_dict(paths["dit"], device="cpu")
            dit_mapped, adaln_t_table = convert_minimax_h3_dit_state_dict(
                dit_raw,
                num_attention_heads=tr_cfg.get("num_attention_heads", 56),
                attention_head_dim=tr_cfg.get("attention_head_dim", 128),
                num_pdd_heads=num_pdd,
            )
            del dit_raw
            if adaln_t_table is not None:
                tr_cfg["time_embed_dim"] = int(adaln_t_table.shape[1])

            with init_empty_weights():
                transformer = MiniMaxH3ImageTransformer2DModel.from_config(tr_cfg)
            dit_missing, dit_unexp = quantized_module_swap_from_state_dict(
                transformer, dit_mapped, compute_dtype=compute_dtype
            )
            del dit_mapped

            from diffusers.models.transformers.transformer_minimax_h3_image import MiniMaxH3ImageRotaryPosEmbed

            transformer.rope = MiniMaxH3ImageRotaryPosEmbed(
                rope_freq_dim=tr_cfg.get("rope_freq_dim", 16),
                rope_theta=tr_cfg.get("rope_theta", 10000.0),
            )
            if adaln_t_table is not None:
                transformer.time_proj = nn.Identity()
                transformer.time_embedder = _PrunedAdalnTableEmbedder(adaln_t_table)
                transformer.norm_out = _PrunedAdaLayerNormOut(
                    transformer.norm_out.norm, transformer.norm_out.linear
                )
                for blk in transformer.transformer_blocks:
                    blk.adaln_proj = _PrunedAdaLayerNormModulation(
                        hidden_size=tr_cfg.get("hidden_size", 5376),
                        orig_linear=blk.adaln_proj.linear,
                    )
            _materialize_remaining_meta(transformer, compute_dtype)

            num_blocks = len(transformer.transformer_blocks)
            dit_resident = min(
                num_blocks,
                min(
                    self.config.profile.resident_vram_blocks,
                    compute_resident_blocks(self.config, num_blocks, estimated_model_vram_gb=20.0),
                ),
            )
            self._dit_offloader = (
                StrataBlockOffloadManager(
                    transformer.transformer_blocks, device=device, resident_vram_blocks=dit_resident
                )
                if dit_resident < num_blocks
                else None
            )
            if on_log:
                on_log(
                    f"[MiniMax-H3-Image DiT] Loaded '{paths['dit'].name}' "
                    f"({dit_resident}/{num_blocks} Counter blocks, pruned={adaln_t_table is not None}, "
                    f"missing={len(dit_missing)}, unexpected={len(dit_unexp)})"
                )

            ivae_cfg = AutoencoderKLMiniMaxH3Image.load_config(str(img_cfg_dir / "vae_diffusers"))
            with init_empty_weights():
                image_vae = AutoencoderKLMiniMaxH3Image.from_config(ivae_cfg)
            ivae_raw = load_safetensors_state_dict(paths["image_vae"], device="cpu")
            ivae_mapped = convert_minimax_h3_video_vae_state_dict(ivae_raw)
            del ivae_raw
            quantized_module_swap_from_state_dict(image_vae, ivae_mapped, compute_dtype=compute_dtype)
            del ivae_mapped
            from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_image import MiniMaxH3ImageRotaryPosEmbed as _VaeRope

            image_vae.decoder.rope = _VaeRope(
                int(ivae_cfg.get("decoder_attention_head_dim", 64) * ivae_cfg.get("decoder_rope_dim_ratio", 0.5)),
                theta=ivae_cfg.get("decoder_rope_theta", 100.0),
            )
            _materialize_remaining_meta(image_vae, compute_dtype)

            scheduler = MiniMaxH3Scheduler.from_pretrained(str(img_cfg_dir / "scheduler"))
            pipe = MiniMaxH3ImageBlocks().init_pipeline(str(img_cfg_dir))
            pipe.update_components(
                text_encoder=self._text_encoder,
                tokenizer=self._tokenizer,
                processor=self._processor,
                vae=image_vae,
                scheduler=scheduler,
                transformer=transformer,
                image_processor=VaeImageProcessor(vae_scale_factor=16, do_normalize=False),
            )
            pipe.set_progress_bar_config(disable=True)
            self._pipe = self._wrap_modular_pipeline_device(pipe, device)
            self._loaded_partition = partition
            return

        # 2. MiniMax-H3 33B Video+Audio DiT (`FL2VA` or `Ref2VA`)
        tr_cfg_file = cfg_dir / ("transformer_ref" if partition == "Ref2VA" else "transformer") / "config.json"
        if not tr_cfg_file.exists():
            tr_cfg_file = cfg_dir / "transformer" / "config.json"
        tr_cfg = json.loads(tr_cfg_file.read_text(encoding="utf-8"))

        dit_raw = load_safetensors_state_dict(paths["dit"], device="cpu")
        dit_mapped, adaln_t_table = convert_minimax_h3_dit_state_dict(
            dit_raw,
            num_attention_heads=tr_cfg.get("num_attention_heads", 56),
            attention_head_dim=tr_cfg.get("attention_head_dim", 128),
        )
        del dit_raw

        if adaln_t_table is not None:
            tr_cfg["time_embed_dim"] = int(adaln_t_table.shape[1])

        with init_empty_weights():
            transformer = MiniMaxH3Transformer3DModel.from_config(tr_cfg)

        dit_missing, dit_unexp = quantized_module_swap_from_state_dict(
            transformer, dit_mapped, compute_dtype=compute_dtype
        )
        del dit_mapped

        from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed

        transformer.rope = MiniMaxH3RotaryPosEmbed(
            rope_freq_dim=tr_cfg.get("rope_freq_dim", 16),
            rope_theta=tr_cfg.get("rope_theta", 10000.0),
        )
        if adaln_t_table is not None:
            transformer.time_proj = nn.Identity()
            transformer.time_embedder = _PrunedAdalnTableEmbedder(adaln_t_table)
            transformer.norm_out = _PrunedAdaLayerNormOut(
                transformer.norm_out.norm, transformer.norm_out.linear
            )
            for blk in transformer.transformer_blocks:
                blk.adaln_proj = _PrunedAdaLayerNormModulation(
                    hidden_size=tr_cfg.get("hidden_size", 5376),
                    orig_linear=blk.adaln_proj.linear,
                )
        _materialize_remaining_meta(transformer, compute_dtype)

        num_blocks = len(transformer.transformer_blocks)
        dit_resident = min(
            num_blocks,
            min(
                self.config.profile.resident_vram_blocks,
                compute_resident_blocks(self.config, num_blocks, estimated_model_vram_gb=20.0),
            ),
        )
        self._dit_offloader = (
            StrataBlockOffloadManager(
                transformer.transformer_blocks, device=device, resident_vram_blocks=dit_resident
            )
            if dit_resident < num_blocks
            else None
        )
        if on_log:
            on_log(
                f"[MiniMax-H3 33B DiT ({partition})] Loaded '{paths['dit'].name}' "
                f"({dit_resident}/{num_blocks} Counter blocks, pruned={adaln_t_table is not None}, "
                f"missing={len(dit_missing)}, unexpected={len(dit_unexp)})"
            )

        # 3. Video VAE (CNN Encoder + 36-layer 3D ViT Decoder)
        vvae_cfg = AutoencoderKLMiniMaxH3.load_config(str(cfg_dir / "vae"))
        with init_empty_weights():
            video_vae = AutoencoderKLMiniMaxH3.from_config(vvae_cfg)
        vvae_raw = load_safetensors_state_dict(paths["video_vae"], device="cpu")
        vvae_mapped = convert_minimax_h3_video_vae_state_dict(vvae_raw)
        del vvae_raw
        quantized_module_swap_from_state_dict(video_vae, vvae_mapped, compute_dtype=compute_dtype)
        del vvae_mapped
        from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import MiniMaxH3VideoRotaryPosEmbed

        video_vae.decoder.rope = MiniMaxH3VideoRotaryPosEmbed(
            int(vvae_cfg.get("decoder_attention_head_dim", 64) * vvae_cfg.get("decoder_rope_dim_ratio", 0.5)),
            theta=vvae_cfg.get("decoder_rope_theta", 100.0),
        )
        _materialize_remaining_meta(video_vae, compute_dtype)

        # 4. Audio VAE (BigVGAN 32kHz Stereo)
        avae_cfg = AutoencoderKLMiniMaxH3Audio.load_config(str(cfg_dir / "audio_vae"))
        audio_vae = AutoencoderKLMiniMaxH3Audio.from_config(avae_cfg)
        _strip_audio_weight_norm(audio_vae)
        avae_raw = load_safetensors_state_dict(paths["audio_vae"], device="cpu")
        avae_sd = {k: v for k, v in avae_raw.items() if k not in ("latents_mean", "latents_std")}
        audio_vae.load_state_dict(avae_sd, strict=False, assign=True)

        scheduler = MiniMaxH3Scheduler.from_pretrained(str(cfg_dir / "scheduler"))
        audio_scheduler = MiniMaxH3Scheduler.from_pretrained(str(cfg_dir / "audio_scheduler"))

        pipe = MiniMaxH3Blocks().init_pipeline(str(cfg_dir))
        pipe.update_components(
            text_encoder=self._text_encoder,
            tokenizer=self._tokenizer,
            processor=self._processor,
            vae=video_vae,
            audio_vae=audio_vae,
            scheduler=scheduler,
            audio_scheduler=audio_scheduler,
            video_processor=VideoProcessor(vae_scale_factor=16, do_normalize=False),
            transformer=transformer,
            transformer_ref=transformer,
        )
        pipe.set_progress_bar_config(disable=True)
        self._pipe = self._wrap_modular_pipeline_device(pipe, device)
        self._loaded_partition = partition

    def _run_stage1_text_encoder(
        self,
        pipe: Any,
        state: Any,
        device: torch.device,
        on_log: Optional[Callable[[str], None]] = None,
    ) -> Tuple[Any, Any]:
        sub_blocks = pipe.blocks.sub_blocks
        self.memory.begin_qwen3_vl_stage(cache_hit=False)
        if self._te_offloader is not None:
            pipe.text_encoder.model.visual.to(device)
            pipe.text_encoder.model.language_model.embed_tokens.to(device)
            pipe.text_encoder.model.language_model.norm.to(device)
            pipe.text_encoder.model.language_model.rotary_emb.to(device)
            self._te_offloader.prepare()
            self._te_offloader.attach_hooks()
        else:
            pipe.text_encoder.to(device)

        if on_log:
            on_log("[MiniMax-H3 Stage 1/3] Running Qwen3-VL-32B Layer-50 multimodal conditioning...")
        pipe, state = sub_blocks["text_encoder"](pipe, state)
        if self._te_offloader is not None:
            self._te_offloader.release_all_to_cpu()
        pipe.text_encoder.to("cpu")
        clear_device_cache(device)
        self.memory.evict_qwen3_vl_stage()
        return pipe, state

    @torch.inference_mode()
    def generate_image(
        self,
        prompt: str,
        width: int,
        height: int,
        seed: int,
        reference_images: Optional[List[Image.Image]] = None,
        output_path: Optional[Path] = None,
        on_log: Optional[Callable[[str], None]] = None,
    ) -> Image.Image:
        """Run full 3-stage Strata MiniMax-H3-Image (6-step PDD T2I or Image-to-Image)."""
        if self.memory.sleeping:
            self.memory.wake()
        self.memory.switch_partition("H3-Image")
        self._ensure_loaded(partition="H3-Image", on_log=on_log)

        device, _ = resolve_compute_device_and_dtype()
        pipe = self._pipe
        transformer = pipe.transformer

        width = max(256, round(width / 32) * 32)
        height = max(256, round(height / 32) * 32)
        gen_dev = "cpu" if device.type == "mps" else device
        generator = torch.Generator(device=gen_dev).manual_seed(int(seed))

        init_inputs: Dict[str, Any] = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "min_generation_area": 1024,
            "generator": generator,
            "output_type": "pil",
        }
        if reference_images:
            init_inputs["image"] = reference_images

        state = _build_minimax_pipeline_state(pipe, init_inputs)
        pipe._strata_device = device
        sub_blocks = pipe.blocks.sub_blocks

        pipe, state = sub_blocks["before_encode"](pipe, state)
        pipe, state = self._run_stage1_text_encoder(pipe, state, device, on_log=on_log)

        pipe.vae.to(device)
        pipe, state = sub_blocks["vae_encoder"](pipe, state)
        pipe.vae.to("cpu")
        clear_device_cache(device)

        if self._dit_offloader is not None:
            for name, child in transformer.named_children():
                if name != "transformer_blocks":
                    child.to(device)
            self._dit_offloader.prepare()
            self._dit_offloader.attach_hooks()
            counter_cnt = self._dit_offloader.resident_vram_blocks
            pantry_cnt = self._dit_offloader.total_blocks - counter_cnt
        else:
            transformer.to(device)
            counter_cnt = len(transformer.transformer_blocks)
            pantry_cnt = 0

        if on_log:
            on_log(
                f"[MiniMax-H3-Image Stage 2/3] Denoising 6 PDD steps @ {width}x{height} "
                f"({counter_cnt}/50 Counter, {pantry_cnt}/50 Pantry)..."
            )
        pipe, state = sub_blocks["denoise"](pipe, state)
        if self._dit_offloader is not None:
            self._dit_offloader.release_all_to_cpu()
        transformer.to("cpu")
        clear_device_cache(device)

        self.memory.begin_vae_stage()
        pipe.vae.to(device)
        if on_log:
            on_log("[MiniMax-H3-Image Stage 3/3] Decoding 24-layer ViT Image VAE...")
        pipe, state = sub_blocks["decode"](pipe, state)
        pipe.vae.to("cpu")
        clear_device_cache(device)
        self.memory.finish_generation()

        pil_img = state.get("images")[0]
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pil_img.save(output_path, quality=95)
        return pil_img

    @torch.inference_mode()
    def generate_video(
        self,
        partition: str,
        prompt: str,
        width: int,
        height: int,
        num_frames: int,
        num_steps: int,
        seed: int,
        start_image: Optional[Image.Image] = None,
        end_image: Optional[Image.Image] = None,
        reference_images: Optional[List[Image.Image]] = None,
        reference_videos: Optional[List[str]] = None,
        reference_audios: Optional[List[str]] = None,
        output_path: Optional[Path] = None,
        enable_audio: bool = True,
        on_log: Optional[Callable[[str], None]] = None,
    ) -> Path:
        """Run full 3-stage Strata MiniMax-H3 joint Video + 32kHz Stereo Audio generation."""
        if self.memory.sleeping:
            self.memory.wake()
        self.memory.switch_partition(partition)
        self._ensure_loaded(partition=partition, on_log=on_log)

        device, _ = resolve_compute_device_and_dtype()
        pipe = self._pipe
        tr_key = "transformer_ref" if partition == "Ref2VA" else "transformer"
        transformer = getattr(pipe, tr_key)

        width = max(256, round(width / 32) * 32)
        height = max(256, round(height / 32) * 32)

        gen_dev = "cpu" if device.type == "mps" else device
        generator = torch.Generator(device=gen_dev).manual_seed(int(seed))
        init_inputs: Dict[str, Any] = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_inference_steps": max(6, int(num_steps)),
            "generator": generator,
            "output_type": "pil",
        }

        if partition == "Ref2VA":
            from diffusers.modular_pipelines.minimax_h3.references import (
                MiniMaxH3AudioReference,
                MiniMaxH3ImageReference,
                MiniMaxH3VideoReference,
            )

            refs: List[Any] = []
            imgs = list(reference_images or [])
            if not imgs:
                for fallback_img in (start_image, end_image):
                    if fallback_img is not None:
                        imgs.append(fallback_img)
            for im in imgs[:9]:
                refs.append(MiniMaxH3ImageReference(image=im.convert("RGB")))
            for v_url in (reference_videos or [])[:3]:
                try:
                    refs.append(MiniMaxH3VideoReference.from_file(v_url))
                except Exception:
                    pass
            for a_url in (reference_audios or [])[:3]:
                if refs:  # Audio references must be paired with at least one visual reference
                    try:
                        refs.append(MiniMaxH3AudioReference.from_file(a_url))
                    except Exception:
                        pass
            if refs:
                init_inputs["references"] = refs
        else:
            if start_image is not None:
                init_inputs["image"] = start_image
            if end_image is not None:
                init_inputs["last_image"] = end_image

        state = _build_minimax_pipeline_state(pipe, init_inputs)
        pipe._strata_device = device
        sub_blocks = pipe.blocks.sub_blocks

        # Stage 0: Layout / pre-encode preparation (lightweight)
        pipe, state = sub_blocks["before_encode"](pipe, state)

        # Stage 1: Qwen3-VL-32B Layer-50 conditioning + immediate VRAM eviction
        pipe, state = self._run_stage1_text_encoder(pipe, state, device, on_log=on_log)

        # Stage 1b: Condition VAE encoding (when start/end/ref media is provided)
        pipe.vae.to(device)
        if getattr(pipe, "audio_vae", None) is not None:
            pipe.audio_vae.to(device)
        pipe, state = sub_blocks["vae_encoder"](pipe, state)
        pipe.vae.to("cpu")
        if getattr(pipe, "audio_vae", None) is not None:
            pipe.audio_vae.to("cpu")
        clear_device_cache(device)

        # Stage 2: 33B DiT Denoising (Strata 30/50 VRAM Counter + 20/50 Pinned RAM Pantry)
        if self._dit_offloader is not None:
            for name, child in transformer.named_children():
                if name != "transformer_blocks":
                    child.to(device)
            self._dit_offloader.prepare()
            self._dit_offloader.attach_hooks()
            counter_cnt = self._dit_offloader.resident_vram_blocks
            pantry_cnt = self._dit_offloader.total_blocks - counter_cnt
        else:
            transformer.to(device)
            counter_cnt = len(transformer.transformer_blocks)
            pantry_cnt = 0

        if on_log:
            on_log(
                f"[MiniMax-H3 Stage 2/3] Denoising 33B DiT ({counter_cnt}/50 Counter, "
                f"{pantry_cnt}/50 Pinned RAM Pantry)..."
            )
        pipe, state = sub_blocks["denoise"](pipe, state)
        if self._dit_offloader is not None:
            self._dit_offloader.release_all_to_cpu()
        transformer.to("cpu")
        clear_device_cache(device)

        # Stage 3: Tiled ViT Video VAE + BigVGAN Audio VAE Decode -> H.264/AAC MP4
        self.memory.begin_vae_stage()
        pipe.vae.to(device)
        pipe.audio_vae.to(device)
        if on_log:
            on_log("[MiniMax-H3 Stage 3/3] Decoding Video ViT VAE + 32kHz BigVGAN Stereo Audio...")
        pipe, state = sub_blocks["decode"](pipe, state)
        pipe.vae.to("cpu")
        pipe.audio_vae.to("cpu")
        clear_device_cache(device)
        self.memory.finish_generation()

        frames = state.get("videos")[0]
        audio_tensor = state.get("audio")
        audio = audio_tensor[0] if (enable_audio and audio_tensor is not None) else None
        sr = int(state.get("sampling_rate") or 32000)
        if output_path is not None:
            encode_frames_to_mp4(
                frames=frames,
                out_path=output_path,
                fps=24,
                audio_waveform=audio,
                audio_sample_rate=sr,
            )
        return output_path or Path("minimax_output.mp4")

    def _evict_active_pipe_only(self) -> None:
        if self._pipe is None:
            return
        if self._dit_offloader is not None:
            self._dit_offloader.release_all_to_cpu()
            self._dit_offloader = None
        for attr in ("transformer", "transformer_ref", "vae", "audio_vae"):
            comp = getattr(self._pipe, attr, None)
            if comp is not None:
                comp.to("cpu")
        self._pipe = None
        self._loaded_partition = None
        clear_device_cache()

    def evict(self) -> None:
        self._evict_active_pipe_only()
        if self._te_offloader is not None:
            self._te_offloader.release_all_to_cpu()
        if self._text_encoder is not None:
            self._text_encoder.to("cpu")
        clear_device_cache()
