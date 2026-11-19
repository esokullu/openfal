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
    ensure_model_configs,
    find_matching_safetensors,
    load_safetensors_state_dict,
    resolve_compute_device_and_dtype,
)
from openfal.runtime.quant import StrataBlockOffloadManager, quantized_module_swap_from_state_dict
from openfal.runtime.attention import configure_qwen_prompt_attention
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


def _swap_swiglu_gate_value(tensor: torch.Tensor) -> torch.Tensor:
    """Reorder fused SwiGLU projections between Comfy `[gate; value]` and Diffusers `[value; gate]`."""
    gate, val = torch.chunk(tensor, 2, dim=0)
    return torch.cat([val, gate], dim=0).contiguous()


def _split_qkv_tensor(
    tensor: torch.Tensor,
    qkv_layout: str = "stacked",
    num_attention_heads: Optional[int] = None,
    attention_head_dim: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split fused QKV tensors (`stacked` [Q;K;V] or head-`interleaved` [H,3,D,...]) into `(to_q, to_k, to_v)`."""
    if qkv_layout == "interleaved":
        rows = tensor.shape[0]
        if (
            num_attention_heads is not None
            and attention_head_dim > 0
            and rows == num_attention_heads * 3 * attention_head_dim
        ):
            n_heads, head_dim = num_attention_heads, attention_head_dim
        elif attention_head_dim > 0 and rows % (3 * attention_head_dim) == 0:
            n_heads, head_dim = rows // (3 * attention_head_dim), attention_head_dim
        elif num_attention_heads is not None and num_attention_heads > 0 and rows % (3 * num_attention_heads) == 0:
            n_heads, head_dim = num_attention_heads, rows // (3 * num_attention_heads)
        else:
            raise ValueError(
                f"Cannot split interleaved QKV tensor of shape {tuple(tensor.shape)} "
                f"with num_attention_heads={num_attention_heads}, attention_head_dim={attention_head_dim}"
            )
        unflat = tensor.unflatten(0, (n_heads, 3, head_dim))
        q, k, v = [w.flatten(0, 1).contiguous() for w in unflat.unbind(dim=1)]
        return q, k, v

    q, k, v = torch.chunk(tensor, 3, dim=0)
    return q.contiguous(), k.contiguous(), v.contiguous()


def _split_qkv_and_swap_swiglu_in_place(
    out: Dict[str, torch.Tensor],
    qkv_token: str,
    qkv_layout: str,
    num_attention_heads: Optional[int],
    attention_head_dim: int,
    swiglu_bases: set[str],
) -> None:
    """In-place reorder fused `.ff.net.0.proj.*` and split fused QKV attention tensors + quant scales."""
    fc1_rows = {k.removesuffix(".weight"): v.shape[0] for k, v in out.items() if k.endswith(".ff.net.0.proj.weight")}
    qkv_weight_suffix = f"{qkv_token}weight"
    qkv_base_trim = len(".weight")
    qkv_rows = {k[:-qkv_base_trim]: v.shape[0] for k, v in out.items() if k.endswith(qkv_weight_suffix)}

    for k in list(out.keys()):
        if k.rsplit(".", 1)[0] in swiglu_bases and k.endswith((".weight", ".bias")):
            out[k] = _swap_swiglu_gate_value(out[k])
        elif k.rsplit(".", 1)[0] in swiglu_bases and k.endswith((".weight_scale", ".scale_weight")):
            base = k.rsplit(".", 1)[0]
            rows = fc1_rows.get(base)
            if out[k].ndim >= 1 and rows and out[k].shape[0] == rows:
                out[k] = _swap_swiglu_gate_value(out[k])
        elif qkv_token in k:
            val = out.pop(k)
            prefix, suffix = k.split(qkv_token, 1)
            qkv_base = f"{prefix}{qkv_token[:-1]}"
            if suffix in ("weight", "bias"):
                q, kk, vv = _split_qkv_tensor(val, qkv_layout, num_attention_heads, attention_head_dim)
            elif suffix in ("weight_scale", "scale_weight") and val.ndim >= 1 and val.shape[0] == qkv_rows.get(qkv_base):
                q, kk, vv = _split_qkv_tensor(val, qkv_layout, num_attention_heads, attention_head_dim)
            else:
                q, kk, vv = val, val.clone(), val.clone()
            out[f"{prefix}.attn.to_q.{suffix}"] = q
            out[f"{prefix}.attn.to_k.{suffix}"] = kk
            out[f"{prefix}.attn.to_v.{suffix}"] = vv


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
    swiglu_bases: set[str] = set()

    for k, v in raw_sd.items():
        if k in ("adaln_t_table", "rope.inv_freq", "scaled_fp8") or k.endswith(
            (".scale_input", ".input_scale", ".comfy_attention.config")
        ):
            continue
        nk = k.replace("blocks.", "transformer_blocks.", 1) if k.startswith("blocks.") else k
        for old, new in rename_map.items():
            nk = nk.replace(old, new)
        out[nk] = v
        if ".mlp.fc1." in k:
            swiglu_bases.add(nk.rsplit(".", 1)[0])

    # Split Comfy `final_layer.video_out` (`[num_pdd_heads * 96, ...]`) for MiniMax-H3-Image.
    # Comfy stores `chunk[0]` as head 0 and `chunks[1..H-1]` as residual deltas `(head_i - head_0)`.
    if num_pdd_heads > 0:
        for suffix in ("weight", "bias"):
            pk = f"proj_out.{suffix}"
            if pk in out and out[pk].shape[0] % num_pdd_heads == 0:
                chunks = torch.chunk(out.pop(pk), num_pdd_heads, dim=0)
                base = chunks[0]
                out[f"pdd_proj_out.0.{suffix}"] = base.contiguous()
                for h_idx in range(1, len(chunks)):
                    # This is a checkpoint layout, not a magnitude heuristic. Native
                    # Diffusers checkpoints already have separate pdd_proj_out keys.
                    head_tensor = (base.float() + chunks[h_idx].float()).to(base.dtype)
                    out[f"pdd_proj_out.{h_idx}.{suffix}"] = head_tensor.contiguous()

    _split_qkv_and_swap_swiglu_in_place(
        out,
        qkv_token=".attn.qkv_proj.",
        qkv_layout=qkv_layout,
        num_attention_heads=num_attention_heads,
        attention_head_dim=attention_head_dim,
        swiglu_bases=swiglu_bases,
    )
    return out, adaln_t_table


def convert_minimax_h3_video_vae_state_dict(
    raw_sd: Dict[str, torch.Tensor],
    num_attention_heads: Optional[int] = None,
    attention_head_dim: int = 64,
    qkv_layout: str = "interleaved",
) -> Dict[str, torch.Tensor]:
    """
    Convert `Comfy-Org/MiniMax-H3` Video VAE or `Comfy-Org/MiniMax-H3-Image` Image VAE keys
    to Diffusers `AutoencoderKLMiniMaxH3` / `AutoencoderKLMiniMaxH3Image`, splitting fused
    head-interleaved `to_qkv` (`(num_heads, 3, head_dim)`) and reordering SwiGLU `[value; gate]`.
    """
    out: Dict[str, torch.Tensor] = {}
    swiglu_bases: set[str] = set()
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
        elif ".attn.to_out." in k and ".attn.to_out.0." not in k:
            nk = k.replace(".attn.to_out.", ".attn.to_out.0.")
        elif ".ff.w1." in k:
            nk = k.replace(".ff.w1.", ".ff.net.0.proj.")
            swiglu_bases.add(nk.rsplit(".", 1)[0])
        elif ".ff.w2." in k:
            nk = k.replace(".ff.w2.", ".ff.net.2.")
        else:
            nk = k
        out[nk] = v

    _split_qkv_and_swap_swiglu_in_place(
        out,
        qkv_token=".attn.to_qkv.",
        qkv_layout=qkv_layout,
        num_attention_heads=num_attention_heads,
        attention_head_dim=attention_head_dim,
        swiglu_bases=swiglu_bases,
    )
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


def _retain_conditioning_hidden_state(model: nn.Module, layer_index: int = 50) -> None:
    """Capture only the requested pre-final-norm language hidden state.

    The normal Transformers recorder retains every layer. Its last recorded state
    can also be replaced by the final norm when a pruned tail is a custom identity
    layer. Capture the real decoder output directly so both memory and semantics
    are independent of that recorder's treatment of pruned layers.
    """
    if getattr(model, "_openfal_conditioning_capture", False):
        return
    original_forward = model.forward
    conditioning_layer = model.language_model.layers[layer_index - 1]

    def forward(*args, **kwargs):
        if not kwargs.get("output_hidden_states", False):
            return original_forward(*args, **kwargs)
        captured = []

        def capture_layer(module, inputs, output):
            captured.append(output[0] if isinstance(output, tuple) else output)

        handle = conditioning_layer.register_forward_hook(capture_layer)
        kwargs["output_hidden_states"] = False
        try:
            output = original_forward(*args, **kwargs)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError(f"Expected one Qwen3-VL layer-{layer_index} hidden state, got {len(captured)}")
        output.hidden_states = (None,) * layer_index + (captured[0],)
        return output

    model.forward = forward
    model._openfal_conditioning_capture = True


def _validate_checkpoint_keys(
    component: str,
    missing: List[str],
    unexpected: List[str],
    allowed_missing: Tuple[str, ...] = (),
) -> None:
    """Reject incomplete checkpoints instead of generating with fabricated weights."""
    unexplained = sorted(set(missing) - set(allowed_missing))
    if unexplained or unexpected:
        raise RuntimeError(
            f"{component} checkpoint is incompatible: missing={unexplained}, unexpected={unexpected}. "
            "Reinstall the matching model with openfal download."
        )


_PRUNED_TIME_EMBEDDER_KEYS = (
    "time_embedder.linear_1.weight",
    "time_embedder.linear_1.bias",
    "time_embedder.linear_2.weight",
    "time_embedder.linear_2.bias",
)


def _materialize_remaining_meta(
    module: nn.Module, dtype: torch.dtype, allowed_parameters: Tuple[str, ...] = ()
) -> None:
    """Initialize only explicitly unused pruned parameters; reject other meta tensors."""
    from openfal.runtime.quant import _get_parent_and_attr

    for name, param in list(module.named_parameters(recurse=True)):
        if param.is_meta:
            if name not in allowed_parameters:
                raise RuntimeError(f"Unloaded model parameter: {name}")
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
                raise RuntimeError(f"Uninitialized model buffer: {bname}")


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


def _resample_reference_audio(reference: Any, sample_rate: int = 32000) -> Any:
    """Normalize decoded reference audio with PyAV, which also decodes the media.

    Diffusers otherwise imports torchaudio when a reference has a different sample
    rate. Keeping this at the file boundary avoids a second binary dependency that
    must match the user's exact PyTorch/CUDA build.
    """
    if not reference.has_audio or reference.sample_rate in (None, sample_rate):
        return reference
    import av

    waveform = reference.audio.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if waveform.ndim != 2 or waveform.shape[0] not in (1, 2):
        raise ValueError("MiniMax-H3 reference audio must be mono or stereo")
    if reference.sample_rate <= 0:
        raise ValueError("MiniMax-H3 reference audio must have a positive sample rate")
    layout = "mono" if waveform.shape[0] == 1 else "stereo"
    frame = av.AudioFrame.from_ndarray(waveform.numpy(), format="fltp", layout=layout)
    frame.sample_rate = int(reference.sample_rate)
    resampler = av.AudioResampler(format="fltp", layout=layout, rate=sample_rate)
    converted = resampler.resample(frame) + resampler.resample(None)
    if not converted:
        raise ValueError("MiniMax-H3 reference audio is empty after resampling")
    reference.audio = torch.cat([torch.from_numpy(item.to_ndarray()) for item in converted], dim=-1)
    reference.sample_rate = sample_rate
    return reference


def _load_minimax_references(
    images: Optional[List[Image.Image]] = None,
    videos: Optional[List[str]] = None,
    audios: Optional[List[str]] = None,
) -> List[Any]:
    """Load every requested reference, or fail before generation can ignore one."""
    from diffusers.modular_pipelines.minimax_h3.references import (
        MiniMaxH3AudioReference,
        MiniMaxH3ImageReference,
        MiniMaxH3VideoReference,
    )

    images, videos, audios = list(images or []), list(videos or []), list(audios or [])
    if not (images or videos or audios):
        raise ValueError("MiniMax-H3 reference-to-video requires at least one image or video reference")
    for kind, refs, limit in (("image", images, 9), ("video", videos, 3), ("audio", audios, 3)):
        if len(refs) > limit:
            raise ValueError(f"MiniMax-H3 supports at most {limit} {kind} references per request")
    if len(images) + len(videos) + len(audios) > 12:
        raise ValueError("MiniMax-H3 supports at most 12 references in total")
    if audios and not (images or videos):
        raise ValueError("MiniMax-H3 audio references require at least one image or video reference")
    refs = [MiniMaxH3ImageReference(image=im.convert("RGB")) for im in images]
    for kind, paths, reference_class in (
        ("video", videos, MiniMaxH3VideoReference),
        ("audio", audios, MiniMaxH3AudioReference),
    ):
        for index, path in enumerate(paths, 1):
            try:
                refs.append(_resample_reference_audio(reference_class.from_file(path)))
            except Exception as exc:
                raise ValueError(f"Unable to decode MiniMax-H3 {kind} reference {index}: {exc}") from exc
    return refs


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
            dit = find_matching_safetensors(self.weights_dir / "H3-Image" / "diffusion_models", ["minimax_h3_image"])
            if dit is None:
                dit = find_matching_safetensors(self.weights_dir / "H3-Image", ["minimax_h3_image_pruned"])

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
        target = ensure_model_configs("minimax_h3", self.weights_dir, on_log=on_log)
        self._cfg_dir = target
        return target

    def _ensure_image_configs(self, on_log: Optional[Callable[[str], None]] = None) -> Path:
        if self._img_cfg_dir is not None:
            return self._img_cfg_dir
        target = ensure_model_configs("minimax_h3_image", self.weights_dir, on_log=on_log)
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

        te_missing, te_unexpected = quantized_module_swap_from_state_dict(
            text_encoder,
            te_mapped,
            compute_dtype=compute_dtype,
            embedding_names=("model.language_model.embed_tokens",),
        )
        del te_mapped
        unused_norm = ("model.language_model.norm.weight",) if not has_layer_50 else ()
        _validate_checkpoint_keys("MiniMax-H3 Qwen3-VL", te_missing, te_unexpected, unused_norm)

        text_encoder.model.language_model.rotary_emb = Qwen3VLTextRotaryEmbedding(config=te_cfg.text_config)
        try:
            text_encoder.model.visual.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(te_cfg.vision_config)
        except (TypeError, AttributeError):
            head_dim = te_cfg.vision_config.hidden_size // te_cfg.vision_config.num_heads
            text_encoder.model.visual.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)
        _materialize_remaining_meta(text_encoder, compute_dtype, unused_norm)
        text_encoder.eval()
        _retain_conditioning_hidden_state(text_encoder.model)
        configure_qwen_prompt_attention(text_encoder)

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
        # Adopt the reference qwen-vl-utils per-frame budget, supported by current
        # Transformers and becoming its default. This does not resize VAE refs.
        if hasattr(self._processor.video_processor, "cap_pixels_per_frame"):
            self._processor.video_processor.cap_pixels_per_frame = True
        if on_log:
            on_log(
                f"[MiniMax-H3 Qwen3-VL-32B] Loaded '{te_path.name}' "
                f"({te_resident}/{len(te_layers)} layers Counter, all active weights verified)"
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

        check_inference_dependencies(("diffusers", "transformers", "accelerate", "torchvision"))

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
            _validate_checkpoint_keys(
                "MiniMax-H3 Image DiT", dit_missing, dit_unexp,
                _PRUNED_TIME_EMBEDDER_KEYS if adaln_t_table is not None else (),
            )

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
            transformer.eval()

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
                    "all active weights verified)"
                )

            ivae_cfg = AutoencoderKLMiniMaxH3Image.load_config(str(img_cfg_dir / "vae_diffusers"))
            with init_empty_weights():
                image_vae = AutoencoderKLMiniMaxH3Image.from_config(ivae_cfg)
            ivae_raw = load_safetensors_state_dict(paths["image_vae"], device="cpu")
            ivae_mapped = convert_minimax_h3_video_vae_state_dict(
                ivae_raw,
                num_attention_heads=int(ivae_cfg.get("decoder_num_attention_heads", 16)),
                attention_head_dim=int(ivae_cfg.get("decoder_attention_head_dim", 64)),
            )
            del ivae_raw
            ivae_missing, ivae_unexpected = quantized_module_swap_from_state_dict(
                image_vae, ivae_mapped, compute_dtype=compute_dtype
            )
            _validate_checkpoint_keys("MiniMax-H3 Image VAE", ivae_missing, ivae_unexpected)
            del ivae_mapped
            from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_image import MiniMaxH3ImageRotaryPosEmbed as _VaeRope

            image_vae.decoder.rope = _VaeRope(
                int(ivae_cfg.get("decoder_attention_head_dim", 64) * ivae_cfg.get("decoder_rope_dim_ratio", 0.5)),
                theta=ivae_cfg.get("decoder_rope_theta", 100.0),
                num_axes=2,
            )
            _materialize_remaining_meta(image_vae, compute_dtype)
            image_vae.eval()

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
        _validate_checkpoint_keys(
            "MiniMax-H3 Video DiT", dit_missing, dit_unexp,
            _PRUNED_TIME_EMBEDDER_KEYS if adaln_t_table is not None else (),
        )

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
        transformer.eval()

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
                "all active weights verified)"
            )

        # 3. Video VAE (CNN Encoder + 36-layer 3D ViT Decoder)
        vvae_cfg = AutoencoderKLMiniMaxH3.load_config(str(cfg_dir / "vae"))
        with init_empty_weights():
            video_vae = AutoencoderKLMiniMaxH3.from_config(vvae_cfg)
        vvae_raw = load_safetensors_state_dict(paths["video_vae"], device="cpu")
        vvae_mapped = convert_minimax_h3_video_vae_state_dict(
            vvae_raw,
            num_attention_heads=int(vvae_cfg.get("decoder_num_attention_heads", 32)),
            attention_head_dim=int(vvae_cfg.get("decoder_attention_head_dim", 64)),
        )
        del vvae_raw
        vvae_missing, vvae_unexpected = quantized_module_swap_from_state_dict(
            video_vae, vvae_mapped, compute_dtype=compute_dtype
        )
        _validate_checkpoint_keys("MiniMax-H3 Video VAE", vvae_missing, vvae_unexpected)
        del vvae_mapped
        from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import MiniMaxH3VideoRotaryPosEmbed

        video_vae.decoder.rope = MiniMaxH3VideoRotaryPosEmbed(
            int(vvae_cfg.get("decoder_attention_head_dim", 64) * vvae_cfg.get("decoder_rope_dim_ratio", 0.5)),
            theta=vvae_cfg.get("decoder_rope_theta", 100.0),
        )
        _materialize_remaining_meta(video_vae, compute_dtype)
        video_vae.eval()

        # 4. Audio VAE (BigVGAN 32kHz Stereo)
        avae_cfg = AutoencoderKLMiniMaxH3Audio.load_config(str(cfg_dir / "audio_vae"))
        audio_vae = AutoencoderKLMiniMaxH3Audio.from_config(avae_cfg)
        _strip_audio_weight_norm(audio_vae)
        avae_raw = load_safetensors_state_dict(paths["audio_vae"], device="cpu")
        avae_sd = {k: v for k, v in avae_raw.items() if k not in ("latents_mean", "latents_std")}
        audio_vae.load_state_dict(avae_sd, strict=True, assign=True)
        audio_vae.eval()

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
        refs = None
        if partition == "Ref2VA":
            imgs = list(reference_images or [])
            if not imgs:
                imgs = [im for im in (start_image, end_image) if im is not None]
            refs = _load_minimax_references(imgs, reference_videos, reference_audios)
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
        import gc

        if self._dit_offloader is not None:
            self._dit_offloader.release_all_to_cpu()
            self._dit_offloader = None
        for attr in ("transformer", "transformer_ref", "vae", "audio_vae"):
            comp = getattr(self._pipe, attr, None)
            if comp is not None:
                comp.to("cpu")
        self._pipe = None
        self._loaded_partition = None
        gc.collect()
        clear_device_cache()

    def evict(self) -> None:
        self._evict_active_pipe_only()
        if self._te_offloader is not None:
            self._te_offloader.release_all_to_cpu()
        if self._text_encoder is not None:
            self._text_encoder.to("cpu")
        clear_device_cache()
