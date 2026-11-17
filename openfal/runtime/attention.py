"""Memory-bounded attention for the streamed Qwen multimodal conditioner."""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F


_QWEN_ATTENTION_NAME = "openfal_qwen_sdpa"


def qwen_prompt_attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Qwen prefill SDPA with explicit grouped KV expansion and fused CUDA kernels.

    Transformers enables native GQA for unmasked Qwen prompts. On Windows builds
    without Flash Attention this excludes memory-efficient SDPA, so long visual
    prompts can fall back to a quadratic attention matrix and exhaust VRAM.
    Repeating KV heads is the same grouped-query computation and lets the fused
    memory-efficient kernel handle it with memory linear in prompt length.
    """
    if kwargs.get("softcap") is not None or kwargs.get("position_bias") is not None:
        raise ValueError("Qwen prompt attention does not support softcap or position bias")

    groups = query.shape[1] // key.shape[1]
    if query.shape[1] % key.shape[1] or key.shape[1] != value.shape[1]:
        raise ValueError("Qwen query heads must be divisible by matching key/value heads")
    if groups > 1:
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)

    causal = getattr(module, "is_causal", True) if is_causal is None else is_causal
    causal = query.shape[2] > 1 and attention_mask is None and causal
    # Match Transformers' prefill handling for an empty static KV cache.
    if causal and key.shape[2] > query.shape[2]:
        key = key[:, :, : query.shape[2], :]
        value = value[:, :, : query.shape[2], :]

    backend_context = nullcontext()
    if query.device.type == "cuda":
        from torch.nn.attention import SDPBackend, sdpa_kernel

        backends = [SDPBackend.EFFICIENT_ATTENTION]
        if torch.backends.cuda.is_flash_attention_available():
            backends.insert(0, SDPBackend.FLASH_ATTENTION)
        # Never silently allocate the quadratic math fallback for reference video.
        backend_context = sdpa_kernel(backends)

    with backend_context:
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=dropout,
            scale=scaling,
            is_causal=causal,
        )
    return output.transpose(1, 2).contiguous(), None


def configure_qwen_prompt_attention(model: torch.nn.Module) -> None:
    """Select this backend only for this model's Qwen language backbone."""
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

    AttentionInterface.register(_QWEN_ATTENTION_NAME, qwen_prompt_attention)
    # A custom name without a mask formatter silently drops padding/causal masks.
    AttentionMaskInterface.register(_QWEN_ATTENTION_NAME, sdpa_mask)
    model.set_attn_implementation({"text_config": _QWEN_ATTENTION_NAME})
