"""Attention parity and long-reference CUDA memory regressions."""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from openfal.runtime.attention import configure_qwen_prompt_attention, qwen_prompt_attention


@pytest.mark.parametrize("masked", [False, True])
def test_prompt_attention_matches_grouped_query_math(masked):
    torch.manual_seed(31)
    q = torch.randn(2, 8, 13, 16)
    k = torch.randn(2, 2, 13, 16)
    v = torch.randn_like(k)
    mask = torch.ones(2, 1, 13, 13, dtype=torch.bool).tril() if masked else None
    if mask is not None:
        mask[0, :, :, 4] = False
    expected = F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, is_causal=mask is None, enable_gqa=True, scale=0.37
    ).transpose(1, 2)
    actual, weights = qwen_prompt_attention(SimpleNamespace(is_causal=True), q, k, v, mask, scaling=0.37)
    torch.testing.assert_close(actual, expected)
    assert weights is None


def test_backend_registration_preserves_qwen_padding_and_vision_backend():
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    config = Qwen3VLConfig(
        text_config={
            "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
            "vocab_size": 32,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [1, 1, 2]},
        },
        vision_config={
            "hidden_size": 32, "intermediate_size": 64, "depth": 1,
            "num_heads": 4, "out_hidden_size": 32, "deepstack_visual_indexes": [],
        },
    )
    torch.manual_seed(7)
    model = Qwen3VLForConditionalGeneration(config).eval()
    vision_backend = model.config.vision_config._attn_implementation
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    mask = torch.tensor([[1, 1, 1, 1], [0, 0, 1, 1]])
    with torch.inference_mode():
        expected = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        configure_qwen_prompt_attention(model)
        actual = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    torch.testing.assert_close(actual, expected)
    assert model.config.vision_config._attn_implementation == vision_backend
    assert model.config.text_config._attn_implementation == "openfal_qwen_sdpa"


@pytest.mark.skipif(os.environ.get("OPENFAL_TEST_CUDA") != "1", reason="opt-in CUDA memory test")
def test_long_multimodal_prompt_uses_linear_memory_cuda_attention():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    # Qwen3-VL-32B's actual 64 query / 8 KV heads, with a visual prompt long
    # enough that a float32 math attention matrix alone would exceed 8 GiB.
    tokens = 6145
    torch.manual_seed(31)
    q = torch.randn(1, 64, tokens, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, 8, tokens, 128, dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        result, _ = qwen_prompt_attention(SimpleNamespace(is_causal=True), q, k, v, None)
        torch.cuda.synchronize()
    assert torch.isfinite(result).all()
    assert torch.cuda.max_memory_allocated() - baseline < 1024**3
    # Causal prefix is independent of all following tokens; compare an actual
    # fused result to the grouped-query CPU reference without a quadratic GPU run.
    reference = F.scaled_dot_product_attention(
        q[:, :, :73].float().cpu(), k[:, :, :73].float().cpu(), v[:, :, :73].float().cpu(),
        is_causal=True, enable_gqa=True,
    ).transpose(1, 2)
    torch.testing.assert_close(result[:, :73].float().cpu(), reference, rtol=0.02, atol=0.02)
