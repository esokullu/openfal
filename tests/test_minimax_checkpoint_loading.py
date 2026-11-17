"""Parity and failure checks for Comfy MiniMax checkpoints, without model weights."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from openfal.runtime.minimax_h3_runner import (
    _materialize_remaining_meta,
    _load_minimax_references,
    _retain_conditioning_hidden_state,
    _PassthroughDecoderLayer,
    _validate_checkpoint_keys,
    convert_minimax_h3_dit_state_dict,
    convert_minimax_h3_video_vae_state_dict,
)


def test_vae_conversion_preserves_fused_attention_and_swiglu_math():
    """Compare to the original head-interleaved QKV and gate-first FFN operations."""
    torch.manual_seed(11)
    dim, heads, head_dim = 8, 2, 4
    x = torch.randn(2, 3, dim)
    qkv_w, qkv_b = torch.randn(3 * dim, dim), torch.randn(3 * dim)
    ff_w, ff_b = torch.randn(4 * dim, dim), torch.randn(4 * dim)
    raw = {
        "decoder.transformer_blocks.0.attn.to_qkv.weight": qkv_w,
        "decoder.transformer_blocks.0.attn.to_qkv.bias": qkv_b,
        "decoder.transformer_blocks.0.ff.w1.weight": ff_w,
        "decoder.transformer_blocks.0.ff.w1.bias": ff_b,
    }
    mapped = convert_minimax_h3_video_vae_state_dict(raw, heads, head_dim)
    reference_qkv = F.linear(x, qkv_w, qkv_b).view(2, 3, heads, 3 * head_dim).chunk(3, dim=-1)
    for name, expected in zip(("q", "k", "v"), reference_qkv):
        key = f"decoder.transformer_blocks.0.attn.to_{name}"
        actual = F.linear(x, mapped[f"{key}.weight"], mapped[f"{key}.bias"])
        torch.testing.assert_close(actual.view(2, 3, heads, head_dim), expected)
    gate, value = F.linear(x, ff_w, ff_b).chunk(2, dim=-1)
    key = "decoder.transformer_blocks.0.ff.net.0.proj"
    converted_value, converted_gate = F.linear(x, mapped[f"{key}.weight"], mapped[f"{key}.bias"]).chunk(2, dim=-1)
    torch.testing.assert_close(converted_value * F.silu(converted_gate), F.silu(gate) * value)


def test_pdd_delta_layout_does_not_depend_on_head_norms():
    base, delta = torch.ones(4, 3), torch.full((4, 3), 8.0)
    converted, _ = convert_minimax_h3_dit_state_dict(
        {"final_layer.video_out.weight": torch.cat([base, delta])}, num_pdd_heads=2
    )
    torch.testing.assert_close(converted["pdd_proj_out.1.weight"], base + delta)
    native = {"pdd_proj_out.0.weight": base, "pdd_proj_out.1.weight": delta}
    converted, _ = convert_minimax_h3_dit_state_dict(native, num_pdd_heads=2)
    torch.testing.assert_close(converted["pdd_proj_out.1.weight"], delta)


def test_native_diffusers_weights_are_not_swapped_a_second_time():
    key = "transformer_blocks.0.ff.net.0.proj.weight"
    weight = torch.arange(24.0).reshape(8, 3)
    native_dit = {key: weight}
    converted, _ = convert_minimax_h3_dit_state_dict(native_dit)
    torch.testing.assert_close(converted[key], weight)
    output_key = "decoder.transformer_blocks.0.attn.to_out.0.weight"
    native_vae = {f"decoder.{key}": weight, output_key: weight}
    converted = convert_minimax_h3_video_vae_state_dict(native_vae)
    torch.testing.assert_close(converted[f"decoder.{key}"], weight)
    torch.testing.assert_close(converted[output_key], weight)


def test_incomplete_checkpoint_cannot_fill_active_weights_with_zeros():
    with pytest.raises(RuntimeError, match="checkpoint is incompatible"):
        _validate_checkpoint_keys("test DiT", ["proj_in.weight"], [])
    with pytest.raises(RuntimeError, match="checkpoint is incompatible"):
        _validate_checkpoint_keys("test DiT", [], ["unknown.weight"])
    _validate_checkpoint_keys("pruned encoder", ["norm.weight"], [], ("norm.weight",))
    module = nn.Linear(4, 4, device="meta")
    with pytest.raises(RuntimeError, match="Unloaded model parameter"):
        _materialize_remaining_meta(module, torch.float32)


def test_bad_reference_media_is_not_silently_discarded(monkeypatch):
    from PIL import Image
    from diffusers.modular_pipelines.minimax_h3.references import MiniMaxH3VideoReference

    def fail_decode(path):
        raise OSError("corrupt video")

    monkeypatch.setattr(MiniMaxH3VideoReference, "from_file", fail_decode)
    with pytest.raises(ValueError, match="requires at least one image or video reference"):
        _load_minimax_references()
    with pytest.raises(ValueError, match="Unable to decode MiniMax-H3 video reference 1"):
        _load_minimax_references(videos=["broken.mp4"])
    with pytest.raises(ValueError, match="require at least one image or video"):
        _load_minimax_references(audios=["voice.wav"])
    with pytest.raises(ValueError, match="at most 9 image references"):
        _load_minimax_references(images=[Image.new("RGB", (1, 1))] * 10)
    with pytest.raises(ValueError, match="at most 12 references in total"):
        _load_minimax_references(
            images=[Image.new("RGB", (1, 1))] * 9,
            videos=["clip.mp4"] * 3,
            audios=["voice.wav"],
        )


def test_reference_wav_resamples_to_native_rate_without_torchaudio(tmp_path, monkeypatch):
    import builtins
    import wave

    import numpy as np
    from PIL import Image

    # Real 44.1 kHz stereo WAV: different tones make channel swaps/mixing visible.
    source_rate, target_rate = 44100, 32000
    time = np.arange(source_rate) / source_rate
    source = 0.25 * np.stack([np.sin(2 * np.pi * 440 * time), np.sin(2 * np.pi * 880 * time)], axis=1)
    path = tmp_path / "stereo_44100.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(source_rate)
        handle.writeframes((source * 32767).astype("<i2").tobytes())

    original_import = builtins.__import__

    def no_torchaudio(name, *args, **kwargs):
        if name == "torchaudio" or name.startswith("torchaudio."):
            raise ImportError("torchaudio deliberately unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torchaudio)
    refs = _load_minimax_references(images=[Image.new("RGB", (32, 32))], audios=[str(path)])
    audio = refs[1]
    assert audio.sample_rate == target_rate
    assert audio.audio.shape == (2, target_rate)
    spectrum = torch.fft.rfft(audio.audio)
    peaks = spectrum.abs().argmax(dim=-1)
    torch.testing.assert_close(peaks, torch.tensor([440, 880]))
    assert torch.isfinite(audio.audio).all()
    assert 0.24 < audio.audio.abs().max() < 0.26


def test_qwen_conditioner_retains_only_exact_prenorm_layer_output():
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    config = Qwen3VLConfig(
        text_config={
            "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 4,
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
    model = Qwen3VLForConditionalGeneration(config).eval().model
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        expected = model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states[2]
        # Mirror the production checkpoint: retained real layers, then a no-op tail.
        model.language_model.layers = nn.ModuleList(
            list(model.language_model.layers[:2]) + [_PassthroughDecoderLayer()]
        )
        _retain_conditioning_hidden_state(model, layer_index=2)
        result = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        assert result.hidden_states[:2] == (None, None)
        torch.testing.assert_close(result.hidden_states[2], expected)
        # This specifically catches accidental substitution by the final RMSNorm.
        assert not torch.allclose(result.hidden_states[2], result.last_hidden_state)
        assert model(input_ids=ids, use_cache=False).hidden_states is None
