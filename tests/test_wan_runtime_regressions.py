"""Regressions for the real Wan/quantized inference paths, without model downloads."""

import os
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from openfal.config import EngineConfig
from openfal.engine.memory_hierarchy import StrataMemoryHierarchy
from openfal.runtime.quant import StrataBlockOffloadManager, StrataFP8ScaledLinear, quantized_module_swap_from_state_dict
from openfal.runtime.wan22_runner import (
    RealWan22Runner, WAN_FLOW_SHIFTS, _materialize_remaining_meta, _set_wan_timesteps,
)


def test_fp8_umt5_matches_dequantized_encoder():
    """Exercise UMT5's actual wo.weight dtype inspection, not just Linear.forward."""
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(91)
    cfg = transformers.UMT5Config(
        vocab_size=64, d_model=32, d_ff=64, d_kv=8, num_heads=4,
        num_layers=2, dropout_rate=0.1, feed_forward_proj="gated-gelu",
    )
    reference = transformers.UMT5EncoderModel(cfg).eval()
    quantized = transformers.UMT5EncoderModel(cfg).eval()
    sd = dict(reference.state_dict())
    dense_sd = dict(sd)
    for name, module in reference.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        weight = sd[f"{name}.weight"]
        scale = weight.abs().max().clamp_min(1e-8) / 448.0
        packed = (weight / scale).to(torch.float8_e4m3fn)
        sd[f"{name}.weight"] = packed
        sd[f"{name}.scale_weight"] = scale
        dense_sd[f"{name}.weight"] = packed.float() * scale
    reference.load_state_dict(dense_sd)
    missing, unexpected = quantized_module_swap_from_state_dict(quantized, sd, compute_dtype=torch.float32)
    assert not missing and not unexpected
    assert isinstance(quantized.encoder.block[0].layer[1].DenseReluDense.wo, StrataFP8ScaledLinear)
    ids = torch.tensor([[3, 15, 29, 1, 0]])
    mask = ids.ne(0)
    with torch.inference_mode():
        actual = quantized(ids, mask).last_hidden_state
        expected = reference(ids, mask).last_hidden_state
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)
    for module in quantized.modules():
        if isinstance(module, StrataFP8ScaledLinear):
            assert module.weight.dtype == torch.float32
            assert module.weight.shape == module.weight_fp8.shape
            assert module.weight_fp8.dtype == torch.float8_e4m3fn


def test_prefetched_tensors_record_compute_stream_before_cpu_eviction(monkeypatch):
    """Transfer-stream allocations must survive asynchronous compute after eviction."""
    manager = StrataBlockOffloadManager(nn.ModuleList([nn.Identity()]), torch.device("cpu"), 0)
    manager.device = torch.device("cuda:0")
    stream, event = Mock(), Mock()
    parameter, buffer = Mock(), Mock()
    manager._staged_gpu_tensors[0] = ({"p": parameter}, {"b": buffer}, event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    monkeypatch.setattr(manager, "_assign_nested_parameter", Mock())
    monkeypatch.setattr(manager, "_assign_nested_buffer", Mock())
    manager._activate_block(0)
    stream.wait_event.assert_called_once_with(event)
    parameter.record_stream.assert_called_once_with(stream)
    buffer.record_stream.assert_called_once_with(stream)
    assert not manager._staged_gpu_tensors


def test_streaming_slot_reclaims_retired_weights_before_prefetch():
    manager = StrataBlockOffloadManager(nn.ModuleList([nn.Identity()]), torch.device("cpu"), 0)
    manager._use_cuda_stream = True
    manager._active_gpu_blocks.add(3)
    retired = Mock()
    retired.query.return_value = False
    manager._retired_compute_events.append(retired)
    manager._wait_for_staging_slot()
    retired.synchronize.assert_called_once()
    assert not manager._retired_compute_events


@pytest.mark.skipif(os.getenv("OPENFAL_TEST_CUDA") != "1", reason="opt-in real CUDA streaming regression")
def test_cuda_streamed_forward_matches_resident_weights():
    """Exercise allocator reuse while earlier blocks still have queued CUDA work."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    class DelayedBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(512, 512, bias=False)

        def forward(self, x):
            if x.is_cuda:
                # Keep compute in flight past the post hook / next prefetch.
                torch.cuda._sleep(5_000_000)
            return x + 0.05 * self.proj(x)

    torch.manual_seed(123)
    blocks = nn.ModuleList([DelayedBlock() for _ in range(12)])
    x = torch.randn(8, 512)
    with torch.inference_mode():
        expected = x.clone()
        for block in blocks:
            expected = block(expected)
        # Initialize CUDA/cuBLAS workspaces before measuring streaming memory.
        blocks[0].cuda()(x.cuda())
        blocks[0].cpu()
        torch.cuda.synchronize()
        manager = StrataBlockOffloadManager(blocks, torch.device("cuda:0"), 0)
        manager.prepare()
        manager.attach_hooks()
        torch.cuda.reset_peak_memory_stats()
        initial_active = torch.cuda.memory_stats()["active_bytes.all.current"]
        try:
            for _ in range(3):
                actual = x.cuda()
                for block in blocks:
                    actual = block(actual)
                torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-4)
            active_peak = torch.cuda.memory_stats()["active_bytes.all.peak"] - initial_active
            one_block_bytes = 512 * 512 * 4
            # Two weight slots plus generous room for activations and allocator
            # bookkeeping; retaining all 12 queued blocks exceeds this bound.
            assert active_peak < 3 * one_block_bytes, active_peak
        finally:
            manager.release_all_to_cpu()


def test_missing_trained_weights_are_rejected():
    with torch.device("meta"):
        model = nn.Linear(4, 4)
    with pytest.raises(RuntimeError, match="uninitialized tensors"):
        _materialize_remaining_meta(model, torch.float32)
    assert model.weight.is_meta


@pytest.mark.parametrize("variant,boundary,high_steps,selected", [
    ("Wan2.2-T2V-A14B", 875, 26, [999, 997, 972, 922, 877, 865, 799, 235]),
    ("Wan2.2-I2V-A14B", 900, 15, [999, 994, 937, 833, 749, 728, 624, 113]),
])
def test_wan_schedule_matches_native_reference(variant, boundary, high_steps, selected):
    """Golden values from Wan-Video/Wan2.2's FlowUniPC scheduler at 40 steps."""
    from diffusers import UniPCMultistepScheduler

    scheduler = UniPCMultistepScheduler(
        prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=WAN_FLOW_SHIFTS[variant],
    )
    _set_wan_timesteps(scheduler, 40, torch.device("cpu"))
    assert scheduler.timesteps[[0, 1, 10, 20, 25, 26, 30, 39]].tolist() == selected
    assert int((scheduler.timesteps >= boundary).sum()) == high_steps
    assert scheduler.sigmas[-1] == 0


def test_wan_vae_loads_fp32_independent_of_transformer_dtype(tmp_path, monkeypatch):
    import diffusers
    from diffusers.loaders import single_file_utils
    import openfal.runtime.wan22_runner as wan_runtime

    class TinyVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(2, dtype=torch.bfloat16))

        @staticmethod
        def load_config(path, **kwargs):
            assert kwargs["local_files_only"]
            return {}

        @staticmethod
        def from_config(config):
            return TinyVAE()

    monkeypatch.setattr(diffusers, "AutoencoderKLWan", TinyVAE)
    monkeypatch.setattr(single_file_utils, "convert_wan_vae_to_diffusers", lambda sd: sd)
    monkeypatch.setattr(wan_runtime, "load_safetensors_state_dict", lambda *args, **kwargs: {
        "weight": torch.ones(2, dtype=torch.float16),
    })
    cfg = EngineConfig(weights_dir=tmp_path)
    runner = RealWan22Runner(cfg, StrataMemoryHierarchy(cfg))
    monkeypatch.setattr(runner, "_ensure_16ch_vae_file", lambda **kwargs: tmp_path / "vae.safetensors")
    runner._load_shared_vae(tmp_path)
    assert runner._vae.weight.dtype == torch.float32
    assert not runner._vae.training


def test_wan_requires_complete_experts_and_local_vae(tmp_path):
    from safetensors.torch import save_file

    cfg = EngineConfig(weights_dir=tmp_path)
    runner = RealWan22Runner(cfg, StrataMemoryHierarchy(cfg))
    root = tmp_path / "Wan2.2-T2V-A14B"
    root.mkdir()
    expert_tensor = {"patch_embedding.weight": torch.zeros(4, 16, 1, 2, 2, dtype=torch.float16)}
    save_file(expert_tensor, str(root / "wan2.2_t2v_high_noise.safetensors"))
    assert not runner.has_weights_for("Wan2.2-T2V-A14B")
    low_path = root / "wan2.2_t2v_low_noise.safetensors"
    low_path.touch()
    assert not runner.has_weights_for("Wan2.2-T2V-A14B")
    save_file(expert_tensor, str(low_path))
    assert runner.has_weights_for("Wan2.2-T2V-A14B")
    with pytest.raises(RuntimeError, match="runtime-assets-only"):
        runner._ensure_16ch_vae_file()

    vae_dir = root / "split_files" / "vae"
    vae_dir.mkdir(parents=True)
    # Even a correctly named file must not pass with the wrong architecture.
    vae_path = vae_dir / "wan_2.1_vae.safetensors"
    save_file({"conv2.weight": torch.zeros(48, 48, 1, 1, 1)}, str(vae_path))
    with pytest.raises(RuntimeError, match="16-channel"):
        runner._ensure_16ch_vae_file()
    save_file({"conv2.weight": torch.zeros(16, 16, 1, 1, 1)}, str(vae_path))
    assert runner._ensure_16ch_vae_file() == vae_path


def test_wan_rejects_unsupported_conditioning_before_loading(tmp_path):
    cfg = EngineConfig(weights_dir=tmp_path)
    runner = RealWan22Runner(cfg, StrataMemoryHierarchy(cfg))
    args = dict(prompt="portrait", width=256, height=256, num_frames=17, fps=16, num_steps=4, seed=1)
    with pytest.raises(NotImplementedError, match="driving video"):
        runner.generate_video(variant="Wan2.2-Animate-14B", **args)
    with pytest.raises(ValueError, match="start image"):
        runner.generate_video(variant="Wan2.2-I2V-A14B", **args)
