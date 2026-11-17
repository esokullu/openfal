"""Portable quantization checks plus actual Metal execution when available."""

from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import openfal.runtime.common as common
import openfal.runtime.quant as quant


FP8_DTYPES = sorted(quant._FP8_DTYPES, key=str)
HAS_MPS = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


@pytest.fixture(autouse=True)
def clear_capability_probe_caches():
    # Retain the real functions even in tests which monkeypatch the resolver's
    # references. A mocked capability must never leak into another test.
    probes = (common.mps_supports_bfloat16, common.mps_supports_fp8)
    for probe in probes:
        probe.cache_clear()
    yield
    for probe in probes:
        probe.cache_clear()


@pytest.mark.parametrize("dtype", FP8_DTYPES)
def test_recursive_parent_cast_preserves_fp8_bytes_and_fp32_scales(dtype):
    raw = torch.linspace(-2, 2, 64).reshape(8, 8).to(dtype)
    scale = torch.tensor([0.003456789] * 8)
    layer = quant.StrataFP8ScaledLinear(raw, scale, compute_dtype=torch.float32)
    parent = nn.Sequential(nn.Sequential(layer))
    before = raw.view(torch.uint8).clone()
    for target in (torch.float16, torch.bfloat16, torch.float32):
        parent.to(dtype=target)
        assert layer.weight_fp8.dtype == dtype
        assert layer._dtype_anchor.dtype == target
        assert layer.weight_scale.dtype == torch.float32
        torch.testing.assert_close(layer.weight_scale[:, 0], scale, rtol=0, atol=0)
        assert torch.equal(layer.weight_fp8.view(torch.uint8), before)
    x = torch.randn(3, 8)
    torch.testing.assert_close(parent(x), F.linear(x, raw.float() * scale[:, None]))


def test_recursive_int8_cast_preserves_scales_and_updates_embedding_compute_dtype():
    raw = torch.arange(32, dtype=torch.int8).reshape(4, 8)
    scale = torch.full((4,), 0.003456789)
    embedding = quant.StrataConvRotEmbedding(raw, scale, convrot=False, compute_dtype=torch.bfloat16)
    linear = quant.StrataConvRotLinear(raw, scale, convrot=False, compute_dtype=torch.bfloat16)
    parent = nn.ModuleList([embedding, linear])
    parent.half().float()
    ids = torch.tensor([0, 2])
    actual = embedding(ids)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, raw[ids].float() * scale[ids, None])
    for child in parent:
        assert child.weight_i8.dtype == torch.int8
        assert child.weight_scale.dtype == torch.float32
        torch.testing.assert_close(child.weight_scale[:, 0], scale, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", FP8_DTYPES)
def test_all_fp8_formats_load_as_quantized_modules(dtype):
    module = nn.Sequential(nn.Linear(8, 4, bias=False))
    missing, extra = quant.quantized_module_swap_from_state_dict(module, {
        "0.weight": torch.ones(4, 8).to(dtype), "0.weight_scale": torch.tensor(0.5),
    }, compute_dtype=torch.float32)
    assert not missing and not extra
    assert isinstance(module[0], quant.StrataFP8ScaledLinear)
    assert module[0].weight_fp8.dtype == dtype


@pytest.mark.parametrize("dtype", FP8_DTYPES)
def test_unsupported_metal_fp8_storage_policy_preserves_cpu_bytes(dtype, monkeypatch):
    raw = torch.ones(4, 8).to(dtype)
    probe = Mock(return_value=False)
    monkeypatch.setattr(quant, "mps_supports_fp8", probe)
    assert quant._storage_device(raw, torch.device("mps")) == torch.device("cpu")
    probe.assert_called_once_with(dtype)
    probe.return_value = True
    assert quant._storage_device(raw, torch.device("mps")) == torch.device("mps")
    assert quant._storage_device(raw, torch.device("cpu")) == torch.device("cpu")


def test_failed_recursive_move_keeps_checkpoint_buffers_recoverable():
    layer = quant.StrataFP8ScaledLinear(torch.ones(4, 8).to(torch.float8_e4m3fn), torch.tensor(0.25))

    def fail(_tensor):
        raise RuntimeError("device move failed")

    with pytest.raises(RuntimeError, match="device move failed"):
        nn.Sequential(layer)._apply(fail)
    assert "weight_fp8" in layer.state_dict()
    assert "weight_scale" in layer.state_dict()
    layer.float()
    torch.testing.assert_close(layer(torch.ones(1, 8)), torch.full((1, 4), 2.0))


@pytest.mark.parametrize("supported,expected", [(True, torch.bfloat16), (False, torch.float32)])
def test_mps_dtype_selection_uses_capability_not_os_assumptions(monkeypatch, supported, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(common, "mps_supports_bfloat16", lambda: supported)
    assert common.resolve_compute_device_and_dtype() == (torch.device("mps"), expected)


@pytest.mark.parametrize("failure", ["allocation", "attention", "conv3d", "nonfinite", "none"])
def test_bfloat16_probe_checks_all_required_kernels_and_numerics(monkeypatch, failure):
    # Exercise probe control flow portably using real CPU tensors; this is not
    # evidence of Metal kernel support (the hardware tests below provide that).
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    real_ones = torch.ones

    def cpu_ones(shape, *, device, dtype):
        assert device == "mps" and dtype == torch.bfloat16
        if failure == "allocation":
            raise TypeError("BF16 allocation unavailable")
        return real_ones(shape, dtype=dtype)

    allocations = Mock(side_effect=cpu_ones)
    monkeypatch.setattr(torch, "ones", allocations)
    if failure == "attention":
        monkeypatch.setattr(F, "scaled_dot_product_attention", Mock(side_effect=NotImplementedError("SDPA unavailable")))
    elif failure == "conv3d":
        monkeypatch.setattr(F, "conv3d", Mock(side_effect=RuntimeError("Conv3D unavailable")))
    elif failure == "nonfinite":
        monkeypatch.setattr(F, "scaled_dot_product_attention", lambda q, k, v: torch.full_like(q, float("nan")))
    expected = failure == "none"
    assert common.mps_supports_bfloat16() is expected
    count = allocations.call_count
    assert common.mps_supports_bfloat16() is expected
    assert allocations.call_count == count  # cache failed probes as well


@pytest.mark.parametrize("dtype", FP8_DTYPES)
@pytest.mark.parametrize("failure", ["copy", "decode", "roundtrip", "none"])
def test_fp8_probe_checks_each_format_and_exact_roundtrip(monkeypatch, dtype, failure):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    real_to = torch.Tensor.to
    transfers = []

    class MetalCopy:
        def __init__(self, tensor):
            self.tensor = tensor

        def to(self, target):
            assert target == torch.float32
            if failure == "decode":
                raise NotImplementedError("FP8 decode unavailable")
            decoded = self.tensor.float()
            return decoded + 1 if failure == "roundtrip" else decoded

    def controlled_to(tensor, *args, **kwargs):
        if args and args[0] == "mps":
            transfers.append(tensor.dtype)
            if failure == "copy":
                raise TypeError("This FP8 format is unavailable")
            return MetalCopy(tensor)
        return real_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", controlled_to)
    assert common.mps_supports_fp8(dtype) is (failure == "none")
    assert common.mps_supports_fp8(dtype) is (failure == "none")
    assert transfers == [dtype]


def test_cpu_cache_cleanup_does_not_initialize_accelerators(monkeypatch):
    cuda = Mock()
    mps = Mock()
    monkeypatch.setattr(torch.cuda, "empty_cache", cuda)
    monkeypatch.setattr(torch.cuda, "ipc_collect", cuda)
    monkeypatch.setattr(torch.mps, "empty_cache", mps)
    common.clear_device_cache(torch.device("cpu"))
    cuda.assert_not_called()
    mps.assert_not_called()


def test_offloader_cleanup_is_idempotent_before_prepare():
    blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
    manager = quant.StrataBlockOffloadManager(blocks, torch.device("cpu"), 1)
    manager.release_all_to_cpu()
    manager.release_all_to_cpu()
    assert not manager._handles and not manager._staged_gpu_tensors
    assert all(parameter.device.type == "cpu" for parameter in blocks.parameters())


def test_offloader_cleanup_recovers_from_partial_prepare(monkeypatch):
    blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
    manager = quant.StrataBlockOffloadManager(blocks, torch.device("cpu"), 0)
    original_to = blocks[1].to
    calls = 0

    def fail_first_move(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("prepare allocation failed")
        return original_to(*args, **kwargs)

    monkeypatch.setattr(blocks[1], "to", fail_first_move)
    with pytest.raises(RuntimeError, match="prepare allocation failed"):
        manager.prepare()
    assert set(manager._pinned_cpu_backing) == {0}
    manager.release_all_to_cpu()
    manager.release_all_to_cpu()
    assert all(parameter.device.type == "cpu" for parameter in blocks.parameters())


@pytest.mark.skipif(not HAS_MPS, reason="actual Apple Metal hardware required")
@pytest.mark.parametrize("kind", ["fp8_e4m3", "fp8_e5m2", "fp8_e4m3_cpu", "fp8_e5m2_cpu", "int8"])
@pytest.mark.parametrize("resident", [0, 1, 3])
def test_actual_mps_quantized_streaming_matches_cpu_reference(kind, resident, monkeypatch):
    device, dtype = common.resolve_compute_device_and_dtype()
    assert device.type == "mps"
    if kind.endswith("_cpu"):
        # Keep exercising CPU decode -> real Metal matmul even after native
        # FP8 storage support becomes available in newer PyTorch releases.
        monkeypatch.setattr(quant, "mps_supports_fp8", lambda _dtype: False)
    torch.manual_seed(17)
    blocks = nn.ModuleList()
    for _ in range(3):
        if kind == "int8":
            layer = quant.StrataConvRotLinear(
                torch.randint(-8, 8, (32, 32), dtype=torch.int8), torch.tensor(0.01),
                group_size=16, compute_dtype=dtype,
            )
        else:
            storage_dtype = torch.float8_e4m3fn if kind.startswith("fp8_e4m3") else torch.float8_e5m2
            layer = quant.StrataFP8ScaledLinear(
                torch.randn(32, 32).to(storage_dtype), torch.tensor(0.1), compute_dtype=dtype,
            )
        blocks.append(nn.Sequential(layer, nn.SiLU()))
    before = [{key: value.detach().clone() for key, value in block.state_dict().items()} for block in blocks]
    x = torch.randn(4, 32, dtype=dtype)
    with torch.inference_mode():
        expected = x.clone()
        for block in blocks:
            expected = block(expected)
        manager = quant.StrataBlockOffloadManager(blocks, device, resident)
        try:
            manager.prepare()
            if kind.endswith("_cpu"):
                assert all(block[0].weight_fp8.device.type == "cpu" for block in blocks)
            manager.attach_hooks()
            for _ in range(2):
                actual = x.to(device)
                for block in blocks:
                    actual = block(actual)
                torch.mps.synchronize()
                torch.testing.assert_close(actual.cpu(), expected, rtol=0.04, atol=0.002)
        finally:
            manager.release_all_to_cpu()
    for block, snapshot in zip(blocks, before):
        for key, value in block.state_dict().items():
            assert value.device.type == "cpu"
            original = snapshot[key]
            if value.dtype in quant._FP8_DTYPES:
                assert torch.equal(value.view(torch.uint8), original.view(torch.uint8))
            else:
                torch.testing.assert_close(value, original, rtol=0, atol=0)
    assert not manager._handles and not manager._staged_gpu_tensors


@pytest.mark.skipif(not HAS_MPS, reason="actual Apple Metal hardware required")
def test_actual_mps_embedding_honors_recursive_compute_dtype():
    raw = torch.arange(32, dtype=torch.int8).reshape(4, 8)
    layer = quant.StrataConvRotEmbedding(raw, torch.tensor(0.1), convrot=False)
    parent = nn.Sequential(layer).to(device="mps", dtype=torch.float32)
    ids = torch.tensor([0, 2], device="mps")
    actual = parent(ids)
    torch.testing.assert_close(actual.cpu(), raw[[0, 2]].float() * 0.1)
    parent.cpu()
    assert layer.weight_i8.device.type == "cpu"
