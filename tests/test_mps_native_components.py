"""Small native components exercise Metal-only operator risks on actual Macs.

CPU cases validate the fixtures everywhere. Metal cases require real hardware;
they deliberately do not emulate a backend or silently move operations to CPU.
"""

import copy

import pytest
import torch

from openfal.runtime.wan_animate_runner import _install_mps_motion_basis


HAS_MPS = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
DEVICES = ["cpu", pytest.param("mps", marks=pytest.mark.skipif(not HAS_MPS, reason="actual Apple Metal hardware required"))]


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(17)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("device", DEVICES)
def test_native_qwen21_complex64_rotary_embedding(device):
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Rope, apply_rotary_emb_qwen

    cpu_rope = QwenImage21Rope(theta=10000, axes_dim=[2, 2, 4])
    mask = torch.tensor([False, False, True, True, True, True])
    x = torch.randn(1, 6, 2, 8)
    with torch.inference_mode():
        reference_freqs = cpu_rope([(1, 2, 2)], mask, torch.device("cpu"))
        expected = apply_rotary_emb_qwen(x, reference_freqs, use_real=False)
        native_rope = copy.deepcopy(cpu_rope)
        actual_freqs = native_rope([(1, 2, 2)], mask.to(device), torch.device(device))
        assert actual_freqs.dtype == torch.complex64
        actual = apply_rotary_emb_qwen(x.to(device), actual_freqs, use_real=False)
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_native_minimax_position_dtype_and_rotary_embedding(device):
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed
    from diffusers.utils.torch_utils import maybe_adjust_dtype_for_device

    # Native MiniMax preprocessing constructs coordinates on CPU in FP64, then
    # adjusts their dtype before transfer. Verify that exact conversion path.
    positions = torch.arange(18, dtype=torch.float64).reshape(6, 3) / 7
    cpu_rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=2)
    with torch.inference_mode():
        expected = cpu_rope(positions)
        target_dtype = maybe_adjust_dtype_for_device(positions.dtype, torch.device(device))
        if device == "mps":
            assert target_dtype == torch.float32
        actual = copy.deepcopy(cpu_rope).to(device)(positions.to(device=device, dtype=target_dtype))
    for observed, reference in zip(actual, expected):
        torch.testing.assert_close(observed.cpu(), reference, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_native_wan_causal_vae_encode_decode(device):
    from diffusers import AutoencoderKLWan

    model = AutoencoderKLWan(
        base_dim=4, z_dim=2, dim_mult=[1, 1, 1, 1], num_res_blocks=1,
        latents_mean=[0.0, 0.0], latents_std=[1.0, 1.0],
    ).eval()
    sample = torch.randn(1, 3, 5, 32, 32)
    native = copy.deepcopy(model).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        expected_latents = model.encode(sample).latent_dist.mode()
        expected_frames = model.decode(expected_latents).sample
        actual_latents = native.encode(sample.to(device)).latent_dist.mode()
        actual_frames = native.decode(actual_latents).sample
    assert actual_frames.shape == sample.shape
    assert torch.isfinite(actual_frames).all()
    torch.testing.assert_close(actual_latents.cpu(), expected_latents, rtol=2e-3, atol=2e-4)
    torch.testing.assert_close(actual_frames.cpu(), expected_frames, rtol=3e-3, atol=3e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_native_animate_motion_encoder_with_cpu_qr_basis(device):
    from diffusers.models.transformers.transformer_wan_animate import WanAnimateMotionEncoder

    model = WanAnimateMotionEncoder(
        size=16, style_dim=8, motion_dim=2, out_dim=8, channels={"16": 4, "8": 4, "4": 4},
    ).eval()
    faces = torch.randn(3, 3, 16, 16)
    with torch.inference_mode():
        expected = model(faces)
        native = copy.deepcopy(model)
        _install_mps_motion_basis(native)
        actual = native.to(device)(faces.to(device))
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-4, atol=2e-5)
