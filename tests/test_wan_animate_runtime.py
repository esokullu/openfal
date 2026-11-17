"""Exercise native Animate conditioning, segmentation and stage cleanup on CPU."""

from types import SimpleNamespace
from unittest.mock import Mock
import os

import numpy as np
import pytest
import torch
from PIL import Image

from openfal.config import EngineConfig
from openfal.engine.memory_hierarchy import StrataMemoryHierarchy
from openfal.runtime.wan_animate_runner import (
    RealWanAnimateRunner,
    _AnimateStages,
    _make_pipeline,
    _make_scheduler,
    _managed_pipeline_calls,
    _install_mps_motion_basis,
)
from openfal.runtime.wan22_runner import RealWan22Runner, _set_wan_timesteps
from openfal.runtime.quant import StrataBlockOffloadManager, quantized_module_swap_from_state_dict


@pytest.fixture
def tiny_pipeline(tmp_path):
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanAnimateTransformer3DModel
    from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel, UMT5Config, UMT5EncoderModel

    class Tokenizer:
        def __call__(self, prompts, max_length, **kwargs):
            ids = torch.zeros(len(prompts), max_length, dtype=torch.long)
            ids[:, :3] = torch.tensor([2, 3, 1])
            return SimpleNamespace(input_ids=ids, attention_mask=(ids > 0).long())

    torch.manual_seed(17)
    vae = AutoencoderKLWan(
        base_dim=4, z_dim=2, dim_mult=[1, 1, 1, 1], num_res_blocks=1,
        latents_mean=[0.0, 0.0], latents_std=[1.0, 1.0],
    ).eval()
    transformer = WanAnimateTransformer3DModel(
        num_attention_heads=2, attention_head_dim=8, in_channels=8, latent_channels=2, out_channels=2,
        text_dim=16, freq_dim=8, ffn_dim=32, num_layers=2, image_dim=16, added_kv_proj_dim=16,
        motion_encoder_size=16, motion_style_dim=8, motion_dim=2, motion_encoder_dim=8,
        motion_encoder_channel_sizes={"16": 4, "8": 4, "4": 4},
        face_encoder_hidden_dim=8, face_encoder_num_heads=2, inject_face_latents_blocks=1,
    ).eval()
    text_encoder = UMT5EncoderModel(UMT5Config(
        vocab_size=16, d_model=16, d_ff=32, d_kv=8, num_heads=2, num_layers=1,
    )).eval()
    clip = CLIPVisionModel(CLIPVisionConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
        image_size=32, patch_size=8,
    )).eval()
    image_processor = CLIPImageProcessor(size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32})
    scheduler = UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=5)
    scheduler.save_pretrained(tmp_path / "scheduler")
    pipe = _make_pipeline(
        tokenizer=Tokenizer(), text_encoder=text_encoder, vae=vae, scheduler=_make_scheduler(tmp_path / "scheduler"),
        image_processor=image_processor, image_encoder=clip, transformer=transformer,
    )
    pipe._strata_device = torch.device("cpu")
    pipe.set_progress_bar_config(disable=True)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield pipe
    finally:
        torch.set_num_threads(old_threads)


@pytest.mark.parametrize("mode", ["animate", "replace"])
def test_native_animate_pipeline_uses_real_conditions_and_preserves_frame_count(tiny_pipeline, tmp_path, mode):
    pipe = tiny_pipeline
    config = EngineConfig(weights_dir=tmp_path)
    shared = RealWan22Runner(config, StrataMemoryHierarchy(config))
    stages = _AnimateStages(shared, pipe, pipe.transformer, None, torch.device("cpu"))
    image = Image.new("RGB", (32, 32), (75, 155, 210))
    # Seven frames require two five-frame segments, exercising VAE <-> DiT
    # transitions, scheduler reset, overlap and removal of the padded tail.
    pose = [Image.new("RGB", (32, 32), (20 * i, 90, 40)) for i in range(7)]
    faces = [Image.new("RGB", (16, 16), (130, 20 * i, 80)) for i in range(7)]
    mask = Image.new("L", (32, 32), 0)
    mask.paste(255, (8, 0, 24, 32))
    background = Image.new("RGB", (32, 32), (10, 40, 80))
    background.paste((0, 0, 0), (8, 0, 24, 32))
    calls = []

    def observe(_module, _args, kwargs):
        calls.append({k: kwargs[k].detach().clone() for k in (
            "encoder_hidden_states_image", "pose_hidden_states", "face_pixel_values", "hidden_states",
        )})

    handle = pipe.transformer.register_forward_pre_hook(observe, with_kwargs=True)
    try:
        with torch.inference_mode(), _managed_pipeline_calls(pipe, stages):
            result = pipe(
                image=image, pose_video=pose, face_video=faces, prompt="a dancer",
                height=32, width=32, segment_frame_length=5, num_inference_steps=2, mode=mode,
                background_video=[background] * 7 if mode == "replace" else None,
                mask_video=[mask] * 7 if mode == "replace" else None,
                motion_encode_batch_size=2, guidance_scale=1.0,
                generator=torch.Generator().manual_seed(11), output_type="np",
            )
    finally:
        handle.remove()
    assert result.frames.shape == (1, 7, 32, 32, 3)
    assert np.isfinite(result.frames).all()
    assert len(calls) == 4
    assert calls[0]["encoder_hidden_states_image"].std() > 0
    assert calls[0]["pose_hidden_states"].std() > 0
    assert not torch.equal(calls[0]["face_pixel_values"][:, :, 0], calls[0]["face_pixel_values"][:, :, 1])
    if mode == "replace":
        # The four mask channels follow generated latent channels; white person
        # regions become unknown (0), retained background remains known (1).
        mask_latents = calls[0]["hidden_states"][:, 2:6, 1:]
        assert mask_latents.min() == 0
        assert mask_latents.max() == 1
    assert stages.current is None
    assert "encode" not in pipe.vae.__dict__
    assert "decode" not in pipe.vae.__dict__
    assert "encode_prompt" not in pipe.__dict__
    assert all(parameter.device.type == "cpu" for parameter in pipe.transformer.parameters())


def test_native_scheduler_reset_matches_wan_reference(tiny_pipeline):
    from diffusers import UniPCMultistepScheduler

    expected = UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=5)
    _set_wan_timesteps(expected, 20, torch.device("cpu"))
    for _ in range(2):
        tiny_pipeline.scheduler.set_timesteps(20, device="cpu")
        torch.testing.assert_close(tiny_pipeline.scheduler.timesteps, expected.timesteps)
        torch.testing.assert_close(tiny_pipeline.scheduler.sigmas, expected.sigmas)


def test_scheduler_reset_discards_populated_multistep_history(tiny_pipeline, tmp_path):
    scheduler = tiny_pipeline.scheduler
    fresh = _make_scheduler(tmp_path / "scheduler")
    sample = torch.randn(1, 2, 3, 4, 4)
    predictions = [torch.randn_like(sample) for _ in range(4)]

    def denoise(instance):
        instance.set_timesteps(4, device="cpu")
        value = sample.clone()
        outputs = []
        for timestep, prediction in zip(instance.timesteps, predictions):
            value = instance.step(prediction, timestep, value, return_dict=False)[0]
            outputs.append(value.clone())
        return outputs

    denoise(scheduler)
    assert scheduler.last_sample is not None
    expected = denoise(fresh)
    for _ in range(2):
        for actual, reference in zip(denoise(scheduler), expected):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["animate", "replace"])
@pytest.mark.parametrize("resident", [0, 1])
@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        os.getenv("OPENFAL_TEST_CUDA") != "1" or not torch.cuda.is_available(),
        reason="set OPENFAL_TEST_CUDA=1 on a CUDA host",
    )),
])
def test_mixed_precision_segment_handoff_matches_native_without_offload(tiny_pipeline, tmp_path, mode, resident, device):
    pipe = tiny_pipeline
    device = torch.device(device)
    pipe._strata_device = device
    # Mirror production: BF16 denoiser/encoders, FP32 VAE, FP8 block storage.
    # Comparing all three segments against the unwrapped native pipeline catches
    # mutations of continuation conditioning or model state during eviction.
    pipe.transformer.to(dtype=torch.bfloat16)
    pipe.text_encoder.to(dtype=torch.bfloat16)
    pipe.image_encoder.to(dtype=torch.bfloat16)
    state = dict(pipe.transformer.state_dict())
    for name, module in pipe.transformer.named_modules():
        if name.startswith("blocks.") and isinstance(module, torch.nn.Linear):
            state[name + ".weight"] = (module.weight.detach().float() / 0.01).to(torch.float8_e4m3fn)
            state[name + ".weight_scale"] = torch.tensor(0.01)
    missing, extra = quantized_module_swap_from_state_dict(pipe.transformer, state, compute_dtype=torch.bfloat16)
    assert not missing and not extra
    image = Image.new("RGB", (32, 32), (75, 155, 210))
    poses = [Image.new("RGB", (32, 32), (15 * i, 90, 40)) for i in range(11)]
    faces = [Image.new("RGB", (16, 16), (130, 15 * i, 80)) for i in range(11)]
    mask = Image.new("L", (32, 32))
    mask.paste(255, (8, 0, 24, 32))
    background = Image.new("RGB", (32, 32), (10, 40, 80))
    background.paste((0, 0, 0), (8, 0, 24, 32))
    args = dict(
        image=image, pose_video=poses, face_video=faces, prompt="a dancer", mode=mode,
        background_video=[background] * 11 if mode == "replace" else None,
        mask_video=[mask] * 11 if mode == "replace" else None,
        height=32, width=32, segment_frame_length=5, num_inference_steps=4,
        motion_encode_batch_size=2, guidance_scale=1.0, output_type="pt",
    )
    seen = []

    def record(_module, _args, kwargs):
        seen.append(kwargs["hidden_states"].detach().clone())

    observer = pipe.transformer.register_forward_pre_hook(record, with_kwargs=True)
    config = EngineConfig(weights_dir=tmp_path)
    shared = RealWan22Runner(config, StrataMemoryHierarchy(config))
    offloader = StrataBlockOffloadManager(pipe.transformer.blocks, device, resident)
    stages = _AnimateStages(shared, pipe, pipe.transformer, offloader, device)
    try:
        with torch.inference_mode():
            for component in (pipe.transformer, pipe.text_encoder, pipe.image_encoder, pipe.vae):
                component.to(device)
            expected = pipe(**args, generator=torch.Generator(device=device).manual_seed(11)).frames
            for component in (pipe.transformer, pipe.text_encoder, pipe.image_encoder, pipe.vae):
                component.to("cpu")
            reference_inputs = seen[:]
            seen.clear()
            with _managed_pipeline_calls(pipe, stages):
                actual = pipe(**args, generator=torch.Generator(device=device).manual_seed(11)).frames
        assert len(seen) == len(reference_inputs) == 12
        for observed, reference in zip(seen, reference_inputs):
            torch.testing.assert_close(observed, reference, rtol=0, atol=0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        observer.remove()
        stages.release()


def test_stage_wrappers_cleanup_when_inference_fails(tiny_pipeline, tmp_path):
    config = EngineConfig(weights_dir=tmp_path)
    shared = RealWan22Runner(config, StrataMemoryHierarchy(config))
    stages = _AnimateStages(shared, tiny_pipeline, tiny_pipeline.transformer, None, torch.device("cpu"))
    with pytest.raises(RuntimeError, match="inference failed"):
        with _managed_pipeline_calls(tiny_pipeline, stages):
            stages.activate("transformer")
            raise RuntimeError("inference failed")
    assert stages.current is None
    assert not tiny_pipeline.transformer._forward_pre_hooks
    assert "encode" not in tiny_pipeline.vae.__dict__
    assert "decode" not in tiny_pipeline.vae.__dict__


def test_cpu_qr_basis_matches_native_motion_decomposition(tiny_pipeline):
    encoder = tiny_pipeline.transformer.motion_encoder
    faces = torch.randn(3, 3, 16, 16)
    with torch.inference_mode():
        expected = encoder(faces)
        _install_mps_motion_basis(encoder)
        actual = encoder(faces)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert "_openfal_motion_basis" not in encoder.state_dict()


@pytest.mark.parametrize("updates,match", [
    ({"face_video": []}, "face_video"),
    ({"mode": "replace"}, "background_video"),
    ({"segment_frame_length": 4}, "segment length"),
    ({"segment_frame_length": 17}, "segment length must be 77"),
    ({"fps": 0}, "fps"),
    ({"width": 31}, "multiples of 16"),
])
def test_invalid_conditioning_rejects_before_model_loading(tmp_path, updates, match):
    config = EngineConfig(weights_dir=tmp_path)
    runner = RealWanAnimateRunner(config, StrataMemoryHierarchy(config))
    runner._ensure_loaded = Mock()
    frame = Image.new("RGB", (32, 32))
    args = dict(image=frame, pose_video=[frame] * 5, face_video=[frame] * 5,
                prompt="dancer", width=32, height=32, fps=16, seed=0, output_path=tmp_path / "video.mp4")
    args.update(updates)
    with pytest.raises(ValueError, match=match):
        runner.generate_video(**args)
    runner._ensure_loaded.assert_not_called()
