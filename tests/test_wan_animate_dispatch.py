"""Wan Animate API/engine dispatch without allocating or executing model weights."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from openfal.config import EngineConfig
from openfal.engine.runner import OpenFalRunner
from openfal.server.app import create_app


@pytest.fixture
def dispatch(tmp_path, monkeypatch):
    import openfal.runtime.common as common
    import openfal.runtime.wan22_runner as wan
    import openfal.runtime.wan_animate_runner as animate
    import openfal.runtime.wan_animate_preprocess as preprocessing

    monkeypatch.setenv("OPENFAL_SYNTHETIC_FALLBACK", "0")
    image = Image.new("RGB", (32, 32), "white")
    # The source ends before the requested128frames; its real117frames must win.
    conditioning = SimpleNamespace(pose_video=[image] * 117, face_video=[image] * 117,
                                   background_video=None, mask_video=None, fps=16)
    prepare = Mock(return_value=conditioning)
    generate = Mock(side_effect=lambda **kwargs: kwargs["output_path"].write_bytes(b"stub-video-for-api-contract"))

    class SharedWan:
        def __init__(self, config, memory):
            self.evict = Mock()

        def has_weights_for(self, variant):
            return variant == "Wan2.2-Animate-14B"

    class Animate:
        def __init__(self, config, memory, shared_runner=None):
            self.shared_runner = shared_runner
            self.generate_video = generate
            self.evict = Mock()

    monkeypatch.setattr(wan, "RealWan22Runner", SharedWan)
    monkeypatch.setattr(animate, "RealWanAnimateRunner", Animate)
    monkeypatch.setattr(preprocessing, "preprocess_wan_animate", prepare)
    monkeypatch.setattr(common, "resolve_compute_device_and_dtype", lambda: (torch.device("cpu"), torch.float32))
    config = EngineConfig(weights_dir=tmp_path / "weights", output_dir=tmp_path / "out", cache_dir=tmp_path / "cache")
    app = create_app(config)
    runner = app.state.runner
    monkeypatch.setattr(runner.face_conditioner, "load_image_from_url_or_path", lambda _: image)
    return SimpleNamespace(app=app, runner=runner, prepare=prepare, generate=generate, conditioning=conditioning)


def _payload(**overrides):
    return {"prompt": "a dancer", "image_url": "identity.png", "driving_video_url": "motion.mp4",
            "duration": 8.0, "fps": 16, "resolution": "480P", "seed": 0, **overrides}


@pytest.mark.parametrize("endpoint,override,mode", [
    ("wan/v2.2/animate", None, "animate"),
    ("wan/v2.2/recast", None, "replace"),
    ("wan/v2.2/animate", "replace", "replace"),
    ("wan/v2.2/recast", "animate", "animate"),
])
def test_animate_dispatch_defaults_and_real_frame_metadata(dispatch, endpoint, override, mode):
    if mode == "replace":
        dispatch.conditioning.background_video = dispatch.conditioning.pose_video
        dispatch.conditioning.mask_video = dispatch.conditioning.pose_video
    response = dispatch.runner.execute(endpoint, _payload(animate_mode=override))
    prepared = dispatch.prepare.call_args
    assert prepared.args[0] == "motion.mp4"
    assert prepared.kwargs["num_frames"] == 128
    assert prepared.kwargs["mode"] == mode
    passed = dispatch.generate.call_args.kwargs
    assert passed["mode"] == mode
    assert passed["num_steps"] == 20
    assert passed["guidance_scale"] == 1.0
    assert passed["seed"] == 0
    assert passed["pose_video"] is dispatch.conditioning.pose_video
    assert response["video"]["num_frames"] == 117
    assert response["video"]["duration"] == 117 / 16
    assert response["video"]["fps"] == 16
    assert not response["video"]["stereo_audio"]
    assert response["strata_engine"]["mode"] == "real_pytorch"
    assert dispatch.runner._real_wan_animate_runner.shared_runner is dispatch.runner._real_wan_runner
    assert dispatch.runner._active_real_family == "wan_animate"


@pytest.mark.parametrize("overrides,error", [
    ({"image_url": None}, "exactly one identity image"),
    ({"reference_image_urls": ["other.png"]}, "exactly one identity image"),
    ({"end_image_url": "identity.png"}, "exactly one identity image"),
    ({"driving_video_url": None}, "exactly one driving_video_url"),
    ({"reference_video_urls": ["other.mp4"]}, "exactly one driving_video_url"),
    ({"reference_audio_urls": ["sound.wav"]}, "audio references"),
    ({"animate_mode": "unknown"}, "animate_mode"),
])
def test_animate_rejects_invalid_media_before_preprocessing(dispatch, overrides, error):
    with pytest.raises(ValueError, match=error):
        dispatch.runner.execute("wan/v2.2/animate", _payload(**overrides))
    dispatch.prepare.assert_not_called()
    dispatch.generate.assert_not_called()


def test_animate_shared_media_aliases_refer_to_one_input(dispatch):
    response = dispatch.runner.execute("wan/v2.2/animate", _payload(
        face_image_url="identity.png", reference_image_urls=["identity.png"],
        target_video_url="motion.mp4", reference_video_urls=["motion.mp4"],
    ))
    assert response["video"]["num_frames"] == 117
    dispatch.generate.assert_called_once()


@pytest.mark.parametrize("endpoint,mode", [("animate", "animate"), ("recast", "replace")])
def test_direct_fal_api_preserves_animate_parameters(dispatch, endpoint, mode):
    with TestClient(dispatch.app) as client:
        result = client.post(f"/fal-ai/wan/v2.2/{endpoint}", json={"input": _payload(
            num_inference_steps=9, guidance_scale=2.0, segment_frame_length=77,
            prev_segment_conditioning_frames=5, motion_encode_batch_size=2,
        )})
    assert result.status_code == 200, result.text
    args = dispatch.generate.call_args.kwargs
    assert args["mode"] == mode
    assert args["num_steps"] == 9
    assert args["guidance_scale"] == 2.0
    assert args["segment_frame_length"] == 77
    assert args["prev_segment_conditioning_frames"] == 5
    assert args["motion_encode_batch_size"] == 2
    assert result.json()["video"]["num_frames"] == 117


@pytest.mark.parametrize("invalid", [
    {"segment_frame_length": 42},
    {"segment_frame_length": 17},
    {"segment_frame_length": 5, "prev_segment_conditioning_frames": 5},
    {"motion_encode_batch_size": 0},
    {"animate_mode": "unknown"},
])
def test_api_rejects_invalid_segment_settings_before_dispatch(dispatch, invalid):
    with TestClient(dispatch.app) as client:
        response = client.post("/queue/wan/v2.2/animate", json=_payload(**invalid))
    assert response.status_code == 422, response.text
    dispatch.prepare.assert_not_called()
    dispatch.generate.assert_not_called()
