from pathlib import Path

from PIL import Image
import pytest

from openfal.config import EngineConfig
from openfal.engine.runner import OpenFalRunner


@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENFAL_SYNTHETIC_FALLBACK", raising=False)
    instance = OpenFalRunner(EngineConfig(
        profile_id="single_5090_32gb",
        weights_dir=tmp_path / "weights",
        output_dir=tmp_path / "outputs",
        cache_dir=tmp_path / "cache",
    ))

    class RuntimeStub:
        calls = []

        def has_weights_for(self, partition):
            return True

        def generate_video(self, **kwargs):
            self.calls.append(kwargs)
            Path(kwargs["output_path"]).write_bytes(b"stub video")

        def evict(self):
            pass

    instance._real_minimax_runner = RuntimeStub()
    return instance


@pytest.mark.parametrize("payload", [
    {"reference_image_urls": ["portrait.png"]},
    {"face_image_url": "portrait.png"},
    {"source_face_url": "portrait.png"},
    {"start_image_url": "first.png", "reference_image_urls": ["portrait.png"]},
])
def test_fl2va_rejects_references_it_cannot_condition_on(runner, payload):
    with pytest.raises(ValueError, match="reference-to-video endpoint"):
        runner.execute("minimax/h3/image-to-video", payload)
    assert not runner._real_minimax_runner.calls


def test_fl2va_rejects_conflicting_start_frame_aliases(runner):
    with pytest.raises(ValueError, match="same starting frame"):
        runner.execute("minimax/h3/image-to-video", {
            "start_image_url": "first.png", "image_url": "other.png",
        })
    assert not runner._real_minimax_runner.calls


@pytest.mark.parametrize("payload,start,end", [
    ({"start_image_url": "first.png"}, "first.png", None),
    ({"image_url": "first.png"}, "first.png", None),
    ({"end_image_url": "last.png"}, None, "last.png"),
    ({"start_image_url": "first.png", "end_image_url": "last.png"}, "first.png", "last.png"),
    ({"start_image_url": "first.png", "image_url": "first.png"}, "first.png", None),
])
def test_fl2va_reuses_decoded_anchors_and_preserves_last_frame_only(runner, monkeypatch, payload, start, end):
    images = {url: Image.new("RGB", (32, 32)) for url in payload.values()}
    loaded = []

    def load_once(url):
        assert url not in loaded, "A validated anchor must not be fetched a second time"
        loaded.append(url)
        return images[url]

    monkeypatch.setattr(runner.face_conditioner, "load_image_from_url_or_path", load_once)
    runner.execute("minimax/h3/image-to-video", payload)
    call, = runner._real_minimax_runner.calls
    assert call["start_image"] is images.get(start)
    assert call["end_image"] is images.get(end)
    assert set(loaded) == set(images)


def test_fl2va_unreadable_anchor_fails_before_generation(runner, monkeypatch):
    monkeypatch.setattr(runner.face_conditioner, "load_image_from_url_or_path", lambda url: None)
    with pytest.raises(ValueError, match="Could not load reference image"):
        runner.execute("minimax/h3/image-to-video", {"end_image_url": "broken.png"})
    assert not runner._real_minimax_runner.calls
