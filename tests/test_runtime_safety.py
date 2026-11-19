"""Concurrency and fail-closed behavior without loading model checkpoints."""

import threading

import pytest

from openfal.config import EngineConfig
from openfal.engine.runner import OpenFalRunner
from openfal.server.queue_manager import FalQueueManager, QueuedJob


def test_missing_weights_never_return_placeholder(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENFAL_SYNTHETIC_FALLBACK", raising=False)
    runner = OpenFalRunner(EngineConfig(
        weights_dir=tmp_path / "weights", output_dir=tmp_path / "outputs",
        cache_dir=tmp_path / "cache",
    ))
    with pytest.raises(RuntimeError, match="No usable model weights"):
        runner.execute("qwen/image-2.1/text-to-image", {"prompt": "test"})
    assert not list((tmp_path / "outputs").glob("*.png"))


def test_video_length_is_not_silently_truncated(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENFAL_SYNTHETIC_FALLBACK", raising=False)
    runner = OpenFalRunner(EngineConfig(
        weights_dir=tmp_path / "weights", output_dir=tmp_path / "outputs",
        cache_dir=tmp_path / "cache",
    ))
    with pytest.raises(ValueError, match="at most 81 frames"):
        runner.execute("wan/v2.2/text-to-video", {"duration": 10, "fps": 16})


def test_queue_serializes_inference_and_honors_waiting_cancellation():
    entered, release = threading.Event(), threading.Event()

    class Runner:
        execution_lock = threading.RLock()
        calls = []

        def execute(self, endpoint_id, **kwargs):
            self.calls.append(endpoint_id)
            entered.set()
            assert release.wait(5)
            return {"ok": True}

    runner = Runner()
    queue = FalQueueManager(runner)
    first = queue.submit_job("first", {})
    try:
        assert entered.wait(5)
        second = queue.submit_job("second", {})
        assert second.status == "IN_QUEUE"
        queue.cancel_job(second.request_id)
        assert second.status == "CANCELLED"
        assert runner.calls == ["first"]
    finally:
        release.set()
    assert first.done_event.wait(5)
    assert first.status == "COMPLETED"
    assert second.status == "CANCELLED"
    assert runner.calls == ["first"]


def test_running_cancellation_does_not_become_completed():
    entered, release = threading.Event(), threading.Event()

    class Runner:
        execution_lock = threading.RLock()

        def execute(self, **kwargs):
            entered.set()
            assert release.wait(5)
            return {"ok": True}

    queue = FalQueueManager(Runner())
    job = queue.submit_job("first", {})
    try:
        assert entered.wait(5)
        queue.cancel_job(job.request_id)
    finally:
        release.set()
    # Acquire after execution to observe the terminal state, not a transient flag.
    with queue._execution_lock:
        assert job.status == "CANCELLED"
        assert job.result is None


def test_completion_cannot_overwrite_a_successful_cancellation(monkeypatch):
    """Force a cancellation into the window between result and status updates."""
    assigning_result = threading.Event()
    cancel_started, cancel_finished = threading.Event(), threading.Event()
    original_setattr = QueuedJob.__setattr__

    def pause_result_assignment(self, name, value):
        if name == "result" and value is not None:
            assigning_result.set()
            assert cancel_started.wait(5)
            # A correct atomic completion keeps cancellation waiting until the
            # terminal state is published. An unlocked worker lets it return
            # CANCELLED here, then incorrectly overwrites that state.
            cancel_finished.wait(0.5)
        original_setattr(self, name, value)

    monkeypatch.setattr(QueuedJob, "__setattr__", pause_result_assignment)

    class Runner:
        execution_lock = threading.RLock()

        def execute(self, **kwargs):
            return {"ok": True}

    queue = FalQueueManager(Runner())
    job = queue.submit_job("first", {})
    assert assigning_result.wait(5)
    observed = []

    def cancel():
        cancel_started.set()
        observed.append(queue.cancel_job(job.request_id).status)
        cancel_finished.set()

    cancelling = threading.Thread(target=cancel)
    cancelling.start()
    cancelling.join(5)
    assert not cancelling.is_alive()
    with queue._execution_lock:
        assert observed == [job.status]
        assert job.status in ("CANCELLED", "COMPLETED")


def test_inference_extra_declares_torchvision():
    from pathlib import Path
    import re

    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    try:
        import tomllib

        inference_deps = tomllib.loads(text)["project"]["optional-dependencies"]["inference"]
        assert any(dep.lower().startswith("torchvision") for dep in inference_deps)
    except ModuleNotFoundError:
        match = re.search(r"inference\s*=\s*\[(.*?)\]", text, re.S)
        assert match and "torchvision" in match.group(1).lower()


def test_torchvision_missing_fails_fast_before_loading_weights(monkeypatch, tmp_path):
    import importlib
    from openfal.engine.memory_hierarchy import StrataMemoryHierarchy
    from openfal.runtime.minimax_h3_runner import RealMiniMaxH3Runner
    from openfal.runtime.qwen_image_runner import RealQwenImage21Runner
    from openfal.runtime.wan_animate_preprocess import preprocess_wan_animate

    real_import = importlib.import_module

    def block_torchvision(name, *args, **kwargs):
        if name == "torchvision" or name.startswith("torchvision."):
            raise ImportError("No module named 'torchvision'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", block_torchvision)
    cfg = EngineConfig(weights_dir=tmp_path / "weights", output_dir=tmp_path / "outputs", cache_dir=tmp_path / "cache")
    mem = StrataMemoryHierarchy(cfg)

    with pytest.raises(RuntimeError, match="torchvision"):
        RealQwenImage21Runner(cfg, mem)._ensure_loaded()
    with pytest.raises(RuntimeError, match="torchvision"):
        RealMiniMaxH3Runner(cfg, mem)._ensure_loaded("FL2VA")
    with pytest.raises(RuntimeError, match="torchvision"):
        preprocess_wan_animate(
            tmp_path / "missing.mp4",
            weights_dir=tmp_path / "weights",
            width=64,
            height=64,
            fps=16,
            num_frames=4,
            mode="replace",
        )
