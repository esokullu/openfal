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
