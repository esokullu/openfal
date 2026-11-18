import os

import huggingface_hub.constants as hf_constants
import pytest

from openfal.engine import downloader

XET_ERROR = (
    "Task error: File reconstruction error: CAS Client Error: "
    "Format error: I/O error: error decoding response body"
)


@pytest.fixture(autouse=True)
def _xet_enabled(monkeypatch):
    monkeypatch.setattr(hf_constants, "HF_HUB_DISABLE_XET", False)
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    monkeypatch.setattr(downloader.time, "sleep", lambda _delay: pytest.fail("unexpected sleep"))


def test_xet_failure_switches_to_https_and_resumes(monkeypatch):
    calls = []
    messages = []

    def action():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError(XET_ERROR)
        return "downloaded"

    result = downloader._run_download_with_retries(
        action, description="demo component", emit=messages.append,
    )

    assert result == "downloaded"
    assert len(calls) == 2
    assert hf_constants.HF_HUB_DISABLE_XET is True
    assert os.environ["HF_HUB_DISABLE_XET"] == "1"
    assert any("plain HTTPS" in message for message in messages)


def test_retryable_https_error_retries_with_backoff(monkeypatch):
    sleeps = []
    monkeypatch.setattr(downloader.time, "sleep", sleeps.append)
    calls = []

    def action():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionResetError("Connection reset by peer")
        return "downloaded"

    assert downloader._run_download_with_retries(
        action, description="demo component", emit=lambda _message: None,
    ) == "downloaded"
    assert len(calls) == 3
    assert sleeps == [10.0, 30.0]


def test_non_retryable_error_fails_immediately():
    calls = []

    def action():
        calls.append(1)
        raise ValueError("401 Unauthorized")

    with pytest.raises(ValueError, match="401"):
        downloader._run_download_with_retries(
            action, description="demo component", emit=lambda _message: None,
        )
    assert len(calls) == 1


def test_component_download_uses_https_fallback(monkeypatch, tmp_path):
    calls = []

    def flaky_snapshot(**kwargs):
        calls.append(kwargs["repo_id"])
        if len(calls) == 1:
            raise RuntimeError(XET_ERROR)
        return kwargs["local_dir"]

    monkeypatch.setattr("huggingface_hub.snapshot_download", flaky_snapshot)
    manager = downloader.StrataModelManager(tmp_path)
    monkeypatch.setattr(manager, "_ensure_component_hf_configs", lambda **_kwargs: None)
    monkeypatch.setattr(manager, "_missing_component_assets", lambda _component: [])

    task = manager.start_download_or_pack(component_id="qwen3_vl", run_async=False)

    assert task.status == "COMPLETED", task.error
    assert len(calls) == 2
    assert hf_constants.HF_HUB_DISABLE_XET is True
