"""Unified-memory lifecycle tests without allocating Metal or CUDA tensors."""

import gc
import threading
from types import SimpleNamespace
from unittest.mock import Mock
import weakref

import pytest
import torch

from openfal.engine.runner import OpenFalRunner
import openfal.runtime.common as common


_SLOTS = ("_real_qwen_runner", "_real_wan_animate_runner", "_real_wan_runner", "_real_minimax_runner")


class _Checkpoint:
    pass


class _Runtime:
    def __init__(self, name, events, shared=None):
        self.name = name
        self.events = events
        self.shared = shared
        self.checkpoint = shared.checkpoint if shared else _Checkpoint()
        # Module forward hooks can form cycles; deleting top-level references
        # alone is insufficient to release unified RAM before the next load.
        self.hook_cycle = self

    def evict(self):
        self.events.append(("evict", self.name))
        if self.shared is not None:
            self.shared.evict()


@pytest.fixture
def lifecycle(monkeypatch):
    events = []
    device = SimpleNamespace(type="mps")
    monkeypatch.setattr(common, "resolve_compute_device_and_dtype", lambda: (torch.device(device.type), torch.float32))
    monkeypatch.setattr(common, "clear_device_cache", lambda target: events.append(("cache", target.type)))
    runner = OpenFalRunner.__new__(OpenFalRunner)
    for slot in _SLOTS:
        setattr(runner, slot, None)
    runner._active_real_family = None
    runner._active_real_partition = None
    runner.execution_lock = threading.RLock()
    runner.memory = SimpleNamespace(finish_generation=Mock())
    return SimpleNamespace(runner=runner, events=events, device=device)


def _install_shared_wan(lifecycle):
    shared = _Runtime("wan", lifecycle.events)
    animate = _Runtime("animate", lifecycle.events, shared=shared)
    lifecycle.runner._real_wan_runner = shared
    lifecycle.runner._real_wan_animate_runner = animate
    return weakref.ref(shared), weakref.ref(animate), weakref.ref(shared.checkpoint)


def test_same_mps_partition_reuses_warm_shared_models(lifecycle):
    references = _install_shared_wan(lifecycle)
    runner = lifecycle.runner
    runner._active_real_family = "wan_animate"
    runner._active_real_partition = "Wan2.2-Animate-14B"
    runner._prepare_real_partition("Wan2.2-Animate-14B")
    assert all(reference() is not None for reference in references)
    assert runner._real_wan_animate_runner.shared is runner._real_wan_runner
    assert not lifecycle.events
    assert runner._active_real_family == "wan_animate"


@pytest.mark.parametrize("old_partition,new_partition", [
    ("Qwen-Image-2.1-Turbo", "Wan2.2-Animate-14B"),
    ("Wan2.2-T2V-A14B", "Wan2.2-I2V-A14B"),
    ("FL2VA", "Ref2VA"),
])
def test_mps_switch_discards_previous_checkpoints_before_cache_release(lifecycle, monkeypatch, old_partition, new_partition):
    runner = lifecycle.runner
    references = list(_install_shared_wan(lifecycle))
    for slot, name in (("_real_qwen_runner", "qwen"), ("_real_minimax_runner", "minimax")):
        runtime = _Runtime(name, lifecycle.events)
        setattr(runner, slot, runtime)
        references.extend((weakref.ref(runtime), weakref.ref(runtime.checkpoint)))
    del runtime
    runner._active_real_family = "wan_animate"
    runner._active_real_partition = old_partition

    def cache_after_checkpoint_release(device):
        assert all(reference() is None for reference in references)
        lifecycle.events.append(("cache", device.type))

    monkeypatch.setattr(common, "clear_device_cache", cache_after_checkpoint_release)
    runner._prepare_real_partition(new_partition)
    assert all(getattr(runner, slot) is None for slot in _SLOTS)
    assert {name for event, name in lifecycle.events if event == "evict"} == {"qwen", "wan", "animate", "minimax"}
    assert lifecycle.events[-1] == ("cache", "mps")
    assert runner._active_real_family is None
    assert runner._active_real_partition == new_partition


def test_explicit_mps_eviction_immediately_releases_shared_owners(lifecycle):
    references = _install_shared_wan(lifecycle)
    runner = lifecycle.runner
    runner._active_real_family = "wan_animate"
    runner._active_real_partition = "Wan2.2-Animate-14B"
    runner.evict()
    assert runner._active_real_family is None
    assert runner._active_real_partition is None
    assert ("evict", "animate") in lifecycle.events
    assert ("evict", "wan") in lifecycle.events
    assert all(reference() is None for reference in references)
    assert all(getattr(runner, slot) is None for slot in _SLOTS)
    assert lifecycle.events[-1] == ("cache", "mps")


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_other_devices_keep_cpu_caches_when_partition_changes(lifecycle, device):
    lifecycle.device.type = device
    references = _install_shared_wan(lifecycle)
    runner = lifecycle.runner
    runner._active_real_family = "wan_animate"
    runner._active_real_partition = "Wan2.2-Animate-14B"
    runner._prepare_real_partition("Qwen-Image-2.1-Turbo")
    assert all(reference() is not None for reference in references)
    assert runner._real_wan_animate_runner.shared is runner._real_wan_runner
    assert runner._active_real_family == "wan_animate"
    assert runner._active_real_partition == "Qwen-Image-2.1-Turbo"
    assert not lifecycle.events


def test_explicit_cuda_eviction_preserves_existing_cpu_cache_policy(lifecycle):
    lifecycle.device.type = "cuda"
    references = _install_shared_wan(lifecycle)
    runner = lifecycle.runner
    runner._active_real_family = "wan_animate"
    runner._active_real_partition = "Wan2.2-Animate-14B"
    runner.evict()
    assert all(reference() is not None for reference in references)
    assert runner._active_real_family is None and runner._active_real_partition is None
    assert ("evict", "animate") in lifecycle.events
    assert ("evict", "wan") in lifecycle.events


def test_failed_cleanup_drops_shared_owners_and_preserves_inference_error(lifecycle, monkeypatch):
    references = _install_shared_wan(lifecycle)
    runner = lifecycle.runner
    runner._active_real_family = "wan_animate"
    runner._active_real_partition = "Wan2.2-Animate-14B"

    def fail_inference(*args):
        raise ValueError("original inference failure")

    def fail_cleanup():
        raise RuntimeError("device cleanup failure")

    monkeypatch.setattr(runner, "_execute", fail_inference)
    runner._real_wan_animate_runner.evict = fail_cleanup
    with pytest.raises(ValueError, match="original inference failure"):
        runner.execute("wan/v2.2/animate", {})
    assert all(getattr(runner, slot) is None for slot in _SLOTS)
    assert runner._active_real_family is None and runner._active_real_partition is None
    runner.memory.finish_generation.assert_called_once()
    gc.collect()
    assert all(reference() is None for reference in references)
