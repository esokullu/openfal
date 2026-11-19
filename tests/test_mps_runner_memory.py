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


def test_minimax_mps_stages_release_text_encoder_and_dit_between_stages(monkeypatch, tmp_path):
    from openfal.config import EngineConfig
    from openfal.engine.memory_hierarchy import StrataMemoryHierarchy
    import openfal.runtime.minimax_h3_runner as mm

    cfg = EngineConfig(profile_id="apple_silicon_64gb", weights_dir=tmp_path)
    runner = mm.RealMiniMaxH3Runner(cfg, StrataMemoryHierarchy(cfg))
    runner._loaded_partition = "Ref2VA"
    runner._active_paths = {"text_encoder": tmp_path / "te.safetensors", "dit": tmp_path / "dit.safetensors"}
    runner._cfg_dir = tmp_path

    # On 64 GB Apple Silicon, the 20 GB DiT fits fully in the 48 GB MPS working set (50/50)
    # without redundant CPU<->MPS Pantry copies.
    assert common.compute_resident_blocks(cfg, 50, estimated_model_vram_gb=20.0) == 50

    class _DummyModule:
        def __init__(self, name):
            self.name = name
            self.devices = []
            self.transformer_blocks = [object()] * 50

        def to(self, device):
            self.devices.append(str(device))
            return self

    refs = {}
    events = []

    class _FakePipe:
        def __init__(self):
            self._component_specs = {"text_encoder": object(), "transformer": object(), "transformer_ref": object()}
            self.text_encoder = None
            self.tokenizer = None
            self.processor = None
            self.transformer = None
            self.transformer_ref = None
            self.blocks = SimpleNamespace(
                sub_blocks={
                    "text_encoder": self._run_te,
                    "denoise": self._run_denoise,
                }
            )

        def update_components(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def _run_te(self, pipe, state):
            assert pipe.text_encoder is not None
            assert pipe.transformer is None and pipe.transformer_ref is None
            events.append("stage1")
            return pipe, state

        def _run_denoise(self, pipe, state):
            # Qwen3-VL-32B must already be garbage-collected before Stage 2 runs.
            assert pipe.text_encoder is None
            assert refs["te"]() is None
            assert pipe.transformer_ref is not None
            events.append("stage2")
            return pipe, state

    def fake_load_te(cfg_dir, te_path, on_log=None):
        te = _DummyModule("qwen3vl")
        refs["te"] = weakref.ref(te)
        runner._text_encoder = te
        runner._tokenizer = object()
        runner._processor = object()

    def fake_load_dit(partition, dit_path, device, compute_dtype, on_log=None):
        assert refs["te"]() is None
        dit = _DummyModule(partition)
        refs["dit"] = weakref.ref(dit)
        return dit

    monkeypatch.setattr(mm, "clear_device_cache", lambda device=None: events.append("clear_cache"))
    monkeypatch.setattr(runner, "_ensure_shared_qwen3vl", fake_load_te)
    monkeypatch.setattr(runner, "_load_partition_dit", fake_load_dit)

    pipe = _FakePipe()
    mps = torch.device("mps")
    pipe, state = runner._run_stage1_text_encoder(pipe, {}, mps)
    assert runner._text_encoder is None and pipe.text_encoder is None
    assert refs["te"]() is None

    pipe, state = runner._run_stage2_denoise(
        "Ref2VA", pipe, state, mps, torch.bfloat16, False, lambda c, p: f"{c}/{p}",
    )
    assert pipe.transformer is None and pipe.transformer_ref is None
    assert refs["dit"]() is None
    assert events == ["stage1", "clear_cache", "stage2", "clear_cache"]


def test_minimax_cuda_stages_retain_cpu_caches_across_stages(monkeypatch, tmp_path):
    from openfal.config import EngineConfig
    from openfal.engine.memory_hierarchy import StrataMemoryHierarchy
    import openfal.runtime.minimax_h3_runner as mm

    cfg = EngineConfig(profile_id="single_5090_32gb", weights_dir=tmp_path)
    runner = mm.RealMiniMaxH3Runner(cfg, StrataMemoryHierarchy(cfg))

    class _DummyModule:
        def __init__(self):
            self.devices = []
            self.transformer_blocks = [object()] * 50

        def to(self, device):
            self.devices.append(str(device))
            return self

    te = _DummyModule()
    dit = _DummyModule()
    runner._text_encoder = te
    pipe = SimpleNamespace(
        text_encoder=te,
        transformer=dit,
        transformer_ref=dit,
        blocks=SimpleNamespace(
            sub_blocks={
                "text_encoder": lambda p, s: (p, s),
                "denoise": lambda p, s: (p, s),
            }
        ),
    )
    monkeypatch.setattr(mm, "clear_device_cache", lambda device=None: None)
    cuda = torch.device("cuda", 0)
    runner._run_stage1_text_encoder(pipe, {}, cuda)
    runner._run_stage2_denoise("FL2VA", pipe, {}, cuda, torch.bfloat16, False, lambda c, p: f"{c}/{p}")
    assert runner._text_encoder is te and pipe.text_encoder is te
    assert pipe.transformer is dit
    assert te.devices[-1] == "cpu" and dit.devices[-1] == "cpu"


def test_wan22_mps_eviction_releases_experts_and_text_encoder(monkeypatch, tmp_path):
    from openfal.config import EngineConfig
    from openfal.engine.memory_hierarchy import StrataMemoryHierarchy
    import openfal.runtime.wan22_runner as wan

    cfg = EngineConfig(profile_id="apple_silicon_64gb", weights_dir=tmp_path)
    runner = wan.RealWan22Runner(cfg, StrataMemoryHierarchy(cfg))

    class _Dummy:
        def to(self, device):
            return self

    expert = _Dummy()
    te = _Dummy()
    vae = _Dummy()
    refs = (weakref.ref(expert), weakref.ref(te), weakref.ref(vae))
    runner._experts["Wan2.2-T2V-A14B:high"] = (expert, None)
    runner._text_encoder = te
    runner._vae = vae
    del expert, te, vae

    monkeypatch.setattr(wan, "resolve_compute_device_and_dtype", lambda: (torch.device("mps"), torch.bfloat16))
    monkeypatch.setattr(wan, "clear_device_cache", lambda device=None: None)
    runner.evict()
    assert not runner._experts and runner._text_encoder is None and runner._vae is None
    assert all(r() is None for r in refs)
