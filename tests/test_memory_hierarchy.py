from pathlib import Path
from openfal.config import EngineConfig, HARDWARE_PROFILES
from openfal.engine.memory_hierarchy import StrataMemoryHierarchy, TierName


def test_hardware_profiles_fit_vram_budgets(tmp_path: Path) -> None:
    for profile_id, profile in HARDWARE_PROFILES.items():
        cfg = EngineConfig(
            profile_id=profile_id,
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
        )
        mem = StrataMemoryHierarchy(cfg)
        telemetry = mem.get_telemetry()

        assert len(telemetry["blocks"]) == 50
        tier1 = telemetry["tiers"]["tier1_vram_counter"]
        assert tier1["used_gb"] <= profile.total_vram_gb
        assert tier1["resident_dit_blocks"] == profile.resident_vram_blocks


def test_dual_5060ti_24gb_and_32gb_pipeline_sharding(tmp_path: Path) -> None:
    cfg = EngineConfig(
        profile_id="dual_5060ti_32gb",
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    mem = StrataMemoryHierarchy(cfg)
    gpu0_blocks = [b for b in mem.blocks if b.assigned_gpu == 0]
    gpu1_blocks = [b for b in mem.blocks if b.assigned_gpu == 1]
    assert len(gpu0_blocks) == 25
    assert len(gpu1_blocks) == 25

    # Switch to 24GB dual-GPU budget profile
    mem.switch_profile("dual_5060ti_24gb")
    t24 = mem.get_telemetry()
    assert t24["profile"]["total_vram_gb"] == 24.0
    assert t24["tiers"]["tier1_vram_counter"]["used_gb"] <= 24.0


def test_sleep_and_wake_vram_eviction(tmp_path: Path) -> None:
    cfg = EngineConfig(
        profile_id="single_5090_32gb",
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )
    mem = StrataMemoryHierarchy(cfg)
    awake_telemetry = mem.get_telemetry()
    assert awake_telemetry["tiers"]["tier1_vram_counter"]["resident_dit_blocks"] == 30

    slept = mem.sleep()
    assert slept["sleeping"] is True
    assert slept["tiers"]["tier1_vram_counter"]["resident_dit_blocks"] == 0
    assert slept["tiers"]["tier1_vram_counter"]["used_gb"] == 0.0
    assert all(b.tier == TierName.RAM_PANTRY for b in mem.blocks)

    woke = mem.wake()
    assert woke["sleeping"] is False
    assert woke["tiers"]["tier1_vram_counter"]["resident_dit_blocks"] == 30
