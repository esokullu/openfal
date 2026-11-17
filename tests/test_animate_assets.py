"""Installer/runtime agreement for Animate's complete local preprocessing models."""

import re
from pathlib import Path

import pytest

from openfal import model_assets
from openfal.engine import downloader
from openfal.model_assets import HF_CONFIGS, WAN_SHARED_FILES, SnapshotAssets, SnapshotFile


def _write(path: Path, size: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


@pytest.fixture
def tiny_assets(monkeypatch):
    groups = tuple(SnapshotAssets(
        id=group.id, repo_id=group.repo_id, revision=group.revision, target_dir=group.target_dir,
        files=tuple(SnapshotFile(path, 2) for path in names),
    ) for group, names in zip(model_assets.animate_asset_groups(), (
        ("image_encoder/config.json", "image_encoder/model.safetensors", "image_processor/preprocessor_config.json"),
        ("process_checkpoint/det/yolov10m.onnx", "process_checkpoint/pose2d/vitpose_h_wholebody.onnx/end2end.onnx",
         "process_checkpoint/pose2d/vitpose_h_wholebody.onnx/onnx__MatMul_001"),
        ("config.json", "model.safetensors", "preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json"),
    )))
    monkeypatch.setattr(model_assets, "animate_asset_groups", lambda: groups)
    monkeypatch.setattr(downloader, "animate_asset_groups", lambda: groups)
    return groups


def _install_group(root, group):
    for asset in group.files:
        _write(root / group.target_dir / asset.path, asset.size)


def _base_assets(root):
    configs = HF_CONFIGS["wan22_animate"]
    for name in configs.required_files:
        _write(root / configs.subdir / name)
    for name in WAN_SHARED_FILES:
        _write(root / name)
    _write(root / "Wan2.2-Animate-14B/split_files/diffusion_models/wan2.2_animate_14B_int8_convrot.safetensors")


def test_shipped_manifest_pins_complete_official_model_snapshots():
    groups = model_assets.animate_asset_groups()
    assert {group.id for group in groups} == {"clip", "motion", "segmentation"}
    for group in groups:
        assert re.fullmatch(r"[a-f0-9]{40}", group.revision)
        assert all(asset.size > 0 for asset in group.files)
        assert len({asset.path for asset in group.files}) == len(group.files)
        assert all(not Path(asset.path).is_absolute() and ".." not in Path(asset.path).parts for asset in group.files)
    motion = next(group for group in groups if group.id == "motion")
    pose = [asset for asset in motion.files if asset.path.startswith("process_checkpoint/pose2d/")]
    assert len(pose) == 394  # Graph plus every external tensor, not just the tiny ONNX graph.
    assert sum(asset.size for asset in pose) == 2549362899
    clip = next(group for group in groups if group.id == "clip")
    assert clip.revision == HF_CONFIGS["wan22_animate"].revision
    segmentation = next(group for group in groups if group.id == "segmentation")
    assert {asset.path for asset in segmentation.files} == {
        "model.safetensors", "config.json", "preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json",
    }


def test_runtime_asset_guard_never_downloads_and_catches_truncated_external_tensor(monkeypatch, tmp_path, tiny_assets):
    def no_network(**kwargs):
        pytest.fail("Runtime asset validation must never download")
    monkeypatch.setattr("huggingface_hub.snapshot_download", no_network)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", no_network)
    with pytest.raises(FileNotFoundError, match="runtime-assets-only"):
        model_assets.require_animate_assets(tmp_path)
    for group in tiny_assets:
        _install_group(tmp_path, group)
    paths = model_assets.require_animate_assets(tmp_path)
    assert paths["pose"].name == "end2end.onnx"
    assert paths["image_encoder"].is_dir()
    assert paths["detector"].is_file()
    sidecar = tmp_path / tiny_assets[1].target_dir / tiny_assets[1].files[-1].path
    sidecar.write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="onnx__MatMul_001"):
        model_assets.require_animate_assets(tmp_path)


def test_installed_status_requires_every_preprocessing_asset(tmp_path, tiny_assets):
    _base_assets(tmp_path)
    manager = downloader.StrataModelManager(tmp_path)
    initial = manager.get_component_status("wan22_animate")
    assert initial["weights_present"] and not initial["installed"]
    assert initial["runtime_supported"]
    for group in tiny_assets:
        _install_group(tmp_path, group)
    status = manager.get_component_status("wan22_animate")
    assert status["installed"] and status["missing_assets"] == []
    missing = tiny_assets[1].files[-1].path
    (tmp_path / tiny_assets[1].target_dir / missing).unlink()
    status = manager.get_component_status("wan22_animate")
    assert not status["installed"]
    assert status["missing_assets"] == [f"{tiny_assets[1].target_dir}/{missing}"]


def test_setup_repairs_only_missing_assets_and_is_idempotent(monkeypatch, tmp_path, tiny_assets):
    _base_assets(tmp_path)
    for group in tiny_assets:
        _install_group(tmp_path, group)
    group = tiny_assets[1]
    invalid = group.files[-1]
    (tmp_path / group.target_dir / invalid.path).write_bytes(b"x")
    requested = []
    def download(**kwargs):
        requested.append(kwargs)
        assert kwargs["allow_patterns"] == [invalid.path]
        assert kwargs["repo_id"] == group.repo_id and kwargs["revision"] == group.revision
        assert kwargs["force_download"] is True
        _write(Path(kwargs["local_dir"]) / invalid.path, invalid.size)
        return kwargs["local_dir"]
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    manager = downloader.StrataModelManager(tmp_path)
    for _ in range(2):
        task = manager.start_download_or_pack(component_id="wan22_animate", mode="runtime_assets", run_async=False)
        assert task.status == "COMPLETED", task.error
    assert len(requested) == 1
    assert manager.get_component_status("wan22_animate")["installed"]


def test_partial_preprocessing_download_fails_setup(monkeypatch, tmp_path, tiny_assets):
    _base_assets(tmp_path)
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **kwargs: kwargs["local_dir"])
    task = downloader.StrataModelManager(tmp_path).start_download_or_pack(
        component_id="wan22_animate", mode="runtime_assets", run_async=False,
    )
    assert task.status == "FAILED"
    assert "Wan Animate clip download is incomplete" in task.error
