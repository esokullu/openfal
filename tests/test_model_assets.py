from pathlib import Path

import pytest

from openfal.engine.downloader import MODEL_COMPONENTS, StrataModelManager
from openfal.model_assets import HF_CONFIGS, WAN_SHARED_FILES
from openfal.runtime.common import (
    ensure_hf_configs, ensure_model_configs, find_first_safetensors, find_matching_safetensors,
)


def _write(root: Path, name: str, contents: bytes = b"{}") -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    return path


def _metadata(root: Path, key: str) -> None:
    assets = HF_CONFIGS[key]
    for name in assets.required_files:
        _write(root / assets.subdir, name)


def test_inference_metadata_missing_is_local_error(monkeypatch, tmp_path):
    def no_download(*args, **kwargs):
        pytest.fail("Inference attempted a network model download")
    monkeypatch.setattr("huggingface_hub.snapshot_download", no_download)
    with pytest.raises(FileNotFoundError, match="runtime-assets-only"):
        ensure_model_configs("qwen_image_21", tmp_path)
    _metadata(tmp_path, "qwen_image_21")
    assert ensure_model_configs("qwen_image_21", tmp_path) == tmp_path / HF_CONFIGS["qwen_image_21"].subdir


def test_metadata_subfolder_and_empty_file(tmp_path):
    _write(tmp_path, "nested/config.json")
    assert ensure_hf_configs("test/model", tmp_path, ["config.json"], [], subfolder="nested") == tmp_path / "nested"
    _write(tmp_path, "nested/config.json", b"")
    with pytest.raises(FileNotFoundError, match="config.json"):
        ensure_hf_configs("test/model", tmp_path, ["config.json"], [], subfolder="nested")


def test_minimax_setup_uses_upstream_processor_filenames(tmp_path):
    _metadata(tmp_path, "minimax_h3")
    files = HF_CONFIGS["minimax_h3"].required_files
    assert "processor/preprocessor_config.json" in files
    assert "processor/processor_config.json" not in files
    assert ensure_model_configs("minimax_h3", tmp_path).is_dir()


@pytest.mark.parametrize("failed_download", [False, True])
def test_setup_cannot_defer_metadata_failure(monkeypatch, tmp_path, failed_download):
    def incomplete_download(**kwargs):
        if failed_download:
            raise OSError("offline")
        return kwargs["local_dir"]
    monkeypatch.setattr("huggingface_hub.snapshot_download", incomplete_download)
    task = StrataModelManager(tmp_path).start_download_or_pack(
        component_id="qwen_image_21", mode="runtime_assets", run_async=False,
    )
    assert task.status == "FAILED"
    assert "offline" in task.error if failed_download else "incomplete" in task.error


def test_minimax_metadata_installed_once_and_repair_skips_checkpoint_download(monkeypatch, tmp_path):
    requested = []
    def metadata_download(**kwargs):
        requested.append(kwargs["repo_id"])
        key = next(key for key, spec in HF_CONFIGS.items() if spec.repo_id == kwargs["repo_id"])
        _metadata(tmp_path, key)
        return kwargs["local_dir"]
    def no_checkpoint_download(**kwargs):
        pytest.fail("MiniMax metadata repair must not download checkpoints")
    monkeypatch.setattr("huggingface_hub.snapshot_download", metadata_download)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", no_checkpoint_download)
    manager = StrataModelManager(tmp_path)
    for _ in range(2):
        task = manager.start_download_or_pack(component_id="minimax-only", mode="runtime_assets", run_async=False)
        assert task.status == "COMPLETED", task.error
    assert requested == ["MiniMaxAI/MiniMax-H3", "MiniMaxAI/MiniMax-H3-Image"]


def test_wan_individual_install_fetches_shared_16ch_vae_and_encoder(monkeypatch, tmp_path):
    _metadata(tmp_path, "wan22_i2v_moe")
    # The 48-channel TI2V-5B decoder must not satisfy the 14B runtime dependency.
    _write(tmp_path, "Wan2.2-T2V-A14B/split_files/vae/wan2.2_vae.safetensors")
    requested = []
    def download(**kwargs):
        requested.append(kwargs["filename"])
        return str(_write(Path(kwargs["local_dir"]), kwargs["filename"]))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    manager = StrataModelManager(tmp_path)
    for _ in range(2):
        task = manager.start_download_or_pack(component_id="wan22_i2v_moe", mode="runtime_assets", run_async=False)
        assert task.status == "COMPLETED", task.error
    assert requested == list(WAN_SHARED_FILES)


def test_installed_requires_both_wan_experts_and_all_assets(tmp_path):
    manager = StrataModelManager(tmp_path)
    root = tmp_path / MODEL_COMPONENTS["wan22_i2v_moe"].subdir
    _write(root, "strata_blocks/block_00.safetensors")
    _write(root, "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors")
    assert not manager.get_component_status("wan22_i2v_moe")["installed"]
    _metadata(tmp_path, "wan22_i2v_moe")
    for name in WAN_SHARED_FILES:
        _write(tmp_path / "Wan2.2-T2V-A14B", name)
    assert not manager.get_component_status("wan22_i2v_moe")["installed"]
    _write(root, "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors")
    assert manager.get_component_status("wan22_i2v_moe")["installed"]
    (tmp_path / HF_CONFIGS["wan22_i2v_moe"].subdir / "tokenizer/spiece.model").unlink()
    status = manager.get_component_status("wan22_i2v_moe")
    assert not status["installed"]
    assert any("spiece.model" in name for name in status["missing_assets"])


def test_minimax_download_expands_dependencies(monkeypatch, tmp_path):
    manager = StrataModelManager(tmp_path)
    requested = []
    def download(**kwargs):
        requested.append(kwargs["spec"].component_id)
    monkeypatch.setattr(manager, "_download_from_hf_hub", download)
    monkeypatch.setattr(manager, "_missing_component_assets", lambda _: [])
    task = manager.start_download_or_pack(component_id="h3-image", run_async=False)
    assert task.status == "COMPLETED"
    assert requested == ["qwen3_vl", "h3-image"]


@pytest.mark.parametrize("quantization", ["int4_awq", "nf4"])
def test_unsupported_quantization_fails_before_any_download(monkeypatch, tmp_path, quantization):
    def forbidden(**kwargs):
        pytest.fail("Unsupported formats must fail before the first network download")
    manager = StrataModelManager(tmp_path)
    monkeypatch.setattr(manager, "_download_from_hf_hub", forbidden)
    task = manager.start_download_or_pack(component_id="all", quantization=quantization, run_async=False)
    assert task.status == "FAILED"
    assert "does not support" in task.error
    assert "--quant fp8_e4m3" in task.error


@pytest.mark.parametrize("name", [
    "minimax_h3_ref2va_pruned-Q4_K.gguf",
    "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "minimax_h3_image_pruned_w4a8.safetensors",
])
def test_unsupported_checkpoints_never_report_installed(tmp_path, name):
    component = "ref2va" if name.endswith("gguf") else ("qwen3_vl" if "nvfp4" in name else "h3-image")
    _write(tmp_path / MODEL_COMPONENTS[component].subdir, name)
    status = StrataModelManager(tmp_path).get_component_status(component)
    assert not status["installed"]
    assert any("Unsupported checkpoint format" in issue for issue in status["missing_assets"])


def test_supported_encoder_with_old_unsupported_files_is_ready_and_selected(tmp_path):
    root = tmp_path / MODEL_COMPONENTS["qwen3_vl"].subdir
    _metadata(tmp_path, "minimax_h3")
    _write(root, "a-old/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")
    _write(root, "a-old/qwen3vl_32b_minimax_h3-Q4_K.gguf")
    supported = _write(root, "z-current/qwen3vl_32b_minimax_h3_int8_convrot.safetensors")
    status = StrataModelManager(tmp_path).get_component_status("qwen3_vl")
    assert status["installed"]
    assert status["missing_assets"] == []
    assert find_matching_safetensors(root, ["qwen3vl_32b"]) == supported
    assert find_first_safetensors(root, [
        "a-old/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", supported.name,
    ]) == supported


def test_recognized_local_bf16_qwen_layout_is_ready(tmp_path):
    from openfal.config import EngineConfig
    from openfal.runtime.qwen_image_runner import RealQwenImage21Runner

    root = tmp_path / MODEL_COMPONENTS["qwen_image_21"].subdir
    _metadata(tmp_path, "qwen_image_21")
    expected = (
        _write(root, "diffusion_models/qwen_image_2.1_bf16.safetensors"),
        _write(root, "text_encoders/qwen3vl_8b_bf16.safetensors"),
        _write(root, "vae/diffusion_pytorch_model.safetensors"),
    )
    _write(root, "unused/qwen3vl_8b_w4a8.safetensors")
    assert StrataModelManager(tmp_path).get_component_status("qwen_image_21")["installed"]
    runtime = RealQwenImage21Runner(EngineConfig(weights_dir=tmp_path), memory=None)
    assert runtime.resolve_weight_paths() == expected


def test_generic_vae_filename_cannot_resolve_to_transformer(tmp_path):
    _write(tmp_path, "transformer/diffusion_pytorch_model.safetensors")
    assert find_first_safetensors(tmp_path, ["vae/diffusion_pytorch_model.safetensors"]) is None
    vae = _write(tmp_path, "nested/vae/diffusion_pytorch_model.safetensors")
    assert find_first_safetensors(tmp_path, ["vae/diffusion_pytorch_model.safetensors"]) == vae


@pytest.mark.parametrize("source,quantization", [
    ("official-bf16", "fp8_e4m3"), ("prequantized", "bf16"),
])
def test_unsupported_official_layout_fails_before_download(monkeypatch, tmp_path, source, quantization):
    manager = StrataModelManager(tmp_path)
    def forbidden(**kwargs):
        pytest.fail("The unsupported official layout must fail before downloading checkpoints")
    monkeypatch.setattr(manager, "_download_from_hf_hub", forbidden)
    task = manager.start_download_or_pack(
        component_id="qwen_image_21", source=source, quantization=quantization, run_async=False,
    )
    assert task.status == "FAILED"
    assert "official BF16 repository shards" in task.error


@pytest.mark.parametrize("option,value", [("quantization", "unknown_format"), ("mode", "unknown_mode")])
def test_invalid_install_options_cannot_silently_fallback(monkeypatch, tmp_path, option, value):
    manager = StrataModelManager(tmp_path)
    def forbidden(**kwargs):
        pytest.fail("Invalid install options must fail before downloads")
    monkeypatch.setattr(manager, "_download_from_hf_hub", forbidden)
    with pytest.raises(ValueError, match=value):
        manager.start_download_or_pack(component_id="qwen_image_21", run_async=False, **{option: value})
    assert manager._tasks == {}
