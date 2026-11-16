from pathlib import Path
import argparse
from openfal.cli import cmd_cleanup
from openfal.engine.downloader import MODEL_COMPONENTS, StrataModelManager, recommend_hardware_profile


def test_prequantized_hf_source_resolution() -> None:
    wan_fp8 = StrataModelManager.resolve_download_source(MODEL_COMPONENTS["wan22_t2v_moe"], "fp8_e4m3")
    assert wan_fp8.repo_id == "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
    assert any("fp8_scaled" in p for p in wan_fp8.allow_patterns)

    fl2va_fp8 = StrataModelManager.resolve_download_source(MODEL_COMPONENTS["fl2va"], "fp8_e4m3")
    assert fl2va_fp8.repo_id == "Comfy-Org/MiniMax-H3"
    assert fl2va_fp8.approx_gb < 25.0

    fl2va_int4 = StrataModelManager.resolve_download_source(MODEL_COMPONENTS["fl2va"], "int4_awq")
    assert fl2va_int4.repo_id == "unsloth/MiniMax-H3-GGUF"
    assert any("Q4_K.gguf" in p for p in fl2va_int4.allow_patterns)

    qwen_fp8 = StrataModelManager.resolve_download_source(MODEL_COMPONENTS["qwen_image_21"], "fp8_e4m3")
    assert qwen_fp8.repo_id == "Comfy-Org/Qwen-Image-2.1"

    fl2va_bf16 = StrataModelManager.resolve_download_source(MODEL_COMPONENTS["fl2va"], "bf16")
    assert fl2va_bf16.repo_id == "MiniMaxAI/MiniMax-H3"


def test_rtx_5090_plus_t400_4gb_excludes_display_gpu() -> None:
    """Reproduces real Windows rig: GPU 0 = RTX 5090 (31.8 GB), GPU 1 = T400 (4.0 GB)."""
    gpus = [
        {"index": 0, "name": "NVIDIA GeForce RTX 5090", "vram_gb": 31.8},
        {"index": 1, "name": "NVIDIA T400 4GB", "vram_gb": 4.0},
    ]
    rec = recommend_hardware_profile(gpus)
    assert rec["recommended_profile_id"] == "single_5090_32gb"
    assert rec["recommended_quantization"] == "fp8_e4m3"
    assert rec["num_gpus"] == 1
    assert rec["total_vram_gb"] == 31.8
    assert rec["recommended_cuda_visible_devices"] == "0"
    assert len(rec["ignored_gpus"]) == 1
    assert rec["ignored_gpus"][0]["index"] == 1


def test_rtx_5090_plus_asymmetric_12gb_prefers_flagship_single_gpu() -> None:
    gpus = [
        {"index": 0, "name": "NVIDIA GeForce RTX 3060", "vram_gb": 12.0},
        {"index": 1, "name": "NVIDIA GeForce RTX 5090", "vram_gb": 31.8},
    ]
    rec = recommend_hardware_profile(gpus)
    assert rec["recommended_profile_id"] == "single_5090_32gb"
    assert rec["recommended_cuda_visible_devices"] == "1"
    assert rec["num_gpus"] == 1


def test_dual_5060ti_16gb_and_dual_12gb_detection() -> None:
    dual_16 = [
        {"index": 0, "name": "NVIDIA GeForce RTX 5060 Ti", "vram_gb": 16.0},
        {"index": 1, "name": "NVIDIA GeForce RTX 5060 Ti", "vram_gb": 16.0},
    ]
    rec32 = recommend_hardware_profile(dual_16)
    assert rec32["recommended_profile_id"] == "dual_5060ti_32gb"
    assert rec32["recommended_quantization"] == "fp8_e4m3"
    assert rec32["recommended_cuda_visible_devices"] == "0,1"

    dual_12 = [
        {"index": 0, "name": "NVIDIA GeForce RTX 4070", "vram_gb": 12.0},
        {"index": 1, "name": "NVIDIA GeForce RTX 3060", "vram_gb": 12.0},
    ]
    rec24 = recommend_hardware_profile(dual_12)
    assert rec24["recommended_profile_id"] == "dual_5060ti_24gb"
    assert rec24["recommended_quantization"] == "int4_awq"
    assert rec24["recommended_cuda_visible_devices"] == "0,1"


def test_single_12gb_and_cuda_visible_devices_mask() -> None:
    single_12 = [{"index": 0, "name": "NVIDIA GeForce RTX 3060", "vram_gb": 12.0}]
    rec12 = recommend_hardware_profile(single_12)
    assert rec12["recommended_profile_id"] == "single_12gb_minimum"
    assert rec12["recommended_quantization"] == "int4_awq"

    dual_16 = [
        {"index": 0, "name": "NVIDIA GeForce RTX 5060 Ti", "vram_gb": 16.0},
        {"index": 1, "name": "NVIDIA GeForce RTX 5060 Ti", "vram_gb": 16.0},
    ]
    masked = recommend_hardware_profile(dual_16, cuda_visible_devices="1")
    assert masked["recommended_profile_id"] == "single_16gb_lowvram"
    assert masked["recommended_cuda_visible_devices"] == "1"


def test_cleanup_incomplete_and_component_verification(tmp_path: Path, monkeypatch) -> None:
    weights_dir = tmp_path / "weights"
    comp_dir = weights_dir / "Wan2.2-T2V-A14B"
    dl_cache = comp_dir / ".cache" / "huggingface" / "download"
    dl_cache.mkdir(parents=True)
    (dl_cache / "blob.incomplete").write_bytes(b"123456")
    finished = comp_dir / "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
    finished.write_bytes(b"abcdef")

    mgr = StrataModelManager(weights_dir)

    # Dry run keeps everything
    dry = mgr.cleanup(dry_run=True)
    assert dry["removed_count"] >= 1
    assert dl_cache.exists()
    assert finished.exists()

    # Incomplete cleanup deletes partial HF cache but preserves finished safetensors
    res_inc = mgr.cleanup(incomplete_only=True, dry_run=False)
    assert res_inc["removed_count"] >= 1
    assert not dl_cache.exists()
    assert finished.exists()

    # CLI verification prompt rejects on "n", deletes on "y"
    args_no = argparse.Namespace(
        weights_dir=str(weights_dir),
        component="wan22_t2v_moe",
        cache=False,
        cache_dir=str(tmp_path / "cache"),
        outputs=False,
        output_dir=str(tmp_path / "out"),
        dry_run=False,
        yes=False,
    )
    monkeypatch.setattr("builtins.input", lambda _: "n")
    cmd_cleanup(args_no)
    assert finished.exists()

    monkeypatch.setattr("builtins.input", lambda _: "y")
    cmd_cleanup(args_no)
    assert not comp_dir.exists()
