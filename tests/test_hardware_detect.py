from pathlib import Path
import argparse
import numpy as np
from safetensors.numpy import load_file as load_safetensors_numpy, save_file as save_safetensors_numpy
from openfal.cli import cmd_autosetup, cmd_cleanup
from openfal.engine.downloader import (
    MODEL_COMPONENTS,
    StrataModelManager,
    recommend_autosetup_plans,
    recommend_hardware_profile,
)


def test_prequantized_and_official_bf16_source_resolution() -> None:
    wan_fp8 = StrataModelManager.resolve_download_source(MODEL_COMPONENTS["wan22_t2v_moe"], "fp8_e4m3")
    assert wan_fp8.repo_id == "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
    assert any("fp8_scaled" in p for p in wan_fp8.allow_patterns)

    # Even with quant=fp8_e4m3 or int4_awq, source='official-bf16' must select upstream official BF16 repo
    wan_official = StrataModelManager.resolve_download_source(
        MODEL_COMPONENTS["wan22_t2v_moe"], "fp8_e4m3", source="official-bf16"
    )
    assert wan_official.repo_id == "Wan-AI/Wan2.2-T2V-A14B"

    fl2va_official = StrataModelManager.resolve_download_source(
        MODEL_COMPONENTS["fl2va"], "int4_awq", source="official-bf16"
    )
    assert fl2va_official.repo_id == "MiniMaxAI/MiniMax-H3"

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


def test_real_local_tensor_quantization_fp8_int4_and_cleanup_raw(tmp_path: Path) -> None:
    weights_dir = tmp_path / "weights"
    comp_dir = weights_dir / "FL2VA"
    comp_dir.mkdir(parents=True)

    rng = np.random.default_rng(123)
    raw_file = comp_dir / "model-00001-of-00001.safetensors"
    save_safetensors_numpy(
        {
            "blocks.0.attn.qkv.weight": rng.standard_normal((64, 256)).astype(np.float32),
            "blocks.27.mlp.fc1.weight": rng.standard_normal((64, 256)).astype(np.float32),
            "blocks.27.norm.weight": np.ones((256,), dtype=np.float32),
        },
        str(raw_file),
    )

    mgr = StrataModelManager(weights_dir)
    # 1. Quantize locally to FP8 E4M3 without removing raw file yet
    t_fp8 = mgr.start_download_or_pack(
        component_id="fl2va",
        quantization="fp8_e4m3",
        mode="strata_repack",
        cleanup_raw=False,
        run_async=False,
    )
    assert t_fp8.status == "COMPLETED"
    assert raw_file.exists()

    blk27 = load_safetensors_numpy(str(comp_dir / "strata_blocks" / "block_27.safetensors"))
    assert "blocks.27.mlp.fc1.weight.qweight" in blk27
    assert blk27["blocks.27.mlp.fc1.weight.qweight"].dtype == np.uint8
    assert "blocks.27.mlp.fc1.weight.scales" in blk27
    assert "blocks.27.norm.weight" in blk27

    # 2. Quantize locally to INT4_AWQ with cleanup_raw=True -> deletes source BF16 safetensors
    t_int4 = mgr.start_download_or_pack(
        component_id="fl2va",
        quantization="int4_awq",
        mode="strata_repack",
        cleanup_raw=True,
        run_async=False,
    )
    assert t_int4.status == "COMPLETED"
    assert not raw_file.exists()

    blk0 = load_safetensors_numpy(str(comp_dir / "strata_blocks" / "block_00.safetensors"))
    assert "blocks.0.attn.qkv.weight.qweight" in blk0
    assert "blocks.0.attn.qkv.weight.qzeros" in blk0
    assert blk0["blocks.0.attn.qkv.weight.qweight"].dtype == np.uint8


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


def test_apple_silicon_unified_memory_profile_selection() -> None:
    mac_64 = [
        {
            "index": 0,
            "name": "Apple M3 Max (Unified Memory MPS)",
            "vram_gb": 48.0,
            "unified_memory": True,
            "total_unified_gb": 64.0,
        }
    ]
    rec64 = recommend_hardware_profile(mac_64)
    assert rec64["recommended_profile_id"] == "apple_silicon_64gb"
    assert rec64["recommended_quantization"] == "int4_awq"
    assert rec64["recommended_cuda_visible_devices"] is None

    mac_36 = [
        {
            "index": 0,
            "name": "Apple M3 Max (Unified Memory MPS)",
            "vram_gb": 27.0,
            "unified_memory": True,
            "total_unified_gb": 36.0,
        }
    ]
    rec36 = recommend_hardware_profile(mac_36)
    assert rec36["recommended_profile_id"] == "apple_silicon_36gb"

    mac_24 = [
        {
            "index": 0,
            "name": "Apple M4 Pro (Unified Memory MPS)",
            "vram_gb": 18.0,
            "unified_memory": True,
            "total_unified_gb": 24.0,
        }
    ]
    rec24 = recommend_hardware_profile(mac_24)
    assert rec24["recommended_profile_id"] == "apple_silicon_24gb"


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


def test_autosetup_plans_and_cli_override(tmp_path: Path) -> None:
    # High disk space (268 GB) + RTX 5090 FP8 -> moderate=alibaba-only, aggressive=all (fp8)
    plans_268 = recommend_autosetup_plans(
        {"free_nvme_gb": 268.0, "recommended_quantization": "fp8_e4m3"}
    )
    assert plans_268["minimal"]["component"] == "qwen_image_21"
    assert plans_268["moderate"]["component"] == "alibaba-only"
    assert plans_268["aggressive"]["component"] == "all"
    assert plans_268["aggressive"]["quantization"] == "fp8_e4m3"

    # Tighter disk space (45 GB) -> moderate=minimax-omni-compact (int4_awq)
    plans_45 = recommend_autosetup_plans(
        {"free_nvme_gb": 45.0, "recommended_quantization": "int4_awq"}
    )
    assert plans_45["moderate"]["component"] == "minimax-omni-compact"
    assert plans_45["moderate"]["quantization"] == "int4_awq"

    # Non-interactive `autosetup -y --pack-only --tier minimal` runs cleanly without input()
    weights_dir = tmp_path / "weights"
    args_auto = argparse.Namespace(
        weights_dir=str(weights_dir),
        tier="minimal",
        yes=True,
        dry_run=False,
        pack_only=True,
        hf_token=None,
    )
    cmd_autosetup(args_auto)
    assert (weights_dir / "Qwen-Image-2.1-Turbo" / "strata_blocks" / "strata_manifest.json").exists()
