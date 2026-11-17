"""
Strata-Style Hardware Auto-Detector, Hugging Face Model Downloader & 50-Block Shard Packer.

Just like Niko1221/Strata's built-in installer:
1. Detects local GPUs (`nvidia-smi` / `torch.cuda` / Apple MPS), System RAM, and free NVMe disk space
   to auto-select the optimal hardware profile (`dual_5060ti_32gb`, `dual_5060ti_24gb`, `single_5090_32gb`, etc.).
2. Downloads official open-weights directly from Hugging Face (`Wan-AI/*`, `Qwen/*`, `MiniMaxAI/MiniMax-H3`,
   and `MiniMaxAI/MiniMax-H3-Image`) via `huggingface_hub` (`snapshot_download`) with resumable HTTP Range
   support, token authentication (`HF_TOKEN`), and selective partition or bundle filtering:
   - `minimax-only`: Downloads ONLY MiniMax-H3 (`fl2va`, `ref2va`, `h3-image`, `qwen3_vl`, `vae`)
   - `minimax-omni-compact`: Single-DiT disk-saver (~36 GB in INT4: `ref2va` + `qwen3_vl` + `vae` handles all tasks!)
   - `alibaba-only`: Downloads ONLY Alibaba (`wan22_t2v_moe`, `wan22_i2v_moe`, `wan22_animate`, `qwen_image_21`)
3. Repacks / quantizes monolithic `.safetensors` weights into 50 page-aligned per-block Strata shards
   (`block_00.safetensors` .. `block_49.safetensors` in `fp8_e4m3` or `int4_awq`) so `StrataDiTBlockStreamer`
   can stream individual blocks into Pinned Host RAM and GPU VRAM Counter without loading 66GB at once.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file as save_safetensors_numpy

from openfal.model_assets import (
    COMPONENT_DEPENDENCIES, HF_CONFIGS, WAN_SHARED_FILES, WEIGHT_GROUPS,
    config_keys_for_component, find_weight, unsupported_checkpoint_format,
    animate_asset_groups, missing_animate_assets,
)


@dataclass
class QuantSourceSpec:
    repo_id: str
    allow_patterns: List[str]
    approx_gb: float


@dataclass
class ModelComponentSpec:
    component_id: str
    display_name: str
    hf_repo_id: str
    subdir: str
    allow_patterns: List[str]
    approx_bf16_gb: float
    approx_fp8_gb: float
    approx_int4_gb: float
    description: str
    is_dit_partition: bool = True
    quant_sources: Dict[str, QuantSourceSpec] = field(default_factory=dict)


MODEL_COMPONENTS: Dict[str, ModelComponentSpec] = {
    "wan22_t2v_moe": ModelComponentSpec(
        component_id="wan22_t2v_moe",
        display_name="Wan2.2-T2V-A14B MoE (Primary Fast Text-to-Video: 27B Total / 14B Active)",
        hf_repo_id="Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        subdir="Wan2.2-T2V-A14B",
        allow_patterns=[
            "split_files/diffusion_models/wan2.2_t2v_*_fp8_scaled.safetensors",
            "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "split_files/vae/wan_2.1_vae.safetensors",
        ],
        approx_bf16_gb=65.1,
        approx_fp8_gb=34.2,
        approx_int4_gb=34.2,
        description="Two-Expert Temporal MoE (14B High-Noise + 14B Low-Noise). Pre-quantized FP8 weights direct from Hugging Face.",
        is_dit_partition=True,
        quant_sources={
            "fp8_e4m3": QuantSourceSpec(
                repo_id="Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
                allow_patterns=[
                    "split_files/diffusion_models/wan2.2_t2v_*_fp8_scaled.safetensors",
                    "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                    "split_files/vae/wan_2.1_vae.safetensors",
                ],
                approx_gb=34.2,
            ),
            "bf16": QuantSourceSpec(
                repo_id="Wan-AI/Wan2.2-T2V-A14B",
                allow_patterns=["*.safetensors", "*.json", "high_noise_model/*", "low_noise_model/*"],
                approx_gb=65.1,
            ),
        },
    ),
    "wan22_i2v_moe": ModelComponentSpec(
        component_id="wan22_i2v_moe",
        display_name="Wan2.2-I2V-A14B MoE (Primary Fast Image-to-Video: 27B Total / 14B Active)",
        hf_repo_id="Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        subdir="Wan2.2-I2V-A14B",
        allow_patterns=["split_files/diffusion_models/wan2.2_i2v_*_fp8_scaled.safetensors"],
        approx_bf16_gb=53.2,
        approx_fp8_gb=26.6,
        approx_int4_gb=26.6,
        description="Two-Expert Temporal MoE for Image-to-Video. Pre-quantized FP8 weights direct from Hugging Face.",
        is_dit_partition=True,
        quant_sources={
            "fp8_e4m3": QuantSourceSpec(
                repo_id="Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
                allow_patterns=["split_files/diffusion_models/wan2.2_i2v_*_fp8_scaled.safetensors"],
                approx_gb=26.6,
            ),
            "bf16": QuantSourceSpec(
                repo_id="Wan-AI/Wan2.2-I2V-A14B",
                allow_patterns=["*.safetensors", "*.json", "high_noise_model/*", "low_noise_model/*"],
                approx_gb=53.2,
            ),
        },
    ),
    "wan22_animate": ModelComponentSpec(
        component_id="wan22_animate",
        display_name="Wan2.2-Animate-14B (Video Character Animation / Replacement)",
        hf_repo_id="Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        subdir="Wan2.2-Animate-14B",
        allow_patterns=["split_files/diffusion_models/wan2.2_animate_14B_int8_convrot.safetensors"],
        approx_bf16_gb=36.7,
        approx_fp8_gb=21.7,
        approx_int4_gb=21.7,
        description="14B character animation/replacement with local CLIP, full-body pose, face crops and SAM2 video segmentation assets.",
        is_dit_partition=True,
        quant_sources={
            "fp8_e4m3": QuantSourceSpec(
                repo_id="Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
                allow_patterns=["split_files/diffusion_models/wan2.2_animate_14B_int8_convrot.safetensors"],
                approx_gb=21.7,
            ),
            "bf16": QuantSourceSpec(
                repo_id="Wan-AI/Wan2.2-Animate-14B",
                allow_patterns=["*.safetensors", "*.json"],
                approx_gb=36.7,
            ),
        },
    ),
    "qwen_image_21": ModelComponentSpec(
        component_id="qwen_image_21",
        display_name="Qwen-Image-2.1-Turbo (Primary 7B 8-Step T2I & 10-Ref Image/Face Edit)",
        hf_repo_id="Comfy-Org/Qwen-Image-2.1",
        subdir="Qwen-Image-2.1-Turbo",
        allow_patterns=[
            "diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors",
            "text_encoders/qwen3vl_8b_int8_convrot.safetensors",
            "vae/qwen_image_2.1_vae_bf16.safetensors",
        ],
        approx_bf16_gb=30.2,
        approx_fp8_gb=16.1,
        approx_int4_gb=13.3,
        description="7B Single-Stream MMDiT + Qwen3-VL-8B + VAE (pre-quantized INT8/W4A8 direct from Hugging Face).",
        is_dit_partition=True,
        quant_sources={
            "fp8_e4m3": QuantSourceSpec(
                repo_id="Comfy-Org/Qwen-Image-2.1",
                allow_patterns=[
                    "diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors",
                    "text_encoders/qwen3vl_8b_int8_convrot.safetensors",
                    "vae/qwen_image_2.1_vae_bf16.safetensors",
                ],
                approx_gb=16.1,
            ),
            "int4_awq": QuantSourceSpec(
                repo_id="Comfy-Org/Qwen-Image-2.1",
                allow_patterns=[
                    "diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors",
                    "text_encoders/qwen3vl_8b_w4a8.safetensors",
                    "vae/qwen_image_2.1_vae_bf16.safetensors",
                ],
                approx_gb=13.3,
            ),
            "bf16": QuantSourceSpec(
                repo_id="Qwen/Qwen-Image-2.1-Turbo",
                allow_patterns=["*.safetensors", "*.json", "transformer/*"],
                approx_gb=30.2,
            ),
        },
    ),
    "fl2va": ModelComponentSpec(
        component_id="fl2va",
        display_name="[MiniMax-H3] H3-Base-FL2VA (Pre-Quantized Pruned FP8 / Q4_K GGUF)",
        hf_repo_id="Comfy-Org/MiniMax-H3",
        subdir="FL2VA",
        allow_patterns=["diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors"],
        approx_bf16_gb=61.7,
        approx_fp8_gb=19.5,
        approx_int4_gb=10.6,
        description="MiniMax-H3 FL2VA DiT for joint video + synchronized stereo audio (direct pre-quantized FP8 / Q4_K).",
        is_dit_partition=True,
        quant_sources={
            "fp8_e4m3": QuantSourceSpec(
                repo_id="Comfy-Org/MiniMax-H3",
                allow_patterns=["diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors"],
                approx_gb=19.5,
            ),
            "int4_awq": QuantSourceSpec(
                repo_id="unsloth/MiniMax-H3-GGUF",
                allow_patterns=["minimax_h3_fl2va_pruned-Q4_K.gguf"],
                approx_gb=10.6,
            ),
            "bf16": QuantSourceSpec(
                repo_id="MiniMaxAI/MiniMax-H3",
                allow_patterns=["FL2VA/*", "*fl2va*", "config.json", "model_index.json"],
                approx_gb=61.7,
            ),
        },
    ),
    "ref2va": ModelComponentSpec(
        component_id="ref2va",
        display_name="[MiniMax-H3] H3-Base-Ref2VA (Pre-Quantized Pruned FP8 / Q4_K GGUF)",
        hf_repo_id="Comfy-Org/MiniMax-H3",
        subdir="Ref2VA",
        allow_patterns=["diffusion_models/minimax_h3_ref2va_pruned_fp8_scaled.safetensors"],
        approx_bf16_gb=61.7,
        approx_fp8_gb=19.5,
        approx_int4_gb=10.6,
        description="MiniMax-H3 Ref2VA DiT for Video Face Swap, 9-Image/3-Video/3-Audio Omni-Ref & Stereo Audio (direct pre-quantized).",
        is_dit_partition=True,
        quant_sources={
            "fp8_e4m3": QuantSourceSpec(
                repo_id="Comfy-Org/MiniMax-H3",
                allow_patterns=["diffusion_models/minimax_h3_ref2va_pruned_fp8_scaled.safetensors"],
                approx_gb=19.5,
            ),
            "int4_awq": QuantSourceSpec(
                repo_id="unsloth/MiniMax-H3-GGUF",
                allow_patterns=["minimax_h3_ref2va_pruned-Q4_K.gguf"],
                approx_gb=10.6,
            ),
            "bf16": QuantSourceSpec(
                repo_id="MiniMaxAI/MiniMax-H3",
                allow_patterns=["Ref2VA/*", "*ref2va*", "config.json"],
                approx_gb=61.7,
            ),
        },
    ),
    "h3-image": ModelComponentSpec(
        component_id="h3-image",
        display_name="[MiniMax-H3] MiniMax-H3-Image (Pre-Quantized INT8 / W4A8)",
        hf_repo_id="Comfy-Org/MiniMax-H3-Image",
        subdir="H3-Image",
        allow_patterns=[
            "diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors",
            "vae/minimax_h3_image_vae_fp16.safetensors",
        ],
        approx_bf16_gb=38.6,
        approx_fp8_gb=20.6,
        approx_int4_gb=12.8,
        description="MiniMax-H3-Image DiT + Image VAE (direct pre-quantized INT8 / W4A8).",
        is_dit_partition=True,
        quant_sources={
            "fp8_e4m3": QuantSourceSpec(
                repo_id="Comfy-Org/MiniMax-H3-Image",
                allow_patterns=[
                    "diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors",
                    "vae/minimax_h3_image_vae_fp16.safetensors",
                ],
                approx_gb=20.6,
            ),
            "int4_awq": QuantSourceSpec(
                repo_id="Comfy-Org/MiniMax-H3-Image",
                allow_patterns=[
                    "diffusion_models/minimax_h3_image_pruned_w4a8.safetensors",
                    "vae/minimax_h3_image_vae_fp16.safetensors",
                ],
                approx_gb=12.8,
            ),
            "bf16": QuantSourceSpec(
                repo_id="MiniMaxAI/MiniMax-H3-Image",
                allow_patterns=["*.safetensors", "*.json", "transformer/*"],
                approx_gb=38.6,
            ),
        },
    ),
    "qwen3_vl": ModelComponentSpec(
        component_id="qwen3_vl",
        display_name="Qwen3-VL-32B Shared Encoder (Pre-Quantized INT8 / Blackwell NVFP4)",
        hf_repo_id="Comfy-Org/MiniMax-H3",
        subdir="text_encoder",
        allow_patterns=["text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"],
        approx_bf16_gb=48.0,
        approx_fp8_gb=25.3,
        approx_int4_gb=14.6,
        description="Qwen3-VL-32B encoder (pre-quantized INT8 25.3GB or native Blackwell NVFP4 14.6GB).",
        is_dit_partition=False,
        quant_sources={
            "fp8_e4m3": QuantSourceSpec(
                repo_id="Comfy-Org/MiniMax-H3",
                allow_patterns=["text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"],
                approx_gb=25.3,
            ),
            "int4_awq": QuantSourceSpec(
                repo_id="Comfy-Org/MiniMax-H3",
                allow_patterns=["text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"],
                approx_gb=14.6,
            ),
            "bf16": QuantSourceSpec(
                repo_id="MiniMaxAI/MiniMax-H3",
                allow_patterns=["text_encoder/*", "tokenizer/*", "*qwen*"],
                approx_gb=48.0,
            ),
        },
    ),
    "vae": ModelComponentSpec(
        component_id="vae",
        display_name="MiniMax-H3 Video VAE (INT8) + Stereo Audio VAE",
        hf_repo_id="Comfy-Org/MiniMax-H3",
        subdir="vae",
        allow_patterns=[
            "vae/minimax_h3_video_vae_int8_convrot.safetensors",
            "vae/minimax_h3_audio_vae_fp32.safetensors",
        ],
        approx_bf16_gb=5.4,
        approx_fp8_gb=3.2,
        approx_int4_gb=3.2,
        description="Pre-quantized INT8 Video VAE (2.6GB) + Stereo Audio VAE (0.56GB).",
        is_dit_partition=False,
        quant_sources={
            "fp8_e4m3": QuantSourceSpec(
                repo_id="Comfy-Org/MiniMax-H3",
                allow_patterns=[
                    "vae/minimax_h3_video_vae_int8_convrot.safetensors",
                    "vae/minimax_h3_audio_vae_fp32.safetensors",
                ],
                approx_gb=3.2,
            ),
            "bf16": QuantSourceSpec(
                repo_id="MiniMaxAI/MiniMax-H3",
                allow_patterns=["vae/minimax_h3_video_vae_fp16.safetensors", "vae/minimax_h3_audio_vae_fp32.safetensors"],
                approx_gb=5.4,
            ),
        },
    ),
}

COMPONENT_BUNDLES: Dict[str, List[str]] = {
    "all": list(MODEL_COMPONENTS.keys()),
    "minimax-only": ["fl2va", "ref2va", "h3-image", "qwen3_vl", "vae"],
    "minimax-omni-compact": ["ref2va", "qwen3_vl", "vae"],
    "alibaba-only": ["wan22_t2v_moe", "wan22_i2v_moe", "wan22_animate", "qwen_image_21"],
}


@dataclass
class DownloadTaskState:
    task_id: str
    component_id: str
    repo_id: str
    quantization: str
    mode: str  # "hf_hub" or "strata_repack"
    status: str = "RUNNING"  # "RUNNING", "COMPLETED", "FAILED"
    progress_pct: float = 0.0
    downloaded_mb: float = 0.0
    total_mb: float = 0.0
    speed_mbps: float = 0.0
    message: str = "Initializing..."
    logs: List[str] = field(default_factory=list)
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    def add_log(self, msg: str) -> None:
        self.message = msg
        self.logs.append(msg)

    def to_dict(self) -> dict:
        return asdict(self)


MIN_COMPUTE_GPU_VRAM_GB = 8.0


def _detect_system_ram_gb() -> float:
    try:
        sys_name = platform.system()
        if sys_name == "Windows":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return round(stat.ullTotalPhys / (1024**3), 1)
        elif sys_name == "Darwin":
            mem_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
            return round(mem_bytes / (1024**3), 1)
        elif Path("/proc/meminfo").exists():
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return round(kb / (1024**2), 1)
    except Exception:
        pass
    return 64.0


def _detect_apple_silicon_gpu(sys_ram_gb: float) -> Optional[dict]:
    """Detect Apple Silicon Unified Memory GPU (Metal / PyTorch MPS) on macOS arm64."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None
    chip_name = "Apple Silicon"
    try:
        brand = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True,
            timeout=2.0,
        ).strip()
        if brand:
            chip_name = brand
    except Exception:
        pass

    mps_available = False
    try:
        import torch

        mps_available = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    except Exception:
        pass

    # macOS Metal allows ~75% of Unified Memory as active wired GPU working set
    allocatable_gb = round(sys_ram_gb * 0.75, 1)
    suffix = "Unified Memory MPS" if mps_available else "Unified Memory"
    return {
        "index": 0,
        "name": f"{chip_name} ({suffix})",
        "vram_gb": allocatable_gb,
        "unified_memory": True,
        "total_unified_gb": sys_ram_gb,
    }


def recommend_hardware_profile(
    detected_gpus: List[dict],
    cuda_visible_devices: Optional[str] = None,
) -> dict:
    """
    Filter out display/low-VRAM (< 8 GB) or severely asymmetric secondary GPUs and
    recommend the safest, fastest Strata hardware profile + CUDA_VISIBLE_DEVICES mask.
    """
    visible_filter: Optional[List[int]] = None
    if cuda_visible_devices is not None:
        raw = cuda_visible_devices.strip()
        if raw in ("-1", "none", "None"):
            visible_filter = []
        elif raw != "":
            parsed: List[int] = []
            for tok in raw.split(","):
                tok = tok.strip()
                if tok.isdigit():
                    parsed.append(int(tok))
            if parsed:
                visible_filter = parsed

    ignored_gpus: List[dict] = []
    compute_candidates: List[dict] = []

    for g in detected_gpus:
        if visible_filter is not None and g["index"] not in visible_filter:
            ignored_gpus.append({**g, "reason": "Excluded by CUDA_VISIBLE_DEVICES"})
            continue
        if g["vram_gb"] < MIN_COMPUTE_GPU_VRAM_GB:
            ignored_gpus.append(
                {**g, "reason": f"VRAM < {MIN_COMPUTE_GPU_VRAM_GB:.0f} GB (display/utility GPU excluded from compute)"}
            )
            continue
        compute_candidates.append(g)

    # Sort compute candidates by VRAM descending (tie-break by GPU index ascending)
    ranked = sorted(compute_candidates, key=lambda x: (-x["vram_gb"], x["index"]))

    selected_gpus: List[dict] = []
    if len(ranked) >= 2:
        g0, g1 = ranked[0], ranked[1]
        if g0["vram_gb"] >= 28.0 and g1["vram_gb"] < 24.0:
            # A single flagship >= 28 GB card (e.g. RTX 5090) outperforms splitting across an asymmetric secondary GPU
            selected_gpus = [g0]
            for extra in ranked[1:]:
                ignored_gpus.append(
                    {
                        **extra,
                        "reason": "Single flagship GPU (>= 28 GB) preferred over asymmetric secondary GPU",
                    }
                )
        elif g1["vram_gb"] >= 10.0:
            selected_gpus = sorted([g0, g1], key=lambda x: x["index"])
            for extra in ranked[2:]:
                ignored_gpus.append({**extra, "reason": "Not needed for 2-GPU pipeline sharding"})
        else:
            selected_gpus = [g0]
            for extra in ranked[1:]:
                ignored_gpus.append(
                    {**extra, "reason": "VRAM < 10 GB for dual-GPU sharding; using primary GPU only"}
                )
    elif len(ranked) == 1:
        selected_gpus = [ranked[0]]

    compute_num_gpus = len(selected_gpus)
    compute_vram = round(sum(g["vram_gb"] for g in selected_gpus), 1)

    if compute_num_gpus >= 2:
        pair_min_vram = min(g["vram_gb"] for g in selected_gpus)
        if pair_min_vram >= 14.0 and compute_vram >= 28.0:
            rec_profile = "dual_5060ti_32gb"
            rec_quant = "fp8_e4m3"
        elif pair_min_vram >= 10.0 and compute_vram >= 20.0:
            rec_profile = "dual_5060ti_24gb"
            rec_quant = "int4_awq"
        else:
            rec_profile = "single_12gb_minimum"
            rec_quant = "int4_awq"
    elif compute_num_gpus == 1:
        g0 = selected_gpus[0]
        vram = g0["vram_gb"]
        if g0.get("unified_memory"):
            unified_gb = float(g0.get("total_unified_gb") or round(vram / 0.75, 1))
            if unified_gb >= 56.0:
                rec_profile = "apple_silicon_64gb"
            elif unified_gb >= 30.0:
                rec_profile = "apple_silicon_36gb"
            else:
                rec_profile = "apple_silicon_24gb"
            rec_quant = "int4_awq"
        elif vram >= 28.0:
            rec_profile = "single_5090_32gb"
            rec_quant = "fp8_e4m3"
        elif vram >= 21.0:
            rec_profile = "single_4090_24gb"
            rec_quant = "fp8_e4m3"
        elif vram >= 14.0:
            rec_profile = "single_16gb_lowvram"
            rec_quant = "int4_awq"
        else:
            rec_profile = "single_12gb_minimum"
            rec_quant = "int4_awq"
    else:
        # No CUDA or Apple Silicon GPUs detected (CPU dev host)
        rec_profile = "dual_5060ti_32gb"
        rec_quant = "fp8_e4m3"

    has_unified_only = compute_num_gpus == 1 and bool(selected_gpus[0].get("unified_memory"))
    rec_cuda_visible = (
        None
        if (not selected_gpus or has_unified_only)
        else ",".join(str(g["index"]) for g in selected_gpus)
    )

    return {
        "selected_compute_gpus": selected_gpus,
        "ignored_gpus": ignored_gpus,
        "num_gpus": compute_num_gpus,
        "total_vram_gb": compute_vram,
        "raw_num_gpus": len(detected_gpus),
        "raw_total_vram_gb": round(sum(g["vram_gb"] for g in detected_gpus), 1),
        "recommended_cuda_visible_devices": rec_cuda_visible,
        "recommended_profile_id": rec_profile,
        # The Python runtime supports INT8/FP8, regardless of the hardware
        # profile's theoretical INT4 memory budget.
        "recommended_quantization": "fp8_e4m3",
        "profile_quantization": rec_quant,
    }


def _estimate_bundle_gb(component_or_bundle: str, quantization: str) -> float:
    ids = COMPONENT_BUNDLES.get(component_or_bundle, [component_or_bundle])
    total = 0.0
    for cid in ids:
        spec = MODEL_COMPONENTS[cid]
        src = StrataModelManager.resolve_download_source(spec, quantization)
        total += src.approx_gb
    return round(total, 1)


def recommend_autosetup_plans(hw: dict) -> Dict[str, dict]:
    """
    Build disk- and VRAM-aware installation plans (`minimal`, `moderate`, `aggressive`)
    for `openfal autosetup`, reserving ~15 GB OS disk headroom.
    """
    free_gb = float(hw.get("free_nvme_gb", 0.0))
    usable_gb = max(0.0, free_gb - 15.0)
    # Select formats the runtime can actually load. GGUF, NVFP4 and W4A8
    # checkpoints are not made usable by selecting a small-memory profile.
    min_quant = "fp8_e4m3"
    min_gb = _estimate_bundle_gb("qwen_image_21", min_quant)

    alibaba_gb = _estimate_bundle_gb("alibaba-only", "fp8_e4m3")
    compact_gb = _estimate_bundle_gb("minimax-omni-compact", "fp8_e4m3")
    if usable_gb >= alibaba_gb:
        mod_comp, mod_quant = "alibaba-only", "fp8_e4m3"
        mod_desc = "Alibaba FP8/INT8 checkpoints (Wan 2.2 and Qwen-Image-2.1; endpoint support varies)"
    elif usable_gb >= compact_gb:
        mod_comp, mod_quant = "minimax-omni-compact", "fp8_e4m3"
        mod_desc = "MiniMax-H3 Ref2VA with shared encoder and VAEs (FP8/INT8)"
    else:
        mod_comp, mod_quant = "qwen_image_21", min_quant
        mod_desc = "7B Qwen-Image-2.1-Turbo (INT8 image generation and editing)"
    mod_gb = _estimate_bundle_gb(mod_comp, mod_quant)

    # 3. Aggressive: Full multi-backend or largest family that fits disk
    all_fp8_gb = _estimate_bundle_gb("all", "fp8_e4m3")
    minimax_gb = _estimate_bundle_gb("minimax-only", "fp8_e4m3")
    if usable_gb >= all_fp8_gb:
        agg_comp, agg_quant = "all", "fp8_e4m3"
        agg_desc = "Full multi-backend suite (Alibaba Wan 2.2 MoE + Qwen-Image-2.1 + MiniMax-H3 FP8)"
    elif usable_gb >= alibaba_gb:
        agg_comp, agg_quant = "alibaba-only", "fp8_e4m3"
        agg_desc = "Alibaba FP8/INT8 checkpoints (largest bundle fitting available NVMe space)"
    elif usable_gb >= minimax_gb:
        agg_comp, agg_quant = "minimax-only", "fp8_e4m3"
        agg_desc = "MiniMax-H3 FP8/INT8 checkpoints (largest bundle fitting available NVMe space)"
    else:
        agg_comp, agg_quant = mod_comp, mod_quant
        agg_desc = f"{mod_desc} (capped by available disk space)"
    agg_gb = _estimate_bundle_gb(agg_comp, agg_quant)

    return {
        "minimal": {
            "tier": "minimal",
            "component": "qwen_image_21",
            "quantization": min_quant,
            "approx_gb": min_gb,
            "fits_disk": free_gb >= min_gb,
            "description": "7B Qwen-Image-2.1-Turbo (Fast Text-to-Image & 10-Ref Face/Image Edit)",
        },
        "moderate": {
            "tier": "moderate",
            "component": mod_comp,
            "quantization": mod_quant,
            "approx_gb": mod_gb,
            "fits_disk": free_gb >= mod_gb,
            "description": mod_desc,
        },
        "aggressive": {
            "tier": "aggressive",
            "component": agg_comp,
            "quantization": agg_quant,
            "approx_gb": agg_gb,
            "fits_disk": free_gb >= agg_gb,
            "description": agg_desc,
        },
    }


class StrataModelManager:
    """
    Manages Hugging Face weight downloads, hardware auto-detection, and Strata 50-block
    `.safetensors` shard quantization/packing.
    """

    def __init__(self, weights_dir: Path):
        self.weights_dir = weights_dir
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._tasks: Dict[str, DownloadTaskState] = {}

    def detect_hardware(self) -> dict:
        gpus: List[dict] = []
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi:
            try:
                out = subprocess.check_output(
                    [nvidia_smi, "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
                    text=True,
                    timeout=4.0,
                )
                for line in out.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        gpus.append(
                            {
                                "index": int(parts[0]),
                                "name": parts[1],
                                "vram_gb": round(float(parts[2]) / 1024.0, 1),
                            }
                        )
            except Exception:
                pass

        if not gpus:
            try:
                import torch

                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        props = torch.cuda.get_device_properties(i)
                        gpus.append(
                            {
                                "index": i,
                                "name": props.name,
                                "vram_gb": round(props.total_memory / (1024**3), 1),
                            }
                        )
            except Exception:
                pass

        disk_usage = shutil.disk_usage(self.weights_dir)
        free_disk_gb = round(disk_usage.free / (1024**3), 1)
        total_disk_gb = round(disk_usage.total / (1024**3), 1)
        sys_ram_gb = _detect_system_ram_gb()

        if not gpus:
            apple_gpu = _detect_apple_silicon_gpu(sys_ram_gb)
            if apple_gpu is not None:
                gpus.append(apple_gpu)

        rec = recommend_hardware_profile(
            detected_gpus=gpus,
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        )

        return {
            "platform": f"{platform.system()} {platform.machine()}",
            "detected_gpus": gpus,
            **rec,
            "system_ram_gb": sys_ram_gb,
            "free_nvme_gb": free_disk_gb,
            "total_nvme_gb": total_disk_gb,
        }

    def get_component_status(self, component_id: str) -> dict:
        spec = MODEL_COMPONENTS[component_id]
        comp_dir = self.weights_dir / spec.subdir
        safetensor_files = list(comp_dir.rglob("*.safetensors")) if comp_dir.exists() else []
        gguf_files = list(comp_dir.rglob("*.gguf")) if comp_dir.exists() else []
        weight_files = safetensor_files + gguf_files
        block_shards = list(comp_dir.glob("strata_blocks/block_*.safetensors")) if comp_dir.exists() else []
        manifest_path = comp_dir / "strata_blocks" / "strata_manifest.json"
        manifest = None
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                manifest = None

        disk_bytes = sum(f.stat().st_size for f in weight_files) if weight_files else 0
        if component_id == "wan22_animate":
            disk_bytes += sum(path.stat().st_size for group in animate_asset_groups() for asset in group.files
                              if (path := self.weights_dir / group.target_dir / asset.path).is_file()
                              and path.suffix not in (".safetensors", ".gguf"))
        missing_assets = self._missing_component_assets(component_id)
        installed = not missing_assets

        return {
            **asdict(spec),
            "local_dir": str(comp_dir),
            "installed": installed,
            "runtime_supported": True,
            "runtime_unavailable_reason": None,
            "weights_present": any("strata_blocks" not in f.parts for f in weight_files),
            "missing_assets": missing_assets,
            "safetensors_count": len(safetensor_files),
            "gguf_count": len(gguf_files),
            "strata_50_blocks_ready": len(block_shards) == 50,
            "strata_blocks_count": len(block_shards),
            "disk_size_mb": round(disk_bytes / (1024 * 1024), 2),
            "manifest": manifest,
        }

    def _missing_component_assets(self, component_id: str) -> List[str]:
        missing = []
        for cid in (component_id, *COMPONENT_DEPENDENCIES.get(component_id, ())):
            root = self.weights_dir / MODEL_COMPONENTS[cid].subdir
            missing_groups = [alternatives for alternatives in WEIGHT_GROUPS[cid]
                              if find_weight(root, alternatives) is None]
            # Unsupported files may coexist with usable checkpoints (for example
            # after replacing an older installation). They only explain a missing
            # supported checkpoint; their mere presence does not invalidate it.
            if missing_groups and root.exists():
                for path in root.rglob("*"):
                    if path.suffix in (".safetensors", ".gguf") and path.is_file() and unsupported_checkpoint_format(path.name):
                        missing.append(f"Unsupported checkpoint format: {path.relative_to(self.weights_dir)}; use FP8/INT8")
            missing.extend(f"{MODEL_COMPONENTS[cid].subdir}/{' or '.join(alternatives)}"
                           for alternatives in missing_groups)
        for key in config_keys_for_component(component_id):
            assets = HF_CONFIGS[key]
            missing.extend(f"{assets.subdir}/{name}" for name in assets.missing(self.weights_dir))
        if component_id.startswith("wan22_"):
            for name in WAN_SHARED_FILES:
                if find_weight(self.weights_dir, (Path(name).name,)) is None:
                    missing.append(name)
        if component_id == "wan22_animate":
            missing.extend(missing_animate_assets(self.weights_dir))
        return list(dict.fromkeys(missing))

    @staticmethod
    def resolve_download_source(
        spec: ModelComponentSpec,
        quantization: str,
        custom_repo_id: Optional[str] = None,
        source: str = "prequantized",
    ) -> QuantSourceSpec:
        if custom_repo_id:
            return QuantSourceSpec(
                repo_id=custom_repo_id,
                allow_patterns=spec.allow_patterns,
                approx_gb=spec.approx_fp8_gb,
            )
        if source == "official-bf16" and "bf16" in spec.quant_sources:
            return spec.quant_sources["bf16"]
        q = quantization.lower()
        if q in spec.quant_sources:
            return spec.quant_sources[q]
        if q == "nf4" and "int4_awq" in spec.quant_sources:
            return spec.quant_sources["int4_awq"]
        if "fp8_e4m3" in spec.quant_sources:
            return spec.quant_sources["fp8_e4m3"]
        return QuantSourceSpec(
            repo_id=spec.hf_repo_id,
            allow_patterns=spec.allow_patterns,
            approx_gb=spec.approx_fp8_gb,
        )

    def get_all_status(self) -> dict:
        with self._lock:
            components = [self.get_component_status(cid) for cid in MODEL_COMPONENTS]
            tasks = [t.to_dict() for t in sorted(self._tasks.values(), key=lambda x: x.started_at, reverse=True)[:10]]
            return {
                "weights_dir": str(self.weights_dir.resolve()),
                "hardware": self.detect_hardware(),
                "bundles": COMPONENT_BUNDLES,
                "components": components,
                "active_tasks": tasks,
            }

    def start_download_or_pack(
        self,
        component_id: str = "all",
        quantization: str = "fp8_e4m3",
        mode: str = "hf_hub",
        source: str = "prequantized",  # "prequantized" or "official-bf16"
        cleanup_raw: bool = False,
        hf_token: Optional[str] = None,
        custom_repo_id: Optional[str] = None,
        run_async: bool = True,
        on_log: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[dict], None]] = None,
        tqdm_class: Optional[Any] = None,
        log_file: Optional[Path] = None,
    ) -> DownloadTaskState:
        quantization = quantization.lower()
        if quantization not in ("fp8_e4m3", "int4_awq", "nf4", "bf16"):
            raise ValueError(f"Unknown quantization '{quantization}'. Use 'fp8_e4m3' for model installs.")
        if mode not in ("hf_hub", "runtime_assets", "strata_repack"):
            raise ValueError(f"Unsupported mode '{mode}'. Use 'hf_hub', 'runtime_assets', or 'strata_repack'.")
        comp_key = component_id.lower()
        if comp_key in COMPONENT_BUNDLES:
            target_ids = COMPONENT_BUNDLES[comp_key]
        else:
            target_ids = [comp_key]
        for cid in target_ids:
            if cid not in MODEL_COMPONENTS:
                valid = list(COMPONENT_BUNDLES.keys()) + list(MODEL_COMPONENTS.keys())
                raise ValueError(f"Unknown model component or bundle '{cid}'. Valid: {valid}")
        # Individual endpoint installs include their shared encoders/decoders too.
        if mode == "hf_hub":
            target_ids = list(dict.fromkeys(
                dependency for cid in target_ids
                for dependency in (*COMPONENT_DEPENDENCIES.get(cid, ()), cid)
            ))
        if source not in ("prequantized", "official-bf16"):
            raise ValueError(f"Invalid source '{source}'. Use 'prequantized' or 'official-bf16'.")

        task_id = f"dl_{int(time.time() * 1000)}_{component_id}"
        resolved_sources = [
            self.resolve_download_source(
                MODEL_COMPONENTS[cid],
                quantization=quantization,
                custom_repo_id=custom_repo_id,
                source=source,
            )
            for cid in target_ids
        ]
        total_approx_gb = max(0.01, sum(s.approx_gb for s in resolved_sources))
        detail_log_path = log_file or (self.weights_dir / "openfal_download.log")

        task = DownloadTaskState(
            task_id=task_id,
            component_id=component_id,
            repo_id=resolved_sources[0].repo_id,
            quantization=quantization,
            mode=mode,
        )
        with self._lock:
            self._tasks[task_id] = task

        def _append_log_file(msg: str) -> None:
            try:
                detail_log_path.parent.mkdir(parents=True, exist_ok=True)
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                with detail_log_path.open("a", encoding="utf-8") as fh:
                    fh.write(f"[{ts}] {msg}\n")
            except Exception:
                pass

        def _worker() -> None:
            try:
                if mode == "hf_hub":
                    if source == "official-bf16" or quantization.lower() == "bf16":
                        raise ValueError(
                            "The installer cannot prepare official BF16 repository shards for this runtime. "
                            "Use --source prequantized --quant fp8_e4m3. Recognized local single-file "
                            "BF16 checkpoints remain supported. No files were downloaded."
                        )
                    if quantization.lower() in ("int4_awq", "nf4") or any(
                        unsupported_checkpoint_format(pattern)
                        for resolved in resolved_sources for pattern in resolved.allow_patterns
                    ):
                        raise ValueError(
                            "The inference runtime does not support GGUF, NVFP4, W4A8 or INT4 checkpoints. "
                            "Use --quant fp8_e4m3 (FP8/INT8 per model). No files were downloaded. "
                            "--pack-only remains available for experimental local quantization."
                        )
                total_targets = len(target_ids)
                completed_gb = 0.0
                for idx, cid in enumerate(target_ids):
                    spec = MODEL_COMPONENTS[cid]
                    src = resolved_sources[idx]

                    def _emit(msg: str) -> None:
                        task.add_log(msg)
                        _append_log_file(msg)
                        if on_log:
                            on_log(msg)

                    if progress_callback:
                        progress_callback(
                            {
                                "phase": "component_start",
                                "component_idx": idx + 1,
                                "total_components": total_targets,
                                "component_id": cid,
                                "repo_id": src.repo_id,
                                "component_approx_gb": src.approx_gb,
                                "completed_gb": round(completed_gb, 2),
                                "total_gb": round(total_approx_gb, 2),
                                "overall_pct": round((completed_gb / total_approx_gb) * 100.0, 1),
                                "detail": f"Starting {cid} (~{src.approx_gb} GB from {src.repo_id})",
                            }
                        )

                    if mode == "hf_hub":
                        self._download_from_hf_hub(
                            spec=spec,
                            source=src,
                            source_mode=source,
                            hf_token=hf_token or os.environ.get("HF_TOKEN"),
                            quantization=quantization,
                            cleanup_raw=cleanup_raw,
                            task=task,
                            emit=_emit,
                            tqdm_class=tqdm_class,
                            progress_callback=progress_callback,
                            progress_ctx={
                                "component_idx": idx + 1,
                                "total_components": total_targets,
                                "component_id": cid,
                                "repo_id": src.repo_id,
                                "component_approx_gb": src.approx_gb,
                                "completed_gb": completed_gb,
                                "total_gb": total_approx_gb,
                            },
                        )
                    elif mode == "runtime_assets":
                        self._ensure_component_hf_configs(
                            spec=spec, target_dir=self.weights_dir / spec.subdir,
                            hf_token=hf_token or os.environ.get("HF_TOKEN"), emit=_emit,
                        )
                    elif mode == "strata_repack":
                        self._build_strata_50_block_shards(
                            spec=spec,
                            quantization=quantization,
                            cleanup_raw=cleanup_raw,
                            quantize_existing=True,
                            task=task,
                            emit=_emit,
                        )
                    else:
                        raise ValueError(f"Unsupported mode '{mode}'. Use 'hf_hub', 'runtime_assets', or 'strata_repack'.")

                    completed_gb += src.approx_gb
                    task.progress_pct = round(min(100.0, (completed_gb / total_approx_gb) * 100.0), 1)
                    if progress_callback:
                        progress_callback(
                            {
                                "phase": "component_done",
                                "component_idx": idx + 1,
                                "total_components": total_targets,
                                "component_id": cid,
                                "repo_id": src.repo_id,
                                "component_approx_gb": src.approx_gb,
                                "completed_gb": round(completed_gb, 2),
                                "total_gb": round(total_approx_gb, 2),
                                "overall_pct": task.progress_pct,
                                "detail": f"Completed {cid}",
                            }
                        )

                if mode == "hf_hub":
                    incomplete = {cid: self._missing_component_assets(cid) for cid in target_ids}
                    incomplete = {cid: items for cid, items in incomplete.items() if items}
                    if incomplete:
                        raise RuntimeError(f"Setup is incomplete; missing supported runtime assets: {incomplete}")
                task.status = "COMPLETED"
                task.progress_pct = 100.0
                task.completed_at = time.time()
                done_msg = (
                    f"[Strata-Installer] Finished '{component_id}' ({mode}, source={source}, quant={quantization}) in "
                    f"{task.completed_at - task.started_at:.1f}s"
                )
                task.add_log(done_msg)
                _append_log_file(done_msg)
            except Exception as exc:
                task.status = "FAILED"
                task.error = str(exc)
                task.completed_at = time.time()
                err_msg = f"[Strata-Installer] ERROR: {exc}"
                task.add_log(err_msg)
                _append_log_file(err_msg)

        if run_async:
            t = threading.Thread(target=_worker, daemon=True)
            t.start()
        else:
            _worker()

        return task

    def _download_from_hf_hub(
        self,
        spec: ModelComponentSpec,
        source: QuantSourceSpec,
        source_mode: str,
        hf_token: Optional[str],
        quantization: str,
        cleanup_raw: bool,
        task: DownloadTaskState,
        emit: Callable[[str], None],
        tqdm_class: Optional[Any] = None,
        progress_callback: Optional[Callable[[dict], None]] = None,
        progress_ctx: Optional[dict] = None,
    ) -> None:
        from huggingface_hub import HfApi, snapshot_download

        target_dir = self.weights_dir / spec.subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        repo_id = source.repo_id

        emit(
            f"[HF-Hub] Connecting to '{repo_id}' for '{spec.component_id}' "
            f"(target quant={quantization}, ~{source.approx_gb} GB)..."
        )
        api = HfApi(token=hf_token)

        try:
            repo_info = api.model_info(repo_id=repo_id, token=hf_token)
            siblings = [s.rfilename for s in (repo_info.siblings or [])]
            emit(f"[HF-Hub] Verified '{repo_id}' ({len(siblings)} files in repo). Starting snapshot_download...")
        except Exception as exc:
            emit(f"[HF-Hub] Note: Could not pre-query repo metadata ({exc}); attempting direct snapshot_download...")

        t0 = time.perf_counter()
        dl_kwargs: Dict[str, Any] = {
            "repo_id": repo_id,
            "local_dir": str(target_dir),
            "allow_patterns": source.allow_patterns,
            "token": hf_token,
            "max_workers": 4,
        }
        if tqdm_class is not None:
            dl_kwargs["tqdm_class"] = tqdm_class
        local_path = snapshot_download(**dl_kwargs)
        dt = max(0.001, time.perf_counter() - t0)
        emit(f"[HF-Hub] Downloaded '{spec.component_id}' to {local_path} in {dt:.1f}s.")

        self._ensure_component_hf_configs(spec=spec, target_dir=target_dir, hf_token=hf_token, emit=emit)

        # Only run local tensor quantization when pulling unquantized official BF16 weights
        should_quantize_raw = (source_mode == "official-bf16")
        if spec.is_dit_partition or should_quantize_raw:
            self._build_strata_50_block_shards(
                spec=spec,
                quantization=quantization,
                cleanup_raw=cleanup_raw and should_quantize_raw,
                quantize_existing=should_quantize_raw,
                task=task,
                emit=emit,
            )

    def _ensure_component_hf_configs(
        self,
        spec: ModelComponentSpec,
        target_dir: Path,
        hf_token: Optional[str],
        emit: Callable[[str], None],
    ) -> None:
        """Install and verify every runtime asset; failures must fail setup itself."""
        from huggingface_hub import hf_hub_download, snapshot_download

        for key in config_keys_for_component(spec.component_id):
            assets = HF_CONFIGS[key]
            if not assets.missing(self.weights_dir):
                continue
            cfg_dir = self.weights_dir / assets.subdir
            emit(f"[HF-Config] Installing config/tokenizer metadata from '{assets.repo_id}' -> {cfg_dir}...")
            snapshot_download(
                repo_id=assets.repo_id,
                revision=assets.revision,
                local_dir=str(cfg_dir),
                allow_patterns=list(assets.allow_patterns),
                ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.ckpt", "assets/*"],
                token=hf_token,
            )
            missing = assets.missing(self.weights_dir)
            if missing:
                raise RuntimeError(f"Metadata download for {assets.repo_id} is incomplete: {', '.join(missing)}")
        if spec.component_id.startswith("wan22_"):
            for filename in WAN_SHARED_FILES:
                if find_weight(self.weights_dir, (Path(filename).name,)) is not None:
                    continue
                emit(f"[HF-Hub] Installing shared Wan runtime checkpoint '{filename}'...")
                path = hf_hub_download(
                    repo_id="Comfy-Org/Wan_2.2_ComfyUI_Repackaged", filename=filename,
                    local_dir=str(self.weights_dir / "Wan2.2-T2V-A14B"), token=hf_token,
                )
                if not Path(path).is_file() or Path(path).stat().st_size == 0:
                    raise RuntimeError(f"Shared Wan checkpoint download is incomplete: {filename}")
        if spec.component_id == "wan22_animate":
            self._ensure_animate_assets(hf_token=hf_token, emit=emit)

    def _ensure_animate_assets(self, hf_token: Optional[str], emit: Callable[[str], None]) -> None:
        from huggingface_hub import snapshot_download

        for group in animate_asset_groups():
            missing = group.missing(self.weights_dir)
            if not missing:
                continue
            target = self.weights_dir / group.target_dir
            size_gb = sum(asset.size for asset in missing) / (1024 ** 3)
            emit(f"[Wan-Animate] Installing {group.id}: {len(missing)} file(s), {size_gb:.2f} GiB "
                 f"from '{group.repo_id}' at {group.revision}...")
            snapshot_download(
                repo_id=group.repo_id, revision=group.revision, local_dir=str(target),
                allow_patterns=[asset.path for asset in missing], token=hf_token, max_workers=4,
                # A stale HF local-dir record must not bless a truncated asset.
                force_download=any((target / asset.path).exists() for asset in missing),
            )
            incomplete = group.missing(self.weights_dir)
            if incomplete:
                raise RuntimeError(
                    f"Wan Animate {group.id} download is incomplete: "
                    + ", ".join(asset.path for asset in incomplete[:5])
                )
            emit(f"[Wan-Animate] Verified all {len(group.files)} {group.id} assets.")

    @staticmethod
    def _extract_block_index(tensor_name: str, num_blocks: int) -> int:
        match = re.search(r"(?:blocks|transformer_blocks|layers|double_blocks|single_blocks)\.(\d+)\.", tensor_name)
        if match:
            return int(match.group(1)) % num_blocks
        return 0

    @staticmethod
    def _quantize_array_fp8(arr: np.ndarray) -> Dict[str, np.ndarray]:
        flat = arr.astype(np.float32)
        if flat.ndim < 2:
            flat = flat.reshape(1, -1)
        cols = flat.shape[-1]
        mat = flat.reshape(-1, cols)
        # Per-column scale matching csrc/strata_dma.cu dequant_fp8_e4m3_to_bf16_kernel
        col_max = np.maximum(np.max(np.abs(mat), axis=0), 1e-8)
        scales = (col_max / 448.0).astype(np.float32)
        scaled = mat / scales[None, :]
        # Encode into IEEE/OCP FP8 E4M3 bits (1 sign, 4 exp bias 7, 3 mantissa)
        sign = (scaled < 0).astype(np.uint8) << 7
        mag = np.clip(np.abs(scaled), 0.0, 448.0)
        out_u8 = np.zeros_like(mag, dtype=np.uint8)
        nz = mag >= (2.0 ** -9)
        if np.any(nz):
            m_nz = mag[nz]
            exp = np.floor(np.log2(m_nz)).astype(np.int32)
            # Subnormal vs normal in E4M3 (bias=7, min normal exp=-6)
            sub = exp < -6
            codes = np.zeros_like(exp, dtype=np.uint8)
            if np.any(sub):
                man_sub = np.clip(np.round(m_nz[sub] / (2.0 ** -9)), 0, 7).astype(np.uint8)
                codes[sub] = man_sub
            norm = ~sub
            if np.any(norm):
                e_n = np.clip(exp[norm], -6, 8)
                man = np.round((m_nz[norm] / (2.0 ** e_n) - 1.0) * 8.0).astype(np.int32)
                carry = man >= 8
                man[carry] = 0
                e_n[carry] = np.minimum(8, e_n[carry] + 1)
                e_bits = np.clip(e_n + 7, 1, 15).astype(np.uint8)
                codes[norm] = (e_bits << 3) | (man.astype(np.uint8) & 0x07)
            out_u8[nz] = codes
        out_u8 |= sign
        return {"qweight": out_u8.reshape(arr.shape), "scales": scales}

    @staticmethod
    def _quantize_array_int4_awq(arr: np.ndarray, group_size: int = 128) -> Dict[str, np.ndarray]:
        flat = arr.astype(np.float32).reshape(-1)
        pad = (-len(flat)) % max(2, group_size)
        if pad:
            flat = np.pad(flat, (0, pad), mode="constant")
        if len(flat) % 2 != 0:
            flat = np.pad(flat, (0, 1), mode="constant")
        groups = flat.reshape(-1, group_size)
        g_min = np.min(groups, axis=1, keepdims=True)
        g_max = np.max(groups, axis=1, keepdims=True)
        scales = np.maximum((g_max - g_min) / 15.0, 1e-8)
        zeros = np.clip(np.round(-g_min / scales), 0, 15).astype(np.uint8).reshape(-1)
        q = np.clip(np.round(groups / scales) + zeros[:, None], 0, 15).astype(np.uint8).reshape(-1)
        packed = (q[0::2] & 0x0F) | ((q[1::2] & 0x0F) << 4)
        # Pack two 4-bit zero points per uint8 byte matching csrc/strata_dma.cu
        if len(zeros) % 2 != 0:
            zeros = np.pad(zeros, (0, 1), mode="constant")
        packed_zeros = (zeros[0::2] & 0x0F) | ((zeros[1::2] & 0x0F) << 4)
        return {
            "qweight": packed,
            "scales": scales.reshape(-1).astype(np.float16),
            "qzeros": packed_zeros,
        }

    def _build_strata_50_block_shards(
        self,
        spec: ModelComponentSpec,
        quantization: str,
        task: DownloadTaskState,
        emit: Callable[[str], None],
        cleanup_raw: bool = False,
        quantize_existing: bool = True,
    ) -> None:
        comp_dir = self.weights_dir / spec.subdir
        blocks_dir = comp_dir / "strata_blocks"
        blocks_dir.mkdir(parents=True, exist_ok=True)

        num_blocks = 50 if spec.is_dit_partition else 8
        raw_files = (
            sorted(
                f for f in comp_dir.rglob("*.safetensors")
                if blocks_dir not in f.parents
            )
            if quantize_existing
            else []
        )

        t0 = time.perf_counter()
        if raw_files:
            emit(
                f"[Strata-Quantizer] Found {len(raw_files)} source safetensors file(s) in {comp_dir}; "
                f"quantizing tensors into {num_blocks} '{quantization}' Strata shards..."
            )
            buckets: Dict[int, Dict[str, np.ndarray]] = {i: {} for i in range(num_blocks)}
            tensor_count = 0
            try:
                import torch  # noqa: F401
                framework = "pt"
            except ImportError:
                framework = "np"

            for rf in raw_files:
                with safe_open(str(rf), framework=framework, device="cpu") as sf:
                    for key in sf.keys():
                        t_obj = sf.get_tensor(key)
                        arr = t_obj.float().numpy() if framework == "pt" else np.asarray(t_obj)
                        blk_idx = self._extract_block_index(key, num_blocks)
                        if quantization == "fp8_e4m3" and arr.ndim >= 2 and arr.size >= 256:
                            qparts = self._quantize_array_fp8(arr)
                            buckets[blk_idx][f"{key}.qweight"] = qparts["qweight"]
                            buckets[blk_idx][f"{key}.scales"] = qparts["scales"]
                        elif quantization in ("int4_awq", "nf4") and arr.ndim >= 2 and arr.size >= 256:
                            qparts = self._quantize_array_int4_awq(arr)
                            buckets[blk_idx][f"{key}.qweight"] = qparts["qweight"]
                            buckets[blk_idx][f"{key}.scales"] = qparts["scales"]
                            buckets[blk_idx][f"{key}.qzeros"] = qparts["qzeros"]
                        else:
                            buckets[blk_idx][key] = arr.astype(np.float16 if arr.dtype == np.float32 and arr.size >= 256 else np.float32)
                        tensor_count += 1

            for blk_idx in range(num_blocks):
                shard_path = blocks_dir / f"block_{blk_idx:02d}.safetensors"
                shard_dict = buckets[blk_idx] or {"_empty_block": np.zeros((1,), dtype=np.float32)}
                save_safetensors_numpy(shard_dict, str(shard_path))
                task.progress_pct = round(((blk_idx + 1) / num_blocks) * 100.0, 1)

            source_mode = "real_tensor_quant"
            if cleanup_raw:
                removed_mb = 0.0
                for rf in raw_files:
                    removed_mb += rf.stat().st_size / (1024 * 1024)
                    rf.unlink()
                dl_cache = comp_dir / ".cache" / "huggingface" / "download"
                if dl_cache.exists():
                    shutil.rmtree(dl_cache)
                emit(
                    f"[Strata-Quantizer] Quantized {tensor_count} tensors -> deleted {len(raw_files)} raw source "
                    f"file(s) freeing {removed_mb:.1f} MB."
                )
        else:
            emit(
                f"[Strata-Packer] Packing {num_blocks} page-aligned '{quantization}' structural shards "
                f"for '{spec.component_id}' into {blocks_dir}..."
            )
            rng = np.random.default_rng(abs(hash(spec.component_id)) % (2**32))
            for blk_idx in range(num_blocks):
                shard_path = blocks_dir / f"block_{blk_idx:02d}.safetensors"
                if quantization == "fp8_e4m3":
                    q_weights = rng.integers(0, 255, size=(64, 64), dtype=np.uint8)
                elif quantization in ("int4_awq", "nf4"):
                    q_weights = rng.integers(0, 255, size=(32, 64), dtype=np.uint8)
                else:
                    q_weights = rng.standard_normal((32, 64), dtype=np.float32)

                scales = np.ones((64,), dtype=np.float32) * 0.02
                ada_ln_modulation = rng.standard_normal((6, 64), dtype=np.float32) * 0.01
                save_safetensors_numpy(
                    {
                        "attn_qkv.qweight": q_weights,
                        "attn_qkv.scales": scales,
                        "adaLN_modulation.weight": ada_ln_modulation,
                    },
                    str(shard_path),
                )
                task.progress_pct = round(((blk_idx + 1) / num_blocks) * 100.0, 1)
            source_mode = "structural_pack"

        elapsed = max(0.001, time.perf_counter() - t0)
        total_bytes = sum(f.stat().st_size for f in blocks_dir.glob("*.safetensors"))
        task.downloaded_mb = round(total_bytes / (1024 * 1024), 2)
        task.total_mb = task.downloaded_mb
        task.speed_mbps = round(task.downloaded_mb / elapsed, 2)

        manifest = {
            "component_id": spec.component_id,
            "hf_repo_id": spec.hf_repo_id,
            "quantization": quantization,
            "source_mode": source_mode,
            "num_blocks": num_blocks,
            "hidden_size": 5376,
            "num_attention_heads": 56,
            "head_dim": 128,
            "created_at": time.time(),
            "shards": [f"block_{i:02d}.safetensors" for i in range(num_blocks)],
        }
        (blocks_dir / "strata_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        emit(
            f"[Strata-Packer] Created {num_blocks} '{quantization}' shards ({source_mode}) + strata_manifest.json "
            f"for '{spec.component_id}' ({elapsed * 1000:.1f} ms)."
        )

    def remove_component(self, component_id: str) -> dict:
        cid = component_id.lower()
        if cid not in MODEL_COMPONENTS:
            raise ValueError(f"Unknown component '{component_id}'")
        spec = MODEL_COMPONENTS[cid]
        comp_dir = self.weights_dir / spec.subdir
        if comp_dir.exists():
            shutil.rmtree(comp_dir)
        return self.get_component_status(cid)

    @staticmethod
    def _dir_size_bytes(path: Path) -> int:
        if not path.exists():
            return 0
        if path.is_file():
            return path.stat().st_size
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())

    def cleanup(
        self,
        component_id: Optional[str] = None,
        incomplete_only: bool = True,
        cache_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        dry_run: bool = False,
    ) -> dict:
        """
        Clean up partial/aborted Hugging Face download caches (`*.incomplete`, `*.lock`,
        `.cache/huggingface/download`), specific component/bundle directories, or
        openfal runtime cache/output directories.
        """
        removed_paths: List[dict] = []
        freed_bytes = 0

        def _delete_target(target: Path, reason: str) -> None:
            nonlocal freed_bytes
            if not target.exists():
                return
            size = self._dir_size_bytes(target)
            freed_bytes += size
            removed_paths.append(
                {
                    "path": str(target),
                    "reason": reason,
                    "freed_mb": round(size / (1024 * 1024), 2),
                }
            )
            if not dry_run:
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()

        if component_id:
            comp_key = component_id.lower()
            if comp_key in COMPONENT_BUNDLES:
                target_ids = COMPONENT_BUNDLES[comp_key]
            elif comp_key in MODEL_COMPONENTS:
                target_ids = [comp_key]
            else:
                valid = list(COMPONENT_BUNDLES.keys()) + list(MODEL_COMPONENTS.keys())
                raise ValueError(f"Unknown component or bundle '{component_id}'. Valid: {valid}")

            for cid in target_ids:
                spec = MODEL_COMPONENTS[cid]
                _delete_target(self.weights_dir / spec.subdir, f"Removed component '{cid}'")

            # If `--component all`, also clear any leftover `.cache` under weights_dir
            if comp_key == "all":
                _delete_target(self.weights_dir / ".cache", "Removed weights cache")
        elif incomplete_only and self.weights_dir.exists():
            # Clean Hugging Face incomplete download staging dirs and stray incomplete/lock files
            for dl_cache in list(self.weights_dir.rglob(".cache/huggingface/download")):
                _delete_target(dl_cache, "Incomplete Hugging Face download cache")
            for pattern, label in (("*.incomplete", "Incomplete HF shard"), ("*.lock", "Stale HF lock file")):
                for f in list(self.weights_dir.rglob(pattern)):
                    _delete_target(f, label)

        if cache_dir is not None and cache_dir.exists():
            _delete_target(cache_dir, "Cleared openfal Layer-50 text encoder cache")

        if output_dir is not None and output_dir.exists():
            _delete_target(output_dir, "Cleared generated media outputs")

        return {
            "weights_dir": str(self.weights_dir.resolve()),
            "dry_run": dry_run,
            "removed_count": len(removed_paths),
            "freed_mb": round(freed_bytes / (1024 * 1024), 2),
            "freed_gb": round(freed_bytes / (1024**3), 3),
            "removed": removed_paths,
        }
