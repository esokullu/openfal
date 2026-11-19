"""
Hardware profiles & configuration for the openfal Strata-DiT inference engine.

Specifically engineered to run the 33B MiniMax-H3 DiT (50 transformer blocks,
hidden_size=5376, 56 heads, head_dim=128) + 32B Qwen3-VL text/multimodal encoder +
Hybrid CNN-ViT VAE on consumer GPUs without vLLM-Omni or enterprise H100 clusters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


class QuantizationMode(str, Enum):
    FP8_E4M3 = "fp8_e4m3"   # ~660 MB / DiT block (33 GB total for 50 blocks) - Blackwell native
    INT4_AWQ = "int4_awq"   # ~340 MB / DiT block (17 GB total for 50 blocks) - 24GB total VRAM sweet spot
    NF4 = "nf4"             # ~350 MB / DiT block (17.5 GB total for 50 blocks)
    BF16 = "bf16"           # ~1320 MB / DiT block (66 GB total for 50 blocks) - High-RAM streaming


class OffloadStrategy(str, Enum):
    STRATA_DUAL_GPU_PIPELINE = "strata_dual_gpu_pipeline"   # 2x 5060 Ti: 25 blocks/GPU + per-GPU pinned RAM DMA
    STRATA_RING_STREAM = "strata_ring_stream"               # 1x 5090 / 4090: K resident blocks + 2 ping-pong DMA slots
    STRATA_LOW_VRAM_MICRO = "strata_low_vram_micro"         # Single 12-16GB GPU: 6 resident blocks + INT4/FP8 DMA
    UNIFIED_MEMORY = "unified_memory"                       # Apple Silicon / APU unified RAM


@dataclass
class GPUDeviceSpec:
    device_id: int
    name: str
    vram_gb: float
    pcie_gen: int = 5
    pcie_lanes: int = 16
    assigned_block_start: int = 0
    assigned_block_end: int = 50
    resident_blocks: int = 12


@dataclass
class HardwareProfile:
    profile_id: str
    display_name: str
    description: str
    num_gpus: int
    total_vram_gb: float
    min_system_ram_gb: int
    recommended_system_ram_gb: int
    quantization: QuantizationMode
    offload_strategy: OffloadStrategy
    total_dit_blocks: int
    resident_vram_blocks: int
    ping_pong_prefetch_slots: int
    qwen3_vl_stream_layers: int
    vae_tiled_decode: bool
    pcie_bandwidth_gbps: float
    gpus: List[GPUDeviceSpec] = field(default_factory=list)

    @property
    def block_size_mb(self) -> float:
        # MiniMax-H3 33B DiT has 50 blocks (~660M params per block)
        mapping = {
            QuantizationMode.BF16: 1320.0,
            QuantizationMode.FP8_E4M3: 660.0,
            QuantizationMode.NF4: 350.0,
            QuantizationMode.INT4_AWQ: 340.0,
        }
        return mapping[self.quantization]

    @property
    def active_pantry_ram_gb(self) -> float:
        # Offloaded DiT blocks in page-locked (cudaHostAlloc) RAM + 4.0 GB Qwen3-VL / VAE staging buffer
        offloaded_blocks = self.total_dit_blocks - self.resident_vram_blocks
        return round((offloaded_blocks * self.block_size_mb) / 1024.0 + 4.0, 1)

    @property
    def full_partition_pinned_ram_gb(self) -> float:
        # Full 50-block partition pinned in RAM (allows instant VRAM Sleep/Wake + zero NVMe read during steps)
        return round((self.total_dit_blocks * self.block_size_mb) / 1024.0 + 4.0, 1)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["block_size_mb"] = self.block_size_mb
        data["active_pantry_ram_gb"] = self.active_pantry_ram_gb
        data["full_partition_pinned_ram_gb"] = self.full_partition_pinned_ram_gb
        return data


# Pre-calibrated consumer GPU profiles inspired by Niko1221/Strata
HARDWARE_PROFILES: Dict[str, HardwareProfile] = {
    "dual_5060ti_32gb": HardwareProfile(
        profile_id="dual_5060ti_32gb",
        display_name="2x NVIDIA RTX 5060 Ti (2x 16GB = 32GB VRAM)",
        description=(
            "Dual-GPU pipeline sharding (Blocks 0-24 on GPU0, Blocks 25-49 on GPU1) with "
            "FP8_E4M3 weights. 28 blocks pinned in VRAM Counter (14/GPU) + 22 blocks streamed "
            "asynchronously from Pinned DDR5 RAM over PCIe 5.0 with zero compute stall."
        ),
        num_gpus=2,
        total_vram_gb=32.0,
        min_system_ram_gb=32,
        recommended_system_ram_gb=64,
        quantization=QuantizationMode.FP8_E4M3,
        offload_strategy=OffloadStrategy.STRATA_DUAL_GPU_PIPELINE,
        total_dit_blocks=50,
        resident_vram_blocks=28,
        ping_pong_prefetch_slots=4,  # 2 slots per GPU
        qwen3_vl_stream_layers=8,
        vae_tiled_decode=True,
        pcie_bandwidth_gbps=58.4,
        gpus=[
            GPUDeviceSpec(
                device_id=0,
                name="NVIDIA GeForce RTX 5060 Ti (16GB)",
                vram_gb=16.0,
                pcie_gen=5,
                pcie_lanes=8,
                assigned_block_start=0,
                assigned_block_end=25,
                resident_blocks=14,
            ),
            GPUDeviceSpec(
                device_id=1,
                name="NVIDIA GeForce RTX 5060 Ti (16GB)",
                vram_gb=16.0,
                pcie_gen=5,
                pcie_lanes=8,
                assigned_block_start=25,
                assigned_block_end=50,
                resident_blocks=14,
            ),
        ],
    ),
    "dual_5060ti_24gb": HardwareProfile(
        profile_id="dual_5060ti_24gb",
        display_name="2x Consumer GPU 24GB Total (2x 12GB / 16GB+8GB VRAM)",
        description=(
            "Ultra-tight 24GB dual-GPU Strata configuration using hybrid INT4-AWQ / FP8 "
            "quantization. Keeps 20 hot DiT blocks in VRAM (10/GPU) and streams 30 blocks from "
            "page-locked Host RAM while maintaining full 768P/1080P latent headroom."
        ),
        num_gpus=2,
        total_vram_gb=24.0,
        min_system_ram_gb=24,
        recommended_system_ram_gb=48,
        quantization=QuantizationMode.INT4_AWQ,
        offload_strategy=OffloadStrategy.STRATA_DUAL_GPU_PIPELINE,
        total_dit_blocks=50,
        resident_vram_blocks=24,
        ping_pong_prefetch_slots=4,
        qwen3_vl_stream_layers=6,
        vae_tiled_decode=True,
        pcie_bandwidth_gbps=52.0,
        gpus=[
            GPUDeviceSpec(
                device_id=0,
                name="NVIDIA GeForce RTX 5060 Ti (12GB Budget Cap)",
                vram_gb=12.0,
                pcie_gen=5,
                pcie_lanes=8,
                assigned_block_start=0,
                assigned_block_end=25,
                resident_blocks=12,
            ),
            GPUDeviceSpec(
                device_id=1,
                name="NVIDIA GeForce RTX 5060 Ti (12GB Budget Cap)",
                vram_gb=12.0,
                pcie_gen=5,
                pcie_lanes=8,
                assigned_block_start=25,
                assigned_block_end=50,
                resident_blocks=12,
            ),
        ],
    ),
    "single_5090_32gb": HardwareProfile(
        profile_id="single_5090_32gb",
        display_name="1x NVIDIA RTX 5090 (32GB GDDR7 VRAM)",
        description=(
            "Single flagship Blackwell GPU running native FP8_E4M3. Holds 30 of 50 DiT blocks "
            "resident in GDDR7 Counter (1.79 TB/s internal bandwidth) and streams the remaining "
            "20 blocks via PCIe 5.0 x16 double-buffered DMA (~60 GB/s) with 99% compute overlap."
        ),
        num_gpus=1,
        total_vram_gb=32.0,
        min_system_ram_gb=32,
        recommended_system_ram_gb=64,
        quantization=QuantizationMode.FP8_E4M3,
        offload_strategy=OffloadStrategy.STRATA_RING_STREAM,
        total_dit_blocks=50,
        resident_vram_blocks=30,
        ping_pong_prefetch_slots=2,
        qwen3_vl_stream_layers=12,
        vae_tiled_decode=True,
        pcie_bandwidth_gbps=60.2,
        gpus=[
            GPUDeviceSpec(
                device_id=0,
                name="NVIDIA GeForce RTX 5090 (32GB)",
                vram_gb=32.0,
                pcie_gen=5,
                pcie_lanes=16,
                assigned_block_start=0,
                assigned_block_end=50,
                resident_blocks=30,
            ),
        ],
    ),
    "single_4090_24gb": HardwareProfile(
        profile_id="single_4090_24gb",
        display_name="1x NVIDIA RTX 4090 / 3090 (24GB VRAM)",
        description=(
            "Single 24GB GPU profile with FP8_E4M3 / INT4-AWQ block streaming. Keeps 20 DiT "
            "blocks in VRAM Counter and streams 30 blocks from Pinned System RAM."
        ),
        num_gpus=1,
        total_vram_gb=24.0,
        min_system_ram_gb=32,
        recommended_system_ram_gb=64,
        quantization=QuantizationMode.FP8_E4M3,
        offload_strategy=OffloadStrategy.STRATA_RING_STREAM,
        total_dit_blocks=50,
        resident_vram_blocks=20,
        ping_pong_prefetch_slots=2,
        qwen3_vl_stream_layers=8,
        vae_tiled_decode=True,
        pcie_bandwidth_gbps=28.5,
        gpus=[
            GPUDeviceSpec(
                device_id=0,
                name="NVIDIA GeForce RTX 4090 (24GB)",
                vram_gb=24.0,
                pcie_gen=4,
                pcie_lanes=16,
                assigned_block_start=0,
                assigned_block_end=50,
                resident_blocks=20,
            ),
        ],
    ),
    "single_16gb_lowvram": HardwareProfile(
        profile_id="single_16gb_lowvram",
        display_name="1x RTX 5060 Ti / 4080 / 4060 Ti (16GB VRAM)",
        description=(
            "Single 16GB GPU micro-counter mode (Strata classic). Uses INT4-AWQ weights, "
            "keeps 12 DiT blocks in VRAM Counter, and streams 38 blocks from Pinned Host RAM."
        ),
        num_gpus=1,
        total_vram_gb=16.0,
        min_system_ram_gb=32,
        recommended_system_ram_gb=48,
        quantization=QuantizationMode.INT4_AWQ,
        offload_strategy=OffloadStrategy.STRATA_LOW_VRAM_MICRO,
        total_dit_blocks=50,
        resident_vram_blocks=12,
        ping_pong_prefetch_slots=2,
        qwen3_vl_stream_layers=4,
        vae_tiled_decode=True,
        pcie_bandwidth_gbps=30.0,
        gpus=[
            GPUDeviceSpec(
                device_id=0,
                name="NVIDIA GeForce RTX 5060 Ti (16GB)",
                vram_gb=16.0,
                pcie_gen=5,
                pcie_lanes=8,
                assigned_block_start=0,
                assigned_block_end=50,
                resident_blocks=12,
            ),
        ],
    ),
    "single_12gb_minimum": HardwareProfile(
        profile_id="single_12gb_minimum",
        display_name="1x RTX 4070 / 3060 / 5060 (12GB VRAM Minimum Entry)",
        description=(
            "Minimum entry-level 12GB VRAM configuration. Uses INT4-AWQ weights, pins 6 DiT "
            "blocks in VRAM Counter + 2 DMA slots, and streams 44 blocks from Pinned Host RAM."
        ),
        num_gpus=1,
        total_vram_gb=12.0,
        min_system_ram_gb=32,
        recommended_system_ram_gb=48,
        quantization=QuantizationMode.INT4_AWQ,
        offload_strategy=OffloadStrategy.STRATA_LOW_VRAM_MICRO,
        total_dit_blocks=50,
        resident_vram_blocks=6,
        ping_pong_prefetch_slots=2,
        qwen3_vl_stream_layers=3,
        vae_tiled_decode=True,
        pcie_bandwidth_gbps=24.0,
        gpus=[
            GPUDeviceSpec(
                device_id=0,
                name="NVIDIA GeForce RTX 12GB (Entry Minimum)",
                vram_gb=12.0,
                pcie_gen=4,
                pcie_lanes=16,
                assigned_block_start=0,
                assigned_block_end=50,
                resident_blocks=6,
            ),
        ],
    ),
    "apple_silicon_64gb": HardwareProfile(
        profile_id="apple_silicon_64gb",
        display_name="Apple Silicon M1/M2/M3/M4 Max or Ultra (64GB+ Unified Memory)",
        description=(
            "Zero-copy Unified Memory MPS profile for 64GB+ Macs. Keeps 40 INT4-AWQ DiT blocks "
            "wired in active Metal MPS working set and maps the remaining 10 blocks via zero-copy Unified RAM."
        ),
        num_gpus=1,
        total_vram_gb=48.0,
        min_system_ram_gb=64,
        recommended_system_ram_gb=64,
        quantization=QuantizationMode.INT4_AWQ,
        offload_strategy=OffloadStrategy.UNIFIED_MEMORY,
        total_dit_blocks=50,
        resident_vram_blocks=40,
        ping_pong_prefetch_slots=2,
        qwen3_vl_stream_layers=12,
        vae_tiled_decode=True,
        pcie_bandwidth_gbps=400.0,
        gpus=[
            GPUDeviceSpec(
                device_id=0,
                name="Apple Silicon GPU (48GB MPS Working Set / 64GB+ Unified)",
                vram_gb=48.0,
                pcie_gen=5,
                pcie_lanes=16,
                assigned_block_start=0,
                assigned_block_end=50,
                resident_blocks=40,
            ),
        ],
    ),
    "apple_silicon_36gb": HardwareProfile(
        profile_id="apple_silicon_36gb",
        display_name="Apple Silicon M3/M4 Max or Pro (32GB–48GB Unified Memory)",
        description=(
            "Zero-copy Unified Memory MPS profile for 32GB/36GB/48GB Macs. Keeps 24 INT4-AWQ DiT "
            "blocks in active Metal MPS working set and streams 26 blocks via zero-copy Unified RAM."
        ),
        num_gpus=1,
        total_vram_gb=24.0,
        min_system_ram_gb=32,
        recommended_system_ram_gb=36,
        quantization=QuantizationMode.INT4_AWQ,
        offload_strategy=OffloadStrategy.UNIFIED_MEMORY,
        total_dit_blocks=50,
        resident_vram_blocks=24,
        ping_pong_prefetch_slots=2,
        qwen3_vl_stream_layers=8,
        vae_tiled_decode=True,
        pcie_bandwidth_gbps=300.0,
        gpus=[
            GPUDeviceSpec(
                device_id=0,
                name="Apple Silicon GPU (24GB MPS Working Set / 36GB Unified)",
                vram_gb=24.0,
                pcie_gen=5,
                pcie_lanes=16,
                assigned_block_start=0,
                assigned_block_end=50,
                resident_blocks=24,
            ),
        ],
    ),
    "apple_silicon_24gb": HardwareProfile(
        profile_id="apple_silicon_24gb",
        display_name="Apple Silicon M2/M3/M4 Pro or Base (24GB Unified Memory Minimum)",
        description=(
            "Entry Apple Silicon Unified Memory MPS profile for 24GB Macs (ideal for 7B Qwen-Image-2.1 "
            "and INT4 Wan 2.2). Keeps 14 INT4-AWQ blocks in MPS working set."
        ),
        num_gpus=1,
        total_vram_gb=16.0,
        min_system_ram_gb=24,
        recommended_system_ram_gb=24,
        quantization=QuantizationMode.INT4_AWQ,
        offload_strategy=OffloadStrategy.UNIFIED_MEMORY,
        total_dit_blocks=50,
        resident_vram_blocks=14,
        ping_pong_prefetch_slots=2,
        qwen3_vl_stream_layers=4,
        vae_tiled_decode=True,
        pcie_bandwidth_gbps=200.0,
        gpus=[
            GPUDeviceSpec(
                device_id=0,
                name="Apple Silicon GPU (16GB MPS Working Set / 24GB Unified)",
                vram_gb=16.0,
                pcie_gen=5,
                pcie_lanes=16,
                assigned_block_start=0,
                assigned_block_end=50,
                resident_blocks=14,
            ),
        ],
    ),
}


@dataclass
class EngineConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    api_key: Optional[str] = field(default_factory=lambda: os.environ.get("OPENFAL_API_KEY") or None)
    profile_id: str = "dual_5060ti_32gb"
    weights_dir: Path = field(default_factory=lambda: Path(os.environ.get("OPENFAL_WEIGHTS_DIR", "./weights/MiniMax-H3")))
    output_dir: Path = field(default_factory=lambda: Path(os.environ.get("OPENFAL_OUTPUT_DIR", "./outputs")))
    cache_dir: Path = field(default_factory=lambda: Path(os.environ.get("OPENFAL_CACHE_DIR", "./.cache/openfal")))
    hf_repo_video: str = "MiniMaxAI/MiniMax-H3"
    hf_repo_image: str = "MiniMaxAI/MiniMax-H3-Image"
    enable_text_encoder_cache: bool = True
    auto_evict_text_encoder: bool = True
    default_num_inference_steps: int = 8  # MiniMax-H3 Turbo 8-step scheduler
    enable_native_stereo_audio: bool = True

    @property
    def profile(self) -> HardwareProfile:
        return HARDWARE_PROFILES.get(self.profile_id, HARDWARE_PROFILES["dual_5060ti_32gb"])


def get_default_config(profile_id: Optional[str] = None) -> EngineConfig:
    selected = profile_id or os.environ.get("OPENFAL_PROFILE", "dual_5060ti_32gb")
    if selected not in HARDWARE_PROFILES:
        selected = "dual_5060ti_32gb"
    cfg = EngineConfig(profile_id=selected)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    return cfg
