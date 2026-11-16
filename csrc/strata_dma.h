// SPDX-License-Identifier: MIT
// Copyright (c) 2026 openfal Contributors
//
// Strata-DiT 3-Tier Memory Hierarchy & Double-Buffered DMA Scheduler for MiniMax-H3 (33B DiT)
// Inspired by Niko1221/Strata:
//   - Tier 1 (The Counter / VRAM): Active latents, pinned resident DiT blocks [0..K-1],
//     and 2 ping-pong async DMA prefetch slots (Slot A / Slot B) per GPU.
//   - Tier 2 (The Pantry / Pinned Host RAM): Page-locked (cudaHostAlloc) storage of all 50
//     MiniMax-H3 DiT blocks (hidden_size=5376, num_heads=56, head_dim=128).
//   - Tier 3 (The Storeroom / NVMe SSD): Memory-mapped inactive partitions (FL2VA, Ref2VA,
//     H3-Image) and LRU Qwen3-VL Layer-50 hidden-state cache.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace openfal {
namespace strata {

enum class MemoryTier : uint8_t {
    VRAM_RESIDENT = 0,     // Pin-locked in GPU VRAM ("The Counter")
    VRAM_PREFETCH_A = 1,   // Ping-pong DMA buffer Slot A in VRAM
    VRAM_PREFETCH_B = 2,   // Ping-pong DMA buffer Slot B in VRAM
    HOST_PINNED_RAM = 3,   // Page-locked DDR5 Host RAM ("The Pantry")
    NVME_MMAP = 4          // NVMe SSD mmap ("The Storeroom")
};

enum class QuantFormat : uint8_t {
    BF16 = 0,       // 2 bytes / param (~1.32 GB per DiT block)
    FP8_E4M3 = 1,   // 1 byte  / param (~660 MB per DiT block) - Blackwell 5060Ti/5090 native
    INT4_AWQ = 2,   // 0.5 byte/ param (~340 MB per DiT block) - Ultra-low VRAM mode
    NF4 = 3         // 0.5 byte/ param
};

struct BlockDescriptor {
    int32_t block_id;             // 0..49 for MiniMax-H3 33B DiT
    int32_t assigned_gpu;         // 0 or 1 (for 2x RTX 5060 Ti pipeline sharding)
    MemoryTier current_tier;
    QuantFormat quant_format;
    size_t byte_size;
    void* host_pinned_ptr;        // Tier 2 pointer (cudaHostAllocMounted)
    void* device_ptr;             // Tier 1 pointer when resident or prefetched
};

struct StrataEngineStats {
    int32_t num_gpus;
    int32_t total_blocks;
    int32_t resident_vram_blocks;
    int32_t host_pinned_blocks;
    size_t vram_counter_bytes;
    size_t host_pantry_bytes;
    size_t nvme_storeroom_bytes;
    double avg_h2d_bandwidth_gbps;
    double compute_overlap_ratio;
    bool sleeping;
};

class StrataDMAScheduler {
public:
    StrataDMAScheduler(
        int32_t num_blocks,
        int32_t resident_blocks,
        int32_t num_gpus,
        QuantFormat quant,
        size_t block_bytes
    );
    ~StrataDMAScheduler();

    // Prefetch block `next_block_id` on `h2d_stream` while `cur_block_id` runs on `compute_stream`
    void prefetch_block_async(int32_t next_block_id, int32_t slot_idx);

    // Wait for `block_id` DMA completion on `compute_stream` before launching DiT attention/FFN
    void wait_block_ready(int32_t block_id);

    // Evict all VRAM Counter blocks back to Pinned Host RAM (0 MB model VRAM while idle)
    void sleep_to_host_ram();

    // Restore pinned resident blocks from Host RAM to VRAM Counter
    void wake_to_vram();

    StrataEngineStats get_stats() const;
    const std::vector<BlockDescriptor>& blocks() const { return blocks_; }

private:
    int32_t num_blocks_;
    int32_t resident_blocks_;
    int32_t num_gpus_;
    QuantFormat quant_;
    size_t block_bytes_;
    bool sleeping_;
    std::vector<BlockDescriptor> blocks_;
};

} // namespace strata
} // namespace openfal
