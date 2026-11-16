// SPDX-License-Identifier: MIT
// Copyright (c) 2026 openfal Contributors
//
// CUDA implementation of Strata-DiT double-buffered PCIe 5.0 H2D block streaming,
// dual-GPU P2P latent relay (for 2x RTX 5060 Ti), and fused FP8_E4M3 / INT4 dequantization.

#include "strata_dma.h"

#ifndef OPENFAL_CPU_ONLY
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace openfal {
namespace strata {

// Fused FP8 (E4M3) + per-channel scale -> BF16 dequantization kernel executed in L2 cache
// immediately after async H2D transfer of a streamed MiniMax-H3 DiT block.
__global__ void dequant_fp8_e4m3_to_bf16_kernel(
    const uint8_t* __restrict__ src_fp8,
    const float* __restrict__ channel_scales,
    __nv_bfloat16* __restrict__ dst_bf16,
    int64_t num_elements,
    int32_t hidden_dim
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    uint8_t raw = src_fp8[idx];
    // Decode IEEE/OCP FP8 E4M3 bits: 1 sign, 4 exp (bias 7), 3 mantissa
    float sign = (raw & 0x80) ? -1.0f : 1.0f;
    int exp = (raw >> 3) & 0x0F;
    int man = raw & 0x07;
    float val = 0.0f;
    if (exp == 0) {
        val = sign * ldexpf(static_cast<float>(man) / 8.0f, -6);
    } else {
        val = sign * ldexpf(1.0f + static_cast<float>(man) / 8.0f, exp - 7);
    }
    int32_t col = static_cast<int32_t>(idx % hidden_dim);
    dst_bf16[idx] = __float2bfloat16(val * channel_scales[col]);
}

// Fused INT4-AWQ (packed 2x4-bit per byte) -> BF16 dequantization kernel for 24GB dual-GPU setups
__global__ void dequant_int4_awq_to_bf16_kernel(
    const uint8_t* __restrict__ packed_int4,
    const __nv_bfloat16* __restrict__ group_scales,
    const uint8_t* __restrict__ group_zeros,
    __nv_bfloat16* __restrict__ dst_bf16,
    int64_t num_packed_bytes,
    int32_t group_size
) {
    int64_t byte_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (byte_idx >= num_packed_bytes) return;

    uint8_t packed = packed_int4[byte_idx];
    int64_t elem0 = byte_idx * 2;
    int64_t group_idx = elem0 / group_size;

    float scale = __bfloat162float(group_scales[group_idx]);
    uint8_t zp_byte = group_zeros[group_idx / 2];
    int zp = (group_idx & 1) ? ((zp_byte >> 4) & 0x0F) : (zp_byte & 0x0F);

    int q0 = (packed & 0x0F) - zp;
    int q1 = ((packed >> 4) & 0x0F) - zp;

    dst_bf16[elem0]     = __float2bfloat16(static_cast<float>(q0) * scale);
    dst_bf16[elem0 + 1] = __float2bfloat16(static_cast<float>(q1) * scale);
}

void launch_fp8_to_bf16_dequant(
    const uint8_t* src_fp8,
    const float* scales,
    void* dst_bf16,
    int64_t num_elements,
    int32_t hidden_dim,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = static_cast<int>((num_elements + threads - 1) / threads);
    dequant_fp8_e4m3_to_bf16_kernel<<<blocks, threads, 0, stream>>>(
        src_fp8, scales, reinterpret_cast<__nv_bfloat16*>(dst_bf16), num_elements, hidden_dim
    );
}

} // namespace strata
} // namespace openfal
#endif
