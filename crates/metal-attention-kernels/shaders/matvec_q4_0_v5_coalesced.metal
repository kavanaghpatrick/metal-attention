#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// v5: Multi-row vectorized Q4_0 matvec with coalesced access
//
// Computes: output[row] = dot(dequant(weight_row), input)
// where weight is stored in Q4_0 format (32 elements per 18-byte block).
//
// 256 threads = 8 simdgroups, each simdgroup handles 1 output row.
// 8 rows per threadgroup => ceil(out_dim/8) threadgroups.
// Vectorized float4 inner loop for better ILP and throughput.
//
// Key improvements over baseline (v1):
// - 8x fewer threadgroup dispatches (8 rows/TG vs 1 row/TG)
// - Vectorized float4 dot products (4 bytes processed per iteration)
// - Better GPU occupancy (256 threads vs 32)
// - Input vector reads amortized across rows via cache reuse
// ---------------------------------------------------------------------------

struct BlockQ4_0 {
    half d;          // scale factor (fp16)
    uchar qs[16];   // 32 x 4-bit quantized values packed into 16 bytes
};

kernel void matvec_q4_0_v5_coalesced(
    device const BlockQ4_0* weight  [[buffer(0)]],
    device const float*     input   [[buffer(1)]],
    device float*           output  [[buffer(2)]],
    constant uint&          out_dim [[buffer(3)]],
    constant uint&          in_dim  [[buffer(4)]],
    uint tgid      [[threadgroup_position_in_grid]],
    uint simd_id   [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    // Each threadgroup processes 8 rows (one per simdgroup)
    const uint ROWS_PER_TG = 8;
    const uint row = tgid * ROWS_PER_TG + simd_id;
    if (row >= out_dim) return;

    const uint n_blocks = in_dim / 32;
    const uint row_offset = row * n_blocks;

    float sum = 0.0f;

    // 32 threads per simdgroup stride over blocks for this row
    for (uint b = simd_lane; b < n_blocks; b += 32) {
        const BlockQ4_0 block = weight[row_offset + b];
        const float scale = float(block.d);
        const uint base = b * 32;

        // Vectorized: process 4 bytes (8 dequantized values) per iteration
        for (uint i = 0; i < 16; i += 4) {
            float4 in_lo = float4(input[base + i],
                                  input[base + i + 1],
                                  input[base + i + 2],
                                  input[base + i + 3]);
            float4 in_hi = float4(input[base + i + 16],
                                  input[base + i + 17],
                                  input[base + i + 18],
                                  input[base + i + 19]);

            float4 lo_vals = float4(int(block.qs[i]   & 0x0F) - 8,
                                    int(block.qs[i+1] & 0x0F) - 8,
                                    int(block.qs[i+2] & 0x0F) - 8,
                                    int(block.qs[i+3] & 0x0F) - 8) * scale;
            float4 hi_vals = float4(int(block.qs[i]   >> 4) - 8,
                                    int(block.qs[i+1] >> 4) - 8,
                                    int(block.qs[i+2] >> 4) - 8,
                                    int(block.qs[i+3] >> 4) - 8) * scale;

            sum += dot(lo_vals, in_lo) + dot(hi_vals, in_hi);
        }
    }

    // 32-wide simdgroup reduction
    sum = simd_sum(sum);

    // Lane 0 of each simdgroup writes its row's result
    if (simd_lane == 0) {
        output[row] = sum;
    }
}
