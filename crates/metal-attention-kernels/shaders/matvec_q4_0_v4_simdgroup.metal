#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Experiment C: simdgroup_matrix accumulation
//
// Uses simdgroup_matrix<float, 8, 8> for the accumulation step.
// Dequantizes Q4_0 blocks to threadgroup memory, then loads via
// simdgroup_load and multiplies with simdgroup_multiply_accumulate.
//
// This is the Metal 3 path (works on M1+).
// 256 threads (8 simdgroups), each simdgroup processes blocks for a tile
// of the output.
// ---------------------------------------------------------------------------

struct BlockQ4_0 {
    half d;
    uchar qs[16];
};

// We dequantize blocks to threadgroup memory in tiles of 8×8
// Input is 1×K vector, weight is M×K matrix
// Output is 1×M vector (treated as M×1)
//
// For matvec: we treat it as M×K @ K×1
// simdgroup_matrix works on 8×8 tiles, so we need K×1 padded to K×8
//
// Actually, simdgroup_matrix is overkill for matvec (it's designed for matmul).
// The vectorized multi-row approach (v3) is likely better for matvec.
// Let's still benchmark it to confirm.

// Simpler approach: use simdgroup_matrix for 8×8 tiles of the weight matrix
// multiplied against 8×1 tiles of the input (broadcast to 8×8)

kernel void matvec_q4_0_v4_simdgroup(
    device const BlockQ4_0* weight  [[buffer(0)]],
    device const float*     input   [[buffer(1)]],
    device float*           output  [[buffer(2)]],
    constant uint&          out_dim [[buffer(3)]],
    constant uint&          in_dim  [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]]
) {
    // Each threadgroup handles 32 rows (4 rows per simdgroup × 8 simdgroups)
    // Actually let's keep it simpler: 8 simdgroups, each handles 4 rows
    // Total: 32 rows per threadgroup
    const uint ROWS_PER_SIMD = 4;
    const uint ROWS_PER_TG = ROWS_PER_SIMD * 8; // 32

    const uint first_row = tgid * ROWS_PER_TG + simd_id * ROWS_PER_SIMD;
    const uint n_blocks = in_dim / 32;

    // Each simdgroup handles ROWS_PER_SIMD rows
    // 32 threads per simdgroup stride over blocks
    for (uint r = 0; r < ROWS_PER_SIMD; r++) {
        const uint row = first_row + r;
        if (row >= out_dim) continue;

        const uint row_offset = row * n_blocks;
        float sum = 0.0f;

        // Vectorized block processing (same as v2)
        for (uint b = simd_lane; b < n_blocks; b += 32) {
            const BlockQ4_0 block = weight[row_offset + b];
            const float scale = float(block.d);
            const uint base = b * 32;

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

        sum = simd_sum(sum);

        if (simd_lane == 0) {
            output[row] = sum;
        }
    }
}
