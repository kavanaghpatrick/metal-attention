#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Experiment B: Multi-row matvec (llama.cpp style)
//
// 256 threads per threadgroup (8 simdgroups).
// Each threadgroup computes N_ROWS output rows.
// Each simdgroup handles the dot product for one block range.
// Uses threadgroup memory for partial sums.
//
// Key insight: More threads per group = better occupancy + amortized dispatch.
// ---------------------------------------------------------------------------

struct BlockQ4_0 {
    half d;
    uchar qs[16];
};

// Each threadgroup processes N_ROWS output rows
// With 256 threads and 32 threads per simdgroup, we have 8 simdgroups
// Each simdgroup accumulates partial sums across blocks for one row
constant constexpr uint N_ROWS = 4;

kernel void matvec_q4_0_v3_multirow(
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
    const uint first_row = tgid * N_ROWS;
    const uint n_blocks = in_dim / 32;

    // Each of 8 simdgroups handles one row (we process min(N_ROWS, remaining))
    // With N_ROWS=4, simdgroups 0-3 handle rows, 4-7 are idle
    // Alternative: all 8 simdgroups cooperate on each row (more parallelism per row)

    // Strategy: all 8 simdgroups cooperate on all N_ROWS rows
    // Each thread processes blocks strided by 256 (total threads)
    // Then simdgroup reduction, then cross-simdgroup reduction via threadgroup memory

    threadgroup float partial_sums[8 * N_ROWS]; // 8 simdgroups × N_ROWS

    for (uint r = 0; r < N_ROWS; r++) {
        const uint row = first_row + r;
        if (row >= out_dim) {
            if (simd_lane == 0) {
                partial_sums[simd_id * N_ROWS + r] = 0.0f;
            }
            continue;
        }

        const uint row_offset = row * n_blocks;
        float sum = 0.0f;

        // All 256 threads stride over blocks
        for (uint b = tid; b < n_blocks; b += 256) {
            const BlockQ4_0 block = weight[row_offset + b];
            const float scale = float(block.d);
            const uint base = b * 32;

            // Vectorized inner loop (same as v2)
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

        // Phase 1: simdgroup reduction (32 threads -> 1 value)
        sum = simd_sum(sum);

        // Phase 2: write simdgroup partial to threadgroup memory
        if (simd_lane == 0) {
            partial_sums[simd_id * N_ROWS + r] = sum;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: simdgroup 0 reduces across all 8 simdgroups for each row
    if (simd_id == 0) {
        for (uint r = 0; r < N_ROWS; r++) {
            const uint row = first_row + r;
            if (row >= out_dim) continue;

            float total = 0.0f;
            if (simd_lane < 8) {
                total = partial_sums[simd_lane * N_ROWS + r];
            }
            total = simd_sum(total);

            if (simd_lane == 0) {
                output[row] = total;
            }
        }
    }
}
