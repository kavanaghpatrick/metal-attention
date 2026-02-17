#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Batched Q4_0 matvec: process up to 3 projections in a single dispatch.
//
// Concatenates output rows: proj_a rows, then proj_b rows, then proj_c rows.
// Each simdgroup handles 1 output row. 256 threads = 8 simdgroups = 8 rows/TG.
//
// The kernel determines which projection a row belongs to based on
// the cumulative dimension boundaries (dim_a, dim_a+dim_b, dim_a+dim_b+dim_c).
//
// Buffer layout:
//   0: weight_a     Q4_0 blocks [dim_a, in_dim/32]
//   1: weight_b     Q4_0 blocks [dim_b, in_dim/32]  (or nullptr if dim_b=0)
//   2: weight_c     Q4_0 blocks [dim_c, in_dim/32]  (or nullptr if dim_c=0)
//   3: input        [in_dim] float
//   4: output_a     [dim_a] float
//   5: output_b     [dim_b] float (or nullptr)
//   6: output_c     [dim_c] float (or nullptr)
//   7: dims         uint4(dim_a, dim_b, dim_c, in_dim)
// ---------------------------------------------------------------------------

struct BlockQ4_0 {
    half d;
    uchar qs[16];
};

kernel void matvec_q4_0_batched(
    device const BlockQ4_0* weight_a  [[buffer(0)]],
    device const BlockQ4_0* weight_b  [[buffer(1)]],
    device const BlockQ4_0* weight_c  [[buffer(2)]],
    device const float*     input     [[buffer(3)]],
    device float*           output_a  [[buffer(4)]],
    device float*           output_b  [[buffer(5)]],
    device float*           output_c  [[buffer(6)]],
    constant uint4&         dims      [[buffer(7)]],  // (dim_a, dim_b, dim_c, in_dim)
    uint tgid      [[threadgroup_position_in_grid]],
    uint simd_id   [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    const uint ROWS_PER_TG = 8;
    const uint global_row = tgid * ROWS_PER_TG + simd_id;

    const uint dim_a = dims.x;
    const uint dim_b = dims.y;
    const uint dim_c = dims.z;
    const uint in_dim = dims.w;
    const uint total_rows = dim_a + dim_b + dim_c;

    if (global_row >= total_rows) return;

    // Determine which projection this row belongs to
    device const BlockQ4_0* weight;
    device float* output;
    uint local_row;

    if (global_row < dim_a) {
        weight = weight_a;
        output = output_a;
        local_row = global_row;
    } else if (global_row < dim_a + dim_b) {
        weight = weight_b;
        output = output_b;
        local_row = global_row - dim_a;
    } else {
        weight = weight_c;
        output = output_c;
        local_row = global_row - dim_a - dim_b;
    }

    const uint n_blocks = in_dim / 32;
    const uint row_offset = local_row * n_blocks;

    float sum = 0.0f;

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
        output[local_row] = sum;
    }
}
