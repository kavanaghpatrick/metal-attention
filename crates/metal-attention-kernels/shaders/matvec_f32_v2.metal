#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// v2: Multi-row vectorized F32 matvec
//
// Computes: output[row] = dot(weight_row, input)
// where weight is stored as dense F32 [out_dim, in_dim] row-major.
//
// 256 threads = 8 simdgroups, each simdgroup handles 1 output row.
// 8 rows per threadgroup => ceil(out_dim/8) threadgroups.
// Vectorized float4 inner loop for coalesced memory reads.
// ---------------------------------------------------------------------------

kernel void matvec_f32_v2(
    device const float* weight  [[buffer(0)]],
    device const float* input   [[buffer(1)]],
    device float*       output  [[buffer(2)]],
    constant uint&      out_dim [[buffer(3)]],
    constant uint&      in_dim  [[buffer(4)]],
    uint tgid      [[threadgroup_position_in_grid]],
    uint simd_id   [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    const uint ROWS_PER_TG = 8;
    const uint row = tgid * ROWS_PER_TG + simd_id;
    if (row >= out_dim) return;

    const uint row_offset = row * in_dim;

    // Vectorized float4 reads: process 4 elements per iteration
    const uint n_float4 = in_dim / 4;
    device const float4* weight_f4 = (device const float4*)(weight + row_offset);
    device const float4* input_f4 = (device const float4*)(input);

    float4 sum4 = float4(0.0f);

    for (uint i = simd_lane; i < n_float4; i += 32) {
        sum4 += weight_f4[i] * input_f4[i];
    }

    float sum = sum4.x + sum4.y + sum4.z + sum4.w;

    // Handle remainder elements (in_dim not divisible by 4)
    uint remainder_start = n_float4 * 4;
    for (uint i = remainder_start + simd_lane; i < in_dim; i += 32) {
        sum += weight[row_offset + i] * input[i];
    }

    // 32-wide simdgroup reduction
    sum = simd_sum(sum);

    if (simd_lane == 0) {
        output[row] = sum;
    }
}
