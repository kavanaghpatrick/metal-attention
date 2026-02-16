#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Q8_0 matvec: multi-row vectorized matrix-vector multiply.
//
// Q8_0 block layout: 2 bytes fp16 scale + 32 signed int8 values = 34 bytes.
// Dequant: value = qs[i] * d
//
// 256 threads = 8 simdgroups, each handles 1 output row (8 rows/TG).
// Vectorized 4-wide inner loop reads 4 int8 values + 4 float inputs.
//
// Buffer layout:
//   0: weight    Q8_0 blocks [out_dim, in_dim/32]
//   1: input     [in_dim] float
//   2: output    [out_dim] float
//   3: out_dim   scalar uint
//   4: in_dim    scalar uint
// ---------------------------------------------------------------------------

struct BlockQ8_0 {
    half d;          // scale factor (fp16)
    char qs[32];     // 32 signed int8 values
};

kernel void matvec_q8_0(
    device const BlockQ8_0* weight  [[buffer(0)]],
    device const float*     input   [[buffer(1)]],
    device float*           output  [[buffer(2)]],
    constant uint&          out_dim [[buffer(3)]],
    constant uint&          in_dim  [[buffer(4)]],
    uint tgid      [[threadgroup_position_in_grid]],
    uint simd_id   [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    const uint ROWS_PER_TG = 8;
    const uint row = tgid * ROWS_PER_TG + simd_id;
    if (row >= out_dim) return;

    const uint n_blocks = in_dim / 32;
    const uint row_offset = row * n_blocks;

    float sum = 0.0f;

    for (uint b = simd_lane; b < n_blocks; b += 32) {
        const BlockQ8_0 block = weight[row_offset + b];
        const float scale = float(block.d);
        const uint base = b * 32;

        // Process 32 elements in groups of 4 for vectorized float4 dot products
        for (uint i = 0; i < 32; i += 4) {
            float4 in_vals = float4(input[base + i],
                                    input[base + i + 1],
                                    input[base + i + 2],
                                    input[base + i + 3]);

            float4 w_vals = float4(float(block.qs[i]),
                                   float(block.qs[i + 1]),
                                   float(block.qs[i + 2]),
                                   float(block.qs[i + 3])) * scale;

            sum += dot(w_vals, in_vals);
        }
    }

    sum = simd_sum(sum);

    if (simd_lane == 0) {
        output[row] = sum;
    }
}
