#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Multi-token batched Q4_0 matvec kernels.
//
// Process batch_size tokens against the same weight matrix in a single dispatch.
// Weights get cached in SLC after the first token, so subsequent tokens read
// from cache — achieving near-linear per-token speedup (3.9x @ batch=4,
// 7.1x @ batch=8 measured on M4).
//
// 256 threads = 8 simdgroups, each handles 1 output row.
// Inner loop processes all batch_size tokens for that row.
//
// Buffer layout:
//   0: weight    Q4_0 blocks [out_dim, in_dim/32]
//   1: input     [batch_size, in_dim] float
//   2: output    [batch_size, out_dim] float
//   3: out_dim   scalar uint
//   4: in_dim    scalar uint
//   5: batch_size scalar uint
// ---------------------------------------------------------------------------

struct BlockQ4_0 {
    half d;
    uchar qs[16];
};

// Variant 1: output = W * input (overwrite)
kernel void multi_token_matvec_q4_0(
    device const BlockQ4_0* weight  [[buffer(0)]],
    device const float*     input   [[buffer(1)]],
    device float*           output  [[buffer(2)]],
    constant uint&          out_dim    [[buffer(3)]],
    constant uint&          in_dim     [[buffer(4)]],
    constant uint&          batch_size [[buffer(5)]],
    uint tgid      [[threadgroup_position_in_grid]],
    uint simd_id   [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    const uint ROWS_PER_TG = 8;
    const uint row = tgid * ROWS_PER_TG + simd_id;
    if (row >= out_dim) return;

    const uint n_blocks = in_dim / 32;
    const uint row_offset = row * n_blocks;

    for (uint tok = 0; tok < batch_size; tok++) {
        device const float* tok_input = input + tok * in_dim;
        float sum = 0.0f;

        for (uint b = simd_lane; b < n_blocks; b += 32) {
            const BlockQ4_0 block = weight[row_offset + b];
            const float scale = float(block.d);
            const uint base = b * 32;

            for (uint i = 0; i < 16; i += 4) {
                float4 in_lo = float4(tok_input[base + i],
                                      tok_input[base + i + 1],
                                      tok_input[base + i + 2],
                                      tok_input[base + i + 3]);
                float4 in_hi = float4(tok_input[base + i + 16],
                                      tok_input[base + i + 17],
                                      tok_input[base + i + 18],
                                      tok_input[base + i + 19]);

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
            output[tok * out_dim + row] = sum;
        }
    }
}

// Variant 2: output += W * input (accumulate for residual connections)
kernel void multi_token_matvec_q4_0_accumulate(
    device const BlockQ4_0* weight  [[buffer(0)]],
    device const float*     input   [[buffer(1)]],
    device float*           output  [[buffer(2)]],
    constant uint&          out_dim    [[buffer(3)]],
    constant uint&          in_dim     [[buffer(4)]],
    constant uint&          batch_size [[buffer(5)]],
    uint tgid      [[threadgroup_position_in_grid]],
    uint simd_id   [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    const uint ROWS_PER_TG = 8;
    const uint row = tgid * ROWS_PER_TG + simd_id;
    if (row >= out_dim) return;

    const uint n_blocks = in_dim / 32;
    const uint row_offset = row * n_blocks;

    for (uint tok = 0; tok < batch_size; tok++) {
        device const float* tok_input = input + tok * in_dim;
        float sum = 0.0f;

        for (uint b = simd_lane; b < n_blocks; b += 32) {
            const BlockQ4_0 block = weight[row_offset + b];
            const float scale = float(block.d);
            const uint base = b * 32;

            for (uint i = 0; i < 16; i += 4) {
                float4 in_lo = float4(tok_input[base + i],
                                      tok_input[base + i + 1],
                                      tok_input[base + i + 2],
                                      tok_input[base + i + 3]);
                float4 in_hi = float4(tok_input[base + i + 16],
                                      tok_input[base + i + 17],
                                      tok_input[base + i + 18],
                                      tok_input[base + i + 19]);

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
            output[tok * out_dim + row] += sum;  // ACCUMULATE
        }
    }
}
