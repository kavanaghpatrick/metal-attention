#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Experiment A: Vectorized Q4_0 matvec with float4 dot products
//
// Same 32 threads/group (1 simdgroup) as baseline, but:
// - Process 2 blocks per iteration (64 elements)
// - Use float4 accumulation for better ILP
// - Vectorized nibble extraction
//
// Minimal change from baseline - just vectorize the inner loop.
// ---------------------------------------------------------------------------

struct BlockQ4_0 {
    half d;
    uchar qs[16];
};

kernel void matvec_q4_0_v2_vec4(
    device const BlockQ4_0* weight  [[buffer(0)]],
    device const float*     input   [[buffer(1)]],
    device float*           output  [[buffer(2)]],
    constant uint&          out_dim [[buffer(3)]],
    constant uint&          in_dim  [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]
) {
    const uint row = tgid;
    if (row >= out_dim) return;

    const uint n_blocks = in_dim / 32;
    const uint row_offset = row * n_blocks;

    float4 sum4 = float4(0.0f);

    for (uint b = tid; b < n_blocks; b += 32) {
        const BlockQ4_0 block = weight[row_offset + b];
        const float scale = float(block.d);
        const uint base = b * 32;

        // Process 4 bytes at a time (8 dequantized values)
        for (uint i = 0; i < 16; i += 4) {
            // Load 4 input pairs as float4
            float4 in_lo = float4(input[base + i],
                                  input[base + i + 1],
                                  input[base + i + 2],
                                  input[base + i + 3]);
            float4 in_hi = float4(input[base + i + 16],
                                  input[base + i + 17],
                                  input[base + i + 18],
                                  input[base + i + 19]);

            // Dequant 4 bytes -> 8 values
            uchar b0 = block.qs[i];
            uchar b1 = block.qs[i + 1];
            uchar b2 = block.qs[i + 2];
            uchar b3 = block.qs[i + 3];

            float4 lo_vals = float4(int(b0 & 0x0F) - 8,
                                    int(b1 & 0x0F) - 8,
                                    int(b2 & 0x0F) - 8,
                                    int(b3 & 0x0F) - 8) * scale;
            float4 hi_vals = float4(int(b0 >> 4) - 8,
                                    int(b1 >> 4) - 8,
                                    int(b2 >> 4) - 8,
                                    int(b3 >> 4) - 8) * scale;

            sum4 += lo_vals * in_lo + hi_vals * in_hi;
        }
    }

    float sum = sum4.x + sum4.y + sum4.z + sum4.w;
    sum = simd_sum(sum);

    if (tid == 0) {
        output[row] = sum;
    }
}
