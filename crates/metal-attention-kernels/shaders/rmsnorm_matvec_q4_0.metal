#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Fused RMSNorm + Q4_0 Dequantize + Matrix-Vector Multiply kernel
//
// Computes: output[row] = dot(dequant(weight_row), rmsnorm(input))
// where rmsnorm(input)[i] = (input[i] / rms) * norm_weight[i]
// and   rms = sqrt(mean(input^2) + eps)
//
// Eliminates the intermediate normalized buffer by computing RMS inline
// and applying normalization during the dot product accumulation.
//
// Q4_0 block layout: 2 bytes fp16 scale (half d) + 16 bytes packed nibbles
//   - Low nibbles (byte & 0x0F) - 8  -> elements [0..15]
//   - High nibbles (byte >> 4) - 8   -> elements [16..31]
//
// Threadgroup: 32 threads (1 simdgroup) per output row.
// Grid: (out_dim, 1, 1) threadgroups.
// Each threadgroup computes one output element via cooperative reduction.
//
// Buffer bindings:
//   0: input          [in_dim]        F32 input vector
//   1: norm_weight    [in_dim]        F32 RMSNorm weight
//   2: weight         [out_dim, n_blocks_per_row] Q4_0 blocks
//   3: output         [out_dim]       F32 output vector
//   4: out_dim        scalar uint
//   5: in_dim         scalar uint
//   6: eps            scalar float
// ---------------------------------------------------------------------------

struct BlockQ4_0 {
    half d;          // scale factor (fp16)
    uchar qs[16];   // 32 x 4-bit quantized values packed into 16 bytes
};

kernel void rmsnorm_matvec_q4_0(
    device const float*     input       [[buffer(0)]],   // [in_dim]
    device const float*     norm_weight [[buffer(1)]],   // [in_dim]
    device const BlockQ4_0* weight      [[buffer(2)]],   // [out_dim, n_blocks_per_row]
    device float*           output      [[buffer(3)]],   // [out_dim]
    constant uint&          out_dim     [[buffer(4)]],
    constant uint&          in_dim      [[buffer(5)]],
    constant float&         eps         [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]
) {
    const uint row = tgid;
    if (row >= out_dim) return;

    // -----------------------------------------------------------------------
    // Phase 1: Cooperative RMS computation across 32 threads
    // Each thread accumulates sum-of-squares over strided elements
    // -----------------------------------------------------------------------
    float partial_ss = 0.0f;
    for (uint i = tid; i < in_dim; i += 32) {
        float val = input[i];
        partial_ss += val * val;
    }

    // simd_sum reduces across all 32 threads in the simdgroup
    float total_ss = simd_sum(partial_ss);
    float inv_rms = 1.0f / sqrt(total_ss / float(in_dim) + eps);

    // -----------------------------------------------------------------------
    // Phase 2: Fused dequant(weight_row) * normalized_input dot product
    // normalized_input[i] = input[i] * inv_rms * norm_weight[i]
    // -----------------------------------------------------------------------
    const uint n_blocks = in_dim / 32;
    const uint row_offset = row * n_blocks;

    float sum = 0.0f;

    for (uint b = tid; b < n_blocks; b += 32) {
        const BlockQ4_0 block = weight[row_offset + b];
        const float scale = float(block.d);
        const uint base = b * 32;

        // Process 16 bytes -> 32 dequantized values
        for (uint i = 0; i < 16; i++) {
            uchar byte_val = block.qs[i];

            // Low nibble -> elements [0..15]
            float dq_lo = float(int(byte_val & 0x0F) - 8) * scale;
            float norm_lo = input[base + i] * inv_rms * norm_weight[base + i];
            sum += dq_lo * norm_lo;

            // High nibble -> elements [16..31]
            float dq_hi = float(int((byte_val >> 4) & 0x0F) - 8) * scale;
            float norm_hi = input[base + i + 16] * inv_rms * norm_weight[base + i + 16];
            sum += dq_hi * norm_hi;
        }
    }

    // 32-wide simdgroup reduction for the dot product
    sum = simd_sum(sum);

    // Thread 0 writes the final result for this output row
    if (tid == 0) {
        output[row] = sum;
    }
}
