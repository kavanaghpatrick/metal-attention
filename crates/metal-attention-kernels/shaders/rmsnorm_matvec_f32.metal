#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Fused RMSNorm + F32 Matrix-Vector Multiply kernel
//
// Computes: output[row] = dot(weight_row, rmsnorm(input))
// where rmsnorm(input)[i] = (input[i] / rms) * norm_weight[i]
// and   rms = sqrt(mean(input^2) + eps)
//
// Eliminates the intermediate normalized buffer by computing RMS inline
// and applying normalization during the dot product accumulation.
//
// For lm_head with tied embeddings (F32 format, not Q4_0).
// Same structure as Q4_0 variant but with direct float weight access
// instead of block dequantization.
//
// Threadgroup: 32 threads (1 simdgroup) per output row.
// Grid: (out_dim, 1, 1) threadgroups.
// Each threadgroup computes one output element via cooperative reduction.
//
// Buffer bindings:
//   0: input          [in_dim]            F32 input vector
//   1: norm_weight    [in_dim]            F32 RMSNorm weight
//   2: weight         [out_dim * in_dim]  F32 weight matrix (row-major)
//   3: output         [out_dim]           F32 output vector
//   4: out_dim        scalar uint
//   5: in_dim         scalar uint
//   6: eps            scalar float
// ---------------------------------------------------------------------------

kernel void rmsnorm_matvec_f32(
    device const float* input       [[buffer(0)]],   // [in_dim]
    device const float* norm_weight [[buffer(1)]],   // [in_dim]
    device const float* weight      [[buffer(2)]],   // [out_dim * in_dim]
    device float*       output      [[buffer(3)]],   // [out_dim]
    constant uint&      out_dim     [[buffer(4)]],
    constant uint&      in_dim      [[buffer(5)]],
    constant float&     eps         [[buffer(6)]],
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
    // Phase 2: Fused weight * normalized_input dot product
    // normalized_input[i] = input[i] * inv_rms * norm_weight[i]
    // No dequantization needed -- direct F32 weight access.
    // -----------------------------------------------------------------------
    const uint row_offset = row * in_dim;

    float sum = 0.0f;

    for (uint i = tid; i < in_dim; i += 32) {
        float w = weight[row_offset + i];
        float norm_val = input[i] * inv_rms * norm_weight[i];
        sum += w * norm_val;
    }

    // 32-wide simdgroup reduction for the dot product
    sum = simd_sum(sum);

    // Thread 0 writes the final result for this output row
    if (tid == 0) {
        output[row] = sum;
    }
}
