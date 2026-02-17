#include <metal_stdlib>
#include "types.h"
using namespace metal;

// ---------------------------------------------------------------------------
// RMSNorm kernel
//
// Computes: output[i] = (input[i] / rms) * weight[i]
// where rms = sqrt(mean(input^2) + eps)
//
// Grid dispatch: (num_tokens, 1, 1)
// Each thread processes one element, threadgroup reduces for RMS computation.
// ---------------------------------------------------------------------------

kernel void rmsnorm(
    device const float* input   [[buffer(0)]],   // [num_tokens, hidden_dim]
    device const float* weight  [[buffer(1)]],   // [hidden_dim]
    device float*       output  [[buffer(2)]],   // [num_tokens, hidden_dim]
    constant LayerParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint hidden_dim = params.hidden_dim;
    const uint token = tid / hidden_dim;
    const uint dim = tid % hidden_dim;
    const uint offset = token * hidden_dim;

    // Compute sum of squares for this token (simplified: each thread does full reduction)
    float ss = 0.0f;
    for (uint i = 0; i < hidden_dim; i++) {
        float val = input[offset + i];
        ss += val * val;
    }
    float rms = sqrt(ss / float(hidden_dim) + params.rms_norm_eps);

    output[offset + dim] = (input[offset + dim] / rms) * weight[dim];
}

// ---------------------------------------------------------------------------
// Optimized RMSNorm kernel with simdgroup cooperative reduction
//
// Computes: output[i] = (input[i] / rms) * weight[i]
// where rms = sqrt(mean(input^2) + eps)
//
// Uses 32 threads (1 simdgroup) with simd_sum for cooperative reduction.
// Each thread processes hidden_dim/32 elements in a strided loop.
//
// Dispatch: grid=(1,1,1), threadgroup=(32,1,1)
// Buffer bindings: input(0), weight(1), output(2), hidden_dim(3), eps(4)
// ---------------------------------------------------------------------------

kernel void rmsnorm_optimized(
    device const float* input      [[buffer(0)]],   // [hidden_dim]
    device const float* weight     [[buffer(1)]],   // [hidden_dim]
    device float*       output     [[buffer(2)]],   // [hidden_dim]
    constant uint&      hidden_dim [[buffer(3)]],
    constant float&     eps        [[buffer(4)]],
    uint tid [[thread_index_in_simdgroup]]
) {
    // Phase 1: Each thread computes partial sum-of-squares over stride
    float partial_ss = 0.0f;
    for (uint i = tid; i < hidden_dim; i += 32) {
        float val = input[i];
        partial_ss += val * val;
    }

    // Phase 2: simd_sum for cooperative reduction across 32 threads
    float total_ss = simd_sum(partial_ss);

    // Phase 3: Compute RMS (broadcast to all threads via simd_sum result)
    float rms = sqrt(total_ss / float(hidden_dim) + eps);

    // Phase 4: Each thread normalizes its stride
    for (uint i = tid; i < hidden_dim; i += 32) {
        output[i] = (input[i] / rms) * weight[i];
    }
}
