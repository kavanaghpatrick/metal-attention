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
