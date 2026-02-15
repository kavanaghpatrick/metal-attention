#include <metal_stdlib>
#include "types.h"
using namespace metal;

// ---------------------------------------------------------------------------
// SwiGLU FFN kernel (stub)
//
// Computes: output = (silu(gate) * up) -- element-wise
// where silu(x) = x * sigmoid(x)
//
// Grid dispatch: (num_tokens * intermediate_dim, 1, 1)
// Each thread computes one output element.
// ---------------------------------------------------------------------------

kernel void ffn_silu(
    device const float* input   [[buffer(0)]],   // [num_tokens, hidden_dim] (unused in stub)
    device const float* gate    [[buffer(1)]],   // [num_tokens, intermediate_dim]
    device const float* up      [[buffer(2)]],   // [num_tokens, intermediate_dim]
    device float*       output  [[buffer(3)]],   // [num_tokens, intermediate_dim]
    constant LayerParams& params [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint total = params.seq_len * params.intermediate_dim;
    if (tid >= total) return;

    float g = gate[tid];
    float silu_g = g / (1.0f + exp(-g));  // silu(x) = x * sigmoid(x)
    output[tid] = silu_g * up[tid];
}
