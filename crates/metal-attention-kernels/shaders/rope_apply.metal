#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// RoPE (Rotary Position Embeddings) single-token decode kernel
//
// Applies rotary position encoding in-place on a flat qk buffer
// of shape [num_heads * head_dim]. Designed for single-token decode
// where position is a scalar (not a sequence dimension).
//
// Each thread handles one (cos, sin) pair for one head.
// Grid dispatch: (num_heads * head_dim / 2), threadgroup: (32)
//
// Uses interleaved pair layout matching the project convention:
//   idx0 = head * head_dim + 2 * pair
//   idx1 = idx0 + 1
//
// RoPE rotation:
//   angle = position / (theta ^ (2 * pair / head_dim))
//   [v0', v1'] = [v0 * cos(angle) - v1 * sin(angle),
//                  v0 * sin(angle) + v1 * cos(angle)]
// ---------------------------------------------------------------------------

kernel void rope_apply(
    device float* qk [[buffer(0)]],              // [num_heads * head_dim] - modified in-place
    constant uint& num_heads [[buffer(1)]],
    constant uint& head_dim [[buffer(2)]],
    constant uint& position [[buffer(3)]],
    constant float& theta [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint half_dim = head_dim / 2;
    uint head = tid / half_dim;
    uint pair = tid % half_dim;

    if (head >= num_heads) return;

    float angle = float(position) / pow(theta, 2.0 * float(pair) / float(head_dim));
    float cos_a = cos(angle);
    float sin_a = sin(angle);

    uint idx0 = head * head_dim + 2 * pair;
    uint idx1 = idx0 + 1;

    float v0 = qk[idx0];
    float v1 = qk[idx1];
    qk[idx0] = v0 * cos_a - v1 * sin_a;
    qk[idx1] = v0 * sin_a + v1 * cos_a;
}
