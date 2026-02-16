#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Dual-buffer RoPE: apply rotary position encoding to Q and K in one dispatch.
//
// Q has num_q_heads heads, K has num_k_heads heads, both with same head_dim.
// Total threads = (num_q_heads + num_k_heads) * head_dim / 2.
// Each thread computes the same angle (position-dependent), applies rotation.
//
// Buffer layout:
//   0: q_buf       [num_q_heads * head_dim]  - modified in-place
//   1: k_buf       [num_k_heads * head_dim]  - modified in-place
//   2: num_q_heads scalar uint
//   3: num_k_heads scalar uint
//   4: head_dim    scalar uint
//   5: position    scalar uint
//   6: theta       scalar float
// ---------------------------------------------------------------------------

kernel void rope_apply_dual(
    device float* q_buf      [[buffer(0)]],
    device float* k_buf      [[buffer(1)]],
    constant uint& num_q_heads [[buffer(2)]],
    constant uint& num_k_heads [[buffer(3)]],
    constant uint& head_dim  [[buffer(4)]],
    constant uint& position  [[buffer(5)]],
    constant float& theta    [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    uint half_dim = head_dim / 2;
    uint total_q_pairs = num_q_heads * half_dim;

    // Determine if this thread handles Q or K
    device float* buf;
    uint local_tid;

    if (tid < total_q_pairs) {
        buf = q_buf;
        local_tid = tid;
    } else {
        buf = k_buf;
        local_tid = tid - total_q_pairs;
    }

    uint head = local_tid / half_dim;
    uint pair = local_tid % half_dim;

    float angle = float(position) / pow(theta, 2.0 * float(pair) / float(head_dim));
    float cos_a = cos(angle);
    float sin_a = sin(angle);

    uint idx0 = head * head_dim + 2 * pair;
    uint idx1 = idx0 + 1;

    float v0 = buf[idx0];
    float v1 = buf[idx1];
    buf[idx0] = v0 * cos_a - v1 * sin_a;
    buf[idx1] = v0 * sin_a + v1 * cos_a;
}
