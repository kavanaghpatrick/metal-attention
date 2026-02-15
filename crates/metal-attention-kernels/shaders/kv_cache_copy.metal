#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// KV cache copy kernel
//
// Copies kv_dim floats from scratch K/V buffers into the KV cache at the
// specified row offset (row_idx).
//
//   k_dst[row_idx * kv_dim + tid] = k_src[tid]
//   v_dst[row_idx * kv_dim + tid] = v_src[tid]
//
// Grid: (kv_dim, 1, 1) threads total.
// Threadgroup: (256, 1, 1).
// ---------------------------------------------------------------------------

kernel void kv_cache_copy(
    device const float* k_src    [[buffer(0)]],
    device const float* v_src    [[buffer(1)]],
    device float*       k_dst    [[buffer(2)]],
    device float*       v_dst    [[buffer(3)]],
    constant uint&      kv_dim   [[buffer(4)]],
    constant uint&      row_idx  [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= kv_dim) return;
    uint offset = row_idx * kv_dim + tid;
    k_dst[offset] = k_src[tid];
    v_dst[offset] = v_src[tid];
}
