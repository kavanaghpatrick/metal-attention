#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Element-wise residual addition kernel
//
// Computes: output[i] = a[i] + b[i]
//
// Grid: (dim, 1, 1) threads total.
// Threadgroup: (256, 1, 1).
// ---------------------------------------------------------------------------

kernel void residual_add(
    device const float* a      [[buffer(0)]],
    device const float* b      [[buffer(1)]],
    device float*       output [[buffer(2)]],
    constant uint&      dim    [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;
    output[tid] = a[tid] + b[tid];
}

// ---------------------------------------------------------------------------
// In-place residual addition: a[i] += b[i]
//
// Eliminates the output buffer + buffer_copy dispatch pair.
// Safe for element-wise ops: each thread reads/writes the same index.
//
// Grid: (dim, 1, 1) threads total.
// Threadgroup: (256, 1, 1).
// ---------------------------------------------------------------------------

kernel void residual_add_inplace(
    device float*       a   [[buffer(0)]],
    device const float* b   [[buffer(1)]],
    constant uint&      dim [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;
    a[tid] += b[tid];
}
