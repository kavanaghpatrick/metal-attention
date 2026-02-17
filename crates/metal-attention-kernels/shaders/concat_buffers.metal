#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// concat_buffers_2: Concatenate 2 input buffers into 1 output buffer
//
// output = [a[0..dim_a], b[0..dim_b]]
// Total output length = dim_a + dim_b
//
// Grid: (dim_a + dim_b, 1, 1) threads total.
// Threadgroup: (256, 1, 1).
// ---------------------------------------------------------------------------

kernel void concat_buffers_2(
    device const float* a      [[buffer(0)]],
    device const float* b      [[buffer(1)]],
    device float*       output [[buffer(2)]],
    constant uint&      dim_a  [[buffer(3)]],
    constant uint&      dim_b  [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = dim_a + dim_b;
    if (gid >= total) return;

    if (gid < dim_a) {
        output[gid] = a[gid];
    } else {
        output[gid] = b[gid - dim_a];
    }
}

// ---------------------------------------------------------------------------
// concat_buffers_3: Concatenate 3 input buffers into 1 output buffer
//
// output = [a[0..dim_a], b[0..dim_b], c[0..dim_c]]
// Total output length = dim_a + dim_b + dim_c
//
// Grid: (dim_a + dim_b + dim_c, 1, 1) threads total.
// Threadgroup: (256, 1, 1).
// ---------------------------------------------------------------------------

kernel void concat_buffers_3(
    device const float* a      [[buffer(0)]],
    device const float* b      [[buffer(1)]],
    device const float* c      [[buffer(2)]],
    device float*       output [[buffer(3)]],
    constant uint&      dim_a  [[buffer(4)]],
    constant uint&      dim_b  [[buffer(5)]],
    constant uint&      dim_c  [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = dim_a + dim_b + dim_c;
    if (gid >= total) return;

    if (gid < dim_a) {
        output[gid] = a[gid];
    } else if (gid < dim_a + dim_b) {
        output[gid] = b[gid - dim_a];
    } else {
        output[gid] = c[gid - dim_a - dim_b];
    }
}
