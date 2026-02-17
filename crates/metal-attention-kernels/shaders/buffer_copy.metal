#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Buffer copy kernel
//
// Copies count floats from src to dst.
//
//   dst[tid] = src[tid]
//
// Grid: (count, 1, 1) threads total.
// Threadgroup: (256, 1, 1).
// ---------------------------------------------------------------------------

kernel void buffer_copy(
    device const float* src   [[buffer(0)]],
    device float*       dst   [[buffer(1)]],
    constant uint&      count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    dst[tid] = src[tid];
}
