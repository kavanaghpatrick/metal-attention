#include <metal_stdlib>
#include "types.h"
using namespace metal;

// ---------------------------------------------------------------------------
// GPU Parallel Prefix Sum (Blelloch Scan) over D*D matrices.
//
// Input:  array of N matrices, each [D, D], stored as flat float arrays.
//         Total input size: N * D * D floats.
// Output: inclusive prefix sums where output[i] = sum(input[0..=i])
//         for each matrix element independently.
//
// Algorithm:
//   Phase 1 (Up-sweep / Reduce): Build partial sums bottom-up.
//   Phase 2 (Down-sweep / Scatter): Distribute sums top-down.
//
// This kernel handles the single-threadgroup case (N fits in one
// threadgroup's shared memory). The host code falls back to CPU for
// very large N if needed (>1024 matrices).
//
// Each thread operates on one element within the D*D matrix,
// and the threadgroup collectively scans across the N dimension.
// ---------------------------------------------------------------------------

// Function constants set by host at PSO compile time.
constant uint HEAD_DIM   [[function_constant(0)]];
constant uint NUM_MATRICES [[function_constant(1)]];

// Maximum matrices we can handle in threadgroup memory.
// For D=64: 64*64*4 = 16KB per matrix. At 32KB limit, we can hold ~2.
// Instead we keep only 2 ping-pong buffers of the scan (current + swap).
// Actually for the scan we process element-wise: each thread handles one
// (d_row, d_col) coordinate and scans along the N dimension.
//
// Approach: Each threadgroup processes ALL N matrices for a subset of
// D*D elements. Threads iterate over N sequentially (prefix sum is
// inherently sequential per element position). This is efficient
// because each thread does N additions for its element coordinate.

kernel void prefix_sum_matrices(
    device const float* input   [[buffer(0)]],   // [N, D, D] flat
    device float* output        [[buffer(1)]],   // [N, D, D] flat
    device const uint* params   [[buffer(2)]],   // [n_matrices, d]
    uint tid [[thread_position_in_grid]]
) {
    const uint n = params[0];   // number of matrices
    const uint d = params[1];   // dimension (D)
    const uint dd = d * d;      // elements per matrix

    // Each thread handles one element position within the D*D matrix
    if (tid >= dd) return;

    const uint elem = tid;  // which element in [D, D]

    // Sequential inclusive prefix sum along the N dimension
    float running = 0.0f;
    for (uint i = 0; i < n; i++) {
        running += input[i * dd + elem];
        output[i * dd + elem] = running;
    }
}
