#include <metal_stdlib>
#include "types.h"
using namespace metal;

// ---------------------------------------------------------------------------
// Matrix multiplication kernel (stub)
//
// Computes C = A * B where:
//   A is [M, K], B is [K, N], C is [M, N]
//
// Naive implementation: each thread computes one output element.
// Grid dispatch: (M * N, 1, 1)
// ---------------------------------------------------------------------------

// Dimensions passed via function constants
constant uint MAT_M [[function_constant(0)]];
constant uint MAT_N [[function_constant(1)]];
constant uint MAT_K [[function_constant(2)]];

kernel void matmul(
    device const float* A [[buffer(0)]],   // [M, K]
    device const float* B [[buffer(1)]],   // [K, N]
    device float*       C [[buffer(2)]],   // [M, N]
    uint tid [[thread_position_in_grid]]
) {
    const uint M = MAT_M;
    const uint N = MAT_N;
    const uint K = MAT_K;

    if (tid >= M * N) return;

    uint row = tid / N;
    uint col = tid % N;

    float acc = 0.0f;
    for (uint k = 0; k < K; k++) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}
