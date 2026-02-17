#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// F32 matrix-vector multiply kernel
//
// Computes: output[row] = dot(weight_row, input)
// where weight is stored as dense F32 [out_dim, in_dim] row-major.
//
// Used for lm_head when weights are tied embeddings (F32, not Q4_0).
//
// Threadgroup: 32 threads (1 simdgroup) per output row.
// Grid: (out_dim, 1, 1) threadgroups.
// Each threadgroup computes one output element via cooperative reduction.
// ---------------------------------------------------------------------------

kernel void matvec_f32(
    device const float* weight  [[buffer(0)]],   // [out_dim, in_dim] row-major F32
    device const float* input   [[buffer(1)]],   // [in_dim] input vector
    device float*       output  [[buffer(2)]],   // [out_dim] output vector
    constant uint&      out_dim [[buffer(3)]],   // number of output rows
    constant uint&      in_dim  [[buffer(4)]],   // number of input elements
    uint tgid [[threadgroup_position_in_grid]],   // which output row
    uint tid  [[thread_index_in_threadgroup]]      // thread within simdgroup [0..31]
) {
    const uint row = tgid;
    if (row >= out_dim) return;

    const uint row_offset = row * in_dim;

    float sum = 0.0f;

    // Each thread strides over elements: thread 0 handles 0,32,64,...
    for (uint i = tid; i < in_dim; i += 32) {
        sum += weight[row_offset + i] * input[i];
    }

    // 32-wide simdgroup reduction
    sum = simd_sum(sum);

    // Thread 0 writes the final dot product for this row
    if (tid == 0) {
        output[row] = sum;
    }
}
