#include <metal_stdlib>
#include "types.h"
using namespace metal;

// ---------------------------------------------------------------------------
// Mamba Selective Scan kernel
//
// Implements the selective SSM recurrence:
//   h_t = A_t * h_{t-1} + B_t * x_t
//   y_t = C_t * h_t + D * x_t
//
// Where A_t, B_t, C_t are input-dependent (selective) and:
//   x_t is the input at time t [d_model]
//   h_t is the hidden state [d_model, d_state]
//   A_t is the decay factor [d_model] (applied per-dimension)
//   B_t is the input gate [d_state]
//   C_t is the output gate [d_state]
//   D   is the skip connection scalar
//
// This kernel processes tokens sequentially (recurrent decode mode).
// Each thread handles a slice of the d_model x d_state state matrix.
//
// Buffer layout:
//   buffer(0): x_in     [seq_len, d_model]   input
//   buffer(1): A_in     [seq_len, d_model]   decay (should be in (0,1))
//   buffer(2): B_in     [seq_len, d_state]   input gate
//   buffer(3): C_in     [seq_len, d_state]   output gate
//   buffer(4): D_param  [1]                  skip connection scalar
//   buffer(5): state_io [d_model, d_state]   hidden state (read+write)
//   buffer(6): output   [seq_len, d_model]   output
//   buffer(7): seq_len  scalar
//   buffer(8): d_model  scalar
//   buffer(9): d_state  scalar
// ---------------------------------------------------------------------------

kernel void ssm_scan(
    device const float* x_in       [[buffer(0)]],   // [seq_len, d_model]
    device const float* A_in       [[buffer(1)]],   // [seq_len, d_model]
    device const float* B_in       [[buffer(2)]],   // [seq_len, d_state]
    device const float* C_in       [[buffer(3)]],   // [seq_len, d_state]
    device const float* D_param    [[buffer(4)]],   // [1] skip connection
    device float*       state_io   [[buffer(5)]],   // [d_model, d_state]
    device float*       output     [[buffer(6)]],   // [seq_len, d_model]
    constant uint&      seq_len    [[buffer(7)]],
    constant uint&      d_model    [[buffer(8)]],
    constant uint&      d_state    [[buffer(9)]],
    uint tid    [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    const uint DM = d_model;
    const uint DS = d_state;
    const float D_skip = D_param[0];

    // Process tokens sequentially (recurrent)
    for (uint t = 0; t < seq_len; t++) {
        const uint x_off = t * DM;
        const uint a_off = t * DM;
        const uint b_off = t * DS;
        const uint c_off = t * DS;

        // Step 1: Update state h[m][s] = A[m] * h[m][s] + B[s] * x[m]
        // Each thread handles multiple (m, s) elements
        const uint total_state = DM * DS;
        for (uint elem = tid; elem < total_state; elem += tg_size) {
            const uint m = elem / DS;  // model dimension index
            const uint s = elem % DS;  // state dimension index

            float h = A_in[a_off + m] * state_io[m * DS + s]
                    + B_in[b_off + s] * x_in[x_off + m];
            state_io[m * DS + s] = h;
        }

        threadgroup_barrier(mem_flags::mem_device);

        // Step 2: Compute output y[m] = sum_s(C[s] * h[m][s]) + D * x[m]
        for (uint m = tid; m < DM; m += tg_size) {
            float acc = 0.0f;
            for (uint s = 0; s < DS; s++) {
                acc += C_in[c_off + s] * state_io[m * DS + s];
            }
            output[x_off + m] = acc + D_skip * x_in[x_off + m];
        }

        threadgroup_barrier(mem_flags::mem_device);
    }
}
