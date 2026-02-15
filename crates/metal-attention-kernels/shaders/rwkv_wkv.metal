#include <metal_stdlib>
#include "types.h"
using namespace metal;

// ---------------------------------------------------------------------------
// RWKV-7 WKV (Weighted Key-Value) operator
//
// The core recurrence for RWKV-7's linear attention mechanism:
//   state_new = diag(w) * state_old + k^T * v   (outer product update)
//   output    = r * state_new                     (query against state)
//
// Where:
//   r (receptance) = query-like vector [head_dim]
//   k (key)        = key vector [head_dim]
//   v (value)      = value vector [head_dim]
//   w (decay)      = per-dimension decay factors [head_dim], in (0,1)
//   state          = hidden state matrix [head_dim, head_dim]
//   output         = result vector [head_dim]
//
// This kernel processes one token at a time (recurrent decode mode).
// For prefill, the host iterates over tokens sequentially.
//
// Layout:
//   Each threadgroup handles one token's WKV computation.
//   Thread mapping: tid handles elements of the head_dim x head_dim state.
//
// Function constants:
//   HEAD_DIM (index 0) - head dimension D
// ---------------------------------------------------------------------------

constant uint FC_HEAD_DIM [[function_constant(0)]];

// Tile dimension for threadgroup memory (compile-time max)
#define TILE_D 64

kernel void rwkv_wkv(
    device const float* r_in      [[buffer(0)]],  // [seq_len, head_dim] receptance
    device const float* k_in      [[buffer(1)]],  // [seq_len, head_dim] key
    device const float* v_in      [[buffer(2)]],  // [seq_len, head_dim] value
    device const float* w_in      [[buffer(3)]],  // [seq_len, head_dim] decay
    device float*       state_io  [[buffer(4)]],  // [head_dim, head_dim] state (read+write)
    device float*       output    [[buffer(5)]],  // [seq_len, head_dim] output
    constant uint&      seq_len   [[buffer(6)]],  // number of tokens to process
    constant uint&      head_dim  [[buffer(7)]],  // head dimension
    uint tid    [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    const uint D = head_dim;
    const uint DD = D * D;

    // Process tokens sequentially (recurrent)
    for (uint t = 0; t < seq_len; t++) {
        const uint token_offset = t * D;

        // Load r, k, v, w for this token into threadgroup memory
        threadgroup float r_tg[TILE_D];
        threadgroup float k_tg[TILE_D];
        threadgroup float v_tg[TILE_D];
        threadgroup float w_tg[TILE_D];

        // Cooperative load of vectors
        for (uint i = tid; i < D; i += tg_size) {
            r_tg[i] = r_in[token_offset + i];
            k_tg[i] = k_in[token_offset + i];
            v_tg[i] = v_in[token_offset + i];
            w_tg[i] = w_in[token_offset + i];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 1: Update state and compute output simultaneously
        // state[i][j] = w[i] * state[i][j] + k[i] * v[j]
        // output[j] = sum_i(r[i] * state_new[i][j])
        //
        // We compute per-column partial sums for the output reduction.

        // Each thread handles multiple (i,j) elements of state
        for (uint elem = tid; elem < DD; elem += tg_size) {
            const uint i = elem / D;  // row
            const uint j = elem % D;  // col

            // State update: decay old state + outer product
            float s = w_tg[i] * state_io[i * D + j] + k_tg[i] * v_tg[j];
            state_io[i * D + j] = s;
        }

        threadgroup_barrier(mem_flags::mem_device);

        // Step 2: Compute output[j] = sum_i(r[i] * state[i][j])
        // Each thread computes output elements
        for (uint j = tid; j < D; j += tg_size) {
            float acc = 0.0f;
            for (uint i = 0; i < D; i++) {
                acc += r_tg[i] * state_io[i * D + j];
            }
            output[token_offset + j] = acc;
        }

        // Barrier before next token to ensure state is consistent
        threadgroup_barrier(mem_flags::mem_device);
    }
}
