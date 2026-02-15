#include <metal_stdlib>
#include "types.h"
using namespace metal;

// ---------------------------------------------------------------------------
// Embedding lookup kernel (stub)
//
// Copies embedding vectors from table based on token IDs.
// Grid dispatch: (seq_len * hidden_dim, 1, 1)
// Each thread copies one element of one embedding vector.
// ---------------------------------------------------------------------------

kernel void embedding_lookup(
    device const float* table   [[buffer(0)]],   // [vocab_size, hidden_dim]
    device const uint*  tokens  [[buffer(1)]],   // [seq_len]
    device float*       output  [[buffer(2)]],   // [seq_len, hidden_dim]
    constant LayerParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint hidden_dim = params.hidden_dim;
    const uint total = params.seq_len * hidden_dim;
    if (tid >= total) return;

    uint token_idx = tid / hidden_dim;
    uint dim = tid % hidden_dim;
    uint token_id = tokens[token_idx];

    output[tid] = table[token_id * hidden_dim + dim];
}
