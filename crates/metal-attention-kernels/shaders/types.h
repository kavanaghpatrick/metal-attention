#ifndef TYPES_H
#define TYPES_H

#include <metal_stdlib>
using namespace metal;

struct AttentionParams {
    uint seq_len;           // sequence length N
    uint head_dim;          // head dimension D
    uint num_heads;         // number of Q heads
    uint num_kv_heads;      // number of KV heads (== num_heads for MHA, < for GQA)
    uint block_r;           // tile rows (queries)
    uint block_c;           // tile columns (keys)
    float scale;            // 1/sqrt(D)
    uint variant;           // 0=standard, 1=RoPE, 2=ALiBi, 3=GQA
    // Paged attention specific
    uint page_size;         // tokens per page
    uint num_pages;         // total pages allocated
    uint max_context_len;   // maximum context length
    uint num_partitions;    // partitioned reduce count
    uint _pad0;
    uint _pad1;
    uint _pad2;
    uint _pad3;
};
// Total: 64 bytes, 4-byte aligned

// Layer-level parameters for transformer block dispatch
struct LayerParams {
    uint hidden_dim;        // model hidden dimension (e.g. 4096)
    uint intermediate_dim;  // FFN intermediate dimension (e.g. 11008)
    uint num_heads;         // number of attention heads
    uint num_kv_heads;      // number of KV heads (GQA)
    uint head_dim;          // per-head dimension
    uint vocab_size;        // vocabulary size for embedding
    uint layer_idx;         // current layer index
    uint num_layers;        // total number of layers
    float rms_norm_eps;     // RMSNorm epsilon (e.g. 1e-5)
    float rope_theta;       // RoPE base frequency (e.g. 10000.0)
    uint seq_len;           // current sequence length
    uint batch_size;        // batch size
    uint _pad0;
    uint _pad1;
    uint _pad2;
    uint _pad3;
};
// Total: 64 bytes, 4-byte aligned

// SSM (State Space Model) parameters for Mamba/Jamba layers
struct SSMParams {
    uint state_dim;         // SSM state dimension (e.g. 16)
    uint hidden_dim;        // model hidden dimension
    uint intermediate_dim;  // SSM intermediate dimension
    uint num_heads;         // number of SSM heads
    uint head_dim;          // per-head dimension
    uint seq_len;           // current sequence length
    uint batch_size;        // batch size
    uint conv_width;        // convolution kernel width (e.g. 4)
    float dt_min;           // minimum delta time
    float dt_max;           // maximum delta time
    uint layer_idx;         // current layer index
    uint num_layers;        // total number of layers
    uint _pad0;
    uint _pad1;
    uint _pad2;
    uint _pad3;
};
// Total: 64 bytes, 4-byte aligned

#endif
