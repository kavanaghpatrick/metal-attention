#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Decode (autoregressive) attention kernel with GQA support
//
// Computes single-token attention: softmax(Q . K^T / scale) . V
// for one query token against the full KV cache.
//
// Q:      [num_heads * head_dim]              - single query token
// K_cache: [kv_len, num_kv_heads * head_dim]  - key cache (row-major)
// V_cache: [kv_len, num_kv_heads * head_dim]  - value cache (row-major)
// Output: [num_heads * head_dim]              - attention output
//
// Grid: (num_heads) threadgroups -- one threadgroup per query head
// Threadgroup: (32) threads -- 1 simdgroup per head
//
// GQA: kv_head = head_id / group_size
//      group_size = num_heads / num_kv_heads
//
// Phase 1: Q.K^T scores -- cooperative over kv_len positions
// Phase 2: Online softmax -- simd_max, exp, simd_sum, normalize
// Phase 3: Weighted V sum -- cooperative over head_dim
// ---------------------------------------------------------------------------

kernel void decode_attention(
    device const float* q        [[buffer(0)]],   // [num_heads * head_dim]
    device const float* k_cache  [[buffer(1)]],   // [kv_len, num_kv_heads * head_dim]
    device const float* v_cache  [[buffer(2)]],   // [kv_len, num_kv_heads * head_dim]
    device float*       output   [[buffer(3)]],   // [num_heads * head_dim]
    constant uint&      num_heads    [[buffer(4)]],
    constant uint&      num_kv_heads [[buffer(5)]],
    constant uint&      head_dim     [[buffer(6)]],
    constant uint&      kv_len       [[buffer(7)]],
    constant float&     scale        [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]],    // which query head
    uint tid  [[thread_index_in_threadgroup]]       // thread within simdgroup [0..31]
) {
    const uint head_id = tgid;
    if (head_id >= num_heads) return;

    // GQA: map query head to KV head
    const uint group_size = num_heads / num_kv_heads;
    const uint kv_head = head_id / group_size;

    // Pointer to this query head's Q vector
    device const float* q_head = q + head_id * head_dim;

    // KV head stride in cache rows
    const uint kv_offset = kv_head * head_dim;
    const uint kv_stride = num_kv_heads * head_dim;  // row stride in cache

    // Threadgroup memory for Q.K^T scores (max 2048 kv positions = 8KB)
    threadgroup float scores[2048];

    // -----------------------------------------------------------------------
    // Phase 1: Compute Q.K^T scores for all kv_len positions
    // Each thread handles a subset of positions, striding by 32
    // -----------------------------------------------------------------------
    for (uint pos = tid; pos < kv_len; pos += 32) {
        device const float* k_row = k_cache + pos * kv_stride + kv_offset;
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dot += q_head[d] * k_row[d];
        }
        scores[pos] = dot * scale;
    }

    // Ensure all threads see the completed scores array
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -----------------------------------------------------------------------
    // Phase 2: Online softmax with numerical stability
    //   1. Find max score (simd_max)
    //   2. Compute exp(score - max) and sum (simd_sum)
    //   3. Normalize scores in-place
    // -----------------------------------------------------------------------

    // Step 2a: Find max score across all positions
    float local_max = -INFINITY;
    for (uint pos = tid; pos < kv_len; pos += 32) {
        local_max = max(local_max, scores[pos]);
    }
    float max_score = simd_max(local_max);

    // Step 2b: Compute exp(score - max) and accumulate sum
    float local_sum = 0.0f;
    for (uint pos = tid; pos < kv_len; pos += 32) {
        float val = exp(scores[pos] - max_score);
        scores[pos] = val;
        local_sum += val;
    }
    float sum_exp = simd_sum(local_sum);

    // Ensure all threads see the exponentiated scores before normalization
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2c: Normalize scores (divide by sum)
    float inv_sum = 1.0f / sum_exp;
    for (uint pos = tid; pos < kv_len; pos += 32) {
        scores[pos] *= inv_sum;
    }

    // Ensure all threads see normalized scores before weighted V sum
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -----------------------------------------------------------------------
    // Phase 3: Weighted V sum
    // Each thread handles a subset of head_dim dimensions, accumulating
    // over all kv positions for each dimension.
    // -----------------------------------------------------------------------
    device float* out_head = output + head_id * head_dim;

    for (uint d = tid; d < head_dim; d += 32) {
        float acc = 0.0f;
        for (uint pos = 0; pos < kv_len; pos++) {
            device const float* v_row = v_cache + pos * kv_stride + kv_offset;
            acc += scores[pos] * v_row[d];
        }
        out_head[d] = acc;
    }
}
