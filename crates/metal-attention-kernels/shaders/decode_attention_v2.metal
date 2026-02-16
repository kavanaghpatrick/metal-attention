#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Optimized decode attention with multi-simdgroup parallelism.
//
// Key improvements over v1:
//   - 256 threads (8 simdgroups) per head vs. 32 (1 simdgroup)
//   - float4 vectorized Q.K^T dot products
//   - fast::exp for softmax computation
//   - Cross-simdgroup reduction via threadgroup memory
//   - All 256 threads active in Phase 3 via position-parallel V accumulation
//
// Grid: (num_heads) threadgroups
// Threadgroup: (256) threads = 8 simdgroups
// ---------------------------------------------------------------------------

kernel void decode_attention_v2(
    device const float* q        [[buffer(0)]],
    device const float* k_cache  [[buffer(1)]],
    device const float* v_cache  [[buffer(2)]],
    device float*       output   [[buffer(3)]],
    constant uint&      num_heads    [[buffer(4)]],
    constant uint&      num_kv_heads [[buffer(5)]],
    constant uint&      head_dim     [[buffer(6)]],
    constant uint&      kv_len       [[buffer(7)]],
    constant float&     scale        [[buffer(8)]],
    uint tgid     [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint simd_id  [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    const uint head_id = tgid;
    if (head_id >= num_heads) return;

    // GQA mapping
    const uint group_size = num_heads / num_kv_heads;
    const uint kv_head = head_id / group_size;
    device const float* q_head = q + head_id * head_dim;
    const uint kv_offset = kv_head * head_dim;
    const uint kv_stride = num_kv_heads * head_dim;

    // Threadgroup memory for scores (max 2048 positions = 8KB)
    threadgroup float scores[2048];

    // Shared memory for cross-simdgroup reductions
    threadgroup float sg_vals[8];

    // -----------------------------------------------------------------------
    // Phase 1: Q.K^T scores — 256 threads stride over positions
    // Vectorized float4 dot product for head_dim elements
    // -----------------------------------------------------------------------
    const uint hd4 = head_dim / 4;  // number of float4 chunks

    for (uint pos = tid; pos < kv_len; pos += 256) {
        device const float4* k_row4 = (device const float4*)(k_cache + pos * kv_stride + kv_offset);
        device const float4* q_head4 = (device const float4*)q_head;
        float dot = 0.0f;
        for (uint d = 0; d < hd4; d++) {
            float4 qv = q_head4[d];
            float4 kv = k_row4[d];
            dot += qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
        }
        // Handle remaining elements if head_dim not divisible by 4
        for (uint d = hd4 * 4; d < head_dim; d++) {
            dot += q_head[d] * (k_cache + pos * kv_stride + kv_offset)[d];
        }
        scores[pos] = dot * scale;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -----------------------------------------------------------------------
    // Phase 2: Softmax — cross-simdgroup reduction
    // -----------------------------------------------------------------------

    // Step 2a: Find max score
    float local_max = -INFINITY;
    for (uint pos = tid; pos < kv_len; pos += 256) {
        local_max = max(local_max, scores[pos]);
    }
    // Simdgroup-level max
    float sg_max = simd_max(local_max);

    // Cross-simdgroup max via threadgroup memory
    if (simd_lane == 0) sg_vals[simd_id] = sg_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 computes global max and broadcasts
    threadgroup float global_max_val;
    if (tid == 0) {
        float m = sg_vals[0];
        for (uint i = 1; i < 8; i++) m = max(m, sg_vals[i]);
        global_max_val = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float max_score = global_max_val;

    // Step 2b: Compute exp(score - max) and accumulate sum
    float local_sum = 0.0f;
    for (uint pos = tid; pos < kv_len; pos += 256) {
        float val = fast::exp(scores[pos] - max_score);
        scores[pos] = val;
        local_sum += val;
    }
    float sg_sum = simd_sum(local_sum);

    // Cross-simdgroup sum
    if (simd_lane == 0) sg_vals[simd_id] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float global_sum_val;
    if (tid == 0) {
        float s = 0.0f;
        for (uint i = 0; i < 8; i++) s += sg_vals[i];
        global_sum_val = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = 1.0f / global_sum_val;

    // Step 2c: Normalize scores
    for (uint pos = tid; pos < kv_len; pos += 256) {
        scores[pos] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -----------------------------------------------------------------------
    // Phase 3: Weighted V sum — all 256 threads active
    //
    // For head_dim=64: 256/64 = 4 threads per dimension.
    // Each thread accumulates over a strided subset of positions, then
    // we reduce the 4 partial sums per dimension via threadgroup memory.
    //
    // Thread mapping: dim = tid % head_dim, chunk = tid / head_dim
    // Thread strides over positions: pos = chunk, chunk + threads_per_dim, ...
    // -----------------------------------------------------------------------
    device float* out_head = output + head_id * head_dim;

    const uint threads_per_dim = 256 / head_dim;  // 4 for head_dim=64
    const uint dim = tid % head_dim;
    const uint chunk = tid / head_dim;

    float acc = 0.0f;
    for (uint pos = chunk; pos < kv_len; pos += threads_per_dim) {
        acc += scores[pos] * v_cache[pos * kv_stride + kv_offset + dim];
    }

    // Store partial sum — reuse scores[] since softmax weights already consumed
    threadgroup float partials[256];
    partials[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First thread of each dimension group reduces and writes output
    if (chunk == 0) {
        float result = partials[dim];
        for (uint c = 1; c < threads_per_dim; c++) {
            result += partials[dim + c * head_dim];
        }
        out_head[dim] = result;
    }
}
