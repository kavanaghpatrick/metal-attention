#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Bandwidth diagnostic kernels
//
// Measures achievable memory bandwidth on this GPU to establish the ceiling
// for our memory-bandwidth-bound Q4_0 matvec workload.
// ---------------------------------------------------------------------------

// Test 1: Pure sequential read bandwidth (float4 coalesced)
// Each thread reads float4 (16 bytes) sequentially. Maximize coalesced reads.
kernel void bandwidth_read_f32(
    device const float4* data [[buffer(0)]],
    device float*        out  [[buffer(1)]],
    constant uint&       n_float4 [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    float4 sum = float4(0.0f);
    // Each thread reads a strided set of float4 values
    for (uint i = tid; i < n_float4; i += 65536) {
        sum += data[i];
    }
    // Prevent optimization: write reduction to output
    if (tid == 0) {
        out[0] = sum.x + sum.y + sum.z + sum.w;
    }
}

// Test 2: Read with Q4_0 block access pattern (matches our matvec)
// Simulates our actual weight read pattern: 18-byte blocks, stride by simdgroup
struct BlockQ4_0 {
    half d;
    uchar qs[16];
};

kernel void bandwidth_read_q4_0(
    device const BlockQ4_0* blocks [[buffer(0)]],
    device float*           out    [[buffer(1)]],
    constant uint&          n_blocks [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]]
) {
    float sum = 0.0f;
    uint row_blocks = n_blocks; // treat as single giant row

    for (uint b = tid; b < row_blocks; b += 32) {
        BlockQ4_0 block = blocks[b + tgid * row_blocks];
        sum += float(block.d);
        // Touch the nibble data to prevent dead-code elimination
        for (uint i = 0; i < 16; i += 4) {
            sum += float(block.qs[i] & 0x0F);
        }
    }

    sum = simd_sum(sum);
    if (tid == 0) {
        out[tgid] = sum;
    }
}

// Test 3: Multi-dispatch overhead measurement
// Trivial kernel that just writes one value - used to measure dispatch overhead
kernel void dispatch_overhead_noop(
    device float* out [[buffer(0)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid == 0) {
        out[0] = 1.0f;
    }
}

// Test 4: High-occupancy bandwidth read (256 threads/group, float4)
// Tests if more threads per group = higher achieved bandwidth
kernel void bandwidth_read_high_occupancy(
    device const float4* data [[buffer(0)]],
    device float*        out  [[buffer(1)]],
    constant uint&       n_float4 [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    uint global_tid = tgid * threads_per_group + tid;
    float4 sum = float4(0.0f);
    uint total_threads = threads_per_group * 1024; // assume 1024 groups

    for (uint i = global_tid; i < n_float4; i += total_threads) {
        sum += data[i];
    }

    // Simdgroup reduction
    float s = sum.x + sum.y + sum.z + sum.w;
    s = simd_sum(s);

    if (tid == 0) {
        out[tgid] = s;
    }
}

// Test 5: Half-precision bandwidth (should be 2x throughput if bandwidth-bound)
kernel void bandwidth_read_f16(
    device const half4* data [[buffer(0)]],
    device float*       out  [[buffer(1)]],
    constant uint&      n_half4 [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    half4 sum = half4(0.0h);
    for (uint i = tid; i < n_half4; i += 65536) {
        sum += data[i];
    }
    if (tid == 0) {
        out[0] = float(sum.x + sum.y + sum.z + sum.w);
    }
}

// ---------------------------------------------------------------------------
// Experiment 6: Fused RMSNorm + Q4_0 matvec megakernel
//
// Tests: Can we eliminate dispatch overhead by fusing two kernels?
// Single threadgroup does: rmsnorm(input) -> matvec(normalized, weights)
// This is a PROBE — we want to measure if dispatch savings matter.
//
// 256 threads = 8 simdgroups, each handling 1 output row.
// Phase A: All 256 threads cooperate on rmsnorm (read input, compute RMS, normalize)
// Phase B: Each simdgroup does one matvec row (same as matvec_q4_0)
// ---------------------------------------------------------------------------

kernel void megakernel_rmsnorm_matvec(
    device const float*     input    [[buffer(0)]],   // [hidden_dim]
    device const float*     weight   [[buffer(1)]],   // [hidden_dim] rmsnorm weight
    device const BlockQ4_0* mat      [[buffer(2)]],   // [out_dim, hidden_dim/32] Q4_0
    device float*           output   [[buffer(3)]],   // [out_dim]
    constant uint&          hidden_dim [[buffer(4)]],
    constant float&         eps      [[buffer(5)]],
    constant uint&          out_dim  [[buffer(6)]],
    uint tgid      [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    // --- Phase A: RMSNorm into threadgroup memory ---
    threadgroup float normalized[1536]; // max hidden_dim (SmolLM=576, fits easily)

    // Step 1: Compute sum of squares
    float local_ss = 0.0f;
    for (uint i = tid; i < hidden_dim; i += 256) {
        float v = input[i];
        local_ss += v * v;
    }
    float sg_ss = simd_sum(local_ss);

    threadgroup float sg_vals[8];
    if (simd_lane == 0) sg_vals[simd_id] = sg_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float rms_scale;
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < 8; i++) total += sg_vals[i];
        rms_scale = rsqrt(total / float(hidden_dim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Normalize and apply weight -> threadgroup memory
    for (uint i = tid; i < hidden_dim; i += 256) {
        normalized[i] = input[i] * rms_scale * weight[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Phase B: Q4_0 matvec from threadgroup memory ---
    const uint ROWS_PER_TG = 8;
    const uint row = tgid * ROWS_PER_TG + simd_id;
    if (row >= out_dim) return;

    const uint n_blocks = hidden_dim / 32;
    const uint row_offset = row * n_blocks;

    float sum = 0.0f;
    for (uint b = simd_lane; b < n_blocks; b += 32) {
        const BlockQ4_0 block = mat[row_offset + b];
        const float scale = float(block.d);
        const uint base = b * 32;

        for (uint i = 0; i < 16; i += 4) {
            // Read from threadgroup memory instead of device memory!
            float4 in_lo = float4(normalized[base + i],
                                  normalized[base + i + 1],
                                  normalized[base + i + 2],
                                  normalized[base + i + 3]);
            float4 in_hi = float4(normalized[base + i + 16],
                                  normalized[base + i + 17],
                                  normalized[base + i + 18],
                                  normalized[base + i + 19]);

            float4 lo_vals = float4(int(block.qs[i]   & 0x0F) - 8,
                                    int(block.qs[i+1] & 0x0F) - 8,
                                    int(block.qs[i+2] & 0x0F) - 8,
                                    int(block.qs[i+3] & 0x0F) - 8) * scale;
            float4 hi_vals = float4(int(block.qs[i]   >> 4) - 8,
                                    int(block.qs[i+1] >> 4) - 8,
                                    int(block.qs[i+2] >> 4) - 8,
                                    int(block.qs[i+3] >> 4) - 8) * scale;

            sum += dot(lo_vals, in_lo) + dot(hi_vals, in_hi);
        }
    }

    sum = simd_sum(sum);
    if (simd_lane == 0) {
        output[row] = sum;
    }
}

// ---------------------------------------------------------------------------
// Experiment 7: Multi-token matmul (batch=N tokens against same weight matrix)
//
// Tests: Does batching tokens amortize dispatch + weight-read overhead?
// Each token still produces a separate output vector, but weight reads
// can be shared across tokens via L2/SLC cache.
//
// 256 threads, 8 simdgroups. Each simdgroup handles 1 output row.
// Inner loop processes all batch_size tokens for that row.
// ---------------------------------------------------------------------------

kernel void bench_multi_token_matvec_q4_0(
    device const BlockQ4_0* weight  [[buffer(0)]],
    device const float*     input   [[buffer(1)]],  // [batch_size, in_dim]
    device float*           output  [[buffer(2)]],  // [batch_size, out_dim]
    constant uint&          out_dim    [[buffer(3)]],
    constant uint&          in_dim     [[buffer(4)]],
    constant uint&          batch_size [[buffer(5)]],
    uint tgid      [[threadgroup_position_in_grid]],
    uint simd_id   [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    const uint ROWS_PER_TG = 8;
    const uint row = tgid * ROWS_PER_TG + simd_id;
    if (row >= out_dim) return;

    const uint n_blocks = in_dim / 32;
    const uint row_offset = row * n_blocks;

    // Process each token in the batch
    for (uint tok = 0; tok < batch_size; tok++) {
        device const float* tok_input = input + tok * in_dim;
        float sum = 0.0f;

        for (uint b = simd_lane; b < n_blocks; b += 32) {
            const BlockQ4_0 block = weight[row_offset + b];
            const float scale = float(block.d);
            const uint base = b * 32;

            for (uint i = 0; i < 16; i += 4) {
                float4 in_lo = float4(tok_input[base + i],
                                      tok_input[base + i + 1],
                                      tok_input[base + i + 2],
                                      tok_input[base + i + 3]);
                float4 in_hi = float4(tok_input[base + i + 16],
                                      tok_input[base + i + 17],
                                      tok_input[base + i + 18],
                                      tok_input[base + i + 19]);

                float4 lo_vals = float4(int(block.qs[i]   & 0x0F) - 8,
                                        int(block.qs[i+1] & 0x0F) - 8,
                                        int(block.qs[i+2] & 0x0F) - 8,
                                        int(block.qs[i+3] & 0x0F) - 8) * scale;
                float4 hi_vals = float4(int(block.qs[i]   >> 4) - 8,
                                        int(block.qs[i+1] >> 4) - 8,
                                        int(block.qs[i+2] >> 4) - 8,
                                        int(block.qs[i+3] >> 4) - 8) * scale;

                sum += dot(lo_vals, in_lo) + dot(hi_vals, in_hi);
            }
        }

        sum = simd_sum(sum);
        if (simd_lane == 0) {
            output[tok * out_dim + row] = sum;
        }
    }
}
