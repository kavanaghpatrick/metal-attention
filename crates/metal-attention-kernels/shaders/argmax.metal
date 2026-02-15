#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Two-stage parallel argmax reduction kernel
//
// Stage 1 (argmax_reduce): 256 threads/group, each thread processes elements
//   at stride across the vocab. Threadgroup shared memory reduction produces
//   one (max_val, max_idx) pair per threadgroup.
//
// Stage 2 (argmax_final): Single threadgroup of 256 threads reduces the
//   partial results from Stage 1 to find the global argmax.
//
// Dispatch Stage 1: ceil(vocab_size / (256 * 4)) threadgroups of 256 threads
// Dispatch Stage 2: 1 threadgroup of 256 threads
// ---------------------------------------------------------------------------

// Shared memory arrays for threadgroup reduction
constant uint THREADS_PER_GROUP = 256;
constant uint ELEMENTS_PER_THREAD = 4;

// ---------------------------------------------------------------------------
// Stage 1: Each threadgroup reduces a chunk of the logits array
// ---------------------------------------------------------------------------
kernel void argmax_reduce(
    device const float* logits       [[buffer(0)]],
    constant uint&      vocab_size   [[buffer(1)]],
    device float*       partial_vals [[buffer(2)]],
    device uint*        partial_idxs [[buffer(3)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]],
    uint num_tg [[threadgroups_per_grid]]
) {
    // Shared memory for threadgroup reduction
    threadgroup float shared_vals[256];
    threadgroup uint  shared_idxs[256];

    // Each thread finds its local max across strided elements
    const uint total_threads = num_tg * THREADS_PER_GROUP;
    const uint global_tid = tgid * THREADS_PER_GROUP + tid;

    float local_max = -INFINITY;
    uint  local_idx = 0;

    for (uint i = global_tid; i < vocab_size; i += total_threads) {
        float val = logits[i];
        if (val > local_max) {
            local_max = val;
            local_idx = i;
        }
    }

    // Write local result to shared memory
    shared_vals[tid] = local_max;
    shared_idxs[tid] = local_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction within threadgroup (sequential halving)
    for (uint stride = THREADS_PER_GROUP / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared_vals[tid + stride] > shared_vals[tid]) {
                shared_vals[tid] = shared_vals[tid + stride];
                shared_idxs[tid] = shared_idxs[tid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 writes the threadgroup result
    if (tid == 0) {
        partial_vals[tgid] = shared_vals[0];
        partial_idxs[tgid] = shared_idxs[0];
    }
}

// ---------------------------------------------------------------------------
// Stage 2: Single threadgroup reduces partial results to global argmax
// ---------------------------------------------------------------------------
kernel void argmax_final(
    device const float* partial_vals [[buffer(0)]],
    device const uint*  partial_idxs [[buffer(1)]],
    constant uint&      num_groups   [[buffer(2)]],
    device uint*        result       [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    // Shared memory for threadgroup reduction
    threadgroup float shared_vals[256];
    threadgroup uint  shared_idxs[256];

    // Each thread handles ceil(num_groups / 256) partial results
    float local_max = -INFINITY;
    uint  local_idx = 0;

    for (uint i = tid; i < num_groups; i += THREADS_PER_GROUP) {
        float val = partial_vals[i];
        if (val > local_max) {
            local_max = val;
            local_idx = partial_idxs[i];
        }
    }

    // Write to shared memory
    shared_vals[tid] = local_max;
    shared_idxs[tid] = local_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction (sequential halving)
    for (uint stride = THREADS_PER_GROUP / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared_vals[tid + stride] > shared_vals[tid]) {
                shared_vals[tid] = shared_vals[tid + stride];
                shared_idxs[tid] = shared_idxs[tid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 writes the global argmax token id
    if (tid == 0) {
        result[0] = shared_idxs[0];
    }
}
