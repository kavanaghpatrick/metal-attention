#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Multi-token batched Q6_K matvec kernels.
//
// Process batch_size tokens against the same weight matrix in a single dispatch.
// Weights cached in SLC after first token for near-linear per-token speedup.
//
// 256 threads = 8 simdgroups, each handles 1 output row.
//
// Buffer layout:
//   0: weight     raw Q6_K bytes [out_dim * ceil(in_dim/256) * 210]
//   1: input      [batch_size, in_dim] float
//   2: output     [batch_size, out_dim] float
//   3: out_dim    scalar uint
//   4: in_dim     scalar uint
//   5: batch_size scalar uint
// ---------------------------------------------------------------------------

#define Q6K_BLOCK_SIZE 256
#define Q6K_BLOCK_BYTES 210

// Variant 1: output = W * input (overwrite)
kernel void multi_token_matvec_q6_k(
    device const uchar* weight     [[buffer(0)]],
    device const float* input      [[buffer(1)]],
    device float*       output     [[buffer(2)]],
    constant uint&      out_dim    [[buffer(3)]],
    constant uint&      in_dim     [[buffer(4)]],
    constant uint&      batch_size [[buffer(5)]],
    uint tgid      [[threadgroup_position_in_grid]],
    uint simd_id   [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    const uint ROWS_PER_TG = 8;
    const uint row = tgid * ROWS_PER_TG + simd_id;
    if (row >= out_dim) return;

    const uint n_blocks = in_dim / Q6K_BLOCK_SIZE;
    const uint row_bytes = n_blocks * Q6K_BLOCK_BYTES;

    for (uint tok = 0; tok < batch_size; tok++) {
        device const float* tok_input = input + tok * in_dim;
        float sum = 0.0f;

        for (uint b = simd_lane; b < n_blocks; b += 32) {
            device const uchar* bp = weight + row * row_bytes + b * Q6K_BLOCK_BYTES;
            device const uchar* ql = bp;
            device const uchar* qh = bp + 128;
            device const char*  sc = (device const char*)(bp + 192);
            float d = float(*(device const half*)(bp + 208));

            const uint base = b * Q6K_BLOCK_SIZE;

            for (uint chunk = 0; chunk < 2; chunk++) {
                const uint ql_off = chunk * 64;
                const uint qh_off = chunk * 32;
                const uint sc_off = chunk * 8;
                const uint inp_off = base + chunk * 128;

                for (uint l = 0; l < 32; l++) {
                    const uint is = l / 16;

                    int q1 = int((ql[ql_off + l]      & 0xF) | (((qh[qh_off + l] >> 0) & 3) << 4)) - 32;
                    int q2 = int((ql[ql_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) - 32;
                    int q3 = int((ql[ql_off + l]      >> 4)  | (((qh[qh_off + l] >> 4) & 3) << 4)) - 32;
                    int q4 = int((ql[ql_off + l + 32] >> 4)  | (((qh[qh_off + l] >> 6) & 3) << 4)) - 32;

                    float sc0 = float(sc[sc_off + is]);
                    float sc1 = float(sc[sc_off + is + 2]);
                    float sc2 = float(sc[sc_off + is + 4]);
                    float sc3 = float(sc[sc_off + is + 6]);

                    sum += d * sc0 * float(q1) * tok_input[inp_off + l];
                    sum += d * sc1 * float(q2) * tok_input[inp_off + l + 32];
                    sum += d * sc2 * float(q3) * tok_input[inp_off + l + 64];
                    sum += d * sc3 * float(q4) * tok_input[inp_off + l + 96];
                }
            }
        }

        sum = simd_sum(sum);
        if (simd_lane == 0) {
            output[tok * out_dim + row] = sum;
        }
    }
}

// Variant 2: output += W * input (accumulate for residual connections)
kernel void multi_token_matvec_q6_k_accumulate(
    device const uchar* weight     [[buffer(0)]],
    device const float* input      [[buffer(1)]],
    device float*       output     [[buffer(2)]],
    constant uint&      out_dim    [[buffer(3)]],
    constant uint&      in_dim     [[buffer(4)]],
    constant uint&      batch_size [[buffer(5)]],
    uint tgid      [[threadgroup_position_in_grid]],
    uint simd_id   [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    const uint ROWS_PER_TG = 8;
    const uint row = tgid * ROWS_PER_TG + simd_id;
    if (row >= out_dim) return;

    const uint n_blocks = in_dim / Q6K_BLOCK_SIZE;
    const uint row_bytes = n_blocks * Q6K_BLOCK_BYTES;

    for (uint tok = 0; tok < batch_size; tok++) {
        device const float* tok_input = input + tok * in_dim;
        float sum = 0.0f;

        for (uint b = simd_lane; b < n_blocks; b += 32) {
            device const uchar* bp = weight + row * row_bytes + b * Q6K_BLOCK_BYTES;
            device const uchar* ql = bp;
            device const uchar* qh = bp + 128;
            device const char*  sc = (device const char*)(bp + 192);
            float d = float(*(device const half*)(bp + 208));

            const uint base = b * Q6K_BLOCK_SIZE;

            for (uint chunk = 0; chunk < 2; chunk++) {
                const uint ql_off = chunk * 64;
                const uint qh_off = chunk * 32;
                const uint sc_off = chunk * 8;
                const uint inp_off = base + chunk * 128;

                for (uint l = 0; l < 32; l++) {
                    const uint is = l / 16;

                    int q1 = int((ql[ql_off + l]      & 0xF) | (((qh[qh_off + l] >> 0) & 3) << 4)) - 32;
                    int q2 = int((ql[ql_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) - 32;
                    int q3 = int((ql[ql_off + l]      >> 4)  | (((qh[qh_off + l] >> 4) & 3) << 4)) - 32;
                    int q4 = int((ql[ql_off + l + 32] >> 4)  | (((qh[qh_off + l] >> 6) & 3) << 4)) - 32;

                    float sc0 = float(sc[sc_off + is]);
                    float sc1 = float(sc[sc_off + is + 2]);
                    float sc2 = float(sc[sc_off + is + 4]);
                    float sc3 = float(sc[sc_off + is + 6]);

                    sum += d * sc0 * float(q1) * tok_input[inp_off + l];
                    sum += d * sc1 * float(q2) * tok_input[inp_off + l + 32];
                    sum += d * sc2 * float(q3) * tok_input[inp_off + l + 64];
                    sum += d * sc3 * float(q4) * tok_input[inp_off + l + 96];
                }
            }
        }

        sum = simd_sum(sum);
        if (simd_lane == 0) {
            output[tok * out_dim + row] += sum;  // ACCUMULATE
        }
    }
}
