#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Q6_K matvec: multi-row matrix-vector multiply for 6-bit K-quant.
//
// Q6_K super-block layout (210 bytes, 256 values):
//   ql[128]:    low 4 bits of 6-bit values
//   qh[64]:     upper 2 bits of 6-bit values
//   scales[16]: int8 sub-block scales (16 sub-blocks of 16 values)
//   d:          fp16 super-block scale (2 bytes)
//
// Bit packing (per chunk of 128 values, l = 0..31):
//   q1 = (ql[l] & 0xF)    | ((qh[l] bits[1:0]) << 4) - 32  -> output at l
//   q2 = (ql[l+32] & 0xF) | ((qh[l] bits[3:2]) << 4) - 32  -> output at l+32
//   q3 = (ql[l] >> 4)     | ((qh[l] bits[5:4]) << 4) - 32  -> output at l+64
//   q4 = (ql[l+32] >> 4)  | ((qh[l] bits[7:6]) << 4) - 32  -> output at l+96
//
// 256 threads = 8 simdgroups, each handles 1 output row (8 rows/TG).
//
// Buffer layout:
//   0: weight    raw Q6_K bytes [out_dim * ceil(in_dim/256) * 210]
//   1: input     [in_dim] float
//   2: output    [out_dim] float
//   3: out_dim   scalar uint
//   4: in_dim    scalar uint
// ---------------------------------------------------------------------------

#define Q6K_BLOCK_SIZE 256
#define Q6K_BLOCK_BYTES 210

kernel void matvec_q6_k(
    device const uchar* weight  [[buffer(0)]],
    device const float* input   [[buffer(1)]],
    device float*       output  [[buffer(2)]],
    constant uint&      out_dim [[buffer(3)]],
    constant uint&      in_dim  [[buffer(4)]],
    uint tgid      [[threadgroup_position_in_grid]],
    uint simd_id   [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    const uint ROWS_PER_TG = 8;
    const uint row = tgid * ROWS_PER_TG + simd_id;
    if (row >= out_dim) return;

    const uint n_blocks = in_dim / Q6K_BLOCK_SIZE;
    const uint row_bytes = n_blocks * Q6K_BLOCK_BYTES;

    float sum = 0.0f;

    for (uint b = simd_lane; b < n_blocks; b += 32) {
        // Raw pointer to this super-block
        device const uchar* bp = weight + row * row_bytes + b * Q6K_BLOCK_BYTES;
        device const uchar* ql = bp;           // 128 bytes
        device const uchar* qh = bp + 128;     // 64 bytes
        device const char*  sc = (device const char*)(bp + 192); // 16 bytes (signed)
        float d = float(*(device const half*)(bp + 208));

        const uint base = b * Q6K_BLOCK_SIZE;

        // Process two 128-element chunks (matches Rust dequantize_q6_k_to_f32 exactly)
        for (uint chunk = 0; chunk < 2; chunk++) {
            const uint ql_off = chunk * 64;
            const uint qh_off = chunk * 32;
            const uint sc_off = chunk * 8;
            const uint inp_off = base + chunk * 128;

            for (uint l = 0; l < 32; l++) {
                const uint is = l / 16; // sub-block half: 0 or 1

                // Reconstruct 6-bit values from ql (4-bit) + qh (2-bit)
                // Each qh byte holds 4 two-bit fields for l, l+32, l+64, l+96
                int q1 = int((ql[ql_off + l]      & 0xF) | (((qh[qh_off + l] >> 0) & 3) << 4)) - 32;
                int q2 = int((ql[ql_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) - 32;
                int q3 = int((ql[ql_off + l]      >> 4)  | (((qh[qh_off + l] >> 4) & 3) << 4)) - 32;
                int q4 = int((ql[ql_off + l + 32] >> 4)  | (((qh[qh_off + l] >> 6) & 3) << 4)) - 32;

                // Per-sub-block scales (8 sub-blocks per chunk, 2 per is group)
                float sc0 = float(sc[sc_off + is]);
                float sc1 = float(sc[sc_off + is + 2]);
                float sc2 = float(sc[sc_off + is + 4]);
                float sc3 = float(sc[sc_off + is + 6]);

                sum += d * sc0 * float(q1) * input[inp_off + l];
                sum += d * sc1 * float(q2) * input[inp_off + l + 32];
                sum += d * sc2 * float(q3) * input[inp_off + l + 64];
                sum += d * sc3 * float(q4) * input[inp_off + l + 96];
            }
        }
    }

    sum = simd_sum(sum);

    if (simd_lane == 0) {
        output[row] = sum;
    }
}
