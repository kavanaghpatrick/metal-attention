#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Q6_K matvec: multi-row vectorized matrix-vector multiply.
//
// Q6_K super-block layout (210 bytes, 256 values):
//   ql[128]   : low 4 bits of each 6-bit value, 2 values per byte
//   qh[64]    : high 2 bits of each 6-bit value, 4 values per byte
//   scales[16]: int8 sub-block scales (16 sub-blocks of 16 values each)
//   d          : fp16 super-block scale
//
// Dequant: value = d * scales[sub_block] * (q6_unsigned - 32)
//   where q6_unsigned = (ql_nibble) | ((qh_2bits) << 4), range [0, 63]
//
// 256 threads = 8 simdgroups, each handles 1 output row (8 rows/TG).
// Vectorized 4-wide inner loop with simd_sum reduction.
//
// Buffer layout:
//   0: weight    Q6_K blocks [out_dim, in_dim/256]
//   1: input     [in_dim] float
//   2: output    [out_dim] float
//   3: out_dim   scalar uint
//   4: in_dim    scalar uint
// ---------------------------------------------------------------------------

struct BlockQ6_K {
    uchar ql[128];    // low 4 bits: 2 values per byte (256 values / 2)
    uchar qh[64];     // high 2 bits: 4 values per byte (256 values / 4)
    char  scales[16]; // int8 sub-block scales (16 sub-blocks of 16 values)
    half  d;          // super-block scale (fp16)
};

kernel void matvec_q6_k(
    device const BlockQ6_K* weight  [[buffer(0)]],
    device const float*     input   [[buffer(1)]],
    device float*           output  [[buffer(2)]],
    constant uint&          out_dim [[buffer(3)]],
    constant uint&          in_dim  [[buffer(4)]],
    uint tgid      [[threadgroup_position_in_grid]],
    uint simd_id   [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    const uint ROWS_PER_TG = 8;
    const uint row = tgid * ROWS_PER_TG + simd_id;
    if (row >= out_dim) return;

    const uint n_blocks = in_dim / 256;  // 256 values per Q6_K super-block
    const uint row_offset = row * n_blocks;

    float sum = 0.0f;

    for (uint b = simd_lane; b < n_blocks; b += 32) {
        device const BlockQ6_K& block = weight[row_offset + b];
        const float d = float(block.d);
        const uint base = b * 256;  // base index into input vector

        // Process 256 values in two chunks of 128.
        // Chunk 0: values [0..127]   -> ql[0..63] low/high nibbles, qh[0..31]
        // Chunk 1: values [128..255] -> ql[64..127] low/high nibbles, qh[32..63]
        for (uint chunk = 0; chunk < 2; chunk++) {
            const uint ql_base = chunk * 64;
            const uint qh_base = chunk * 32;
            const uint val_base = base + chunk * 128;

            // Each chunk has 128 values across 8 sub-blocks of 16 values each
            // Sub-blocks within this chunk: chunk*8 .. chunk*8+7
            const uint sb_base = chunk * 8;

            // Process 128 values in groups of 4
            for (uint i = 0; i < 64; i += 4) {
                // Low nibble values (indices 0..63 within chunk)
                float4 in_lo = float4(input[val_base + i],
                                      input[val_base + i + 1],
                                      input[val_base + i + 2],
                                      input[val_base + i + 3]);

                // High nibble values (indices 64..127 within chunk)
                float4 in_hi = float4(input[val_base + 64 + i],
                                      input[val_base + 64 + i + 1],
                                      input[val_base + 64 + i + 2],
                                      input[val_base + 64 + i + 3]);

                float4 w_lo, w_hi;
                for (uint k = 0; k < 4; k++) {
                    // Extract low 4 bits from ql
                    uchar ql_byte = block.ql[ql_base + i + k];
                    uchar ql_lo = ql_byte & 0x0F;        // low nibble -> value index i+k
                    uchar ql_hi = (ql_byte >> 4) & 0x0F; // high nibble -> value index 64+i+k

                    // Extract high 2 bits from qh
                    // qh packs 4 values per byte: bits [1:0], [3:2], [5:4], [7:6]
                    // For value index j within chunk (0..127):
                    //   qh byte index = j % 64 / 4 = (j/4) for j<64, ((j-64)/4) for j>=64
                    //   Actually: qh maps linearly, 4 values per byte
                    //   Byte qh[j/4], bits (j%4)*2 .. (j%4)*2+1

                    // Value index for low nibble: i+k (within 0..63 of chunk)
                    uint lo_idx = i + k;
                    uchar qh_byte_lo = block.qh[qh_base + lo_idx / 4];
                    uchar qh_lo = (qh_byte_lo >> ((lo_idx % 4) * 2)) & 0x03;

                    // Value index for high nibble: 64+i+k (within 64..127 of chunk)
                    uint hi_idx = 64 + i + k;
                    uchar qh_byte_hi = block.qh[qh_base + hi_idx / 4];
                    uchar qh_hi = (qh_byte_hi >> ((hi_idx % 4) * 2)) & 0x03;

                    // Reconstruct 6-bit unsigned values
                    uchar q6_lo = ql_lo | (qh_lo << 4); // [0..63]
                    uchar q6_hi = ql_hi | (qh_hi << 4); // [0..63]

                    // Sub-block index (each sub-block is 16 values)
                    uint sb_lo = sb_base + lo_idx / 16;
                    uint sb_hi = sb_base + hi_idx / 16;

                    float scale_lo = d * float(block.scales[sb_lo]);
                    float scale_hi = d * float(block.scales[sb_hi]);

                    w_lo[k] = scale_lo * float(int(q6_lo) - 32);
                    w_hi[k] = scale_hi * float(int(q6_hi) - 32);
                }

                sum += dot(w_lo, in_lo);
                sum += dot(w_hi, in_hi);
            }
        }
    }

    sum = simd_sum(sum);

    if (simd_lane == 0) {
        output[row] = sum;
    }
}
