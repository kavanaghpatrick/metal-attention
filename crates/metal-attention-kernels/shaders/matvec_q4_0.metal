#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Fused Q4_0 dequantize + matrix-vector multiply kernel
//
// Computes: output[row] = dot(dequant(weight_row), input)
// where weight is stored in Q4_0 format (32 elements per 18-byte block).
//
// Q4_0 block layout: 2 bytes fp16 scale (half d) + 16 bytes packed nibbles
//   - Low nibbles (byte & 0x0F) - 8  -> elements [0..15]
//   - High nibbles (byte >> 4) - 8   -> elements [16..31]
// Matches dequantize.metal nibble ordering.
//
// Threadgroup: 32 threads (1 simdgroup) per output row.
// Grid: (out_dim, 1, 1) threadgroups.
// Each threadgroup computes one output element via cooperative reduction.
// ---------------------------------------------------------------------------

struct BlockQ4_0 {
    half d;          // scale factor (fp16)
    uchar qs[16];   // 32 x 4-bit quantized values packed into 16 bytes
};

kernel void matvec_q4_0(
    device const BlockQ4_0* weight  [[buffer(0)]],   // [out_dim, n_blocks_per_row] Q4_0 blocks
    device const float*     input   [[buffer(1)]],   // [in_dim] input vector (F32)
    device float*           output  [[buffer(2)]],   // [out_dim] output vector (F32)
    constant uint&          out_dim [[buffer(3)]],   // number of output rows
    constant uint&          in_dim  [[buffer(4)]],   // number of input elements (must be multiple of 32)
    uint tgid [[threadgroup_position_in_grid]],       // which output row
    uint tid  [[thread_index_in_threadgroup]]          // thread within simdgroup [0..31]
) {
    const uint row = tgid;
    if (row >= out_dim) return;

    const uint n_blocks = in_dim / 32;  // number of Q4_0 blocks per row
    const uint row_offset = row * n_blocks;

    float sum = 0.0f;

    // Each thread strides over blocks: thread 0 handles blocks 0,32,64,...
    // thread 1 handles blocks 1,33,65,... etc.
    for (uint b = tid; b < n_blocks; b += 32) {
        const BlockQ4_0 block = weight[row_offset + b];
        const float scale = float(block.d);
        const uint base = b * 32;  // base index into input vector

        // Process 16 bytes -> 32 dequantized values
        for (uint i = 0; i < 16; i++) {
            uchar byte_val = block.qs[i];

            // Low nibble -> input index [base + i] (elements 0..15)
            float lo = float(int(byte_val & 0x0F) - 8) * scale;
            sum += lo * input[base + i];

            // High nibble -> input index [base + i + 16] (elements 16..31)
            float hi = float(int((byte_val >> 4) & 0x0F) - 8) * scale;
            sum += hi * input[base + i + 16];
        }
    }

    // 32-wide simdgroup reduction
    sum = simd_sum(sum);

    // Thread 0 writes the final dot product for this row
    if (tid == 0) {
        output[row] = sum;
    }
}
