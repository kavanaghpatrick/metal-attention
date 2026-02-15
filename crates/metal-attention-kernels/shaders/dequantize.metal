#include <metal_stdlib>
#include "types.h"
using namespace metal;

// ---------------------------------------------------------------------------
// Dequantization kernel: Q4_0 format (stub)
//
// Q4_0 block: 32 elements packed into 18 bytes
//   - 2 bytes: half-precision scale factor
//   - 16 bytes: 32 x 4-bit quantized values (pairs packed into bytes)
//
// Each thread dequantizes one block of 32 elements.
// Grid dispatch: (num_blocks, 1, 1)
// ---------------------------------------------------------------------------

// Q4_0 block size: 32 elements per block
#define Q4_0_BLOCK_SIZE 32
// Q4_0 byte size: 2 (scale as float16) + 16 (32 nibbles) = 18 bytes
#define Q4_0_BYTES_PER_BLOCK 18

kernel void dequantize_q4_0(
    device const uchar* input   [[buffer(0)]],   // packed Q4_0 blocks
    device float*       output  [[buffer(1)]],   // [num_blocks * 32] dequantized values
    uint tid [[thread_position_in_grid]]
) {
    // Each thread handles one Q4_0 block
    const uint block_offset = tid * Q4_0_BYTES_PER_BLOCK;
    const uint out_offset = tid * Q4_0_BLOCK_SIZE;

    // Read scale factor (stored as float16 in first 2 bytes)
    device const half* scale_ptr = (device const half*)(input + block_offset);
    float scale = float(*scale_ptr);

    // Read 16 bytes of quantized data (32 x 4-bit values)
    device const uchar* quants = input + block_offset + 2;

    for (uint i = 0; i < 16; i++) {
        uchar byte_val = quants[i];
        // Low nibble (subtract 8 to center around zero)
        float lo = float(int(byte_val & 0x0F) - 8) * scale;
        // High nibble
        float hi = float(int((byte_val >> 4) & 0x0F) - 8) * scale;
        output[out_offset + i * 2] = lo;
        output[out_offset + i * 2 + 1] = hi;
    }
}
