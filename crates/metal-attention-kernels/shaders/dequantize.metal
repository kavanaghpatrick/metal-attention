#include <metal_stdlib>
#include "types.h"
using namespace metal;

// ---------------------------------------------------------------------------
// Dequantization kernels for GGUF quantization formats
//
// Q4_0: 32 elements per block, 18 bytes (2 fp16 scale + 16 data)
// Q8_0: 32 elements per block, 34 bytes (2 fp16 scale + 32 data)
//
// Each thread dequantizes one block of 32 elements.
// Grid dispatch: (num_blocks, 1, 1)
// ---------------------------------------------------------------------------

// --- Q4_0 ---
// Block: 2 bytes fp16 scale + 16 bytes (32 x 4-bit nibbles)
// Dequant: value = (nibble - 8) * scale
#define Q4_0_BLOCK_SIZE 32
#define Q4_0_BYTES_PER_BLOCK 18

kernel void dequantize_q4_0(
    device const uchar* input   [[buffer(0)]],   // packed Q4_0 blocks
    device float*       output  [[buffer(1)]],   // [num_blocks * 32] dequantized values
    uint tid [[thread_position_in_grid]]
) {
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

// --- Q8_0 ---
// Block: 2 bytes fp16 scale + 32 bytes (32 x int8 values)
// Dequant: value = q * scale
#define Q8_0_BLOCK_SIZE 32
#define Q8_0_BYTES_PER_BLOCK 34

kernel void dequantize_q8_0(
    device const uchar* input   [[buffer(0)]],   // packed Q8_0 blocks
    device float*       output  [[buffer(1)]],   // [num_blocks * 32] dequantized values
    uint tid [[thread_position_in_grid]]
) {
    const uint block_offset = tid * Q8_0_BYTES_PER_BLOCK;
    const uint out_offset = tid * Q8_0_BLOCK_SIZE;

    // Read scale factor (stored as float16 in first 2 bytes)
    device const half* scale_ptr = (device const half*)(input + block_offset);
    float scale = float(*scale_ptr);

    // Read 32 bytes of int8 quantized data
    device const char* quants = (device const char*)(input + block_offset + 2);

    for (uint i = 0; i < Q8_0_BLOCK_SIZE; i++) {
        output[out_offset + i] = float(quants[i]) * scale;
    }
}
