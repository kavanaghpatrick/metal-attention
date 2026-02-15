//! Dequantization dispatch: GPU kernel host code.
//!
//! Dispatches Metal kernels for unpacking quantized GGUF blocks:
//!   Q4_0: 18 bytes/block = 2 (fp16 scale) + 16 (32 x 4-bit values)
//!   Q8_0: 34 bytes/block = 2 (fp16 scale) + 32 (32 x int8 values)
//!
//! Output: 32 float values per block.

use crate::buffer::{alloc_buffer, alloc_buffer_with_data, read_buffer_slice};
use crate::device::GpuDevice;
use crate::pipeline::{PsoCache, PsoKey};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLSize,
};

/// Q4_0 block size: 32 elements per block.
pub const Q4_0_BLOCK_SIZE: usize = 32;
/// Q4_0 byte size: 2 (scale as float16) + 16 (32 nibbles) = 18 bytes.
pub const Q4_0_BYTES_PER_BLOCK: usize = 18;

/// Q8_0 block size: 32 elements per block.
pub const Q8_0_BLOCK_SIZE: usize = 32;
/// Q8_0 byte size: 2 (scale as float16) + 32 (32 int8 values) = 34 bytes.
pub const Q8_0_BYTES_PER_BLOCK: usize = 34;

/// Run Q4_0 dequantization on the GPU.
///
/// # Arguments
/// - `device`: GpuDevice with Metal device, command queue, and shader library
/// - `pso_cache`: PSO cache for kernel compilation
/// - `input`: Raw Q4_0 packed data (num_blocks * 18 bytes)
/// - `num_blocks`: Number of Q4_0 blocks
///
/// # Returns
/// Dequantized float values [num_blocks * 32]
pub fn dispatch_dequantize_q4_0(
    device: &GpuDevice,
    pso_cache: &mut PsoCache,
    input: &[u8],
    num_blocks: usize,
) -> Vec<f32> {
    assert_eq!(
        input.len(),
        num_blocks * Q4_0_BYTES_PER_BLOCK,
        "input length mismatch"
    );

    let total_elements = num_blocks * Q4_0_BLOCK_SIZE;

    let input_buf = alloc_buffer_with_data(&device.device, input);
    let output_buf = alloc_buffer(&device.device, total_elements * std::mem::size_of::<f32>());

    let pso_key = PsoKey::simple("dequantize_q4_0");
    let pso = pso_cache.get_or_compile(&pso_key);

    let command_buffer = device
        .command_queue
        .commandBuffer()
        .expect("Failed to create command buffer");
    let encoder = command_buffer
        .computeCommandEncoder()
        .expect("Failed to create compute encoder");

    encoder.setComputePipelineState(pso);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(&*input_buf), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&*output_buf), 0, 1);
    }

    let max_threads = pso.maxTotalThreadsPerThreadgroup();
    let tg_size = max_threads.min(256);

    let grid_size = MTLSize {
        width: num_blocks,
        height: 1,
        depth: 1,
    };
    let threadgroup_size = MTLSize {
        width: tg_size,
        height: 1,
        depth: 1,
    };

    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
    encoder.endEncoding();

    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    let status = command_buffer.status();
    assert_eq!(
        status,
        objc2_metal::MTLCommandBufferStatus::Completed,
        "dequantize_q4_0 command buffer failed with status {:?}. Error: {:?}",
        status,
        command_buffer.error()
    );

    unsafe { read_buffer_slice(&output_buf, total_elements) }
}

/// Run Q8_0 dequantization on the GPU.
///
/// # Arguments
/// - `device`: GpuDevice with Metal device, command queue, and shader library
/// - `pso_cache`: PSO cache for kernel compilation
/// - `input`: Raw Q8_0 packed data (num_blocks * 34 bytes)
/// - `num_blocks`: Number of Q8_0 blocks
///
/// # Returns
/// Dequantized float values [num_blocks * 32]
pub fn dispatch_dequantize_q8_0(
    device: &GpuDevice,
    pso_cache: &mut PsoCache,
    input: &[u8],
    num_blocks: usize,
) -> Vec<f32> {
    assert_eq!(
        input.len(),
        num_blocks * Q8_0_BYTES_PER_BLOCK,
        "Q8_0 input length mismatch"
    );

    let total_elements = num_blocks * Q8_0_BLOCK_SIZE;

    let input_buf = alloc_buffer_with_data(&device.device, input);
    let output_buf = alloc_buffer(&device.device, total_elements * std::mem::size_of::<f32>());

    let pso_key = PsoKey::simple("dequantize_q8_0");
    let pso = pso_cache.get_or_compile(&pso_key);

    let command_buffer = device
        .command_queue
        .commandBuffer()
        .expect("Failed to create command buffer");
    let encoder = command_buffer
        .computeCommandEncoder()
        .expect("Failed to create compute encoder");

    encoder.setComputePipelineState(pso);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(&*input_buf), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&*output_buf), 0, 1);
    }

    let max_threads = pso.maxTotalThreadsPerThreadgroup();
    let tg_size = max_threads.min(256);

    let grid_size = MTLSize {
        width: num_blocks,
        height: 1,
        depth: 1,
    };
    let threadgroup_size = MTLSize {
        width: tg_size,
        height: 1,
        depth: 1,
    };

    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
    encoder.endEncoding();

    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    let status = command_buffer.status();
    assert_eq!(
        status,
        objc2_metal::MTLCommandBufferStatus::Completed,
        "dequantize_q8_0 command buffer failed with status {:?}. Error: {:?}",
        status,
        command_buffer.error()
    );

    unsafe { read_buffer_slice(&output_buf, total_elements) }
}
