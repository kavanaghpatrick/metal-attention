//! Flash attention dispatch: GPU kernel host code.
//!
//! Loads PSO with function constants for HEAD_DIM, BLOCK_R, BLOCK_C, ALIBI_ENABLED,
//! binds Q/K/V/output buffers + AttentionParams, and dispatches threadgroups.
//!
//! Adapted from proto/src/proto1_flash.rs to use the kernels crate GpuDevice/PsoCache.

use crate::buffer::{alloc_buffer, alloc_buffer_with_data, read_buffer_slice};
use crate::device::GpuDevice;
use crate::pipeline::{PsoCache, PsoKey};
use crate::types::AttentionParams;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
};

/// Run flash attention on the GPU using the Metal kernel.
///
/// Allocates Q/K/V buffers, uploads data, compiles PSO with function constants,
/// dispatches 2D threadgroups (ceil(seq_len/BLOCK_R), num_heads), reads back output.
///
/// # Arguments
/// - `device`: GpuDevice with Metal device, command queue, and shader library
/// - `pso_cache`: PSO cache for kernel compilation (mutably borrowed for caching)
/// - `q`: Query matrix, row-major [seq_len, head_dim], stored as f32
/// - `k`: Key matrix, row-major [seq_len, head_dim], stored as f32
/// - `v`: Value matrix, row-major [seq_len, head_dim], stored as f32
/// - `seq_len`: Number of tokens
/// - `head_dim`: Dimension of each head
/// - `num_heads`: Number of attention heads
///
/// # Returns
/// Output matrix [seq_len, head_dim] as Vec<f32>
///
/// # Panics
/// Panics if input slice lengths don't match seq_len * head_dim, or if the
/// Metal command buffer fails to execute.
pub fn dispatch_flash_attention(
    device: &GpuDevice,
    pso_cache: &mut PsoCache,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    num_heads: usize,
) -> Vec<f32> {
    assert_eq!(q.len(), seq_len * head_dim, "Q length mismatch");
    assert_eq!(k.len(), seq_len * head_dim, "K length mismatch");
    assert_eq!(v.len(), seq_len * head_dim, "V length mismatch");

    let block_r: u32 = 16;
    let block_c: u32 = 64;

    // Create AttentionParams
    let params = AttentionParams::flash(seq_len as u32, head_dim as u32, num_heads as u32);

    // Allocate Metal buffers
    let q_buf = alloc_buffer_with_data(&device.device, q);
    let k_buf = alloc_buffer_with_data(&device.device, k);
    let v_buf = alloc_buffer_with_data(&device.device, v);
    let o_buf = alloc_buffer(&device.device, seq_len * head_dim * std::mem::size_of::<f32>());
    let params_buf = alloc_buffer_with_data(&device.device, std::slice::from_ref(&params));

    // Compile PSO with function constants
    // Function constant indices must match the shader:
    // Index 0 = HEAD_DIM (uint), Index 1 = BLOCK_R (uint), Index 2 = BLOCK_C (uint)
    // Index 4 = ALIBI_ENABLED (bool)
    let pso_key = PsoKey::simple("flash_attention")
        .with_uint(0, head_dim as u32)
        .with_uint(1, block_r)
        .with_uint(2, block_c)
        .with_bool(4, false);

    let pso = pso_cache.get_or_compile(&pso_key);

    // Create command buffer and compute encoder
    let command_buffer = device
        .command_queue
        .commandBuffer()
        .expect("Failed to create command buffer");
    let encoder = command_buffer
        .computeCommandEncoder()
        .expect("Failed to create compute encoder");

    // Set PSO and bind buffers
    encoder.setComputePipelineState(pso);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(&*q_buf), 0, 0); // Q at buffer(0)
        encoder.setBuffer_offset_atIndex(Some(&*k_buf), 0, 1); // K at buffer(1)
        encoder.setBuffer_offset_atIndex(Some(&*v_buf), 0, 2); // V at buffer(2)
        encoder.setBuffer_offset_atIndex(Some(&*o_buf), 0, 3); // O at buffer(3)
        encoder.setBuffer_offset_atIndex(Some(&*params_buf), 0, 4); // params at buffer(4)
    }

    // Dispatch threadgroups:
    // Grid: (ceil(seq_len/BLOCK_R), num_heads, 1)
    // Threadgroup size: 32 threads (one simdgroup)
    let num_row_blocks = (seq_len as u64 + block_r as u64 - 1) / block_r as u64;
    let threadgroups_per_grid = MTLSize {
        width: num_row_blocks as usize,
        height: num_heads,
        depth: 1,
    };
    let threads_per_threadgroup = MTLSize {
        width: 32, // One simdgroup
        height: 1,
        depth: 1,
    };

    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        threadgroups_per_grid,
        threads_per_threadgroup,
    );
    encoder.endEncoding();

    // Commit and wait
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    // Check command buffer status
    let status = command_buffer.status();
    assert_eq!(
        status,
        objc2_metal::MTLCommandBufferStatus::Completed,
        "Flash attention command buffer failed with status {:?}. Error: {:?}",
        status,
        command_buffer.error()
    );

    // Read back output
    unsafe { read_buffer_slice(&o_buf, seq_len * head_dim) }
}
