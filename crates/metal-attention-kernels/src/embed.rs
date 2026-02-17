//! Embedding lookup dispatch: GPU kernel host code.
//!
//! Dispatches the `embedding_lookup` Metal kernel which performs:
//!   output[token_idx, dim] = table[token_id, dim]
//! Simple table lookup for token embeddings.

use crate::buffer::{alloc_buffer, alloc_buffer_with_data, read_buffer_slice};
use crate::device::GpuDevice;
use crate::pipeline::{PsoCache, PsoKey};
use crate::types::LayerParams;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLSize,
};

/// Run embedding lookup on the GPU.
///
/// # Arguments
/// - `device`: GpuDevice with Metal device, command queue, and shader library
/// - `pso_cache`: PSO cache for kernel compilation
/// - `table`: Embedding table [vocab_size, hidden_dim]
/// - `token_ids`: Token IDs [seq_len]
/// - `seq_len`: Number of tokens
/// - `hidden_dim`: Embedding dimension
///
/// # Returns
/// Output tensor [seq_len, hidden_dim] as Vec<f32>
pub fn dispatch_embedding_lookup(
    device: &GpuDevice,
    pso_cache: &mut PsoCache,
    table: &[f32],
    token_ids: &[u32],
    seq_len: usize,
    hidden_dim: usize,
) -> Vec<f32> {
    assert_eq!(token_ids.len(), seq_len, "token_ids length mismatch");

    let total = seq_len * hidden_dim;

    let params = LayerParams {
        hidden_dim: hidden_dim as u32,
        seq_len: seq_len as u32,
        ..Default::default()
    };

    let table_buf = alloc_buffer_with_data(&device.device, table);
    let tokens_buf = alloc_buffer_with_data(&device.device, token_ids);
    let output_buf = alloc_buffer(&device.device, total * std::mem::size_of::<f32>());
    let params_buf = alloc_buffer_with_data(&device.device, std::slice::from_ref(&params));

    let pso_key = PsoKey::simple("embedding_lookup");
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
        encoder.setBuffer_offset_atIndex(Some(&*table_buf), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&*tokens_buf), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&*output_buf), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(&*params_buf), 0, 3);
    }

    let max_threads = pso.maxTotalThreadsPerThreadgroup();
    let tg_size = max_threads.min(256);

    let grid_size = MTLSize {
        width: total,
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
        "embedding_lookup command buffer failed with status {:?}. Error: {:?}",
        status,
        command_buffer.error()
    );

    unsafe { read_buffer_slice(&output_buf, total) }
}
