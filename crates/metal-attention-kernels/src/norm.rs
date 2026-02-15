//! RMSNorm dispatch: GPU kernel host code.
//!
//! Dispatches the `rmsnorm` Metal kernel which computes:
//!   output[i] = (input[i] / rms) * weight[i]
//! where rms = sqrt(mean(input^2) + eps).

use crate::buffer::{alloc_buffer, alloc_buffer_with_data, read_buffer_slice};
use crate::device::GpuDevice;
use crate::pipeline::{PsoCache, PsoKey};
use crate::types::LayerParams;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLSize,
};

/// Run RMSNorm on the GPU.
///
/// # Arguments
/// - `device`: GpuDevice with Metal device, command queue, and shader library
/// - `pso_cache`: PSO cache for kernel compilation
/// - `input`: Input tensor [num_tokens, hidden_dim]
/// - `weight`: Normalization weights [hidden_dim]
/// - `num_tokens`: Number of tokens (rows)
/// - `hidden_dim`: Hidden dimension (columns)
/// - `eps`: Epsilon for numerical stability
///
/// # Returns
/// Output tensor [num_tokens, hidden_dim] as Vec<f32>
pub fn dispatch_rmsnorm(
    device: &GpuDevice,
    pso_cache: &mut PsoCache,
    input: &[f32],
    weight: &[f32],
    num_tokens: usize,
    hidden_dim: usize,
    eps: f32,
) -> Vec<f32> {
    assert_eq!(input.len(), num_tokens * hidden_dim, "input length mismatch");
    assert_eq!(weight.len(), hidden_dim, "weight length mismatch");

    let total_elements = num_tokens * hidden_dim;

    // Build LayerParams
    let params = LayerParams {
        hidden_dim: hidden_dim as u32,
        rms_norm_eps: eps,
        seq_len: num_tokens as u32,
        ..Default::default()
    };

    // Allocate Metal buffers
    let input_buf = alloc_buffer_with_data(&device.device, input);
    let weight_buf = alloc_buffer_with_data(&device.device, weight);
    let output_buf = alloc_buffer(&device.device, total_elements * std::mem::size_of::<f32>());
    let params_buf = alloc_buffer_with_data(&device.device, std::slice::from_ref(&params));

    // Compile PSO (no function constants needed for this simple version)
    let pso_key = PsoKey::simple("rmsnorm");
    let pso = pso_cache.get_or_compile(&pso_key);

    // Create command buffer and compute encoder
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
        encoder.setBuffer_offset_atIndex(Some(&*weight_buf), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&*output_buf), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(&*params_buf), 0, 3);
    }

    // Dispatch: one thread per element
    let max_threads = pso.maxTotalThreadsPerThreadgroup();
    let tg_size = max_threads.min(256);

    let grid_size = MTLSize {
        width: total_elements,
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
        "rmsnorm command buffer failed with status {:?}. Error: {:?}",
        status,
        command_buffer.error()
    );

    unsafe { read_buffer_slice(&output_buf, total_elements) }
}
