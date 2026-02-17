//! SwiGLU FFN activation dispatch: GPU kernel host code.
//!
//! Dispatches the `ffn_silu` Metal kernel which computes the element-wise part:
//!   output[i] = silu(gate[i]) * up[i]
//! where silu(x) = x * sigmoid(x).
//!
//! The matrix multiplications (gate_proj, up_proj, down_proj) are done separately
//! using the matmul kernel; this kernel handles only the activation + element-wise multiply.

use crate::buffer::{alloc_buffer, alloc_buffer_with_data, read_buffer_slice};
use crate::device::GpuDevice;
use crate::pipeline::{PsoCache, PsoKey};
use crate::types::LayerParams;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLSize,
};

/// Run SwiGLU activation on the GPU (element-wise silu(gate) * up).
///
/// # Arguments
/// - `device`: GpuDevice with Metal device, command queue, and shader library
/// - `pso_cache`: PSO cache for kernel compilation
/// - `gate`: Gate projection output [num_tokens, intermediate_dim]
/// - `up`: Up projection output [num_tokens, intermediate_dim]
/// - `num_tokens`: Number of tokens
/// - `intermediate_dim`: FFN intermediate dimension
///
/// # Returns
/// Output tensor [num_tokens, intermediate_dim] as Vec<f32>
pub fn dispatch_ffn_silu(
    device: &GpuDevice,
    pso_cache: &mut PsoCache,
    gate: &[f32],
    up: &[f32],
    num_tokens: usize,
    intermediate_dim: usize,
) -> Vec<f32> {
    let total = num_tokens * intermediate_dim;
    assert_eq!(gate.len(), total, "gate length mismatch");
    assert_eq!(up.len(), total, "up length mismatch");

    // Build LayerParams (ffn_silu uses seq_len and intermediate_dim)
    let params = LayerParams {
        intermediate_dim: intermediate_dim as u32,
        seq_len: num_tokens as u32,
        ..Default::default()
    };

    // Allocate Metal buffers
    // Note: input buffer(0) is unused in the kernel but required by the signature
    let input_buf = alloc_buffer(&device.device, 4); // dummy, unused
    let gate_buf = alloc_buffer_with_data(&device.device, gate);
    let up_buf = alloc_buffer_with_data(&device.device, up);
    let output_buf = alloc_buffer(&device.device, total * std::mem::size_of::<f32>());
    let params_buf = alloc_buffer_with_data(&device.device, std::slice::from_ref(&params));

    let pso_key = PsoKey::simple("ffn_silu");
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
        encoder.setBuffer_offset_atIndex(Some(&*gate_buf), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&*up_buf), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(&*output_buf), 0, 3);
        encoder.setBuffer_offset_atIndex(Some(&*params_buf), 0, 4);
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
        "ffn_silu command buffer failed with status {:?}. Error: {:?}",
        status,
        command_buffer.error()
    );

    unsafe { read_buffer_slice(&output_buf, total) }
}
