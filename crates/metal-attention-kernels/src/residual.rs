//! Residual (element-wise) addition dispatch: GPU kernel host code.
//!
//! Dispatches the `residual_add` Metal kernel which computes:
//!   output[i] = a[i] + b[i]
//! for each element in the input vectors.

use crate::buffer::{alloc_buffer, alloc_buffer_with_data, read_buffer_slice};
use crate::device::GpuDevice;
use crate::dispatch::{set_buffer, set_bytes};
use crate::pipeline::{PsoCache, PsoKey};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
};

/// Run element-wise residual addition on the GPU.
///
/// Computes `output[i] = a[i] + b[i]` for all elements.
///
/// # Arguments
/// - `device`: GpuDevice with Metal device, command queue, and shader library
/// - `pso_cache`: PSO cache for kernel compilation
/// - `a`: First input vector [dim] as f32
/// - `b`: Second input vector [dim] as f32
///
/// # Returns
/// Output vector [dim] as Vec<f32>
///
/// # Panics
/// Panics if input lengths don't match or the command buffer fails.
pub fn dispatch_residual_add(
    device: &GpuDevice,
    pso_cache: &mut PsoCache,
    a: &[f32],
    b: &[f32],
) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "input lengths must match");
    let dim = a.len();

    // Allocate Metal buffers
    let a_buf = alloc_buffer_with_data(&device.device, a);
    let b_buf = alloc_buffer_with_data(&device.device, b);
    let output_buf = alloc_buffer(&device.device, dim * std::mem::size_of::<f32>());

    // Compile PSO
    let pso_key = PsoKey::simple("residual_add");
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
    set_buffer(&encoder, &a_buf, 0, 0);
    set_buffer(&encoder, &b_buf, 0, 1);
    set_buffer(&encoder, &output_buf, 0, 2);

    // Set dimension constant
    let dim_u32 = dim as u32;
    set_bytes(&encoder, &dim_u32, 3);

    // Dispatch: grid=(dim), threadgroup=(256)
    let grid_size = MTLSize {
        width: dim,
        height: 1,
        depth: 1,
    };
    let threadgroup_size = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };

    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
    encoder.endEncoding();

    // Commit and wait
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    // Check command buffer status
    let status = command_buffer.status();
    assert_eq!(
        status,
        objc2_metal::MTLCommandBufferStatus::Completed,
        "residual_add command buffer failed with status {:?}. Error: {:?}",
        status,
        command_buffer.error()
    );

    // Read back output
    unsafe { read_buffer_slice(&output_buf, dim) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::GpuDevice;
    use crate::pipeline::PsoCache;

    #[test]
    fn test_residual_add_basic() {
        let gpu = GpuDevice::new();
        let mut pso_cache = PsoCache::new(gpu.library.clone());

        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![10.0f32, 20.0, 30.0];
        let expected = vec![11.0f32, 22.0, 33.0];

        let result = dispatch_residual_add(&gpu, &mut pso_cache, &a, &b);

        assert_eq!(result.len(), expected.len());
        for i in 0..expected.len() {
            assert_eq!(
                result[i], expected[i],
                "index {}: GPU={}, expected={}",
                i, result[i], expected[i]
            );
        }
    }
}
