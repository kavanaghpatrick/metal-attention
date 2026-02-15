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
    assert_eq!(
        input.len(),
        num_tokens * hidden_dim,
        "input length mismatch"
    );
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

/// Run optimized RMSNorm with simdgroup cooperative reduction on the GPU.
///
/// Computes `output[i] = (input[i] / rms) * weight[i]`
/// where `rms = sqrt(mean(input^2) + eps)`.
///
/// Uses 32 threads (1 simdgroup) with `simd_sum` for cooperative reduction.
/// Each thread processes `hidden_dim/32` elements in a strided loop.
///
/// # Arguments
/// - `device`: GpuDevice with Metal device, command queue, and shader library
/// - `pso_cache`: PSO cache for kernel compilation
/// - `input`: Input vector [hidden_dim] as f32
/// - `weight`: Normalization weights [hidden_dim] as f32
/// - `eps`: Epsilon for numerical stability
///
/// # Returns
/// Output vector [hidden_dim] as Vec<f32>
///
/// # Panics
/// Panics if input and weight lengths don't match or the command buffer fails.
pub fn dispatch_rmsnorm_optimized(
    device: &GpuDevice,
    pso_cache: &mut PsoCache,
    input: &[f32],
    weight: &[f32],
    eps: f32,
) -> Vec<f32> {
    let hidden_dim = input.len();
    assert_eq!(
        input.len(),
        weight.len(),
        "input and weight length mismatch"
    );

    // Allocate Metal buffers
    let input_buf = alloc_buffer_with_data(&device.device, input);
    let weight_buf = alloc_buffer_with_data(&device.device, weight);
    let output_buf = alloc_buffer(&device.device, hidden_dim * std::mem::size_of::<f32>());

    // Compile PSO
    let pso_key = PsoKey::simple("rmsnorm_optimized");
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
        encoder.setBuffer_offset_atIndex(Some(&*input_buf), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&*weight_buf), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&*output_buf), 0, 2);
    }

    // Set dimension and epsilon constants via setBytes
    let hidden_dim_u32 = hidden_dim as u32;
    let size_u32 = std::mem::size_of::<u32>();
    let size_f32 = std::mem::size_of::<f32>();
    unsafe {
        let ptr_dim = std::ptr::NonNull::new(
            &hidden_dim_u32 as *const u32 as *mut std::ffi::c_void,
        )
        .expect("dim pointer is null");
        encoder.setBytes_length_atIndex(ptr_dim, size_u32, 3);

        let ptr_eps =
            std::ptr::NonNull::new(&eps as *const f32 as *mut std::ffi::c_void)
                .expect("eps pointer is null");
        encoder.setBytes_length_atIndex(ptr_eps, size_f32, 4);
    }

    // Dispatch: grid=(1,1,1), threadgroup=(32,1,1)
    let grid_size = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    let threadgroup_size = MTLSize {
        width: 32,
        height: 1,
        depth: 1,
    };

    encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
    encoder.endEncoding();

    // Commit and wait
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    // Check command buffer status
    let status = command_buffer.status();
    assert_eq!(
        status,
        objc2_metal::MTLCommandBufferStatus::Completed,
        "rmsnorm_optimized command buffer failed with status {:?}. Error: {:?}",
        status,
        command_buffer.error()
    );

    unsafe { read_buffer_slice(&output_buf, hidden_dim) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::GpuDevice;
    use crate::pipeline::PsoCache;

    /// CPU reference RMSNorm: output[i] = (input[i] / rms) * weight[i]
    /// where rms = sqrt(mean(input^2) + eps)
    fn cpu_rmsnorm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        let n = input.len();
        let ss: f32 = input.iter().map(|x| x * x).sum();
        let rms = (ss / n as f32 + eps).sqrt();
        input
            .iter()
            .zip(weight.iter())
            .map(|(x, w)| (x / rms) * w)
            .collect()
    }

    #[test]
    fn test_rmsnorm_optimized_known_vector() {
        let gpu = GpuDevice::new();
        let mut pso_cache = PsoCache::new(gpu.library.clone());

        // Use [3.0, 4.0] — sum of squares = 9 + 16 = 25, mean_ss = 12.5
        // rms = sqrt(12.5 + 1e-5) ≈ 3.535534...
        // With weight = [1.0, 1.0]: output = [3/rms, 4/rms]
        let hidden_dim = 32; // must be >= 32 for 32-thread dispatch
        let mut input = vec![0.0f32; hidden_dim];
        input[0] = 3.0;
        input[1] = 4.0;
        let weight = vec![1.0f32; hidden_dim];
        let eps = 1e-5f32;

        let expected = cpu_rmsnorm(&input, &weight, eps);
        let result = dispatch_rmsnorm_optimized(&gpu, &mut pso_cache, &input, &weight, eps);

        assert_eq!(result.len(), hidden_dim);
        for i in 0..hidden_dim {
            let diff = (result[i] - expected[i]).abs();
            assert!(
                diff < 1e-5,
                "index {}: GPU={}, CPU={}, diff={}",
                i,
                result[i],
                expected[i],
                diff
            );
        }

        // Verify analytically: ss=25, mean_ss=25/32, rms=sqrt(25/32+1e-5)
        let mean_ss = 25.0f32 / 32.0;
        let rms = (mean_ss + eps).sqrt();
        let expected_0 = 3.0 / rms;
        let expected_1 = 4.0 / rms;
        assert!(
            (result[0] - expected_0).abs() < 1e-5,
            "element 0: GPU={}, analytical={}",
            result[0],
            expected_0
        );
        assert!(
            (result[1] - expected_1).abs() < 1e-5,
            "element 1: GPU={}, analytical={}",
            result[1],
            expected_1
        );

        eprintln!(
            "test_rmsnorm_optimized_known_vector: rms={:.6}, out[0]={:.6}, out[1]={:.6}",
            rms, result[0], result[1]
        );
    }
}
