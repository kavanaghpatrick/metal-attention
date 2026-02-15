//! GPU parallel prefix sum dispatch over D*D matrices.
//!
//! Computes inclusive prefix sums: output[i] = sum(input[0..=i])
//! for each element position across N matrices of dimension [D, D].
//!
//! GPU kernel parallelizes across D*D element positions; each thread
//! performs a sequential scan along the N dimension.

use crate::buffer::{alloc_buffer, alloc_buffer_with_data, read_buffer_slice};
use crate::device::GpuDevice;
use crate::pipeline::{PsoCache, PsoKey};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLSize,
};

/// Compute inclusive prefix sum over N matrices of dimension [D, D] on the GPU.
///
/// Input layout: `[n_matrices, d, d]` as flat f32 array.
/// Output: `[n_matrices, d, d]` where `output[i] = sum(input[0..=i])` element-wise.
///
/// # Arguments
/// - `device`: GpuDevice with Metal device, command queue, and shader library
/// - `pso_cache`: PSO cache for kernel compilation
/// - `input`: Flat input array of `n_matrices * d * d` floats
/// - `n_matrices`: Number of matrices N
/// - `d`: Matrix dimension (D, so each matrix is D*D)
///
/// # Returns
/// Output array `[n_matrices, d, d]` as Vec<f32>
pub fn dispatch_prefix_sum(
    device: &GpuDevice,
    pso_cache: &mut PsoCache,
    input: &[f32],
    n_matrices: usize,
    d: usize,
) -> Vec<f32> {
    let dd = d * d;
    assert_eq!(
        input.len(),
        n_matrices * dd,
        "Input length must be n_matrices * d * d"
    );

    // Allocate Metal buffers
    let input_buf = alloc_buffer_with_data(&device.device, input);
    let output_buf = alloc_buffer(&device.device, n_matrices * dd * std::mem::size_of::<f32>());

    // Params buffer: [n_matrices, d]
    let params: [u32; 2] = [n_matrices as u32, d as u32];
    let params_buf = alloc_buffer_with_data(&device.device, &params);

    // Compile PSO (no function constants needed -- params passed via buffer)
    let pso_key = PsoKey::simple("prefix_sum_matrices");
    let pso = pso_cache.get_or_compile(&pso_key);

    let max_threads = pso.maxTotalThreadsPerThreadgroup();
    let threads_per_tg = dd.min(max_threads);

    // Create command buffer and compute encoder
    let command_buffer = device
        .command_queue
        .commandBuffer()
        .expect("Failed to create command buffer for prefix_sum");
    let encoder = command_buffer
        .computeCommandEncoder()
        .expect("Failed to create compute encoder for prefix_sum");

    encoder.setComputePipelineState(pso);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(&*input_buf), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&*output_buf), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&*params_buf), 0, 2);
    }

    // Dispatch: one thread per D*D element
    let num_threadgroups = dd.div_ceil(threads_per_tg);
    let threadgroups = MTLSize {
        width: num_threadgroups,
        height: 1,
        depth: 1,
    };
    let tg_size = MTLSize {
        width: threads_per_tg,
        height: 1,
        depth: 1,
    };

    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, tg_size);
    encoder.endEncoding();

    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    let status = command_buffer.status();
    assert_eq!(
        status,
        objc2_metal::MTLCommandBufferStatus::Completed,
        "prefix_sum command buffer failed with status {:?}. Error: {:?}",
        status,
        command_buffer.error()
    );

    unsafe { read_buffer_slice(&output_buf, n_matrices * dd) }
}

/// CPU reference implementation of inclusive prefix sum over D*D matrices.
///
/// output[i][r][c] = sum(input[0..=i][r][c]) for all element positions.
pub fn cpu_prefix_sum(input: &[f32], n_matrices: usize, d: usize) -> Vec<f32> {
    let dd = d * d;
    assert_eq!(input.len(), n_matrices * dd);

    let mut output = vec![0.0f32; n_matrices * dd];

    // First matrix: just copy
    output[..dd].copy_from_slice(&input[..dd]);

    // Subsequent matrices: prefix sum element-wise
    for i in 1..n_matrices {
        for elem in 0..dd {
            output[i * dd + elem] = output[(i - 1) * dd + elem] + input[i * dd + elem];
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::GpuDevice;
    use crate::pipeline::PsoCache;

    /// Simple deterministic pseudo-random number generator.
    struct SimpleRng {
        state: u64,
    }

    impl SimpleRng {
        fn new(seed: u64) -> Self {
            Self {
                state: seed.wrapping_add(1),
            }
        }

        fn next_u64(&mut self) -> u64 {
            self.state ^= self.state << 13;
            self.state ^= self.state >> 7;
            self.state ^= self.state << 17;
            self.state
        }

        fn next_f32(&mut self) -> f32 {
            (self.next_u64() & 0xFFFFFF) as f32 / 0xFFFFFF as f32
        }

        fn next_f32_range(&mut self, lo: f32, hi: f32) -> f32 {
            lo + self.next_f32() * (hi - lo)
        }
    }

    #[test]
    fn test_prefix_sum_cpu_reference() {
        // 3 matrices, each 2x2
        let input = vec![
            1.0, 2.0, 3.0, 4.0, // matrix 0
            10.0, 20.0, 30.0, 40.0, // matrix 1
            100.0, 200.0, 300.0, 400.0, // matrix 2
        ];

        let output = cpu_prefix_sum(&input, 3, 2);

        // matrix 0: [1,2,3,4]
        assert_eq!(output[0], 1.0);
        assert_eq!(output[1], 2.0);
        assert_eq!(output[2], 3.0);
        assert_eq!(output[3], 4.0);

        // matrix 1: [1+10, 2+20, 3+30, 4+40] = [11, 22, 33, 44]
        assert_eq!(output[4], 11.0);
        assert_eq!(output[5], 22.0);
        assert_eq!(output[6], 33.0);
        assert_eq!(output[7], 44.0);

        // matrix 2: [11+100, 22+200, 33+300, 44+400] = [111, 222, 333, 444]
        assert_eq!(output[8], 111.0);
        assert_eq!(output[9], 222.0);
        assert_eq!(output[10], 333.0);
        assert_eq!(output[11], 444.0);
    }

    #[test]
    fn test_prefix_sum_gpu_vs_cpu_small() {
        let gpu = GpuDevice::new();
        let mut pso_cache = PsoCache::new(gpu.library.clone());

        let n = 4;
        let d = 4;
        let dd = d * d;

        let mut rng = SimpleRng::new(123);
        let input: Vec<f32> = (0..n * dd).map(|_| rng.next_f32_range(-1.0, 1.0)).collect();

        let cpu_out = cpu_prefix_sum(&input, n, d);
        let gpu_out = dispatch_prefix_sum(&gpu, &mut pso_cache, &input, n, d);

        assert_eq!(cpu_out.len(), gpu_out.len());
        let atol = 1e-4;
        for i in 0..cpu_out.len() {
            let diff = (cpu_out[i] - gpu_out[i]).abs();
            assert!(
                diff < atol,
                "Prefix sum mismatch at index {}: CPU={}, GPU={}, diff={}",
                i,
                cpu_out[i],
                gpu_out[i],
                diff
            );
        }
    }

    #[test]
    fn test_prefix_sum_gpu_vs_cpu_medium() {
        let gpu = GpuDevice::new();
        let mut pso_cache = PsoCache::new(gpu.library.clone());

        // Medium: 16 matrices of 8x8
        let n = 16;
        let d = 8;
        let dd = d * d;

        let mut rng = SimpleRng::new(456);
        let input: Vec<f32> = (0..n * dd).map(|_| rng.next_f32_range(-0.5, 0.5)).collect();

        let cpu_out = cpu_prefix_sum(&input, n, d);
        let gpu_out = dispatch_prefix_sum(&gpu, &mut pso_cache, &input, n, d);

        assert_eq!(cpu_out.len(), gpu_out.len());
        let atol = 1e-3;
        for i in 0..cpu_out.len() {
            let diff = (cpu_out[i] - gpu_out[i]).abs();
            assert!(
                diff < atol,
                "Prefix sum mismatch at index {}: CPU={}, GPU={}, diff={}",
                i,
                cpu_out[i],
                gpu_out[i],
                diff
            );
        }
    }

    #[test]
    fn test_prefix_sum_single_matrix() {
        let gpu = GpuDevice::new();
        let mut pso_cache = PsoCache::new(gpu.library.clone());

        // Single matrix: prefix sum is identity
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gpu_out = dispatch_prefix_sum(&gpu, &mut pso_cache, &input, 1, 2);
        assert_eq!(gpu_out, input);
    }
}
