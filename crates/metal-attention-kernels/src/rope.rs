//! RoPE (Rotary Position Embeddings) dispatch: GPU kernel host code.
//!
//! Applies rotary position encoding to Q and K tensors in-place.
//! Dispatches the `apply_rope` Metal kernel with AttentionParams.

use crate::buffer::alloc_buffer_with_data;
use crate::device::GpuDevice;
use crate::pipeline::{PsoCache, PsoKey};
use crate::types::AttentionParams;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
};

/// Apply RoPE (Rotary Position Embeddings) on the GPU.
///
/// Modifies Q and K tensors in-place by applying rotary position encoding.
/// Each (token, dim_pair) gets a rotation based on position and frequency.
///
/// # Arguments
/// - `device`: GpuDevice with Metal device, command queue, and shader library
/// - `pso_cache`: PSO cache for kernel compilation
/// - `q`: Query matrix [seq_len, head_dim], modified in-place
/// - `k`: Key matrix [seq_len, head_dim], modified in-place
/// - `seq_len`: Number of tokens
/// - `head_dim`: Dimension per head (must be even)
///
/// # Returns
/// Tuple of (q_rotated, k_rotated) as Vec<f32>
pub fn dispatch_rope(
    device: &GpuDevice,
    pso_cache: &mut PsoCache,
    q: &[f32],
    k: &[f32],
    seq_len: usize,
    head_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(q.len(), seq_len * head_dim, "Q length mismatch");
    assert_eq!(k.len(), seq_len * head_dim, "K length mismatch");
    assert_eq!(head_dim % 2, 0, "head_dim must be even for RoPE");

    let params = AttentionParams {
        seq_len: seq_len as u32,
        head_dim: head_dim as u32,
        ..AttentionParams::default()
    };

    // Allocate buffers (mutable -- rope modifies in-place)
    let q_buf = alloc_buffer_with_data(&device.device, q);
    let k_buf = alloc_buffer_with_data(&device.device, k);
    let params_buf = alloc_buffer_with_data(&device.device, std::slice::from_ref(&params));

    // Compile PSO
    let pso_key = PsoKey::simple("apply_rope");
    let pso = pso_cache.get_or_compile(&pso_key);

    // Create command buffer and encoder
    let command_buffer = device
        .command_queue
        .commandBuffer()
        .expect("Failed to create command buffer");
    let encoder = command_buffer
        .computeCommandEncoder()
        .expect("Failed to create compute encoder");

    encoder.setComputePipelineState(pso);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(&*q_buf), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&*k_buf), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&*params_buf), 0, 2);
    }

    // Dispatch: grid (seq_len, head_dim/2), threads per threadgroup chosen by Metal
    let grid_size = MTLSize {
        width: seq_len,
        height: head_dim / 2,
        depth: 1,
    };
    let threadgroup_size = MTLSize {
        width: std::cmp::min(seq_len, 16),
        height: std::cmp::min(head_dim / 2, 16),
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
        "RoPE command buffer failed with status {:?}. Error: {:?}",
        status,
        command_buffer.error()
    );

    // Read back modified buffers
    let q_out = unsafe { crate::buffer::read_buffer_slice(&q_buf, seq_len * head_dim) };
    let k_out = unsafe { crate::buffer::read_buffer_slice(&k_buf, seq_len * head_dim) };

    (q_out, k_out)
}

/// CPU reference implementation of RoPE for testing.
pub fn cpu_rope(q: &mut [f32], k: &mut [f32], seq_len: usize, head_dim: usize) {
    let theta_base: f32 = 10000.0;
    for token in 0..seq_len {
        for pair in 0..(head_dim / 2) {
            let angle = token as f32 / theta_base.powf(2.0 * pair as f32 / head_dim as f32);
            let cos_a = angle.cos();
            let sin_a = angle.sin();

            let idx0 = token * head_dim + 2 * pair;
            let idx1 = idx0 + 1;

            // Rotate Q
            let q0 = q[idx0];
            let q1 = q[idx1];
            q[idx0] = q0 * cos_a - q1 * sin_a;
            q[idx1] = q0 * sin_a + q1 * cos_a;

            // Rotate K
            let k0 = k[idx0];
            let k1 = k[idx1];
            k[idx0] = k0 * cos_a - k1 * sin_a;
            k[idx1] = k0 * sin_a + k1 * cos_a;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple xorshift RNG for test data generation.
    struct TestRng(u64);
    impl TestRng {
        fn new(seed: u64) -> Self {
            Self(seed.wrapping_add(1))
        }
        fn next_f32(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 & 0xFFFFFF) as f32 / 0xFFFFFF as f32 * 2.0 - 1.0
        }
    }

    #[test]
    fn test_rope_gpu_vs_cpu() {
        let seq_len = 8;
        let head_dim = 16;
        let mut rng = TestRng::new(42);

        let q: Vec<f32> = (0..seq_len * head_dim).map(|_| rng.next_f32()).collect();
        let k: Vec<f32> = (0..seq_len * head_dim).map(|_| rng.next_f32()).collect();

        // CPU reference
        let mut q_cpu = q.clone();
        let mut k_cpu = k.clone();
        cpu_rope(&mut q_cpu, &mut k_cpu, seq_len, head_dim);

        // GPU dispatch
        let gpu = GpuDevice::new();
        let mut pso_cache = PsoCache::new(gpu.library.clone());
        let (q_gpu, k_gpu) = dispatch_rope(&gpu, &mut pso_cache, &q, &k, seq_len, head_dim);

        // Compare
        let atol = 1e-4;
        for i in 0..q_cpu.len() {
            let diff = (q_cpu[i] - q_gpu[i]).abs();
            assert!(
                diff < atol,
                "Q mismatch at {}: cpu={}, gpu={}, diff={}",
                i,
                q_cpu[i],
                q_gpu[i],
                diff
            );
        }
        for i in 0..k_cpu.len() {
            let diff = (k_cpu[i] - k_gpu[i]).abs();
            assert!(
                diff < atol,
                "K mismatch at {}: cpu={}, gpu={}, diff={}",
                i,
                k_cpu[i],
                k_gpu[i],
                diff
            );
        }
    }
}
