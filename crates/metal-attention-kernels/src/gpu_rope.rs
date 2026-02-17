//! GPU RoPE (Rotary Position Embeddings) single-token decode dispatch.
//!
//! Dispatches the `rope_apply` Metal kernel which applies rotary position
//! encoding in-place on a flat qk buffer [num_heads * head_dim].
//! Designed for single-token decode where position is a scalar.
//!
//! Uses interleaved pair layout: idx0 = head*head_dim + 2*pair, idx1 = idx0+1.

use crate::buffer::{alloc_buffer_with_data, read_buffer_slice};
use crate::device::GpuDevice;
use crate::dispatch::{set_buffer, set_bytes};
use crate::pipeline::{PsoCache, PsoKey};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
};

/// Apply RoPE position encoding on the GPU for single-token decode.
///
/// Modifies the qk buffer in-place by applying rotary position encoding
/// at the given position. Each (head, pair) gets a rotation based on
/// position and frequency.
///
/// # Arguments
/// - `device`: GpuDevice with Metal device, command queue, and shader library
/// - `pso_cache`: PSO cache for kernel compilation
/// - `qk`: Input vector [num_heads * head_dim] as f32, modified in-place via GPU
/// - `num_heads`: Number of attention heads
/// - `head_dim`: Dimension per head (must be even)
/// - `position`: Token position for angle computation
/// - `theta`: RoPE frequency base (typically 10000.0)
///
/// # Returns
/// Rotated vector [num_heads * head_dim] as Vec<f32>
///
/// # Panics
/// Panics if qk length doesn't match num_heads * head_dim, head_dim is odd,
/// or the command buffer fails.
pub fn dispatch_rope_apply(
    device: &GpuDevice,
    pso_cache: &mut PsoCache,
    qk: &[f32],
    num_heads: usize,
    head_dim: usize,
    position: usize,
    theta: f32,
) -> Vec<f32> {
    assert_eq!(
        qk.len(),
        num_heads * head_dim,
        "qk length mismatch: got {}, expected {}",
        qk.len(),
        num_heads * head_dim
    );
    assert_eq!(head_dim % 2, 0, "head_dim must be even for RoPE");

    // Allocate Metal buffer (mutable -- RoPE modifies in-place)
    let qk_buf = alloc_buffer_with_data(&device.device, qk);

    // Compile PSO
    let pso_key = PsoKey::simple("rope_apply");
    let pso = pso_cache.get_or_compile(&pso_key);

    // Create command buffer and compute encoder
    let command_buffer = device
        .command_queue
        .commandBuffer()
        .expect("Failed to create command buffer");
    let encoder = command_buffer
        .computeCommandEncoder()
        .expect("Failed to create compute encoder");

    // Set PSO and bind buffers/constants
    encoder.setComputePipelineState(pso);
    set_buffer(&encoder, &qk_buf, 0, 0);

    let num_heads_u32 = num_heads as u32;
    let head_dim_u32 = head_dim as u32;
    let position_u32 = position as u32;
    set_bytes(&encoder, &num_heads_u32, 1);
    set_bytes(&encoder, &head_dim_u32, 2);
    set_bytes(&encoder, &position_u32, 3);
    set_bytes(&encoder, &theta, 4);

    // Dispatch: grid=(num_heads * head_dim / 2), threadgroup=(32)
    let total_pairs = num_heads * head_dim / 2;
    let grid_size = MTLSize {
        width: total_pairs,
        height: 1,
        depth: 1,
    };
    let threadgroup_size = MTLSize {
        width: 32,
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
        "rope_apply command buffer failed with status {:?}. Error: {:?}",
        status,
        command_buffer.error()
    );

    // Read back modified buffer
    unsafe { read_buffer_slice(&qk_buf, num_heads * head_dim) }
}

/// CPU reference implementation of single-token RoPE for testing.
///
/// Applies RoPE in-place on a flat qk buffer [num_heads * head_dim]
/// using interleaved pair layout matching the GPU kernel.
pub fn cpu_rope_apply(
    qk: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    position: usize,
    theta: f32,
) {
    let half_dim = head_dim / 2;
    for head in 0..num_heads {
        for pair in 0..half_dim {
            let angle = position as f32 / theta.powf(2.0 * pair as f32 / head_dim as f32);
            let cos_a = angle.cos();
            let sin_a = angle.sin();

            let idx0 = head * head_dim + 2 * pair;
            let idx1 = idx0 + 1;

            let v0 = qk[idx0];
            let v1 = qk[idx1];
            qk[idx0] = v0 * cos_a - v1 * sin_a;
            qk[idx1] = v0 * sin_a + v1 * cos_a;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::GpuDevice;
    use crate::pipeline::PsoCache;

    const THETA: f32 = 10000.0;

    #[test]
    fn test_rope_apply_position_0_identity() {
        // At position=0, angle=0 for all pairs => cos=1, sin=0 => output == input
        let gpu = GpuDevice::new();
        let mut pso_cache = PsoCache::new(gpu.library.clone());

        let num_heads = 2;
        let head_dim = 8;
        let position = 0;

        // Use distinct values so we can verify identity
        let qk: Vec<f32> = (0..num_heads * head_dim)
            .map(|i| (i as f32 + 1.0) * 0.1)
            .collect();

        let result = dispatch_rope_apply(
            &gpu,
            &mut pso_cache,
            &qk,
            num_heads,
            head_dim,
            position,
            THETA,
        );

        assert_eq!(result.len(), qk.len());
        for i in 0..qk.len() {
            let diff = (result[i] - qk[i]).abs();
            assert!(
                diff < 1e-5,
                "Position 0 should be identity. Index {}: input={}, output={}, diff={}",
                i,
                qk[i],
                result[i],
                diff
            );
        }
    }

    #[test]
    fn test_rope_apply_position_1_vs_cpu() {
        // At position=1, verify GPU matches CPU reference
        let gpu = GpuDevice::new();
        let mut pso_cache = PsoCache::new(gpu.library.clone());

        let num_heads = 4;
        let head_dim = 8;
        let position = 1;

        // Use distinct values
        let qk: Vec<f32> = (0..num_heads * head_dim)
            .map(|i| (i as f32 - 16.0) * 0.05)
            .collect();

        // GPU result
        let gpu_result = dispatch_rope_apply(
            &gpu,
            &mut pso_cache,
            &qk,
            num_heads,
            head_dim,
            position,
            THETA,
        );

        // CPU reference
        let mut cpu_result = qk.clone();
        cpu_rope_apply(&mut cpu_result, num_heads, head_dim, position, THETA);

        assert_eq!(gpu_result.len(), cpu_result.len());
        let mut max_diff = 0.0f32;
        for i in 0..cpu_result.len() {
            let diff = (gpu_result[i] - cpu_result[i]).abs();
            max_diff = max_diff.max(diff);
            assert!(
                diff < 1e-4,
                "Mismatch at index {}: GPU={}, CPU={}, diff={}",
                i,
                gpu_result[i],
                cpu_result[i],
                diff
            );
        }

        // Verify rotation actually happened (position 1 should differ from input for pair 0)
        // pair=0: angle = 1.0 / 10000^0 = 1.0, cos(1)~0.5403, sin(1)~0.8415
        let diff_from_input: f32 = gpu_result
            .iter()
            .zip(qk.iter())
            .map(|(g, q)| (g - q).abs())
            .sum();
        assert!(
            diff_from_input > 0.01,
            "Rotation should change values at position 1, but total diff={}",
            diff_from_input
        );

        eprintln!(
            "test_rope_apply_position_1_vs_cpu: max_diff={:.6}, rotation verified",
            max_diff
        );
    }

    #[test]
    fn test_rope_apply_multi_position() {
        // Verify rotation angles at positions 0, 1, 10, 100
        let gpu = GpuDevice::new();
        let mut pso_cache = PsoCache::new(gpu.library.clone());

        let num_heads = 4;
        let head_dim = 64; // realistic SmolLM head_dim

        let qk: Vec<f32> = (0..num_heads * head_dim)
            .map(|i| (i as f32 - 128.0) * 0.01)
            .collect();

        for position in [0usize, 1, 10, 100] {
            let gpu_result = dispatch_rope_apply(
                &gpu,
                &mut pso_cache,
                &qk,
                num_heads,
                head_dim,
                position,
                THETA,
            );

            let mut cpu_result = qk.clone();
            cpu_rope_apply(&mut cpu_result, num_heads, head_dim, position, THETA);

            assert_eq!(gpu_result.len(), cpu_result.len());
            let mut max_diff = 0.0f32;
            for i in 0..cpu_result.len() {
                let diff = (gpu_result[i] - cpu_result[i]).abs();
                max_diff = max_diff.max(diff);
                assert!(
                    diff < 1e-4,
                    "position={}, index {}: GPU={}, CPU={}, diff={}",
                    position,
                    i,
                    gpu_result[i],
                    cpu_result[i],
                    diff
                );
            }

            // At position 0, output should be identity (cos=1, sin=0)
            if position == 0 {
                for i in 0..qk.len() {
                    let diff = (gpu_result[i] - qk[i]).abs();
                    assert!(
                        diff < 1e-5,
                        "position=0 should be identity, index {}: input={}, output={}, diff={}",
                        i,
                        qk[i],
                        gpu_result[i],
                        diff
                    );
                }
            } else {
                // At non-zero positions, output should differ from input
                let total_diff: f32 = gpu_result
                    .iter()
                    .zip(qk.iter())
                    .map(|(g, q)| (g - q).abs())
                    .sum();
                assert!(
                    total_diff > 0.01,
                    "position={}: rotation should change values, but total diff={}",
                    position,
                    total_diff
                );
            }

            eprintln!(
                "test_rope_apply_multi_position: position={}, max_diff={:.6}",
                position, max_diff
            );
        }
    }
}
