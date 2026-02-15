//! GQA (Grouped Query Attention) remap dispatch: GPU kernel host code.
//!
//! Expands KV heads to match Q head count by repeating each KV head
//! `group_size` times. Dispatches the `gqa_remap` Metal kernel.

use crate::buffer::{alloc_buffer, alloc_buffer_with_data, read_buffer_slice};
use crate::device::GpuDevice;
use crate::pipeline::{PsoCache, PsoKey};
use crate::types::AttentionParams;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
};

/// Expand KV heads to match Q head count (GQA remap) on the GPU.
///
/// Takes K (or V) with `num_kv_heads` and expands to `num_heads` by
/// repeating each KV head `group_size = num_heads / num_kv_heads` times.
///
/// # Arguments
/// - `device`: GpuDevice with Metal device, command queue, and shader library
/// - `pso_cache`: PSO cache for kernel compilation
/// - `k_full`: Input K/V matrix [num_kv_heads, seq_len, head_dim]
/// - `seq_len`: Number of tokens
/// - `head_dim`: Dimension per head
/// - `num_heads`: Number of Q heads (output)
/// - `num_kv_heads`: Number of KV heads (input)
///
/// # Returns
/// Expanded matrix [num_heads, seq_len, head_dim] as Vec<f32>
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gqa_remap(
    device: &GpuDevice,
    pso_cache: &mut PsoCache,
    k_full: &[f32],
    seq_len: usize,
    head_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
) -> Vec<f32> {
    assert_eq!(
        k_full.len(),
        num_kv_heads * seq_len * head_dim,
        "k_full length mismatch"
    );
    assert!(
        num_heads.is_multiple_of(num_kv_heads),
        "num_heads must be divisible by num_kv_heads"
    );

    let params = AttentionParams::gqa(seq_len as u32, head_dim as u32, num_heads as u32, num_kv_heads as u32);

    let k_buf = alloc_buffer_with_data(&device.device, k_full);
    let out_size = num_heads * seq_len * head_dim * std::mem::size_of::<f32>();
    let out_buf = alloc_buffer(&device.device, out_size);
    let params_buf = alloc_buffer_with_data(&device.device, std::slice::from_ref(&params));

    let pso_key = PsoKey::simple("gqa_remap");
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
        encoder.setBuffer_offset_atIndex(Some(&*k_buf), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&*out_buf), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&*params_buf), 0, 2);
    }

    // Grid: (num_heads, seq_len, head_dim)
    let grid_size = MTLSize {
        width: num_heads,
        height: seq_len,
        depth: head_dim,
    };
    let threadgroup_size = MTLSize {
        width: std::cmp::min(num_heads, 4),
        height: std::cmp::min(seq_len, 4),
        depth: std::cmp::min(head_dim, 16),
    };

    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
    encoder.endEncoding();

    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    let status = command_buffer.status();
    assert_eq!(
        status,
        objc2_metal::MTLCommandBufferStatus::Completed,
        "GQA remap command buffer failed with status {:?}. Error: {:?}",
        status,
        command_buffer.error()
    );

    unsafe { read_buffer_slice(&out_buf, num_heads * seq_len * head_dim) }
}

/// CPU reference implementation of GQA remap for testing.
pub fn cpu_gqa_remap(
    k_full: &[f32],
    seq_len: usize,
    head_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
) -> Vec<f32> {
    let group_size = num_heads / num_kv_heads;
    let mut expanded = vec![0.0f32; num_heads * seq_len * head_dim];

    for q_head in 0..num_heads {
        let kv_head = q_head / group_size;
        for token in 0..seq_len {
            for dim in 0..head_dim {
                let src = (kv_head * seq_len + token) * head_dim + dim;
                let dst = (q_head * seq_len + token) * head_dim + dim;
                expanded[dst] = k_full[src];
            }
        }
    }

    expanded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gqa_remap_gpu_vs_cpu() {
        let seq_len = 4;
        let head_dim = 8;
        let num_heads = 8;
        let num_kv_heads = 2;

        // Generate deterministic test data
        let k_full: Vec<f32> = (0..num_kv_heads * seq_len * head_dim)
            .map(|i| i as f32 * 0.01)
            .collect();

        // CPU reference
        let cpu_out = cpu_gqa_remap(&k_full, seq_len, head_dim, num_heads, num_kv_heads);

        // GPU dispatch
        let gpu = GpuDevice::new();
        let mut pso_cache = PsoCache::new(gpu.library.clone());
        let gpu_out = dispatch_gqa_remap(
            &gpu,
            &mut pso_cache,
            &k_full,
            seq_len,
            head_dim,
            num_heads,
            num_kv_heads,
        );

        assert_eq!(cpu_out.len(), gpu_out.len());

        let atol = 1e-6;
        for i in 0..cpu_out.len() {
            let diff = (cpu_out[i] - gpu_out[i]).abs();
            assert!(
                diff < atol,
                "GQA remap mismatch at {}: cpu={}, gpu={}, diff={}",
                i, cpu_out[i], gpu_out[i], diff
            );
        }
    }

    #[test]
    fn test_gqa_remap_mha() {
        // When num_heads == num_kv_heads, output == input (identity remap)
        let seq_len = 4;
        let head_dim = 8;
        let num_heads = 4;
        let num_kv_heads = 4;

        let k_full: Vec<f32> = (0..num_kv_heads * seq_len * head_dim)
            .map(|i| i as f32 * 0.1)
            .collect();

        let cpu_out = cpu_gqa_remap(&k_full, seq_len, head_dim, num_heads, num_kv_heads);
        assert_eq!(cpu_out, k_full, "MHA remap should be identity");
    }
}
