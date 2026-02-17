//! Decode (autoregressive) attention dispatch: GPU kernel host code.
//!
//! Dispatches the `decode_attention` Metal kernel which computes single-token
//! attention: softmax(Q . K^T / scale) . V for one query token against the
//! full KV cache. Supports Grouped Query Attention (GQA).
//!
//! Grid=(num_heads), Threadgroup=(32) -- one simdgroup per query head.

use crate::buffer::{alloc_buffer, alloc_buffer_with_data, read_buffer_slice};
use crate::device::GpuDevice;
use crate::dispatch::{set_buffer, set_bytes};
use crate::pipeline::{PsoCache, PsoKey};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
};

/// Run decode attention on the GPU for a single query token.
///
/// Computes: output = softmax(Q . K^T / scale) . V
///
/// # Arguments
/// - `device`: GpuDevice with Metal device, command queue, and shader library
/// - `pso_cache`: PSO cache for kernel compilation
/// - `q`: Query vector [num_heads * head_dim] as f32
/// - `k_cache`: Key cache [kv_len, num_kv_heads * head_dim] as f32 (row-major)
/// - `v_cache`: Value cache [kv_len, num_kv_heads * head_dim] as f32 (row-major)
/// - `num_heads`: Number of query attention heads
/// - `num_kv_heads`: Number of KV heads (for GQA, num_heads / num_kv_heads = group_size)
/// - `head_dim`: Dimension per head
/// - `kv_len`: Number of KV cache positions filled
///
/// # Returns
/// Output vector [num_heads * head_dim] as Vec<f32>
///
/// # Panics
/// Panics if dimensions are inconsistent, kv_len exceeds 2048, num_heads is not
/// divisible by num_kv_heads, or the command buffer fails.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_decode_attention(
    device: &GpuDevice,
    pso_cache: &mut PsoCache,
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
) -> Vec<f32> {
    assert_eq!(
        q.len(),
        num_heads * head_dim,
        "q length mismatch: got {}, expected {}",
        q.len(),
        num_heads * head_dim
    );
    let kv_dim = num_kv_heads * head_dim;
    assert_eq!(
        k_cache.len(),
        kv_len * kv_dim,
        "k_cache length mismatch: got {}, expected {}",
        k_cache.len(),
        kv_len * kv_dim
    );
    assert_eq!(
        v_cache.len(),
        kv_len * kv_dim,
        "v_cache length mismatch: got {}, expected {}",
        v_cache.len(),
        kv_len * kv_dim
    );
    assert!(kv_len <= 2048, "kv_len {} exceeds max 2048", kv_len);
    assert_eq!(
        num_heads % num_kv_heads,
        0,
        "num_heads ({}) must be divisible by num_kv_heads ({})",
        num_heads,
        num_kv_heads
    );

    let output_len = num_heads * head_dim;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    // Allocate Metal buffers
    let q_buf = alloc_buffer_with_data(&device.device, q);
    let k_buf = alloc_buffer_with_data(&device.device, k_cache);
    let v_buf = alloc_buffer_with_data(&device.device, v_cache);
    let out_buf = alloc_buffer(&device.device, output_len * std::mem::size_of::<f32>());

    // Compile PSO
    let pso_key = PsoKey::simple("decode_attention");
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
    set_buffer(&encoder, &q_buf, 0, 0);
    set_buffer(&encoder, &k_buf, 0, 1);
    set_buffer(&encoder, &v_buf, 0, 2);
    set_buffer(&encoder, &out_buf, 0, 3);

    // Set constants via setBytes
    let num_heads_u32 = num_heads as u32;
    let num_kv_heads_u32 = num_kv_heads as u32;
    let head_dim_u32 = head_dim as u32;
    let kv_len_u32 = kv_len as u32;
    set_bytes(&encoder, &num_heads_u32, 4);
    set_bytes(&encoder, &num_kv_heads_u32, 5);
    set_bytes(&encoder, &head_dim_u32, 6);
    set_bytes(&encoder, &kv_len_u32, 7);
    set_bytes(&encoder, &scale, 8);

    // Dispatch: one threadgroup per query head, 32 threads per threadgroup
    let threadgroups_per_grid = MTLSize {
        width: num_heads,
        height: 1,
        depth: 1,
    };
    let threads_per_threadgroup = MTLSize {
        width: 32,
        height: 1,
        depth: 1,
    };

    encoder
        .dispatchThreadgroups_threadsPerThreadgroup(threadgroups_per_grid, threads_per_threadgroup);
    encoder.endEncoding();

    // Commit and wait
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    // Check command buffer status
    let status = command_buffer.status();
    assert_eq!(
        status,
        objc2_metal::MTLCommandBufferStatus::Completed,
        "decode_attention command buffer failed with status {:?}. Error: {:?}",
        status,
        command_buffer.error()
    );

    // Read back output
    unsafe { read_buffer_slice(&out_buf, output_len) }
}

/// CPU reference implementation of decode attention for testing.
///
/// Computes: output = softmax(Q . K^T / sqrt(head_dim)) . V
/// with GQA support (multiple Q heads share KV heads).
#[cfg(test)]
fn cpu_decode_attention(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
) -> Vec<f32> {
    let group_size = num_heads / num_kv_heads;
    let kv_dim = num_kv_heads * head_dim;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; num_heads * head_dim];

    for head in 0..num_heads {
        let kv_head = head / group_size;
        let q_offset = head * head_dim;

        // Compute Q.K^T scores
        let mut scores = vec![0.0f32; kv_len];
        for pos in 0..kv_len {
            let k_offset = pos * kv_dim + kv_head * head_dim;
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[q_offset + d] * k_cache[k_offset + d];
            }
            scores[pos] = dot * scale;
        }

        // Softmax with numerical stability
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max_score).exp();
            sum_exp += *s;
        }
        for s in scores.iter_mut() {
            *s /= sum_exp;
        }

        // Weighted V sum
        let out_offset = head * head_dim;
        for d in 0..head_dim {
            let mut acc = 0.0f32;
            for pos in 0..kv_len {
                let v_offset = pos * kv_dim + kv_head * head_dim;
                acc += scores[pos] * v_cache[v_offset + d];
            }
            output[out_offset + d] = acc;
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::GpuDevice;
    use crate::pipeline::PsoCache;

    #[test]
    fn test_decode_attention_basic() {
        // 2 heads, head_dim=4, kv_len=3, no GQA (num_kv_heads=2)
        let gpu = GpuDevice::new();
        let mut pso_cache = PsoCache::new(gpu.library.clone());

        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let kv_len = 3;

        // Q: [2 * 4] = [8] -- two query heads
        let q = vec![
            1.0, 0.0, 0.0, 0.0, // head 0: [1,0,0,0]
            0.0, 1.0, 0.0, 0.0, // head 1: [0,1,0,0]
        ];

        // K cache: [3, 2*4] = [3, 8] -- 3 positions, 2 kv heads
        let k_cache = vec![
            // pos 0: kv_head0=[1,0,0,0], kv_head1=[0,0,1,0]
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            // pos 1: kv_head0=[0,1,0,0], kv_head1=[0,0,0,1]
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            // pos 2: kv_head0=[0,0,1,0], kv_head1=[1,0,0,0]
            0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ];

        // V cache: [3, 2*4] = [3, 8] -- distinct values per position
        let v_cache = vec![
            // pos 0
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, // pos 1
            1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, // pos 2
            2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8,
        ];

        // CPU reference
        let expected = cpu_decode_attention(
            &q,
            &k_cache,
            &v_cache,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_len,
        );

        // GPU dispatch
        let result = dispatch_decode_attention(
            &gpu,
            &mut pso_cache,
            &q,
            &k_cache,
            &v_cache,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_len,
        );

        assert_eq!(result.len(), num_heads * head_dim);
        let mut max_diff = 0.0f32;
        for i in 0..result.len() {
            let diff = (result[i] - expected[i]).abs();
            max_diff = max_diff.max(diff);
            assert!(
                diff < 1e-3,
                "Index {}: GPU={}, CPU={}, diff={}",
                i,
                result[i],
                expected[i],
                diff
            );
        }

        eprintln!(
            "test_decode_attention_basic: max_diff={:.6}, {} outputs verified",
            max_diff,
            result.len()
        );
    }

    #[test]
    fn test_decode_attention_gqa() {
        // 9 Q heads, 3 KV heads, head_dim=64, kv_len=5
        // group_size = 9/3 = 3: Q heads {0,1,2}->KV head 0, {3,4,5}->KV head 1, {6,7,8}->KV head 2
        let gpu = GpuDevice::new();
        let mut pso_cache = PsoCache::new(gpu.library.clone());

        let num_heads = 9;
        let num_kv_heads = 3;
        let head_dim = 64;
        let kv_len = 5;

        let kv_dim = num_kv_heads * head_dim; // 192

        // Deterministic pseudo-random data
        let q: Vec<f32> = (0..num_heads * head_dim)
            .map(|i| 0.01 * ((i * 7 + 3) % 100) as f32 - 0.5)
            .collect();

        let k_cache: Vec<f32> = (0..kv_len * kv_dim)
            .map(|i| 0.01 * ((i * 13 + 7) % 100) as f32 - 0.5)
            .collect();

        let v_cache: Vec<f32> = (0..kv_len * kv_dim)
            .map(|i| 0.01 * ((i * 17 + 11) % 100) as f32 - 0.5)
            .collect();

        // CPU reference
        let expected = cpu_decode_attention(
            &q,
            &k_cache,
            &v_cache,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_len,
        );

        // GPU dispatch
        let result = dispatch_decode_attention(
            &gpu,
            &mut pso_cache,
            &q,
            &k_cache,
            &v_cache,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_len,
        );

        assert_eq!(result.len(), num_heads * head_dim);

        let mut max_diff = 0.0f32;
        for i in 0..result.len() {
            let diff = (result[i] - expected[i]).abs();
            max_diff = max_diff.max(diff);
            assert!(
                diff < 1e-3,
                "Index {}: GPU={}, CPU={}, diff={}",
                i,
                result[i],
                expected[i],
                diff
            );
        }

        // Verify GQA grouping: Q heads in same group should produce DIFFERENT outputs
        // (they read same KV but have different Q vectors)
        let head0_out = &result[0..head_dim];
        let head1_out = &result[head_dim..2 * head_dim];
        let head3_out = &result[3 * head_dim..4 * head_dim];

        // Heads 0 and 1 share KV head 0 but have different Q -> different output
        let diff_same_group: f32 = head0_out
            .iter()
            .zip(head1_out.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff_same_group > 0.01,
            "Heads 0 and 1 (same KV group) should have different outputs, diff={}",
            diff_same_group
        );

        // Heads 0 and 3 use different KV heads -> should also differ
        let diff_cross_group: f32 = head0_out
            .iter()
            .zip(head3_out.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff_cross_group > 0.01,
            "Heads 0 and 3 (different KV groups) should have different outputs, diff={}",
            diff_cross_group
        );

        eprintln!(
            "test_decode_attention_gqa: max_diff={:.6}, {} outputs verified, GQA grouping correct",
            max_diff,
            result.len()
        );
    }

    #[test]
    fn test_decode_attention_single_head() {
        // Trivial case: 1 head, 1 kv_head, head_dim=4, kv_len=1
        // With a single KV position, softmax output = 1.0, so output = V[0]
        let gpu = GpuDevice::new();
        let mut pso_cache = PsoCache::new(gpu.library.clone());

        let num_heads = 1;
        let num_kv_heads = 1;
        let head_dim = 4;
        let kv_len = 1;

        let q = vec![1.0, 0.0, 0.0, 0.0];
        let k_cache = vec![0.5, 0.5, 0.0, 0.0]; // single KV position
        let v_cache = vec![0.1, 0.2, 0.3, 0.4]; // expected output (softmax trivially = 1.0)

        let expected = cpu_decode_attention(
            &q,
            &k_cache,
            &v_cache,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_len,
        );

        let result = dispatch_decode_attention(
            &gpu,
            &mut pso_cache,
            &q,
            &k_cache,
            &v_cache,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_len,
        );

        assert_eq!(result.len(), num_heads * head_dim);
        for i in 0..result.len() {
            let diff = (result[i] - expected[i]).abs();
            assert!(
                diff < 1e-3,
                "Index {}: GPU={}, CPU={}, diff={}",
                i,
                result[i],
                expected[i],
                diff
            );
        }

        // With single KV position, output should equal V[0] exactly
        for i in 0..head_dim {
            let diff = (result[i] - v_cache[i]).abs();
            assert!(
                diff < 1e-3,
                "Single KV: output[{}]={}, expected V[{}]={}, diff={}",
                i,
                result[i],
                i,
                v_cache[i],
                diff
            );
        }
    }

    #[test]
    fn test_decode_attention_long_context() {
        // kv_len=512, verify no threadgroup memory overflow (scores[2048] limit)
        let gpu = GpuDevice::new();
        let mut pso_cache = PsoCache::new(gpu.library.clone());

        let num_heads = 9;
        let num_kv_heads = 3;
        let head_dim = 64;
        let kv_len = 512;

        let kv_dim = num_kv_heads * head_dim; // 192

        // Deterministic pseudo-random data
        let q: Vec<f32> = (0..num_heads * head_dim)
            .map(|i| 0.01 * ((i * 7 + 3) % 100) as f32 - 0.5)
            .collect();

        let k_cache: Vec<f32> = (0..kv_len * kv_dim)
            .map(|i| 0.01 * ((i * 13 + 7) % 100) as f32 - 0.5)
            .collect();

        let v_cache: Vec<f32> = (0..kv_len * kv_dim)
            .map(|i| 0.01 * ((i * 17 + 11) % 100) as f32 - 0.5)
            .collect();

        let expected = cpu_decode_attention(
            &q,
            &k_cache,
            &v_cache,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_len,
        );

        let result = dispatch_decode_attention(
            &gpu,
            &mut pso_cache,
            &q,
            &k_cache,
            &v_cache,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_len,
        );

        assert_eq!(result.len(), num_heads * head_dim);
        let mut max_diff = 0.0f32;
        for i in 0..result.len() {
            let diff = (result[i] - expected[i]).abs();
            max_diff = max_diff.max(diff);
            assert!(
                diff < 1e-3,
                "Index {}: GPU={}, CPU={}, diff={}",
                i,
                result[i],
                expected[i],
                diff
            );
        }

        eprintln!(
            "test_decode_attention_long_context: kv_len={}, max_diff={:.6}, {} outputs verified",
            kv_len,
            max_diff,
            result.len()
        );
    }
}
