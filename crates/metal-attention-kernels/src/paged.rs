//! Paged attention V2 dispatch: two-pass GPU kernel host code.
//!
//! Pass 1 (paged_attention_partition): Each threadgroup processes one query block
//!   x one partition of KV pages, producing partial attention output with online
//!   softmax state (O_partial, m_partial, l_partial).
//!
//! Pass 2 (paged_attention_reduce): Combines partial outputs from all partitions
//!   using log-sum-exp reduction to produce the final attention output.
//!
//! Uses block-table indirection for non-contiguous KV page access.

use crate::buffer::{alloc_buffer, alloc_buffer_with_data, read_buffer_slice};
use crate::device::GpuDevice;
use crate::pipeline::{PsoCache, PsoKey};
use crate::types::AttentionParams;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLSize,
};

/// Run paged attention on the GPU using two Metal kernel passes.
///
/// The KV cache is organized as pages (fixed-size token blocks) with a block
/// table mapping logical page indices to physical page locations.
///
/// KV cache layout expected by the shader:
///   `KV_cache[phys_page * 2 * page_size * head_dim]` = K data for page
///   `KV_cache[phys_page * 2 * page_size * head_dim + page_size * head_dim]` = V data
///
/// # Arguments
/// - `device`: GpuDevice
/// - `pso_cache`: PSO cache
/// - `q`: Query matrix [seq_len, head_dim]
/// - `kv_cache`: Interleaved KV page data [num_phys_pages, 2, page_size, head_dim]
/// - `block_table`: Logical -> physical page mapping [num_logical_pages]
/// - `seq_len`: Number of query tokens
/// - `head_dim`: Dimension per head
/// - `page_size`: Tokens per page
/// - `context_len`: Total number of KV tokens (context length)
/// - `num_partitions`: Number of partitions for the two-pass reduce
///
/// # Returns
/// Output matrix [seq_len, head_dim] as Vec<f32>
#[allow(clippy::too_many_arguments)]
pub fn dispatch_paged_attention(
    device: &GpuDevice,
    pso_cache: &mut PsoCache,
    q: &[f32],
    kv_cache: &[f32],
    block_table: &[u32],
    seq_len: usize,
    head_dim: usize,
    page_size: usize,
    context_len: usize,
    num_partitions: usize,
) -> Vec<f32> {
    assert_eq!(q.len(), seq_len * head_dim, "Q length mismatch");

    let block_r: usize = 16; // TILE_Q in shader
    let num_query_blocks = (seq_len + block_r - 1) / block_r;
    let num_logical_pages = block_table.len();

    // Build AttentionParams for the shader
    let params = AttentionParams::paged(
        seq_len as u32,
        head_dim as u32,
        1, // single head for now
        page_size as u32,
        num_logical_pages as u32,
        context_len as u32,
        num_partitions as u32,
    );

    // Allocate Metal buffers
    let q_buf = alloc_buffer_with_data(&device.device, q);
    let kv_buf = alloc_buffer_with_data(&device.device, kv_cache);
    let pt_buf = alloc_buffer_with_data(&device.device, block_table);
    let params_buf = alloc_buffer_with_data(&device.device, std::slice::from_ref(&params));

    // Partial output buffers for pass 1
    // Layout: [num_query_blocks * num_partitions, BLOCK_R, head_dim]
    let num_pb = num_query_blocks * num_partitions;
    let o_partial_size = num_pb * block_r * head_dim;
    let ml_partial_size = num_pb * block_r;

    let o_partial_buf = alloc_buffer(
        &device.device,
        o_partial_size * std::mem::size_of::<f32>(),
    );
    let m_partial_buf = alloc_buffer(
        &device.device,
        ml_partial_size * std::mem::size_of::<f32>(),
    );
    let l_partial_buf = alloc_buffer(
        &device.device,
        ml_partial_size * std::mem::size_of::<f32>(),
    );

    // ====================================================================
    // Pass 1: paged_attention_partition
    // Grid: (num_query_blocks, num_partitions)
    // Each threadgroup = 32 threads (one simdgroup)
    // ====================================================================
    {
        let pso_key = PsoKey::simple("paged_attention_partition")
            .with_uint(0, head_dim as u32)
            .with_uint(1, page_size as u32);
        let pso = pso_cache.get_or_compile(&pso_key);

        let command_buffer = device
            .command_queue
            .commandBuffer()
            .expect("Failed to create command buffer for paged_attention_partition");
        let encoder = command_buffer
            .computeCommandEncoder()
            .expect("Failed to create compute encoder for paged_attention_partition");

        encoder.setComputePipelineState(pso);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&*q_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&*kv_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&*pt_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&*o_partial_buf), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&*m_partial_buf), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(&*l_partial_buf), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(&*params_buf), 0, 6);
        }

        let threadgroups = MTLSize {
            width: num_query_blocks,
            height: num_partitions,
            depth: 1,
        };
        let tg_size = MTLSize {
            width: 32,
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
            "paged_attention_partition failed with status {:?}. Error: {:?}",
            status,
            command_buffer.error()
        );
    }

    // ====================================================================
    // Pass 2: paged_attention_reduce
    // Grid: (num_query_blocks, 1)
    // Each threadgroup = TILE_Q * TILE_D threads
    // ====================================================================
    let o_final_buf = alloc_buffer(
        &device.device,
        seq_len * head_dim * std::mem::size_of::<f32>(),
    );

    {
        let pso_key = PsoKey::simple("paged_attention_reduce")
            .with_uint(0, head_dim as u32)
            .with_uint(1, page_size as u32);
        let pso = pso_cache.get_or_compile(&pso_key);

        let command_buffer = device
            .command_queue
            .commandBuffer()
            .expect("Failed to create command buffer for paged_attention_reduce");
        let encoder = command_buffer
            .computeCommandEncoder()
            .expect("Failed to create compute encoder for paged_attention_reduce");

        encoder.setComputePipelineState(pso);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&*o_partial_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&*m_partial_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&*l_partial_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&*o_final_buf), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&*params_buf), 0, 4);
        }

        // Threads per threadgroup: TILE_Q * TILE_D = 16 * 64 = 1024
        let tile_q: usize = 16;
        let threads_needed = tile_q * head_dim;
        let max_threads = pso.maxTotalThreadsPerThreadgroup();
        let threads_per_tg = threads_needed.min(max_threads);

        let threadgroups = MTLSize {
            width: num_query_blocks,
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
            "paged_attention_reduce failed with status {:?}. Error: {:?}",
            status,
            command_buffer.error()
        );
    }

    unsafe { read_buffer_slice(&o_final_buf, seq_len * head_dim) }
}

/// CPU reference implementation of paged attention for testing.
///
/// Standard softmax attention: O = softmax(Q * K^T / sqrt(d)) * V
/// but reading K/V through block-table page indirection.
#[allow(clippy::too_many_arguments)]
pub fn cpu_paged_attention(
    q: &[f32],
    kv_cache: &[f32],
    block_table: &[u32],
    seq_len: usize,
    head_dim: usize,
    page_size: usize,
    context_len: usize,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; seq_len * head_dim];

    for q_row in 0..seq_len {
        let q_off = q_row * head_dim;

        // Compute attention scores for all context positions
        let mut scores = vec![-f32::INFINITY; context_len];
        let mut max_score = -f32::INFINITY;

        for kv_pos in 0..context_len {
            // Find which page and slot
            let logical_page = kv_pos / page_size;
            let slot = kv_pos % page_size;
            let phys_page = block_table[logical_page] as usize;

            // K offset in interleaved layout: [phys_page, 2, page_size, head_dim]
            // K starts at offset 0 within the page
            let k_base = phys_page * 2 * page_size * head_dim + slot * head_dim;

            // Dot product Q[q_row] . K[kv_pos]
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[q_off + d] * kv_cache[k_base + d];
            }
            scores[kv_pos] = dot * scale;
            max_score = max_score.max(scores[kv_pos]);
        }

        // Softmax
        let mut sum_exp = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max_score).exp();
            sum_exp += *s;
        }
        if sum_exp > 0.0 {
            for s in scores.iter_mut() {
                *s /= sum_exp;
            }
        }

        // Weighted sum of V
        for kv_pos in 0..context_len {
            let logical_page = kv_pos / page_size;
            let slot = kv_pos % page_size;
            let phys_page = block_table[logical_page] as usize;

            // V offset: after K in the interleaved layout
            let v_base =
                phys_page * 2 * page_size * head_dim + page_size * head_dim + slot * head_dim;

            for d in 0..head_dim {
                output[q_off + d] += scores[kv_pos] * kv_cache[v_base + d];
            }
        }
    }

    output
}

/// Build an interleaved KV cache from separate K and V page pools.
///
/// Input layout (from PagedKVCache):
///   k_pages: [num_pages, page_size, head_dim]
///   v_pages: [num_pages, page_size, head_dim]
///
/// Output layout (for shader):
///   kv_cache: [num_pages, 2, page_size, head_dim]
///   (K then V per page)
pub fn interleave_kv_pages(
    k_pages: &[f32],
    v_pages: &[f32],
    num_pages: usize,
    page_size: usize,
    head_dim: usize,
) -> Vec<f32> {
    let page_elems = page_size * head_dim;
    let mut kv = vec![0.0f32; num_pages * 2 * page_elems];

    for p in 0..num_pages {
        let k_src = p * page_elems;
        let v_src = p * page_elems;
        let kv_dst = p * 2 * page_elems;

        // K part
        kv[kv_dst..kv_dst + page_elems].copy_from_slice(&k_pages[k_src..k_src + page_elems]);
        // V part
        kv[kv_dst + page_elems..kv_dst + 2 * page_elems]
            .copy_from_slice(&v_pages[v_src..v_src + page_elems]);
    }

    kv
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::GpuDevice;
    use crate::flash::dispatch_flash_attention;
    #[allow(unused_imports)]
    use crate::kv_cache::{DenseKVCache, PagedKVCache};
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

    /// Test: CPU paged attention matches dense attention on small inputs.
    #[test]
    fn test_paged_cpu_vs_dense_cpu() {
        let head_dim = 4;
        let page_size = 4;
        let seq_len = 4; // query length
        let context_len = 4; // KV length (same for self-attention)

        let mut rng = SimpleRng::new(42);

        // Generate Q, K, V
        let q: Vec<f32> = (0..seq_len * head_dim)
            .map(|_| rng.next_f32_range(-0.5, 0.5))
            .collect();
        let k: Vec<f32> = (0..context_len * head_dim)
            .map(|_| rng.next_f32_range(-0.5, 0.5))
            .collect();
        let v: Vec<f32> = (0..context_len * head_dim)
            .map(|_| rng.next_f32_range(-0.5, 0.5))
            .collect();

        // Dense reference
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut dense_output = vec![0.0f32; seq_len * head_dim];
        for qr in 0..seq_len {
            let mut scores = vec![0.0f32; context_len];
            let mut max_s = -f32::INFINITY;
            for kc in 0..context_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[qr * head_dim + d] * k[kc * head_dim + d];
                }
                scores[kc] = dot * scale;
                max_s = max_s.max(scores[kc]);
            }
            let mut sum_exp = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max_s).exp();
                sum_exp += *s;
            }
            for s in scores.iter_mut() {
                *s /= sum_exp;
            }
            for kc in 0..context_len {
                for d in 0..head_dim {
                    dense_output[qr * head_dim + d] += scores[kc] * v[kc * head_dim + d];
                }
            }
        }

        // Build paged KV cache with 1 page (all tokens fit)
        let num_pages = (context_len + page_size - 1) / page_size;
        let page_elems = page_size * head_dim;
        let mut k_pages = vec![0.0f32; num_pages * page_elems];
        let mut v_pages = vec![0.0f32; num_pages * page_elems];

        // Fill pages
        for t in 0..context_len {
            let page = t / page_size;
            let slot = t % page_size;
            k_pages[page * page_elems + slot * head_dim..page * page_elems + (slot + 1) * head_dim]
                .copy_from_slice(&k[t * head_dim..(t + 1) * head_dim]);
            v_pages[page * page_elems + slot * head_dim..page * page_elems + (slot + 1) * head_dim]
                .copy_from_slice(&v[t * head_dim..(t + 1) * head_dim]);
        }

        // Identity block table
        let block_table: Vec<u32> = (0..num_pages as u32).collect();

        let kv_cache = interleave_kv_pages(&k_pages, &v_pages, num_pages, page_size, head_dim);
        let paged_output =
            cpu_paged_attention(&q, &kv_cache, &block_table, seq_len, head_dim, page_size, context_len);

        let atol = 1e-5;
        for i in 0..dense_output.len() {
            let diff = (dense_output[i] - paged_output[i]).abs();
            assert!(
                diff < atol,
                "CPU paged vs dense mismatch at {}: dense={}, paged={}, diff={}",
                i,
                dense_output[i],
                paged_output[i],
                diff
            );
        }
    }

    /// Test: GPU paged attention (2-pass) vs CPU reference on small input.
    #[test]
    fn test_paged_gpu_vs_cpu() {
        let gpu = GpuDevice::new();
        let mut pso_cache = PsoCache::new(gpu.library.clone());

        let head_dim = 64; // TILE_D in shader
        let page_size = 16; // TILE_PAGE in shader
        let seq_len = 16; // one query block
        let context_len = 16; // one page of context

        let mut rng = SimpleRng::new(789);

        let q: Vec<f32> = (0..seq_len * head_dim)
            .map(|_| rng.next_f32_range(-0.3, 0.3))
            .collect();

        // Build paged cache with identity block table
        let num_pages = (context_len + page_size - 1) / page_size;
        let page_elems = page_size * head_dim;

        let mut k_pages = vec![0.0f32; num_pages * page_elems];
        let mut v_pages = vec![0.0f32; num_pages * page_elems];

        for t in 0..context_len {
            let page = t / page_size;
            let slot = t % page_size;
            for d in 0..head_dim {
                k_pages[page * page_elems + slot * head_dim + d] = rng.next_f32_range(-0.3, 0.3);
                v_pages[page * page_elems + slot * head_dim + d] = rng.next_f32_range(-0.3, 0.3);
            }
        }

        let block_table: Vec<u32> = (0..num_pages as u32).collect();
        let kv_cache = interleave_kv_pages(&k_pages, &v_pages, num_pages, page_size, head_dim);

        // CPU reference
        let cpu_out =
            cpu_paged_attention(&q, &kv_cache, &block_table, seq_len, head_dim, page_size, context_len);

        // GPU paged attention (with 1 partition = single pass + reduce)
        let gpu_out = dispatch_paged_attention(
            &gpu,
            &mut pso_cache,
            &q,
            &kv_cache,
            &block_table,
            seq_len,
            head_dim,
            page_size,
            context_len,
            1, // single partition
        );

        assert_eq!(cpu_out.len(), gpu_out.len());
        let atol = 5e-3; // GPU float precision tolerance
        for i in 0..cpu_out.len() {
            let diff = (cpu_out[i] - gpu_out[i]).abs();
            assert!(
                diff < atol,
                "Paged GPU vs CPU mismatch at {}: CPU={}, GPU={}, diff={}",
                i,
                cpu_out[i],
                gpu_out[i],
                diff
            );
        }
    }

    /// Test: GPU paged attention produces same output as GPU flash attention
    /// when given the same Q/K/V data.
    #[test]
    fn test_paged_vs_dense_gpu() {
        let gpu = GpuDevice::new();
        let mut pso_cache = PsoCache::new(gpu.library.clone());

        let head_dim = 64;
        let page_size = 16;
        let seq_len = 16;
        let context_len = 16;

        let mut rng = SimpleRng::new(999);

        let q: Vec<f32> = (0..seq_len * head_dim)
            .map(|_| rng.next_f32_range(-0.3, 0.3))
            .collect();
        let k: Vec<f32> = (0..context_len * head_dim)
            .map(|_| rng.next_f32_range(-0.3, 0.3))
            .collect();
        let v: Vec<f32> = (0..context_len * head_dim)
            .map(|_| rng.next_f32_range(-0.3, 0.3))
            .collect();

        // Dense flash attention
        let dense_out =
            dispatch_flash_attention(&gpu, &mut pso_cache, &q, &k, &v, seq_len, head_dim, 1);

        // Build paged KV cache from the same K/V
        let num_pages = (context_len + page_size - 1) / page_size;
        let page_elems = page_size * head_dim;

        let mut k_pages = vec![0.0f32; num_pages * page_elems];
        let mut v_pages = vec![0.0f32; num_pages * page_elems];

        for t in 0..context_len {
            let page = t / page_size;
            let slot = t % page_size;
            k_pages[page * page_elems + slot * head_dim..page * page_elems + (slot + 1) * head_dim]
                .copy_from_slice(&k[t * head_dim..(t + 1) * head_dim]);
            v_pages[page * page_elems + slot * head_dim..page * page_elems + (slot + 1) * head_dim]
                .copy_from_slice(&v[t * head_dim..(t + 1) * head_dim]);
        }

        let block_table: Vec<u32> = (0..num_pages as u32).collect();
        let kv_cache = interleave_kv_pages(&k_pages, &v_pages, num_pages, page_size, head_dim);

        let paged_out = dispatch_paged_attention(
            &gpu,
            &mut pso_cache,
            &q,
            &kv_cache,
            &block_table,
            seq_len,
            head_dim,
            page_size,
            context_len,
            1,
        );

        assert_eq!(dense_out.len(), paged_out.len());
        let atol = 1e-2; // Allow wider tolerance since GPU implementations may differ slightly
        let mut max_diff = 0.0f32;
        for i in 0..dense_out.len() {
            let diff = (dense_out[i] - paged_out[i]).abs();
            max_diff = max_diff.max(diff);
            assert!(
                diff < atol,
                "Paged vs dense GPU mismatch at {}: dense={}, paged={}, diff={}",
                i,
                dense_out[i],
                paged_out[i],
                diff
            );
        }
        eprintln!("Paged vs Dense GPU max_diff: {max_diff}");
    }
}
