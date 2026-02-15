//! FLA linear attention dispatch: chunk_h + chunk_o GPU kernels with CPU prefix sum.
//!
//! Two-pass dispatch:
//! 1. `chunk_h`: Each threadgroup computes delta_H[chunk] = K_chunk^T * V_chunk (D x D)
//! 2. CPU prefix sum: H_cumulative[c] = H_cumulative[c-1] + delta_H[c]
//! 3. `chunk_o`: Each threadgroup computes O_chunk = Q_chunk * H_cumulative (C x D)
//!
//! Adapted from proto/src/proto6_fla.rs to use the kernels crate GpuDevice/PsoCache.

use crate::buffer::{alloc_buffer, alloc_buffer_with_data, read_buffer_slice};
use crate::device::GpuDevice;
use crate::pipeline::{PsoCache, PsoKey};
use crate::types::AttentionParams;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLSize,
};

/// Run chunk-based linear attention on the GPU using two Metal kernels.
///
/// Two-pass dispatch:
/// 1. `chunk_h`: Each threadgroup computes delta_H[chunk] = K_chunk^T * V_chunk (D x D)
/// 2. CPU prefix sum: H_cumulative[c] = sum(delta_H[0..=c])
/// 3. `chunk_o`: Each threadgroup computes O_chunk = Q_chunk * H_cumulative (C x D)
///
/// # Arguments
/// - `device`: GpuDevice with Metal device, command queue, and shader library
/// - `pso_cache`: PSO cache for kernel compilation
/// - `q`: Query matrix [seq_len, head_dim]
/// - `k`: Key matrix [seq_len, head_dim]
/// - `v`: Value matrix [seq_len, head_dim]
/// - `seq_len`: Number of tokens (must be divisible by chunk_size)
/// - `head_dim`: Dimension of each head
/// - `chunk_size`: Number of tokens per chunk
///
/// # Returns
/// Output matrix [seq_len, head_dim] as Vec<f32>
///
/// # Panics
/// Panics if seq_len is not divisible by chunk_size, or if input lengths don't match.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_linear_attention(
    device: &GpuDevice,
    pso_cache: &mut PsoCache,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    chunk_size: usize,
) -> Vec<f32> {
    assert_eq!(q.len(), seq_len * head_dim, "Q length mismatch");
    assert_eq!(k.len(), seq_len * head_dim, "K length mismatch");
    assert_eq!(v.len(), seq_len * head_dim, "V length mismatch");
    assert!(
        seq_len.is_multiple_of(chunk_size),
        "seq_len must be divisible by chunk_size"
    );

    let num_chunks = seq_len / chunk_size;

    // Build AttentionParams
    let params = AttentionParams {
        seq_len: seq_len as u32,
        head_dim: head_dim as u32,
        ..Default::default()
    };

    // Allocate Metal buffers for input data
    let k_buf = alloc_buffer_with_data(&device.device, k);
    let v_buf = alloc_buffer_with_data(&device.device, v);
    let q_buf = alloc_buffer_with_data(&device.device, q);
    let params_buf = alloc_buffer_with_data(&device.device, std::slice::from_ref(&params));

    // H_deltas buffer: [num_chunks, head_dim, head_dim]
    let h_deltas_size = num_chunks * head_dim * head_dim;
    let h_deltas_buf = alloc_buffer(&device.device, h_deltas_size * std::mem::size_of::<f32>());

    // ====================================================================
    // Pass 1: chunk_h kernel - compute delta_H per chunk
    // ====================================================================
    {
        // Compile chunk_h PSO with function constants: HEAD_DIM=index(0), CHUNK_SIZE=index(4)
        let chunk_h_key = PsoKey::simple("chunk_h")
            .with_uint(0, head_dim as u32)
            .with_uint(4, chunk_size as u32);
        let chunk_h_pso = pso_cache.get_or_compile(&chunk_h_key);

        // Determine threadgroup size: min(D*D, maxTotalThreadsPerThreadgroup)
        let max_threads_h = chunk_h_pso.maxTotalThreadsPerThreadgroup();
        let threads_per_tg_h = (head_dim * head_dim).min(max_threads_h);

        let command_buffer = device
            .command_queue
            .commandBuffer()
            .expect("Failed to create command buffer for chunk_h");

        let encoder = command_buffer
            .computeCommandEncoder()
            .expect("Failed to create compute encoder for chunk_h");

        encoder.setComputePipelineState(chunk_h_pso);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&*k_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&*v_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&*h_deltas_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&*params_buf), 0, 3);
        }

        let threadgroups = MTLSize {
            width: num_chunks,
            height: 1,
            depth: 1,
        };
        let threads_per_tg = MTLSize {
            width: threads_per_tg_h,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
        encoder.endEncoding();

        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        let status = command_buffer.status();
        assert_eq!(
            status,
            objc2_metal::MTLCommandBufferStatus::Completed,
            "chunk_h command buffer failed with status {:?}. Error: {:?}",
            status,
            command_buffer.error()
        );
    }

    // ====================================================================
    // CPU prefix sum: H_cumulative[c] = sum(delta_H[0..=c])
    // ====================================================================
    let h_deltas: Vec<f32> = unsafe { read_buffer_slice(&h_deltas_buf, h_deltas_size) };

    let dd = head_dim * head_dim;
    let mut h_cumulative = vec![0.0f32; num_chunks * dd];

    // H_cumulative[0] = H_deltas[0]
    h_cumulative[..dd].copy_from_slice(&h_deltas[..dd]);

    // H_cumulative[c] = H_cumulative[c-1] + H_deltas[c]
    for c in 1..num_chunks {
        for elem in 0..dd {
            h_cumulative[c * dd + elem] =
                h_cumulative[(c - 1) * dd + elem] + h_deltas[c * dd + elem];
        }
    }

    // Upload H_cumulative to GPU
    let h_cumulative_buf = alloc_buffer_with_data(&device.device, &h_cumulative);

    // Output buffer
    let o_buf = alloc_buffer(
        &device.device,
        seq_len * head_dim * std::mem::size_of::<f32>(),
    );

    // ====================================================================
    // Pass 2: chunk_o kernel - compute O = Q * H_cumulative
    // ====================================================================
    {
        let chunk_o_key = PsoKey::simple("chunk_o")
            .with_uint(0, head_dim as u32)
            .with_uint(4, chunk_size as u32);
        let chunk_o_pso = pso_cache.get_or_compile(&chunk_o_key);

        // Determine threadgroup size: min(C*D, maxTotalThreadsPerThreadgroup)
        let max_threads_o = chunk_o_pso.maxTotalThreadsPerThreadgroup();
        let threads_per_tg_o = (chunk_size * head_dim).min(max_threads_o);

        let command_buffer = device
            .command_queue
            .commandBuffer()
            .expect("Failed to create command buffer for chunk_o");

        let encoder = command_buffer
            .computeCommandEncoder()
            .expect("Failed to create compute encoder for chunk_o");

        encoder.setComputePipelineState(chunk_o_pso);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&*q_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&*h_cumulative_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&*o_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&*params_buf), 0, 3);
        }

        let threadgroups = MTLSize {
            width: num_chunks,
            height: 1,
            depth: 1,
        };
        let threads_per_tg = MTLSize {
            width: threads_per_tg_o,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
        encoder.endEncoding();

        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        let status = command_buffer.status();
        assert_eq!(
            status,
            objc2_metal::MTLCommandBufferStatus::Completed,
            "chunk_o command buffer failed with status {:?}. Error: {:?}",
            status,
            command_buffer.error()
        );
    }

    // Read back output
    unsafe { read_buffer_slice(&o_buf, seq_len * head_dim) }
}
