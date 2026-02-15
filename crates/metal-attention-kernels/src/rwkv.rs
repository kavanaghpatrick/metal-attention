//! RWKV-7 WKV dispatch: GPU kernel host code.
//!
//! Dispatches the `rwkv_wkv` Metal kernel for state update and output computation.
//! The kernel processes tokens sequentially (recurrent mode), updating the
//! [head_dim, head_dim] state matrix in place.

use crate::buffer::{alloc_buffer, alloc_buffer_with_data, read_buffer_slice};
use crate::device::GpuDevice;
use crate::pipeline::{PsoCache, PsoKey};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLSize,
};

/// Run the RWKV-7 WKV operator on the GPU.
///
/// Processes `seq_len` tokens sequentially through the recurrent state update:
///   state[i][j] = w[i] * state[i][j] + k[i] * v[j]
///   output[j]   = sum_i(r[i] * state_new[i][j])
///
/// # Arguments
/// - `device`: GpuDevice with Metal device, command queue, and shader library
/// - `pso_cache`: PSO cache for kernel compilation
/// - `r`: Receptance (query-like) vectors [seq_len, head_dim]
/// - `k`: Key vectors [seq_len, head_dim]
/// - `v`: Value vectors [seq_len, head_dim]
/// - `w`: Decay factors [seq_len, head_dim], values in (0, 1)
/// - `state`: Initial state matrix [head_dim, head_dim], mutated in place
/// - `seq_len`: Number of tokens to process
/// - `head_dim`: Dimension of each head
///
/// # Returns
/// Tuple of (output [seq_len, head_dim], final_state [head_dim, head_dim])
#[allow(clippy::too_many_arguments)]
pub fn dispatch_rwkv_wkv(
    device: &GpuDevice,
    pso_cache: &mut PsoCache,
    r: &[f32],
    k: &[f32],
    v: &[f32],
    w: &[f32],
    state: &[f32],
    seq_len: usize,
    head_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n = seq_len * head_dim;
    assert_eq!(r.len(), n, "R length mismatch");
    assert_eq!(k.len(), n, "K length mismatch");
    assert_eq!(v.len(), n, "V length mismatch");
    assert_eq!(w.len(), n, "W length mismatch");
    assert_eq!(state.len(), head_dim * head_dim, "State length mismatch");

    // Allocate Metal buffers
    let r_buf = alloc_buffer_with_data(&device.device, r);
    let k_buf = alloc_buffer_with_data(&device.device, k);
    let v_buf = alloc_buffer_with_data(&device.device, v);
    let w_buf = alloc_buffer_with_data(&device.device, w);
    let state_buf = alloc_buffer_with_data(&device.device, state);
    let o_buf = alloc_buffer(&device.device, n * std::mem::size_of::<f32>());

    // Scalar parameters
    let seq_len_u32 = seq_len as u32;
    let head_dim_u32 = head_dim as u32;
    let seq_len_buf = alloc_buffer_with_data(&device.device, std::slice::from_ref(&seq_len_u32));
    let head_dim_buf = alloc_buffer_with_data(&device.device, std::slice::from_ref(&head_dim_u32));

    // Compile PSO with function constant for HEAD_DIM
    let pso_key = PsoKey::simple("rwkv_wkv").with_uint(0, head_dim as u32);

    let pso = pso_cache.get_or_compile(&pso_key);

    // Determine threadgroup size
    let max_threads = pso.maxTotalThreadsPerThreadgroup();
    let threads_per_tg = (head_dim * head_dim).min(max_threads);

    // Create command buffer and compute encoder
    let command_buffer = device
        .command_queue
        .commandBuffer()
        .expect("Failed to create command buffer for rwkv_wkv");
    let encoder = command_buffer
        .computeCommandEncoder()
        .expect("Failed to create compute encoder for rwkv_wkv");

    encoder.setComputePipelineState(pso);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(&*r_buf), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&*k_buf), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&*v_buf), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(&*w_buf), 0, 3);
        encoder.setBuffer_offset_atIndex(Some(&*state_buf), 0, 4);
        encoder.setBuffer_offset_atIndex(Some(&*o_buf), 0, 5);
        encoder.setBuffer_offset_atIndex(Some(&*seq_len_buf), 0, 6);
        encoder.setBuffer_offset_atIndex(Some(&*head_dim_buf), 0, 7);
    }

    // Single threadgroup -- kernel processes all tokens sequentially
    let threadgroups = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    let threads_per_tg = MTLSize {
        width: threads_per_tg,
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
        "rwkv_wkv command buffer failed with status {:?}. Error: {:?}",
        status,
        command_buffer.error()
    );

    // Read back output and final state
    let output = unsafe { read_buffer_slice(&o_buf, n) };
    let final_state = unsafe { read_buffer_slice(&state_buf, head_dim * head_dim) };

    (output, final_state)
}

/// CPU reference implementation of the RWKV WKV operator for testing.
///
/// Matches the GPU kernel's behavior exactly:
///   state[i][j] = w[i] * state[i][j] + k[i] * v[j]
///   output[j]   = sum_i(r[i] * state_new[i][j])
pub fn cpu_rwkv_wkv(
    r: &[f32],
    k: &[f32],
    v: &[f32],
    w: &[f32],
    state: &mut [f32],
    seq_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; seq_len * head_dim];

    for t in 0..seq_len {
        let off = t * head_dim;

        // Update state: state[i][j] = w[i] * state[i][j] + k[i] * v[j]
        for i in 0..head_dim {
            for j in 0..head_dim {
                state[i * head_dim + j] =
                    w[off + i] * state[i * head_dim + j] + k[off + i] * v[off + j];
            }
        }

        // Compute output: output[j] = sum_i(r[i] * state[i][j])
        for j in 0..head_dim {
            let mut acc = 0.0f32;
            for i in 0..head_dim {
                acc += r[off + i] * state[i * head_dim + j];
            }
            output[off + j] = acc;
        }
    }

    output
}
