//! Matrix multiplication dispatch: GPU kernel host code.
//!
//! Dispatches the `matmul` Metal kernel which computes:
//!   C[M, N] = A[M, K] * B[K, N]
//!
//! Uses function constants for M, N, K dimensions.

use crate::buffer::{alloc_buffer, alloc_buffer_with_data, read_buffer_slice};
use crate::device::GpuDevice;
use crate::pipeline::{PsoCache, PsoKey};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLSize,
};

/// Run matrix multiplication on the GPU: C = A * B.
///
/// # Arguments
/// - `device`: GpuDevice with Metal device, command queue, and shader library
/// - `pso_cache`: PSO cache for kernel compilation
/// - `a`: Matrix A [M, K], row-major
/// - `b`: Matrix B [K, N], row-major
/// - `m`: Number of rows in A / rows in C
/// - `n`: Number of columns in B / columns in C
/// - `k`: Number of columns in A / rows in B
///
/// # Returns
/// Output matrix C [M, N] as Vec<f32>
pub fn dispatch_matmul(
    device: &GpuDevice,
    pso_cache: &mut PsoCache,
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "A length mismatch");
    assert_eq!(b.len(), k * n, "B length mismatch");

    let total = m * n;

    let a_buf = alloc_buffer_with_data(&device.device, a);
    let b_buf = alloc_buffer_with_data(&device.device, b);
    let c_buf = alloc_buffer(&device.device, total * std::mem::size_of::<f32>());

    // Function constants: index 0 = MAT_M, index 1 = MAT_N, index 2 = MAT_K
    let pso_key = PsoKey::simple("matmul")
        .with_uint(0, m as u32)
        .with_uint(1, n as u32)
        .with_uint(2, k as u32);
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
        encoder.setBuffer_offset_atIndex(Some(&*a_buf), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&*b_buf), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&*c_buf), 0, 2);
    }

    let max_threads = pso.maxTotalThreadsPerThreadgroup();
    let tg_size = max_threads.min(256);

    let grid_size = MTLSize {
        width: total,
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
        "matmul command buffer failed with status {:?}. Error: {:?}",
        status,
        command_buffer.error()
    );

    unsafe { read_buffer_slice(&c_buf, total) }
}
