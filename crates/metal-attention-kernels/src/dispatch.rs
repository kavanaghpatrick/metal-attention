//! Compute command encoder helpers for Metal GPU dispatch.
//!
//! Provides convenience functions for encoding compute commands:
//! buffer binding, bytes upload, and threadgroup dispatch (1D and 2D).

use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize,
};
use std::ptr::NonNull;

/// Bind a buffer at the given argument index on a compute encoder.
///
/// # Safety
/// The encoder must be in a valid encoding state.
pub fn set_buffer(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    buffer: &ProtocolObject<dyn MTLBuffer>,
    offset: usize,
    index: usize,
) {
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(buffer), offset, index);
    }
}

/// Upload inline bytes at the given argument index on a compute encoder.
///
/// Uses `setBytes` which copies the data into the command buffer — suitable
/// for small data like parameter structs (< 4 KB).
pub fn set_bytes<T: Copy>(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    data: &T,
    index: usize,
) {
    let size = std::mem::size_of::<T>();
    unsafe {
        let ptr = NonNull::new(data as *const T as *mut std::ffi::c_void)
            .expect("data pointer is null");
        encoder.setBytes_length_atIndex(ptr, size, index);
    }
}

/// Upload a byte slice at the given argument index.
pub fn set_bytes_slice(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    data: &[u8],
    index: usize,
) {
    unsafe {
        let ptr = NonNull::new(data.as_ptr() as *mut std::ffi::c_void)
            .expect("data pointer is null");
        encoder.setBytes_length_atIndex(ptr, data.len(), index);
    }
}

/// Encode a 1D compute dispatch: set pipeline, bind buffers, dispatch threadgroups.
///
/// `buffers` is a slice of (buffer, index) pairs to bind at the given argument indices.
/// `total_threads` is the number of threads to dispatch.
/// Threadgroup size is clamped to min(maxTotalThreadsPerThreadgroup, 256).
pub fn dispatch_1d(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    buffers: &[(&ProtocolObject<dyn MTLBuffer>, usize)],
    total_threads: usize,
) {
    encoder.setComputePipelineState(pipeline);

    unsafe {
        for (buffer, index) in buffers {
            encoder.setBuffer_offset_atIndex(Some(*buffer), 0, *index);
        }
    }

    let threads_per_tg = pipeline.maxTotalThreadsPerThreadgroup().min(256);
    let grid_size = MTLSize {
        width: total_threads,
        height: 1,
        depth: 1,
    };
    let tg_size = MTLSize {
        width: threads_per_tg,
        height: 1,
        depth: 1,
    };

    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, tg_size);
}

/// Encode a 2D compute dispatch for attention grids.
///
/// `buffers` is a slice of (buffer, index) pairs to bind at the given argument indices.
/// `width` and `height` define the 2D grid dimensions (e.g., seq_len x heads).
/// Threadgroup size defaults to 16x16 (256 threads), clamped to pipeline max.
pub fn dispatch_2d(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    buffers: &[(&ProtocolObject<dyn MTLBuffer>, usize)],
    width: usize,
    height: usize,
) {
    encoder.setComputePipelineState(pipeline);

    unsafe {
        for (buffer, index) in buffers {
            encoder.setBuffer_offset_atIndex(Some(*buffer), 0, *index);
        }
    }

    let max_threads = pipeline.maxTotalThreadsPerThreadgroup();
    let tg_side = if max_threads >= 256 {
        16
    } else {
        (max_threads as f64).sqrt() as usize
    };

    let grid_size = MTLSize {
        width,
        height,
        depth: 1,
    };
    let tg_size = MTLSize {
        width: tg_side,
        height: tg_side,
        depth: 1,
    };

    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, tg_size);
}

/// Dispatch threadgroups with explicit grid and threadgroup sizes.
///
/// Lower-level helper when the caller manages pipeline state and buffer
/// binding separately.
pub fn dispatch_threadgroups(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    grid_size: MTLSize,
    threadgroup_size: MTLSize,
) {
    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
}
