//! Buffer allocation and pooling for Metal GPU buffers.
//!
//! Provides convenience functions for buffer allocation, data upload,
//! zero-copy mmap buffers, and a simple buffer pool with acquire/release.

use std::collections::HashMap;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

/// Allocate a Metal buffer of `size` bytes with StorageModeShared.
pub fn alloc_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    size: usize,
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    let options = MTLResourceOptions::StorageModeShared;
    device
        .newBufferWithLength_options(size, options)
        .expect("Failed to allocate Metal buffer")
}

/// Allocate a Metal buffer of `size` bytes with StorageModePrivate.
///
/// Private buffers are only accessible by the GPU, which allows the driver
/// to place them in dedicated GPU memory for faster access. Cannot be read
/// from CPU — use for scratch/intermediate buffers that never leave the GPU.
pub fn alloc_buffer_private(
    device: &ProtocolObject<dyn MTLDevice>,
    size: usize,
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    let options = MTLResourceOptions::StorageModePrivate;
    device
        .newBufferWithLength_options(size, options)
        .expect("Failed to allocate Private Metal buffer")
}

/// Allocate a Metal buffer initialized with the given data slice.
pub fn alloc_buffer_with_data<T: Copy>(
    device: &ProtocolObject<dyn MTLDevice>,
    data: &[T],
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    let size = std::mem::size_of_val(data);
    let options = MTLResourceOptions::StorageModeShared;

    unsafe {
        let ptr =
            NonNull::new(data.as_ptr() as *mut std::ffi::c_void).expect("data pointer is null");
        device
            .newBufferWithBytes_length_options(ptr, size, options)
            .expect("Failed to allocate Metal buffer with data")
    }
}

/// Create a zero-copy buffer wrapping an existing memory region.
///
/// Uses `newBufferWithBytesNoCopy` to create a Metal buffer that directly
/// references the provided memory without copying. Ideal for mmap'd weight files.
///
/// # Safety
/// - `ptr` must point to a valid memory region of at least `len` bytes
/// - The memory must remain valid for the lifetime of the returned buffer
/// - The memory must be page-aligned (4096 bytes on Apple Silicon)
pub unsafe fn create_weight_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    ptr: *mut std::ffi::c_void,
    len: usize,
) -> Option<Retained<ProtocolObject<dyn MTLBuffer>>> {
    let non_null = NonNull::new(ptr)?;
    let options = MTLResourceOptions::StorageModeShared;
    device.newBufferWithBytesNoCopy_length_options_deallocator(non_null, len, options, None)
}

/// Read back a single value of type T from a Metal buffer at offset 0.
///
/// # Safety
/// The buffer must contain at least `size_of::<T>()` bytes, and the
/// data must be a valid representation of T.
pub unsafe fn read_buffer<T: Copy>(buffer: &ProtocolObject<dyn MTLBuffer>) -> T {
    let ptr = buffer.contents().as_ptr() as *const T;
    *ptr
}

/// Read back a slice of T values from a Metal buffer.
///
/// # Safety
/// The buffer must contain at least `count * size_of::<T>()` bytes.
pub unsafe fn read_buffer_slice<T: Copy>(
    buffer: &ProtocolObject<dyn MTLBuffer>,
    count: usize,
) -> Vec<T> {
    let ptr = buffer.contents().as_ptr() as *const T;
    let slice = std::slice::from_raw_parts(ptr, count);
    slice.to_vec()
}

/// Size bucket for the buffer pool (rounded up to power-of-two).
fn bucket_size(size: usize) -> usize {
    // Minimum bucket: 4 KB (Metal page alignment)
    let min_bucket = 4096;
    if size <= min_bucket {
        return min_bucket;
    }
    size.next_power_of_two()
}

/// Simple buffer pool that reuses Metal buffers by size bucket.
///
/// Buffers are grouped by power-of-two size buckets. When a buffer is released
/// back to the pool, it becomes available for future acquisitions of the same
/// or smaller size within the same bucket.
pub struct BufferPool {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    /// Free buffers grouped by bucket size.
    free: HashMap<usize, Vec<Retained<ProtocolObject<dyn MTLBuffer>>>>,
}

impl BufferPool {
    /// Create a new empty buffer pool.
    pub fn new(device: Retained<ProtocolObject<dyn MTLDevice>>) -> Self {
        Self {
            device,
            free: HashMap::new(),
        }
    }

    /// Acquire a buffer of at least `size` bytes.
    ///
    /// Returns a pooled buffer if one is available in the matching size bucket,
    /// otherwise allocates a new one.
    pub fn acquire(&mut self, size: usize) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        let bucket = bucket_size(size);
        if let Some(buffers) = self.free.get_mut(&bucket) {
            if let Some(buf) = buffers.pop() {
                return buf;
            }
        }
        alloc_buffer(&self.device, bucket)
    }

    /// Release a buffer back to the pool for reuse.
    pub fn release(&mut self, buffer: Retained<ProtocolObject<dyn MTLBuffer>>) {
        let size = buffer.length();
        let bucket = bucket_size(size);
        self.free.entry(bucket).or_default().push(buffer);
    }

    /// Number of free buffers across all buckets.
    pub fn free_count(&self) -> usize {
        self.free.values().map(|v| v.len()).sum()
    }

    /// Clear all pooled buffers, releasing GPU memory.
    pub fn clear(&mut self) {
        self.free.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::GpuDevice;

    #[test]
    fn test_buffer_roundtrip() {
        let gpu = GpuDevice::new();

        // Test alloc_buffer_with_data + read_buffer_slice roundtrip
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let buffer = alloc_buffer_with_data(&gpu.device, &input);

        let output: Vec<f32> = unsafe { read_buffer_slice(&buffer, input.len()) };
        assert_eq!(input, output, "buffer roundtrip should preserve data");

        // Test alloc_buffer + manual write + read_buffer single value
        let buf = alloc_buffer(&gpu.device, std::mem::size_of::<u32>());
        unsafe {
            let ptr = buf.contents().as_ptr() as *mut u32;
            *ptr = 42;
        }
        let val: u32 = unsafe { read_buffer(&buf) };
        assert_eq!(val, 42, "single value roundtrip should preserve data");
    }

    #[test]
    fn test_buffer_pool_acquire_release() {
        let gpu = GpuDevice::new();
        let mut pool = BufferPool::new(gpu.device.clone());

        assert_eq!(pool.free_count(), 0);

        // Acquire a buffer
        let buf = pool.acquire(1024);
        assert!(buf.length() >= 1024);

        // Release it back
        pool.release(buf);
        assert_eq!(pool.free_count(), 1);

        // Acquire again — should reuse the pooled buffer
        let buf2 = pool.acquire(1024);
        assert!(buf2.length() >= 1024);
        assert_eq!(pool.free_count(), 0, "Should have reused pooled buffer");

        // Clear pool
        pool.release(buf2);
        pool.clear();
        assert_eq!(pool.free_count(), 0);
    }

    #[test]
    fn test_bucket_size() {
        assert_eq!(bucket_size(1), 4096);
        assert_eq!(bucket_size(4096), 4096);
        assert_eq!(bucket_size(4097), 8192);
        assert_eq!(bucket_size(8192), 8192);
        assert_eq!(bucket_size(10000), 16384);
    }
}
