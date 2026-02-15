//! GPU-resident key/value cache for decode attention.
//!
//! `GpuKVCache` stores K and V projections for all sequence positions
//! in Metal buffers. `GpuKVCacheSet` wraps one cache per transformer layer.
//!
//! For POC, append uses CPU-side memcpy via `contents()` pointer since
//! K/V vectors are small (kv_dim * 4 bytes = 768 bytes for SmolLM).
//! This avoids breaking the compute encoder or needing a separate copy kernel.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice};

use metal_attention_kernels::buffer::alloc_buffer;

/// GPU-resident KV cache for a single transformer layer.
///
/// Stores key and value projections as contiguous F32 buffers
/// with shape `[max_len, kv_dim]`. New K/V rows are appended
/// at position `len` via CPU memcpy.
pub struct GpuKVCache {
    /// Key cache buffer: `[max_len, kv_dim]` F32.
    k_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Value cache buffer: `[max_len, kv_dim]` F32.
    v_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Current number of filled positions.
    len: usize,
    /// Maximum sequence length (e.g., 2048).
    max_len: usize,
    /// KV dimension (num_kv_heads * head_dim, e.g., 3*64=192 for SmolLM).
    kv_dim: usize,
}

impl GpuKVCache {
    /// Create a new KV cache with pre-allocated buffers.
    ///
    /// # Arguments
    /// - `device`: Metal device for buffer allocation.
    /// - `max_len`: Maximum sequence length (e.g., 2048).
    /// - `kv_dim`: KV dimension (num_kv_heads * head_dim).
    pub fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        max_len: usize,
        kv_dim: usize,
    ) -> Self {
        let buf_size = max_len * kv_dim * std::mem::size_of::<f32>();
        let k_buf = alloc_buffer(device, buf_size);
        let v_buf = alloc_buffer(device, buf_size);

        Self {
            k_buf,
            v_buf,
            len: 0,
            max_len,
            kv_dim,
        }
    }

    /// Append a single K/V row to the cache via CPU-side memcpy.
    ///
    /// Copies `kv_dim` floats from `k_vec` and `v_vec` into the cache
    /// at position `self.len`, then increments `len`.
    ///
    /// For POC this uses CPU memcpy via `contents()` pointer on
    /// StorageModeShared buffers. The K/V vectors are small
    /// (kv_dim * 4 = 768 bytes for SmolLM) so this is negligible overhead.
    ///
    /// # Panics
    /// - If the cache is full (`len >= max_len`).
    /// - If `k_vec` or `v_vec` buffers are smaller than `kv_dim * 4` bytes.
    pub fn append_kv(
        &mut self,
        k_vec_buf: &ProtocolObject<dyn MTLBuffer>,
        v_vec_buf: &ProtocolObject<dyn MTLBuffer>,
    ) {
        assert!(
            self.len < self.max_len,
            "KV cache full: len={} >= max_len={}",
            self.len,
            self.max_len
        );

        let row_bytes = self.kv_dim * std::mem::size_of::<f32>();
        let offset = self.len * row_bytes;

        assert!(
            k_vec_buf.length() >= row_bytes,
            "k_vec_buf too small: {} < {}",
            k_vec_buf.length(),
            row_bytes
        );
        assert!(
            v_vec_buf.length() >= row_bytes,
            "v_vec_buf too small: {} < {}",
            v_vec_buf.length(),
            row_bytes
        );

        unsafe {
            let k_src = k_vec_buf.contents().as_ptr() as *const u8;
            let k_dst = (self.k_buf.contents().as_ptr() as *mut u8).add(offset);
            std::ptr::copy_nonoverlapping(k_src, k_dst, row_bytes);

            let v_src = v_vec_buf.contents().as_ptr() as *const u8;
            let v_dst = (self.v_buf.contents().as_ptr() as *mut u8).add(offset);
            std::ptr::copy_nonoverlapping(v_src, v_dst, row_bytes);
        }

        self.len += 1;
    }

    /// Get the key cache buffer.
    pub fn k_buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.k_buf
    }

    /// Get the value cache buffer.
    pub fn v_buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.v_buf
    }

    /// Current number of filled sequence positions.
    pub fn current_len(&self) -> usize {
        self.len
    }

    /// Maximum sequence length.
    pub fn max_len(&self) -> usize {
        self.max_len
    }

    /// KV dimension (num_kv_heads * head_dim).
    pub fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    /// Reset the cache (clear all stored KV pairs).
    pub fn reset(&mut self) {
        self.len = 0;
    }
}

/// Set of KV caches, one per transformer layer.
pub struct GpuKVCacheSet {
    caches: Vec<GpuKVCache>,
}

impl GpuKVCacheSet {
    /// Create a new KV cache set with one cache per layer.
    ///
    /// # Arguments
    /// - `device`: Metal device for buffer allocation.
    /// - `num_layers`: Number of transformer layers.
    /// - `max_len`: Maximum sequence length (e.g., 2048).
    /// - `kv_dim`: KV dimension (num_kv_heads * head_dim).
    pub fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        num_layers: usize,
        max_len: usize,
        kv_dim: usize,
    ) -> Self {
        let caches = (0..num_layers)
            .map(|_| GpuKVCache::new(device, max_len, kv_dim))
            .collect();

        Self { caches }
    }

    /// Get a mutable reference to the cache for a given layer.
    pub fn cache_mut(&mut self, layer_idx: usize) -> &mut GpuKVCache {
        &mut self.caches[layer_idx]
    }

    /// Get an immutable reference to the cache for a given layer.
    pub fn cache(&self, layer_idx: usize) -> &GpuKVCache {
        &self.caches[layer_idx]
    }

    /// Number of layers.
    pub fn num_layers(&self) -> usize {
        self.caches.len()
    }

    /// Reset all caches (clear all stored KV pairs).
    pub fn reset(&mut self) {
        for cache in &mut self.caches {
            cache.reset();
        }
    }
}
