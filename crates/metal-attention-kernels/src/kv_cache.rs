//! Dense KV cache for softmax attention.
//!
//! Stores K and V projections in contiguous CPU buffers.
//! Supports append (for incremental decode) and full-slice reads (for attention).
//! GPU buffer management deferred to full inference pipeline phase.

/// Dense KV cache storing K and V as contiguous f32 slices.
///
/// Layout: K is `[max_seq, head_dim]`, V is `[max_seq, head_dim]`.
/// `current_len` tracks how many positions are populated.
#[derive(Debug, Clone)]
pub struct DenseKVCache {
    /// Key buffer: [max_seq, head_dim] row-major.
    k: Vec<f32>,
    /// Value buffer: [max_seq, head_dim] row-major.
    v: Vec<f32>,
    /// Number of populated positions (0..current_len are valid).
    current_len: usize,
    /// Maximum sequence length this cache can hold.
    max_seq: usize,
    /// Per-head dimension.
    head_dim: usize,
}

impl DenseKVCache {
    /// Create a new empty KV cache.
    pub fn new(max_seq: usize, head_dim: usize) -> Self {
        Self {
            k: vec![0.0; max_seq * head_dim],
            v: vec![0.0; max_seq * head_dim],
            current_len: 0,
            max_seq,
            head_dim,
        }
    }

    /// Append new K/V vectors to the cache.
    ///
    /// `k_new` and `v_new` must have length `num_tokens * head_dim`.
    /// Panics if appending would exceed `max_seq`.
    pub fn append(&mut self, k_new: &[f32], v_new: &[f32]) {
        let num_tokens = k_new.len() / self.head_dim;
        assert_eq!(
            k_new.len(),
            num_tokens * self.head_dim,
            "k_new length must be a multiple of head_dim"
        );
        assert_eq!(
            k_new.len(),
            v_new.len(),
            "k_new and v_new must have the same length"
        );
        assert!(
            self.current_len + num_tokens <= self.max_seq,
            "KV cache overflow: {} + {} > {}",
            self.current_len,
            num_tokens,
            self.max_seq
        );

        let start = self.current_len * self.head_dim;
        let end = start + num_tokens * self.head_dim;
        self.k[start..end].copy_from_slice(k_new);
        self.v[start..end].copy_from_slice(v_new);
        self.current_len += num_tokens;
    }

    /// Get the filled K slice: `[current_len * head_dim]`.
    pub fn k_slice(&self) -> &[f32] {
        &self.k[..self.current_len * self.head_dim]
    }

    /// Get the filled V slice: `[current_len * head_dim]`.
    pub fn v_slice(&self) -> &[f32] {
        &self.v[..self.current_len * self.head_dim]
    }

    /// Current number of cached positions.
    pub fn len(&self) -> usize {
        self.current_len
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.current_len == 0
    }

    /// Maximum sequence length.
    pub fn max_len(&self) -> usize {
        self.max_seq
    }

    /// Head dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Reset the cache to empty (does not deallocate).
    pub fn reset(&mut self) {
        self.current_len = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_new() {
        let cache = DenseKVCache::new(128, 64);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.max_len(), 128);
        assert_eq!(cache.head_dim(), 64);
        assert_eq!(cache.k_slice().len(), 0);
        assert_eq!(cache.v_slice().len(), 0);
    }

    #[test]
    fn test_kv_cache_append_single() {
        let mut cache = DenseKVCache::new(128, 4);
        let k = vec![1.0, 2.0, 3.0, 4.0];
        let v = vec![5.0, 6.0, 7.0, 8.0];
        cache.append(&k, &v);

        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());
        assert_eq!(cache.k_slice(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(cache.v_slice(), &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_kv_cache_append_multiple() {
        let mut cache = DenseKVCache::new(128, 4);

        // Append 3 tokens at once
        let k = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0];
        let v = vec![5.0, 6.0, 7.0, 8.0, 50.0, 60.0, 70.0, 80.0, 500.0, 600.0, 700.0, 800.0];
        cache.append(&k, &v);

        assert_eq!(cache.len(), 3);
        assert_eq!(cache.k_slice().len(), 12);
        assert_eq!(cache.v_slice().len(), 12);

        // Verify data integrity
        assert_eq!(cache.k_slice()[0], 1.0);
        assert_eq!(cache.k_slice()[4], 10.0);
        assert_eq!(cache.k_slice()[8], 100.0);
        assert_eq!(cache.v_slice()[0], 5.0);
        assert_eq!(cache.v_slice()[4], 50.0);
        assert_eq!(cache.v_slice()[8], 500.0);
    }

    #[test]
    fn test_kv_cache_incremental_append() {
        let mut cache = DenseKVCache::new(128, 4);

        // Append one at a time (decode pattern)
        cache.append(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(cache.len(), 1);

        cache.append(&[10.0, 20.0, 30.0, 40.0], &[50.0, 60.0, 70.0, 80.0]);
        assert_eq!(cache.len(), 2);

        // Verify both tokens are accessible
        assert_eq!(cache.k_slice(), &[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]);
        assert_eq!(cache.v_slice(), &[5.0, 6.0, 7.0, 8.0, 50.0, 60.0, 70.0, 80.0]);
    }

    #[test]
    fn test_kv_cache_reset() {
        let mut cache = DenseKVCache::new(128, 4);
        cache.append(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(cache.len(), 1);

        cache.reset();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.k_slice().len(), 0);
    }

    #[test]
    #[should_panic(expected = "KV cache overflow")]
    fn test_kv_cache_overflow() {
        let mut cache = DenseKVCache::new(2, 4);
        cache.append(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]);
        cache.append(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]);
        // This should panic
        cache.append(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]);
    }
}
