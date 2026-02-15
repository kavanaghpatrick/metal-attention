//! Dense KV cache for softmax attention.
//!
//! Stores K and V projections in contiguous CPU buffers.
//! Supports append (for incremental decode) and full-slice reads (for attention).
//! GPU buffer management deferred to full inference pipeline phase.

/// Default page size in tokens for paged KV cache.
pub const PAGE_SIZE: usize = 16;

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

/// Paged KV cache with block-table indirection for PagedAttention.
///
/// Pages are fixed-size blocks of `page_size` tokens. A block table maps
/// logical page indices to physical page IDs in a contiguous page pool.
/// This enables non-contiguous KV storage with O(1) page allocation/free.
///
/// KV data layout per page: interleaved `[K: page_size * head_dim, V: page_size * head_dim]`.
/// The GPU shader loads pages via `page_table[logical] -> physical` indirection.
#[derive(Debug, Clone)]
pub struct PagedKVCache {
    /// Contiguous K page storage: `[num_allocated_pages, page_size, head_dim]`.
    k_pool: Vec<f32>,
    /// Contiguous V page storage: `[num_allocated_pages, page_size, head_dim]`.
    v_pool: Vec<f32>,
    /// Block table: logical page index -> physical page ID.
    block_table: Vec<u32>,
    /// Free list of physical page IDs available for allocation.
    free_list: Vec<u32>,
    /// Total number of physical pages allocated in the pool.
    total_pages: usize,
    /// Number of logical pages currently in use.
    num_logical_pages: usize,
    /// Number of tokens in the current (last) page.
    current_page_fill: usize,
    /// Total tokens appended across all pages.
    current_len: usize,
    /// Tokens per page.
    page_size: usize,
    /// Per-head dimension.
    head_dim: usize,
}

impl PagedKVCache {
    /// Create a new paged KV cache.
    ///
    /// `max_pages` is the maximum number of physical pages in the pool.
    /// `page_size` is the number of tokens per page (default: PAGE_SIZE).
    /// `head_dim` is the per-head dimension.
    pub fn new(max_pages: usize, page_size: usize, head_dim: usize) -> Self {
        let page_elems = page_size * head_dim;
        // Pre-allocate all pages up front (pool based)
        let k_pool = vec![0.0f32; max_pages * page_elems];
        let v_pool = vec![0.0f32; max_pages * page_elems];

        // All pages start as free
        let free_list: Vec<u32> = (0..max_pages as u32).rev().collect();

        Self {
            k_pool,
            v_pool,
            block_table: Vec::new(),
            free_list,
            total_pages: max_pages,
            num_logical_pages: 0,
            current_page_fill: 0,
            current_len: 0,
            page_size,
            head_dim,
        }
    }

    /// Allocate a physical page from the free list.
    ///
    /// Returns the physical page ID, or panics if no pages are available.
    pub fn allocate_page(&mut self) -> u32 {
        self.free_list
            .pop()
            .expect("PagedKVCache: no free pages available")
    }

    /// Free a physical page, returning it to the free list.
    pub fn free_page(&mut self, page_id: u32) {
        assert!(
            (page_id as usize) < self.total_pages,
            "Invalid page_id: {}",
            page_id
        );
        self.free_list.push(page_id);
    }

    /// Append K and V vectors for new tokens.
    ///
    /// `k_new` and `v_new` must have length `num_tokens * head_dim`.
    /// Automatically allocates new pages as needed.
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

        let page_elems = self.page_size * self.head_dim;
        let mut offset = 0;

        for _ in 0..num_tokens {
            // Need a new page?
            if self.num_logical_pages == 0 || self.current_page_fill == self.page_size {
                let phys = self.allocate_page();
                self.block_table.push(phys);
                self.num_logical_pages += 1;
                self.current_page_fill = 0;
            }

            let phys = self.block_table[self.num_logical_pages - 1] as usize;
            let slot = self.current_page_fill;
            let pool_offset = phys * page_elems + slot * self.head_dim;

            // Copy one token of K and V
            self.k_pool[pool_offset..pool_offset + self.head_dim]
                .copy_from_slice(&k_new[offset..offset + self.head_dim]);
            self.v_pool[pool_offset..pool_offset + self.head_dim]
                .copy_from_slice(&v_new[offset..offset + self.head_dim]);

            self.current_page_fill += 1;
            self.current_len += 1;
            offset += self.head_dim;
        }
    }

    /// Get the block table (logical page -> physical page mapping) for GPU dispatch.
    pub fn block_table(&self) -> &[u32] {
        &self.block_table
    }

    /// Get the contiguous K page pool for GPU dispatch.
    ///
    /// Layout: `[total_pages, page_size, head_dim]`.
    pub fn k_pages(&self) -> &[f32] {
        &self.k_pool
    }

    /// Get the contiguous V page pool for GPU dispatch.
    ///
    /// Layout: `[total_pages, page_size, head_dim]`.
    pub fn v_pages(&self) -> &[f32] {
        &self.v_pool
    }

    /// Current number of cached tokens.
    pub fn len(&self) -> usize {
        self.current_len
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.current_len == 0
    }

    /// Page size (tokens per page).
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Head dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Number of logical pages in use.
    pub fn num_pages(&self) -> usize {
        self.num_logical_pages
    }

    /// Number of free pages remaining.
    pub fn free_pages(&self) -> usize {
        self.free_list.len()
    }

    /// Reset the cache to empty (does not deallocate pool).
    pub fn reset(&mut self) {
        // Return all block table pages to free list
        for &phys in &self.block_table {
            self.free_list.push(phys);
        }
        self.block_table.clear();
        self.num_logical_pages = 0;
        self.current_page_fill = 0;
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

    // ===================== PagedKVCache tests =====================

    #[test]
    fn test_paged_kv_cache_new() {
        let cache = PagedKVCache::new(8, 4, 4);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.page_size(), 4);
        assert_eq!(cache.head_dim(), 4);
        assert_eq!(cache.num_pages(), 0);
        assert_eq!(cache.free_pages(), 8);
        assert!(cache.block_table().is_empty());
    }

    #[test]
    fn test_paged_kv_cache_allocate_free() {
        let mut cache = PagedKVCache::new(4, 4, 4);
        assert_eq!(cache.free_pages(), 4);

        let p0 = cache.allocate_page();
        assert_eq!(cache.free_pages(), 3);

        let p1 = cache.allocate_page();
        assert_eq!(cache.free_pages(), 2);
        assert_ne!(p0, p1);

        cache.free_page(p0);
        assert_eq!(cache.free_pages(), 3);

        cache.free_page(p1);
        assert_eq!(cache.free_pages(), 4);
    }

    #[test]
    fn test_paged_kv_cache_append_single_page() {
        let head_dim = 4;
        let page_size = 4;
        let mut cache = PagedKVCache::new(4, page_size, head_dim);

        // Append 2 tokens (fits in one page)
        let k = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let v = vec![5.0, 6.0, 7.0, 8.0, 50.0, 60.0, 70.0, 80.0];
        cache.append(&k, &v);

        assert_eq!(cache.len(), 2);
        assert_eq!(cache.num_pages(), 1);
        assert_eq!(cache.block_table().len(), 1);
        assert_eq!(cache.free_pages(), 3);

        // Verify data in the page pool
        let phys = cache.block_table()[0] as usize;
        let page_elems = page_size * head_dim;
        let k_start = phys * page_elems;
        assert_eq!(cache.k_pages()[k_start..k_start + head_dim], [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(
            cache.k_pages()[k_start + head_dim..k_start + 2 * head_dim],
            [10.0, 20.0, 30.0, 40.0]
        );
        assert_eq!(cache.v_pages()[k_start..k_start + head_dim], [5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_paged_kv_cache_append_spans_pages() {
        let head_dim = 4;
        let page_size = 2; // small page so 3 tokens -> 2 pages
        let mut cache = PagedKVCache::new(4, page_size, head_dim);

        // Append 3 tokens
        let k: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let v: Vec<f32> = (100..112).map(|i| i as f32).collect();
        cache.append(&k, &v);

        assert_eq!(cache.len(), 3);
        assert_eq!(cache.num_pages(), 2); // page 0: 2 tokens, page 1: 1 token
        assert_eq!(cache.block_table().len(), 2);
        assert_eq!(cache.free_pages(), 2);
    }

    #[test]
    fn test_paged_kv_cache_block_table_correct() {
        let head_dim = 2;
        let page_size = 2;
        let mut cache = PagedKVCache::new(8, page_size, head_dim);

        // Append 5 tokens -> 3 pages
        for i in 0..5 {
            let k = vec![(i * 2) as f32, (i * 2 + 1) as f32];
            let v = vec![(i * 2 + 100) as f32, (i * 2 + 101) as f32];
            cache.append(&k, &v);
        }

        assert_eq!(cache.len(), 5);
        assert_eq!(cache.num_pages(), 3);

        // Each block table entry is a valid physical page
        for &phys in cache.block_table() {
            assert!((phys as usize) < 8, "Physical page ID out of range");
        }

        // All block table entries should be distinct
        let mut seen = std::collections::HashSet::new();
        for &phys in cache.block_table() {
            assert!(seen.insert(phys), "Duplicate physical page in block table");
        }
    }

    #[test]
    fn test_paged_kv_cache_reset() {
        let mut cache = PagedKVCache::new(4, 2, 4);
        cache.append(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.num_pages(), 1);

        cache.reset();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.num_pages(), 0);
        assert_eq!(cache.free_pages(), 4);
    }

    #[test]
    #[should_panic(expected = "no free pages")]
    fn test_paged_kv_cache_overflow() {
        let mut cache = PagedKVCache::new(1, 2, 2); // 1 page, 2 tokens per page
        // Fill the one page
        cache.append(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]);
        // This requires a new page but none are free -> panic
        cache.append(&[1.0, 2.0], &[5.0, 6.0]);
    }
}
