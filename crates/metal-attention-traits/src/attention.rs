//! Softmax attention trait with KV cache, GQA, and position encoding support.
//!
//! Extends `SequenceBlock` with attention-specific operations.

use crate::sequence::SequenceBlock;
use crate::types::{BlockConfig, TensorView};

/// Position encoding variant for softmax attention.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PositionEncoding {
    None,
    RoPE { theta_base: f32 },
    ALiBi,
}

/// KV cache strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KVCacheMode {
    /// Contiguous buffer, simple indexing.
    Dense,
    /// PagedAttention V2 with block table indirection.
    Paged { page_size: u32 },
}

/// GQA configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GQAConfig {
    /// Number of Q heads per KV head group. 1 = MHA, num_heads = MQA.
    pub group_size: usize,
}

/// A softmax attention block with O(N^2) compute and KV cache.
///
/// Processes sequences using scaled dot-product attention with
/// Flash Attention kernel implementation (tiled, online softmax,
/// simdgroup_matrix).
///
/// Examples: standard multi-head attention, GQA, MQA.
pub trait SoftmaxAttention: SequenceBlock {
    /// Current sequence length in the KV cache.
    fn cached_length(&self, state: &Self::State) -> usize;

    /// Maximum sequence length this cache can hold.
    fn max_length(&self, state: &Self::State) -> usize;

    /// Position encoding used by this attention implementation.
    fn position_encoding(&self) -> PositionEncoding;

    /// KV cache mode (dense or paged).
    fn cache_mode(&self) -> KVCacheMode;

    /// GQA configuration.
    fn gqa_config(&self) -> GQAConfig;

    /// Prefill: compute attention over full prompt and populate KV cache.
    /// Uses Flash Attention kernel for O(N^2) attention with tiling.
    fn prefill_attention(
        &self,
        q: &TensorView,
        k: &TensorView,
        v: &TensorView,
        state: &mut Self::State,
        config: &BlockConfig,
    ) -> TensorView;

    /// Decode: compute single-query attention against full KV cache.
    /// Appends new K/V to cache, computes attention over all cached positions.
    fn decode_attention(
        &self,
        q: &TensorView,
        k: &TensorView,
        v: &TensorView,
        state: &mut Self::State,
        config: &BlockConfig,
    ) -> TensorView;
}
