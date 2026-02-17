//! Linear sequence model trait for O(N) / O(1) per-token processing.
//!
//! Extends `SequenceBlock` with chunk-based prefill and recurrent decode.

use crate::sequence::SequenceBlock;
use crate::types::{BlockConfig, TensorView};

/// Position encoding configuration for linear models (most use none).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearPositionEncoding {
    None,
    /// RWKV-style token shift.
    TokenShift,
}

/// A sequence block with O(N) or O(1) per-token processing.
///
/// LinearSequenceModels maintain a fixed-size hidden state that is updated
/// per token (or per chunk during prefill). No KV cache is needed --
/// the entire context is compressed into the state matrix.
///
/// Examples: FLA linear attention, Mamba SSM, RG-LRU, RWKV-7 blocks.
pub trait LinearSequenceModel: SequenceBlock {
    /// Process input in chunks during prefill (parallel over chunks).
    /// chunk_size is selected based on head_dim and 32KB threadgroup memory budget.
    fn prefill_chunked(
        &self,
        input: &TensorView,
        state: &mut Self::State,
        config: &BlockConfig,
        chunk_size: usize,
    ) -> TensorView;

    /// Single-token recurrent update during decode. O(D^2) or O(d_model * d_state).
    fn decode_step(
        &self,
        input: &TensorView,
        state: &mut Self::State,
        config: &BlockConfig,
    ) -> TensorView;

    /// Maximum supported head dimension for this implementation.
    fn max_head_dim(&self) -> usize;

    /// Optimal chunk size for the given head dimension (respecting 32KB threadgroup limit).
    fn optimal_chunk_size(&self, head_dim: usize) -> usize;
}
