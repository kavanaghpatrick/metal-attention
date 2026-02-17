//! Root trait for sequence-processing blocks.
//!
//! `SequenceBlock` is the base trait that all model components implement,
//! whether they use softmax attention, linear attention, or SSM mechanisms.

use crate::types::{BlockConfig, TensorView};

/// Root trait for any sequence-processing block.
///
/// A SequenceBlock processes a sequence of token embeddings, producing
/// a transformed sequence of the same shape. It may maintain internal
/// state (e.g., recurrent hidden state, KV cache).
pub trait SequenceBlock: Send + Sync {
    /// Per-layer persistent state type (KV cache for attention, hidden state for SSM).
    type State: Clone + Send;

    /// Initialize empty state for a new sequence.
    fn init_state(&self, config: &BlockConfig) -> Self::State;

    /// Process a full sequence (prefill). Updates state in place.
    /// Input shape: [seq_len, hidden_size]
    /// Output shape: [seq_len, hidden_size]
    fn forward_prefill(
        &self,
        input: &TensorView,
        state: &mut Self::State,
        config: &BlockConfig,
    ) -> TensorView;

    /// Process a single token (decode). Updates state in place.
    /// Input shape: [1, hidden_size]
    /// Output shape: [1, hidden_size]
    fn forward_decode(
        &self,
        input: &TensorView,
        state: &mut Self::State,
        config: &BlockConfig,
    ) -> TensorView;

    /// Memory footprint of this block's state in bytes.
    fn state_size_bytes(&self, config: &BlockConfig) -> usize;
}
