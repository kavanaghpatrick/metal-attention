//! HybridModel: engine that wraps model layers + embeddings + lm_head.
//!
//! Constructs a full model from GGUF-loaded weights (or random weights for testing).
//! Supports schedule-driven dispatch through RWKV-7 blocks.

use metal_attention_models::rwkv7::{Rwkv7Block, Rwkv7State};
use metal_attention_models::registry::ModelConfig;
use metal_attention_gguf::ModelArchitecture;
use metal_attention_traits::types::BlockConfig;
use metal_attention_traits::sequence::SequenceBlock;

use crate::sampling::SimpleRng;

/// A model layer variant for enum-based dispatch.
///
/// Using an enum instead of `dyn SequenceBlock` because
/// `SequenceBlock` has associated types that prevent object safety.
pub enum ModelLayer {
    /// RWKV-7 linear recurrent block.
    Rwkv7(Rwkv7Block),
    // Future: Flash, Jamba, Griffin, etc.
}

/// Per-layer state, mirroring the ModelLayer enum.
pub enum LayerState {
    Rwkv7(Rwkv7State),
}

impl Clone for LayerState {
    fn clone(&self) -> Self {
        match self {
            LayerState::Rwkv7(s) => LayerState::Rwkv7(s.clone()),
        }
    }
}

/// Full model state across all layers.
pub struct ModelState {
    pub layer_states: Vec<LayerState>,
}

impl Clone for ModelState {
    fn clone(&self) -> Self {
        Self {
            layer_states: self.layer_states.clone(),
        }
    }
}

/// The hybrid model engine tying together embeddings, layers, and lm_head.
pub struct HybridModel {
    /// Token embedding matrix: [vocab_size, hidden_size], row-major.
    pub embed_weight: Vec<f32>,
    /// Layer normalization weight before lm_head: [hidden_size].
    pub final_norm_weight: Vec<f32>,
    /// Language model head projection: [vocab_size, hidden_size], row-major.
    pub lm_head_weight: Vec<f32>,
    /// Sequence processing layers.
    pub layers: Vec<ModelLayer>,
    /// Model configuration.
    pub config: ModelConfig,
    /// Block config used for SequenceBlock trait methods.
    pub block_config: BlockConfig,
}

impl HybridModel {
    /// Construct a model with random weights for testing.
    ///
    /// Creates `num_layers` RWKV-7 blocks with random weights,
    /// plus random embedding, norm, and lm_head weights.
    pub fn random(
        vocab_size: usize,
        hidden_size: usize,
        head_dim: usize,
        num_heads: usize,
        num_layers: usize,
        seed: u64,
    ) -> Self {
        let mut rng = SimpleRng::new(seed);

        // Embedding: small random init
        let scale = 1.0 / (hidden_size as f32).sqrt();
        let embed_weight: Vec<f32> = (0..vocab_size * hidden_size)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale)
            .collect();

        // Final norm: initialize to 1.0 (RMSNorm identity)
        let final_norm_weight = vec![1.0f32; hidden_size];

        // LM head: small random init
        let lm_head_weight: Vec<f32> = (0..vocab_size * hidden_size)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale)
            .collect();

        // Layers
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let block = Rwkv7Block::random(hidden_size, head_dim, num_heads, seed + i as u64 * 31 + 7);
            layers.push(ModelLayer::Rwkv7(block));
        }

        let config = ModelConfig {
            architecture: ModelArchitecture::Rwkv,
            hidden_size,
            head_dim,
            num_heads,
            num_layers,
        };

        let block_config = BlockConfig {
            hidden_size,
            head_dim,
            num_heads,
            num_kv_heads: num_heads,
            layer_index: 0,
        };

        Self {
            embed_weight,
            final_norm_weight,
            lm_head_weight,
            layers,
            config,
            block_config,
        }
    }

    /// Initialize fresh model state for all layers.
    pub fn init_state(&self) -> ModelState {
        let layer_states = self
            .layers
            .iter()
            .map(|layer| match layer {
                ModelLayer::Rwkv7(block) => {
                    LayerState::Rwkv7(block.init_state(&self.block_config))
                }
            })
            .collect();
        ModelState { layer_states }
    }

    /// Look up token embedding: returns [hidden_size] vector.
    pub fn embed(&self, token_id: u32) -> Vec<f32> {
        let hs = self.config.hidden_size;
        let start = (token_id as usize) * hs;
        if start + hs > self.embed_weight.len() {
            // Out of vocab: return zeros
            return vec![0.0f32; hs];
        }
        self.embed_weight[start..start + hs].to_vec()
    }

    /// Apply RMSNorm: y = x * weight / rms(x).
    pub fn rmsnorm(&self, x: &[f32]) -> Vec<f32> {
        let n = x.len();
        let eps = 1e-5f32;
        let rms = (x.iter().map(|&v| v * v).sum::<f32>() / n as f32 + eps).sqrt();
        x.iter()
            .zip(self.final_norm_weight.iter())
            .map(|(&xi, &wi)| xi / rms * wi)
            .collect()
    }

    /// Project hidden state to logits via lm_head: [hidden_size] -> [vocab_size].
    pub fn lm_head(&self, hidden: &[f32]) -> Vec<f32> {
        let vocab_size = self.vocab_size();
        let hs = self.config.hidden_size;
        let mut logits = vec![0.0f32; vocab_size];
        for i in 0..vocab_size {
            let mut acc = 0.0f32;
            for j in 0..hs {
                acc += self.lm_head_weight[i * hs + j] * hidden[j];
            }
            logits[i] = acc;
        }
        logits
    }

    /// Forward pass for a single token through all layers.
    ///
    /// Returns logits [vocab_size].
    pub fn forward_token(&self, token_id: u32, state: &mut ModelState) -> Vec<f32> {
        // 1. Embed
        let mut hidden = self.embed(token_id);

        // 2. Per-layer forward
        for (i, layer) in self.layers.iter().enumerate() {
            match layer {
                ModelLayer::Rwkv7(block) => {
                    let LayerState::Rwkv7(ref mut layer_state) = state.layer_states[i];
                    hidden = block.process_token(&hidden, layer_state);
                }
            }
        }

        // 3. Final normalization
        hidden = self.rmsnorm(&hidden);

        // 4. LM head projection -> logits
        self.lm_head(&hidden)
    }

    /// Get the vocabulary size from the embedding matrix dimensions.
    pub fn vocab_size(&self) -> usize {
        self.embed_weight.len() / self.config.hidden_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_model_construction() {
        let model = HybridModel::random(32, 16, 16, 1, 2, 42);
        assert_eq!(model.vocab_size(), 32);
        assert_eq!(model.layers.len(), 2);
        assert_eq!(model.config.hidden_size, 16);
    }

    #[test]
    fn test_embed_lookup() {
        let model = HybridModel::random(32, 16, 16, 1, 1, 42);
        let emb = model.embed(0);
        assert_eq!(emb.len(), 16);
        // Should not be all zeros (random init)
        assert!(emb.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn test_embed_out_of_vocab() {
        let model = HybridModel::random(32, 16, 16, 1, 1, 42);
        let emb = model.embed(999);
        assert_eq!(emb.len(), 16);
        assert!(emb.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_rmsnorm_output_shape() {
        let model = HybridModel::random(32, 16, 16, 1, 1, 42);
        let input: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let output = model.rmsnorm(&input);
        assert_eq!(output.len(), 16);
        for &v in &output {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_lm_head_output_shape() {
        let model = HybridModel::random(32, 16, 16, 1, 1, 42);
        let hidden: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let logits = model.lm_head(&hidden);
        assert_eq!(logits.len(), 32);
        for &v in &logits {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_forward_token_produces_valid_logits() {
        let model = HybridModel::random(32, 16, 16, 1, 2, 42);
        let mut state = model.init_state();
        let logits = model.forward_token(5, &mut state);
        assert_eq!(logits.len(), 32);
        for &v in &logits {
            assert!(v.is_finite(), "logit must be finite, got {}", v);
        }
    }

    #[test]
    fn test_init_state_layer_count() {
        let model = HybridModel::random(32, 16, 16, 1, 4, 42);
        let state = model.init_state();
        assert_eq!(state.layer_states.len(), 4);
    }
}
