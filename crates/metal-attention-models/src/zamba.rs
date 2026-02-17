//! Zamba hybrid model: 6:1 Mamba:SharedAttention + LoRA projectors.
//!
//! Zamba interleaves Mamba SSM blocks with a single shared FlashAttention block
//! in a 6:1 ratio. The key innovation is that one attention block is reused
//! (shared weights) at every attention position, with small LoRA (Low-Rank
//! Adaptation) projectors per depth position to specialize the shared attention.
//!
//! Architecture:
//!   - 6 Mamba blocks per 1 shared attention block
//!   - Shared attention: one FlashAttentionLayer instance, applied at every 7th layer
//!   - LoRA projectors: small rank-r adaptation per depth position (simplified)
//!
//! This reduces parameter count compared to having unique attention blocks.

use crate::flash_attn::FlashAttentionState;
use crate::llama::LlamaLayer;
use crate::mamba::{MambaBlock, MambaState};
use metal_attention_traits::schedule::{LayerSchedule, LayerType};
use metal_attention_traits::sequence::SequenceBlock;
use metal_attention_traits::types::BlockConfig;

/// LoRA (Low-Rank Adaptation) projector for shared attention specialization.
///
/// Applies a low-rank residual: output = input + scale * (B * (A * input))
/// where A: [rank, hidden_size] and B: [hidden_size, rank].
pub struct LoraProjector {
    /// Hidden size (model dimension).
    pub hidden_size: usize,
    /// LoRA rank (bottleneck dimension).
    pub rank: usize,
    /// Down-projection: [rank, hidden_size], row-major.
    pub w_a: Vec<f32>,
    /// Up-projection: [hidden_size, rank], row-major.
    pub w_b: Vec<f32>,
    /// Scaling factor for the LoRA residual.
    pub scale: f32,
}

impl LoraProjector {
    /// Create a LoRA projector with random weights for testing.
    pub fn random(hidden_size: usize, rank: usize, seed: u64) -> Self {
        let mut rng = SimpleRng::new(seed);
        let scale_init = 1.0 / (rank as f32).sqrt();

        let w_a: Vec<f32> = (0..rank * hidden_size)
            .map(|_| rng.next_f32_range(-scale_init, scale_init))
            .collect();
        // B is initialized to near-zero so LoRA starts as identity
        let w_b: Vec<f32> = (0..hidden_size * rank)
            .map(|_| rng.next_f32_range(-0.01, 0.01))
            .collect();

        Self {
            hidden_size,
            rank,
            w_a,
            w_b,
            scale: 1.0,
        }
    }

    /// Apply LoRA: output = input + scale * B * (A * input).
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let hs = self.hidden_size;
        let r = self.rank;

        // Down-project: A * input -> [rank]
        let down = matvec(&self.w_a, input, r, hs);

        // Up-project: B * down -> [hidden_size]
        let up = matvec(&self.w_b, &down, hs, r);

        // Residual: input + scale * up
        let mut output = Vec::with_capacity(hs);
        for i in 0..hs {
            output.push(input[i] + self.scale * up[i]);
        }
        output
    }
}

/// State for a single Zamba layer (either Mamba or SharedAttention).
#[derive(Clone)]
pub enum ZambaLayerState {
    /// Mamba SSM state.
    Mamba(MambaState),
    /// SharedAttention state (KV cache).
    Attention(FlashAttentionState),
}

/// A single Zamba layer: either Mamba SSM or SharedAttention (with LoRA).
///
/// Mamba layers use MambaBlock for sequence mixing.
/// Attention layers use a shared FlashAttention block with a per-position LoRA projector.
pub enum ZambaLayer {
    /// Mamba SSM block.
    Mamba(MambaBlock),
    /// Shared attention with per-position LoRA projector.
    /// The `attn_index` identifies which shared attention instance this maps to
    /// (all map to the same shared weights, but have unique LoRA + KV cache).
    Attention {
        /// Index into the shared attention positions (for LoRA lookup).
        attn_index: usize,
    },
}

impl ZambaLayer {
    /// Check whether this is a Mamba layer.
    pub fn is_mamba(&self) -> bool {
        matches!(self, ZambaLayer::Mamba(_))
    }

    /// Check whether this is an Attention layer.
    pub fn is_attention(&self) -> bool {
        matches!(self, ZambaLayer::Attention { .. })
    }
}

/// Full Zamba model containing layers, shared attention, and LoRA projectors.
pub struct ZambaModel {
    /// All layers (Mamba or Attention references).
    pub layers: Vec<ZambaLayer>,
    /// The single shared attention block (LlamaLayer includes RMSNorm + Attention + FFN).
    pub shared_attention: LlamaLayer,
    /// Per-attention-position LoRA projectors.
    pub lora_projectors: Vec<LoraProjector>,
    /// Layer schedule.
    pub schedule: LayerSchedule,
}

impl ZambaModel {
    /// Initialize state for all layers.
    pub fn init_states(&self, config: &BlockConfig) -> Vec<ZambaLayerState> {
        self.layers
            .iter()
            .map(|layer| match layer {
                ZambaLayer::Mamba(block) => ZambaLayerState::Mamba(block.init_state(config)),
                ZambaLayer::Attention { .. } => {
                    ZambaLayerState::Attention(self.shared_attention.attention.init_state(config))
                }
            })
            .collect()
    }

    /// Process a single token through one layer.
    ///
    /// For Mamba layers: runs the MambaBlock.
    /// For Attention layers: applies LoRA, runs shared attention, residual.
    pub fn process_token(
        &self,
        layer_index: usize,
        input: &[f32],
        state: &mut ZambaLayerState,
    ) -> Vec<f32> {
        match (&self.layers[layer_index], state) {
            (ZambaLayer::Mamba(block), ZambaLayerState::Mamba(mamba_state)) => {
                // 1. Mamba SSM
                let ssm_out = block.process_token(input, mamba_state);

                // 2. Residual connection
                add(input, &ssm_out)
            }
            (ZambaLayer::Attention { attn_index }, ZambaLayerState::Attention(attn_state)) => {
                // 1. Apply LoRA projector to specialize input for this position
                let lora_input = self.lora_projectors[*attn_index].forward(input);

                // 2. Run shared attention (LlamaLayer)
                let mut llama_state = crate::llama::LlamaState {
                    attn_state: attn_state.clone(),
                };

                let output = if attn_state.kv_cache.is_empty() {
                    self.shared_attention
                        .process_prefill(&lora_input, &mut llama_state, 1)
                } else {
                    self.shared_attention
                        .process_token(&lora_input, &mut llama_state)
                };

                *attn_state = llama_state.attn_state;
                output
            }
            _ => panic!("ZambaLayer/ZambaLayerState type mismatch"),
        }
    }

    /// Process a single token through all layers.
    ///
    /// Returns the output hidden state [hidden_size].
    pub fn forward_all(&self, input: &[f32], states: &mut [ZambaLayerState]) -> Vec<f32> {
        let mut hidden = input.to_vec();
        for (i, state) in states.iter_mut().enumerate().take(self.layers.len()) {
            hidden = self.process_token(i, &hidden, state);
        }
        hidden
    }
}

/// Build a full Zamba model using 6:1 Mamba:SharedAttention schedule.
///
/// Returns a ZambaModel with the appropriate mix of Mamba and shared Attention layers.
#[allow(clippy::too_many_arguments)]
pub fn build_zamba_model(
    hidden_size: usize,
    head_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    num_layers: usize,
    d_state: usize,
    lora_rank: usize,
    seed: u64,
) -> ZambaModel {
    let schedule = LayerSchedule::periodic(num_layers, 6);
    let mut layers = Vec::with_capacity(num_layers);
    let mut attn_index = 0;
    let attn_count = schedule.attention_count();

    // Build shared attention block (one instance, shared weights)
    let shared_attention =
        LlamaLayer::random(hidden_size, head_dim, num_heads, num_kv_heads, seed + 9999);

    // Build LoRA projectors (one per attention position)
    let mut lora_projectors = Vec::with_capacity(attn_count);
    for a in 0..attn_count {
        lora_projectors.push(LoraProjector::random(
            hidden_size,
            lora_rank,
            seed + 5000 + (a as u64) * 37,
        ));
    }

    // Build layers
    for i in 0..num_layers {
        let layer_seed = seed + (i as u64) * 31 + 7;
        match schedule.layer_type(i).unwrap() {
            LayerType::Linear => {
                let block = MambaBlock::random(hidden_size, d_state, layer_seed);
                layers.push(ZambaLayer::Mamba(block));
            }
            LayerType::Attention => {
                layers.push(ZambaLayer::Attention { attn_index });
                attn_index += 1;
            }
        }
    }

    ZambaModel {
        layers,
        shared_attention,
        lora_projectors,
        schedule,
    }
}

/// Matrix-vector multiply: y = W * x, W is [out_dim, in_dim].
fn matvec(w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; out_dim];
    for i in 0..out_dim {
        let mut acc = 0.0f32;
        for j in 0..in_dim {
            acc += w[i * in_dim + j] * x[j];
        }
        y[i] = acc;
    }
    y
}

/// Element-wise addition.
fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
}

/// Simple deterministic pseudo-random number generator for test weight initialization.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() & 0xFFFFFF) as f32 / 0xFFFFFF as f32
    }

    fn next_f32_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_f32() * (hi - lo)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal_attention_traits::schedule::LayerType;

    fn make_config(
        hidden_size: usize,
        head_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
    ) -> BlockConfig {
        BlockConfig {
            hidden_size,
            head_dim,
            num_heads,
            num_kv_heads,
            layer_index: 0,
        }
    }

    #[test]
    fn test_zamba_layer_schedule_6_1() {
        let hidden_size = 16;
        let head_dim = 8;
        let num_heads = 2;
        let num_kv_heads = 2;
        let num_layers = 14;
        let d_state = 4;
        let lora_rank = 4;

        let model = build_zamba_model(
            hidden_size,
            head_dim,
            num_heads,
            num_kv_heads,
            num_layers,
            d_state,
            lora_rank,
            42,
        );

        assert_eq!(model.layers.len(), num_layers);

        // Verify 6:1 pattern
        for (i, layer) in model.layers.iter().enumerate() {
            let expected_type = model.schedule.layer_type(i).unwrap();
            match expected_type {
                LayerType::Linear => {
                    assert!(
                        layer.is_mamba(),
                        "Layer {} should be Mamba, got Attention",
                        i
                    );
                }
                LayerType::Attention => {
                    assert!(
                        layer.is_attention(),
                        "Layer {} should be Attention, got Mamba",
                        i
                    );
                }
            }
        }

        // Count: 12 Mamba, 2 Attention for 14 layers
        let mamba_count = model.layers.iter().filter(|l| l.is_mamba()).count();
        let attn_count = model.layers.iter().filter(|l| l.is_attention()).count();
        assert_eq!(mamba_count, 12, "Should have 12 Mamba layers");
        assert_eq!(attn_count, 2, "Should have 2 Attention layers");

        // Verify shared attention: only one attention block
        // Verify LoRA projectors match attention count
        assert_eq!(
            model.lora_projectors.len(),
            attn_count,
            "Should have one LoRA projector per attention position"
        );
    }

    #[test]
    fn test_zamba_shared_attention_reuse() {
        // Verify that all attention layers share the same underlying weights
        let hidden_size = 16;
        let head_dim = 8;
        let num_heads = 2;
        let num_kv_heads = 2;
        let num_layers = 14;
        let d_state = 4;
        let lora_rank = 4;

        let model = build_zamba_model(
            hidden_size,
            head_dim,
            num_heads,
            num_kv_heads,
            num_layers,
            d_state,
            lora_rank,
            42,
        );

        // The shared_attention field is a single LlamaLayer -- all attention positions
        // reference it. Verify the shared attention has the correct dimensions.
        assert_eq!(model.shared_attention.hidden_size, hidden_size);
        assert_eq!(model.shared_attention.attention.num_heads, num_heads);
        assert_eq!(model.shared_attention.attention.num_kv_heads, num_kv_heads);
        assert_eq!(model.shared_attention.attention.head_dim, head_dim);

        // All Attention layers reference attn_index 0..N-1
        let attn_indices: Vec<usize> = model
            .layers
            .iter()
            .filter_map(|l| {
                if let ZambaLayer::Attention { attn_index } = l {
                    Some(*attn_index)
                } else {
                    None
                }
            })
            .collect();

        // Should be sequential: 0, 1
        for (i, &idx) in attn_indices.iter().enumerate() {
            assert_eq!(idx, i, "Attention index {} should be {}", idx, i);
        }
    }

    #[test]
    fn test_zamba_lora_projector() {
        let hidden_size = 16;
        let rank = 4;
        let lora = LoraProjector::random(hidden_size, rank, 42);

        let mut rng = SimpleRng::new(100);
        let input: Vec<f32> = (0..hidden_size)
            .map(|_| rng.next_f32_range(-0.5, 0.5))
            .collect();

        let output = lora.forward(&input);

        assert_eq!(output.len(), hidden_size);
        for (i, &val) in output.iter().enumerate() {
            assert!(val.is_finite(), "LoRA output[{}] not finite: {}", i, val);
        }

        // Output should differ from input (LoRA adds a residual)
        let differs = input
            .iter()
            .zip(output.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-10);
        assert!(
            differs,
            "LoRA output should differ from input (residual added)"
        );
    }

    #[test]
    fn test_zamba_mamba_layer_forward() {
        let hidden_size = 16;
        let head_dim = 8;
        let num_heads = 2;
        let num_kv_heads = 2;
        let num_layers = 7;
        let d_state = 4;
        let lora_rank = 4;

        let model = build_zamba_model(
            hidden_size,
            head_dim,
            num_heads,
            num_kv_heads,
            num_layers,
            d_state,
            lora_rank,
            42,
        );

        let config = make_config(hidden_size, head_dim, num_heads, num_kv_heads);
        let mut states = model.init_states(&config);

        let mut rng = SimpleRng::new(100);
        let input: Vec<f32> = (0..hidden_size)
            .map(|_| rng.next_f32_range(-0.5, 0.5))
            .collect();

        // Process through first layer (Mamba)
        assert!(model.layers[0].is_mamba());
        let output = model.process_token(0, &input, &mut states[0]);
        assert_eq!(output.len(), hidden_size);
        for (i, &val) in output.iter().enumerate() {
            assert!(
                val.is_finite(),
                "Mamba layer output[{}] not finite: {}",
                i,
                val
            );
        }
    }

    #[test]
    fn test_zamba_hybrid_dispatch() {
        // Build a 7-layer Zamba model (6 Mamba + 1 Attention) and run through all
        let hidden_size = 16;
        let head_dim = 8;
        let num_heads = 2;
        let num_kv_heads = 2;
        let num_layers = 7;
        let d_state = 4;
        let lora_rank = 4;

        let model = build_zamba_model(
            hidden_size,
            head_dim,
            num_heads,
            num_kv_heads,
            num_layers,
            d_state,
            lora_rank,
            42,
        );

        let config = make_config(hidden_size, head_dim, num_heads, num_kv_heads);
        let mut states = model.init_states(&config);

        let mut rng = SimpleRng::new(300);
        let input: Vec<f32> = (0..hidden_size)
            .map(|_| rng.next_f32_range(-0.5, 0.5))
            .collect();

        let output = model.forward_all(&input, &mut states);

        assert_eq!(output.len(), hidden_size);
        for (i, &val) in output.iter().enumerate() {
            assert!(val.is_finite(), "Final output[{}] not finite: {}", i, val);
        }

        // Verify the schedule was correct
        assert_eq!(model.schedule.linear_count(), 6);
        assert_eq!(model.schedule.attention_count(), 1);
    }

    #[test]
    fn test_zamba_architecture_detection() {
        // Verify Zamba architecture is detected from GGUF metadata
        use metal_attention_gguf::ModelArchitecture;

        let arch = ModelArchitecture::from_str_name("zamba");
        assert_eq!(arch, ModelArchitecture::Zamba);

        let arch = ModelArchitecture::from_str_name("zamba2");
        assert_eq!(arch, ModelArchitecture::Zamba);
    }
}
