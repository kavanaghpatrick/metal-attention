//! Jamba hybrid model: 7:1 Mamba:Attention + MoE FFN.
//!
//! Jamba interleaves Mamba SSM blocks with FlashAttention blocks in a 7:1 ratio.
//! Each layer consists of either:
//!   - MambaBlock (linear sequence model) + MoE FFN
//!   - FlashAttention + MoE FFN
//!
//! The layer type is determined by LayerSchedule::periodic(total_layers, 7).
//! MoE routing uses top-2 expert selection from 16 experts (simplified).

use crate::flash_attn::FlashAttentionState;
use crate::llama::LlamaLayer;
use crate::mamba::{MambaBlock, MambaState};
use metal_attention_traits::schedule::{LayerSchedule, LayerType};
use metal_attention_traits::sequence::SequenceBlock;
use metal_attention_traits::types::BlockConfig;

/// State for a single Jamba layer (either Mamba or Attention variant).
#[derive(Clone)]
pub enum JambaLayerState {
    /// Mamba SSM state.
    Mamba(MambaState),
    /// FlashAttention state (KV cache).
    Attention(FlashAttentionState),
}

/// A single Jamba layer: either Mamba SSM or FlashAttention + SwiGLU FFN.
///
/// Attention layers use a full LlamaLayer (RMSNorm + Attention + FFN).
/// Mamba layers use MambaBlock for the sequence mixing, plus a simplified MoE FFN.
pub enum JambaLayer {
    /// Mamba SSM block + MoE FFN.
    Mamba {
        block: MambaBlock,
        moe: MoEFFN,
    },
    /// FlashAttention block (includes SwiGLU FFN from LlamaLayer).
    Attention(LlamaLayer),
}

impl JambaLayer {
    /// Initialize state for this layer.
    pub fn init_state(&self, config: &BlockConfig) -> JambaLayerState {
        match self {
            JambaLayer::Mamba { block, .. } => {
                JambaLayerState::Mamba(block.init_state(config))
            }
            JambaLayer::Attention(llama) => {
                JambaLayerState::Attention(llama.attention.init_state(config))
            }
        }
    }

    /// Process a single token through this layer.
    ///
    /// Returns the output vector [hidden_size].
    pub fn process_token(&self, input: &[f32], state: &mut JambaLayerState) -> Vec<f32> {
        match (self, state) {
            (JambaLayer::Mamba { block, moe }, JambaLayerState::Mamba(mamba_state)) => {
                // 1. Mamba SSM for sequence mixing
                let ssm_out = block.process_token(input, mamba_state);

                // 2. Residual connection
                let hidden = add(input, &ssm_out);

                // 3. MoE FFN
                let ffn_out = moe.forward(&hidden);

                // 4. Residual connection
                add(&hidden, &ffn_out)
            }
            (JambaLayer::Attention(llama), JambaLayerState::Attention(attn_state)) => {
                // Full Llama-style processing: RMSNorm -> Attention -> Residual -> RMSNorm -> FFN -> Residual
                // For the first token we need to do a prefill to populate KV cache
                let mut llama_state = crate::llama::LlamaState {
                    attn_state: attn_state.clone(),
                };

                let output = if attn_state.kv_cache.len() == 0 {
                    // First token: prefill path
                    let out = llama.process_prefill(input, &mut llama_state, 1);
                    out
                } else {
                    // Subsequent tokens: decode path
                    llama.process_token(input, &mut llama_state)
                };

                *attn_state = llama_state.attn_state;
                output
            }
            _ => panic!("JambaLayer/JambaLayerState type mismatch"),
        }
    }

    /// Check whether this is a Mamba layer.
    pub fn is_mamba(&self) -> bool {
        matches!(self, JambaLayer::Mamba { .. })
    }

    /// Check whether this is an Attention layer.
    pub fn is_attention(&self) -> bool {
        matches!(self, JambaLayer::Attention(_))
    }
}

/// Mixture of Experts FFN.
///
/// Simplified MoE with top-2 expert selection from `num_experts` experts.
/// Each expert is a small SwiGLU FFN. The router selects the top-2 experts
/// based on a learned gating weight, and the output is a weighted sum.
pub struct MoEFFN {
    /// Number of experts.
    pub num_experts: usize,
    /// Hidden size (input/output dimension).
    pub hidden_size: usize,
    /// Expert intermediate size.
    pub expert_intermediate_size: usize,

    /// Router weights: [num_experts, hidden_size], row-major.
    pub w_router: Vec<f32>,

    /// Expert gate projections: num_experts x [expert_intermediate_size, hidden_size].
    pub w_expert_gate: Vec<Vec<f32>>,
    /// Expert up projections: num_experts x [expert_intermediate_size, hidden_size].
    pub w_expert_up: Vec<Vec<f32>>,
    /// Expert down projections: num_experts x [hidden_size, expert_intermediate_size].
    pub w_expert_down: Vec<Vec<f32>>,
}

impl MoEFFN {
    /// Create a MoE FFN with random weights for testing.
    pub fn random(hidden_size: usize, num_experts: usize, seed: u64) -> Self {
        let expert_intermediate_size = hidden_size * 2; // Smaller than standard FFN
        let mut rng = SimpleRng::new(seed);

        let scale = 1.0 / (hidden_size as f32).sqrt();

        // Router weights
        let w_router: Vec<f32> = (0..num_experts * hidden_size)
            .map(|_| rng.next_f32_range(-scale, scale))
            .collect();

        // Expert weights
        let mut w_expert_gate = Vec::with_capacity(num_experts);
        let mut w_expert_up = Vec::with_capacity(num_experts);
        let mut w_expert_down = Vec::with_capacity(num_experts);

        for _ in 0..num_experts {
            w_expert_gate.push(
                (0..expert_intermediate_size * hidden_size)
                    .map(|_| rng.next_f32_range(-scale, scale))
                    .collect(),
            );
            w_expert_up.push(
                (0..expert_intermediate_size * hidden_size)
                    .map(|_| rng.next_f32_range(-scale, scale))
                    .collect(),
            );
            w_expert_down.push(
                (0..hidden_size * expert_intermediate_size)
                    .map(|_| rng.next_f32_range(-scale, scale))
                    .collect(),
            );
        }

        Self {
            num_experts,
            hidden_size,
            expert_intermediate_size,
            w_router,
            w_expert_gate,
            w_expert_up,
            w_expert_down,
        }
    }

    /// Route and compute MoE FFN output.
    ///
    /// 1. Compute router logits
    /// 2. Select top-2 experts
    /// 3. Run SwiGLU FFN for each selected expert
    /// 4. Weighted sum of expert outputs
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let hs = self.hidden_size;
        let ne = self.num_experts;

        // 1. Router logits: [num_experts]
        let router_logits = matvec(&self.w_router, input, ne, hs);

        // 2. Top-2 expert selection
        let (top1_idx, top2_idx, top1_weight, top2_weight) = top2_gating(&router_logits);

        // 3. Run selected experts through SwiGLU FFN
        let expert1_out = self.expert_forward(top1_idx, input);
        let expert2_out = self.expert_forward(top2_idx, input);

        // 4. Weighted combination
        let mut output = vec![0.0f32; hs];
        for i in 0..hs {
            output[i] = top1_weight * expert1_out[i] + top2_weight * expert2_out[i];
        }
        output
    }

    /// Run a single expert's SwiGLU FFN.
    fn expert_forward(&self, expert_idx: usize, input: &[f32]) -> Vec<f32> {
        let hs = self.hidden_size;
        let is = self.expert_intermediate_size;

        let gate = matvec(&self.w_expert_gate[expert_idx], input, is, hs);
        let up = matvec(&self.w_expert_up[expert_idx], input, is, hs);

        // SwiGLU: silu(gate) * up
        let hidden: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(&g, &u)| silu(g) * u)
            .collect();

        // Down projection
        matvec(&self.w_expert_down[expert_idx], &hidden, hs, is)
    }
}

/// Select top-2 experts from router logits with softmax gating.
///
/// Returns (top1_idx, top2_idx, top1_weight, top2_weight) where weights sum to 1.
fn top2_gating(logits: &[f32]) -> (usize, usize, f32, f32) {
    let n = logits.len();

    // Find top-2 indices
    let mut top1_idx = 0;
    let mut top2_idx = 1;
    if logits[1] > logits[0] {
        top1_idx = 1;
        top2_idx = 0;
    }

    for i in 2..n {
        if logits[i] > logits[top1_idx] {
            top2_idx = top1_idx;
            top1_idx = i;
        } else if logits[i] > logits[top2_idx] {
            top2_idx = i;
        }
    }

    // Softmax over top-2 logits for weights
    let max_logit = logits[top1_idx].max(logits[top2_idx]);
    let exp1 = (logits[top1_idx] - max_logit).exp();
    let exp2 = (logits[top2_idx] - max_logit).exp();
    let sum = exp1 + exp2;

    (top1_idx, top2_idx, exp1 / sum, exp2 / sum)
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

/// SiLU activation: x * sigmoid(x).
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Element-wise addition.
fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
}

/// Build a full Jamba model's layers using 7:1 Mamba:Attention schedule.
///
/// Returns a Vec of JambaLayer with the appropriate mix of Mamba and Attention layers.
pub fn build_jamba_layers(
    hidden_size: usize,
    head_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    num_layers: usize,
    d_state: usize,
    num_experts: usize,
    seed: u64,
) -> (Vec<JambaLayer>, LayerSchedule) {
    let schedule = LayerSchedule::periodic(num_layers, 7);
    let mut layers = Vec::with_capacity(num_layers);

    for i in 0..num_layers {
        let layer_seed = seed + (i as u64) * 31 + 7;
        match schedule.layer_type(i).unwrap() {
            LayerType::Linear => {
                let block = MambaBlock::random(hidden_size, d_state, layer_seed);
                let moe = MoEFFN::random(hidden_size, num_experts, layer_seed + 1000);
                layers.push(JambaLayer::Mamba { block, moe });
            }
            LayerType::Attention => {
                let llama = LlamaLayer::random(
                    hidden_size,
                    head_dim,
                    num_heads,
                    num_kv_heads,
                    layer_seed,
                );
                layers.push(JambaLayer::Attention(llama));
            }
        }
    }

    (layers, schedule)
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

    fn make_config(hidden_size: usize, head_dim: usize, num_heads: usize, num_kv_heads: usize) -> BlockConfig {
        BlockConfig {
            hidden_size,
            head_dim,
            num_heads,
            num_kv_heads,
            layer_index: 0,
        }
    }

    #[test]
    fn test_jamba_layer_schedule_7_1() {
        let hidden_size = 16;
        let head_dim = 8;
        let num_heads = 2;
        let num_kv_heads = 2;
        let num_layers = 16;
        let d_state = 4;
        let num_experts = 16;

        let (layers, schedule) = build_jamba_layers(
            hidden_size, head_dim, num_heads, num_kv_heads,
            num_layers, d_state, num_experts, 42,
        );

        assert_eq!(layers.len(), num_layers);

        // Verify 7:1 pattern: layers 0-6 Mamba, layer 7 Attention, 8-14 Mamba, 15 Attention
        for (i, layer) in layers.iter().enumerate() {
            let expected_type = schedule.layer_type(i).unwrap();
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

        // Count: 14 Mamba, 2 Attention for 16 layers
        let mamba_count = layers.iter().filter(|l| l.is_mamba()).count();
        let attn_count = layers.iter().filter(|l| l.is_attention()).count();
        assert_eq!(mamba_count, 14, "Should have 14 Mamba layers");
        assert_eq!(attn_count, 2, "Should have 2 Attention layers");
    }

    #[test]
    fn test_jamba_mamba_layer_forward() {
        let hidden_size = 16;
        let d_state = 4;
        let num_experts = 4; // Fewer experts for faster testing

        let block = MambaBlock::random(hidden_size, d_state, 42);
        let moe = MoEFFN::random(hidden_size, num_experts, 43);
        let layer = JambaLayer::Mamba { block, moe };

        let config = make_config(hidden_size, 8, 2, 2);
        let mut state = layer.init_state(&config);

        let mut rng = SimpleRng::new(100);
        let input: Vec<f32> = (0..hidden_size)
            .map(|_| rng.next_f32_range(-0.5, 0.5))
            .collect();

        let output = layer.process_token(&input, &mut state);
        assert_eq!(output.len(), hidden_size);
        for (i, &val) in output.iter().enumerate() {
            assert!(val.is_finite(), "Mamba layer output[{}] not finite: {}", i, val);
        }
    }

    #[test]
    fn test_jamba_attention_layer_forward() {
        let hidden_size = 32;
        let head_dim = 16;
        let num_heads = 2;
        let num_kv_heads = 2;

        let llama = LlamaLayer::random(hidden_size, head_dim, num_heads, num_kv_heads, 42);
        let layer = JambaLayer::Attention(llama);

        let config = make_config(hidden_size, head_dim, num_heads, num_kv_heads);
        let mut state = layer.init_state(&config);

        let mut rng = SimpleRng::new(200);
        let input: Vec<f32> = (0..hidden_size)
            .map(|_| rng.next_f32_range(-0.5, 0.5))
            .collect();

        let output = layer.process_token(&input, &mut state);
        assert_eq!(output.len(), hidden_size);
        for (i, &val) in output.iter().enumerate() {
            assert!(val.is_finite(), "Attention layer output[{}] not finite: {}", i, val);
        }
    }

    #[test]
    fn test_jamba_hybrid_dispatch_alternates() {
        // Build a 16-layer Jamba model and run a token through all layers
        let hidden_size = 16;
        let head_dim = 8;
        let num_heads = 2;
        let num_kv_heads = 2;
        let num_layers = 16;
        let d_state = 4;
        let num_experts = 4;

        let (layers, schedule) = build_jamba_layers(
            hidden_size, head_dim, num_heads, num_kv_heads,
            num_layers, d_state, num_experts, 42,
        );

        let config = make_config(hidden_size, head_dim, num_heads, num_kv_heads);
        let mut states: Vec<JambaLayerState> = layers
            .iter()
            .map(|l| l.init_state(&config))
            .collect();

        let mut rng = SimpleRng::new(300);
        let input: Vec<f32> = (0..hidden_size)
            .map(|_| rng.next_f32_range(-0.5, 0.5))
            .collect();

        let mut hidden = input;
        for (i, layer) in layers.iter().enumerate() {
            hidden = layer.process_token(&hidden, &mut states[i]);
            assert_eq!(hidden.len(), hidden_size, "Layer {} output size mismatch", i);
            for (j, &val) in hidden.iter().enumerate() {
                assert!(
                    val.is_finite(),
                    "Layer {} output[{}] not finite: {}",
                    i, j, val
                );
            }
        }

        // Verify the schedule was correct
        assert_eq!(schedule.linear_count(), 14);
        assert_eq!(schedule.attention_count(), 2);
    }

    #[test]
    fn test_moe_routing_top2() {
        let hidden_size = 16;
        let num_experts = 16;
        let moe = MoEFFN::random(hidden_size, num_experts, 42);

        let mut rng = SimpleRng::new(100);
        let input: Vec<f32> = (0..hidden_size)
            .map(|_| rng.next_f32_range(-0.5, 0.5))
            .collect();

        let output = moe.forward(&input);

        // Output should be valid (hidden_size)
        assert_eq!(output.len(), hidden_size);
        for (i, &val) in output.iter().enumerate() {
            assert!(val.is_finite(), "MoE output[{}] not finite: {}", i, val);
        }
    }

    #[test]
    fn test_moe_top2_gating_weights_sum_to_one() {
        let logits = vec![1.0, 3.0, 0.5, 2.0, -1.0];
        let (top1, top2, w1, w2) = top2_gating(&logits);

        assert_eq!(top1, 1, "Top-1 should be index 1 (highest logit 3.0)");
        assert_eq!(top2, 3, "Top-2 should be index 3 (second highest logit 2.0)");
        assert!(
            (w1 + w2 - 1.0).abs() < 1e-6,
            "Gating weights should sum to 1.0, got {}",
            w1 + w2
        );
        assert!(w1 > w2, "Top-1 weight should be larger than top-2");
    }
}
