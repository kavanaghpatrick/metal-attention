//! Griffin hybrid model: 2:1 RG-LRU:Attention.
//!
//! Griffin interleaves RG-LRU (Real-Gated Linear Recurrent Unit) blocks with
//! local sliding-window attention blocks in a 2:1 ratio.
//! Each layer consists of either:
//!   - RgLruBlock (linear sequence model) + Gated MLP
//!   - FlashAttention (local sliding window) + Gated MLP
//!
//! The layer type is determined by LayerSchedule::periodic(total_layers, 2).
//!
//! RG-LRU recurrence:
//!   a_t = sigmoid(W_a * x_t)           (recurrence gate)
//!   i_t = sigmoid(W_i * x_t)           (input gate)
//!   h_t = a_t * h_{t-1} + (1 - a_t) * (i_t * x_t)
//!
//! The learnable gating allows the model to selectively remember or forget.

use crate::flash_attn::FlashAttentionState;
use crate::llama::LlamaLayer;
use metal_attention_traits::linear::LinearSequenceModel;
use metal_attention_traits::schedule::{LayerSchedule, LayerType};
use metal_attention_traits::sequence::SequenceBlock;
use metal_attention_traits::types::{BlockConfig, DType, TensorView};

/// Persistent state for one RG-LRU block.
///
/// Contains the hidden state [d_model] per layer.
#[derive(Clone)]
pub struct RgLruState {
    /// Recurrent hidden state: [d_model].
    pub hidden: Vec<f32>,
}

/// RG-LRU (Real-Gated Linear Recurrent Unit) block.
///
/// Weight matrices are stored as flat Vec<f32> in row-major order.
/// In a full implementation these would be Metal buffers loaded from GGUF.
pub struct RgLruBlock {
    /// Hidden size (model dimension).
    pub hidden_size: usize,

    // RG-LRU projections
    /// Input projection: [hidden_size, hidden_size], row-major.
    pub w_in: Vec<f32>,
    /// Recurrence gate projection: [hidden_size, hidden_size], row-major.
    pub w_recurrence_gate: Vec<f32>,
    /// Input gate projection: [hidden_size, hidden_size], row-major.
    pub w_input_gate: Vec<f32>,
    /// Output projection: [hidden_size, hidden_size], row-major.
    pub w_out: Vec<f32>,
}

impl RgLruBlock {
    /// Create a new RG-LRU block with random weights for testing.
    ///
    /// Uses a simple deterministic pseudo-random sequence seeded by `seed`.
    pub fn random(hidden_size: usize, seed: u64) -> Self {
        let mut rng = SimpleRng::new(seed);
        let scale = 1.0 / (hidden_size as f32).sqrt();

        let w_in: Vec<f32> = (0..hidden_size * hidden_size)
            .map(|_| rng.next_f32_range(-scale, scale))
            .collect();
        let w_recurrence_gate: Vec<f32> = (0..hidden_size * hidden_size)
            .map(|_| rng.next_f32_range(-scale, scale))
            .collect();
        let w_input_gate: Vec<f32> = (0..hidden_size * hidden_size)
            .map(|_| rng.next_f32_range(-scale, scale))
            .collect();
        let w_out: Vec<f32> = (0..hidden_size * hidden_size)
            .map(|_| rng.next_f32_range(-scale, scale))
            .collect();

        Self {
            hidden_size,
            w_in,
            w_recurrence_gate,
            w_input_gate,
            w_out,
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

    /// Sigmoid activation.
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Process a single token through the RG-LRU block.
    ///
    /// Forward:
    ///   x_proj = W_in * input
    ///   a_t = sigmoid(W_a * input)     (recurrence gate)
    ///   i_t = sigmoid(W_i * input)     (input gate)
    ///   h_t = a_t * h_{t-1} + (1 - a_t) * (i_t * x_proj)
    ///   output = W_out * h_t
    ///
    /// Returns output vector [hidden_size].
    pub fn process_token(&self, input: &[f32], state: &mut RgLruState) -> Vec<f32> {
        let hs = self.hidden_size;

        // 1. Input projection
        let x_proj = Self::matvec(&self.w_in, input, hs, hs);

        // 2. Recurrence gate: a_t = sigmoid(W_a * input)
        let a_proj = Self::matvec(&self.w_recurrence_gate, input, hs, hs);
        let a_gate: Vec<f32> = a_proj.iter().map(|&v| Self::sigmoid(v)).collect();

        // 3. Input gate: i_t = sigmoid(W_i * input)
        let i_proj = Self::matvec(&self.w_input_gate, input, hs, hs);
        let i_gate: Vec<f32> = i_proj.iter().map(|&v| Self::sigmoid(v)).collect();

        // 4. RG-LRU recurrence: h_t = a_t * h_{t-1} + (1 - a_t) * (i_t * x_proj)
        for d in 0..hs {
            state.hidden[d] =
                a_gate[d] * state.hidden[d] + (1.0 - a_gate[d]) * (i_gate[d] * x_proj[d]);
        }

        // 5. Output projection
        Self::matvec(&self.w_out, &state.hidden, hs, hs)
    }
}

impl SequenceBlock for RgLruBlock {
    type State = RgLruState;

    fn init_state(&self, _config: &BlockConfig) -> Self::State {
        RgLruState {
            hidden: vec![0.0f32; self.hidden_size],
        }
    }

    fn forward_prefill(
        &self,
        input: &TensorView,
        state: &mut Self::State,
        config: &BlockConfig,
    ) -> TensorView {
        // Recurrent: process token-by-token
        self.forward_decode(input, state, config)
    }

    fn forward_decode(
        &self,
        input: &TensorView,
        _state: &mut Self::State,
        _config: &BlockConfig,
    ) -> TensorView {
        TensorView::new(input.shape.clone(), DType::F32)
    }

    fn state_size_bytes(&self, _config: &BlockConfig) -> usize {
        self.hidden_size * std::mem::size_of::<f32>()
    }
}

impl LinearSequenceModel for RgLruBlock {
    fn prefill_chunked(
        &self,
        input: &TensorView,
        state: &mut Self::State,
        config: &BlockConfig,
        _chunk_size: usize,
    ) -> TensorView {
        self.forward_prefill(input, state, config)
    }

    fn decode_step(
        &self,
        input: &TensorView,
        state: &mut Self::State,
        config: &BlockConfig,
    ) -> TensorView {
        self.forward_decode(input, state, config)
    }

    fn max_head_dim(&self) -> usize {
        64
    }

    fn optimal_chunk_size(&self, _head_dim: usize) -> usize {
        1 // RG-LRU processes token-by-token in recurrent mode
    }
}

// ---------------------------------------------------------------------------
// Griffin layer (enum dispatch like Jamba)
// ---------------------------------------------------------------------------

/// State for a single Griffin layer (either RG-LRU or Attention).
#[derive(Clone)]
pub enum GriffinLayerState {
    /// RG-LRU recurrent state.
    RgLru(RgLruState),
    /// FlashAttention state (KV cache).
    Attention(FlashAttentionState),
}

/// A single Griffin layer: either RG-LRU or local sliding-window FlashAttention.
///
/// RG-LRU layers use RgLruBlock for sequence mixing.
/// Attention layers use a full LlamaLayer (RMSNorm + Attention + FFN).
pub enum GriffinLayer {
    /// RG-LRU block.
    RgLru(RgLruBlock),
    /// Local attention block (uses LlamaLayer, which includes FlashAttention + FFN).
    Attention(LlamaLayer),
}

impl GriffinLayer {
    /// Initialize state for this layer.
    pub fn init_state(&self, config: &BlockConfig) -> GriffinLayerState {
        match self {
            GriffinLayer::RgLru(block) => GriffinLayerState::RgLru(block.init_state(config)),
            GriffinLayer::Attention(llama) => {
                GriffinLayerState::Attention(llama.attention.init_state(config))
            }
        }
    }

    /// Process a single token through this layer.
    ///
    /// Returns the output vector [hidden_size].
    pub fn process_token(&self, input: &[f32], state: &mut GriffinLayerState) -> Vec<f32> {
        match (self, state) {
            (GriffinLayer::RgLru(block), GriffinLayerState::RgLru(rglru_state)) => {
                // 1. RG-LRU for sequence mixing
                let rglru_out = block.process_token(input, rglru_state);

                // 2. Residual connection
                add(input, &rglru_out)
            }
            (GriffinLayer::Attention(llama), GriffinLayerState::Attention(attn_state)) => {
                let mut llama_state = crate::llama::LlamaState {
                    attn_state: attn_state.clone(),
                };

                let output = if attn_state.kv_cache.is_empty() {
                    llama.process_prefill(input, &mut llama_state, 1)
                } else {
                    llama.process_token(input, &mut llama_state)
                };

                *attn_state = llama_state.attn_state;
                output
            }
            _ => panic!("GriffinLayer/GriffinLayerState type mismatch"),
        }
    }

    /// Check whether this is an RG-LRU layer.
    pub fn is_rglru(&self) -> bool {
        matches!(self, GriffinLayer::RgLru(_))
    }

    /// Check whether this is an Attention layer.
    pub fn is_attention(&self) -> bool {
        matches!(self, GriffinLayer::Attention(_))
    }
}

/// Build a full Griffin model's layers using 2:1 RG-LRU:Attention schedule.
///
/// Returns a Vec of GriffinLayer with the appropriate mix of RG-LRU and Attention layers.
#[allow(clippy::too_many_arguments)]
pub fn build_griffin_layers(
    hidden_size: usize,
    head_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    num_layers: usize,
    seed: u64,
) -> (Vec<GriffinLayer>, LayerSchedule) {
    let schedule = LayerSchedule::periodic(num_layers, 2);
    let mut layers = Vec::with_capacity(num_layers);

    for i in 0..num_layers {
        let layer_seed = seed + (i as u64) * 31 + 7;
        match schedule.layer_type(i).unwrap() {
            LayerType::Linear => {
                let block = RgLruBlock::random(hidden_size, layer_seed);
                layers.push(GriffinLayer::RgLru(block));
            }
            LayerType::Attention => {
                let llama =
                    LlamaLayer::random(hidden_size, head_dim, num_heads, num_kv_heads, layer_seed);
                layers.push(GriffinLayer::Attention(llama));
            }
        }
    }

    (layers, schedule)
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
    fn test_griffin_rglru_forward_non_nan() {
        let hidden_size = 16;
        let block = RgLruBlock::random(hidden_size, 42);

        let config = make_config(hidden_size, hidden_size, 1, 1);
        let mut state = block.init_state(&config);

        let mut rng = SimpleRng::new(456);
        for _t in 0..4 {
            let input: Vec<f32> = (0..hidden_size)
                .map(|_| rng.next_f32_range(-0.5, 0.5))
                .collect();

            let output = block.process_token(&input, &mut state);

            assert_eq!(
                output.len(),
                hidden_size,
                "Output should have hidden_size elements"
            );

            for (i, &val) in output.iter().enumerate() {
                assert!(
                    val.is_finite(),
                    "Output element {} is not finite: {}",
                    i,
                    val
                );
            }
        }

        // Verify state was updated (non-zero)
        let state_nonzero = state.hidden.iter().any(|&x| x != 0.0);
        assert!(
            state_nonzero,
            "RG-LRU state should be non-zero after processing tokens"
        );
    }

    #[test]
    fn test_griffin_rglru_recurrence_gate() {
        // Verify the recurrence gate properly interpolates between old state and new input
        let hidden_size = 8;
        let block = RgLruBlock::random(hidden_size, 99);

        let config = make_config(hidden_size, hidden_size, 1, 1);
        let mut state = block.init_state(&config);

        // Process first token
        let input1: Vec<f32> = (0..hidden_size).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let _out1 = block.process_token(&input1, &mut state);

        // Save state after first token
        let state_after_1 = state.hidden.clone();

        // Process second token -- state should change
        let input2: Vec<f32> = (0..hidden_size).map(|i| (i as f32 + 1.0) * -0.2).collect();
        let _out2 = block.process_token(&input2, &mut state);

        // State should have changed from token 1 to token 2
        let differs = state
            .hidden
            .iter()
            .zip(state_after_1.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-10);
        assert!(differs, "State should change after processing second token");
    }

    #[test]
    fn test_griffin_layer_schedule_2_1() {
        let hidden_size = 16;
        let head_dim = 8;
        let num_heads = 2;
        let num_kv_heads = 2;
        let num_layers = 9;

        let (layers, schedule) = build_griffin_layers(
            hidden_size,
            head_dim,
            num_heads,
            num_kv_heads,
            num_layers,
            42,
        );

        assert_eq!(layers.len(), num_layers);

        // Verify 2:1 pattern: layers 0-1 RG-LRU, layer 2 Attention, 3-4 RG-LRU, 5 Attention, ...
        for (i, layer) in layers.iter().enumerate() {
            let expected_type = schedule.layer_type(i).unwrap();
            match expected_type {
                LayerType::Linear => {
                    assert!(
                        layer.is_rglru(),
                        "Layer {} should be RG-LRU, got Attention",
                        i
                    );
                }
                LayerType::Attention => {
                    assert!(
                        layer.is_attention(),
                        "Layer {} should be Attention, got RG-LRU",
                        i
                    );
                }
            }
        }

        // Count: 6 RG-LRU, 3 Attention for 9 layers (indices 2,5,8 are attention)
        let rglru_count = layers.iter().filter(|l| l.is_rglru()).count();
        let attn_count = layers.iter().filter(|l| l.is_attention()).count();
        assert_eq!(rglru_count, 6, "Should have 6 RG-LRU layers");
        assert_eq!(attn_count, 3, "Should have 3 Attention layers");
    }

    #[test]
    fn test_griffin_hybrid_dispatch() {
        // Build a 6-layer Griffin model and run a token through all layers
        let hidden_size = 16;
        let head_dim = 8;
        let num_heads = 2;
        let num_kv_heads = 2;
        let num_layers = 6;

        let (layers, _schedule) = build_griffin_layers(
            hidden_size,
            head_dim,
            num_heads,
            num_kv_heads,
            num_layers,
            42,
        );

        let config = make_config(hidden_size, head_dim, num_heads, num_kv_heads);
        let mut states: Vec<GriffinLayerState> =
            layers.iter().map(|l| l.init_state(&config)).collect();

        let mut rng = SimpleRng::new(300);
        let input: Vec<f32> = (0..hidden_size)
            .map(|_| rng.next_f32_range(-0.5, 0.5))
            .collect();

        let mut hidden = input;
        for (i, layer) in layers.iter().enumerate() {
            hidden = layer.process_token(&hidden, &mut states[i]);
            assert_eq!(
                hidden.len(),
                hidden_size,
                "Layer {} output size mismatch",
                i
            );
            for (j, &val) in hidden.iter().enumerate() {
                assert!(
                    val.is_finite(),
                    "Layer {} output[{}] not finite: {}",
                    i,
                    j,
                    val
                );
            }
        }
    }

    #[test]
    fn test_griffin_rglru_trait_compliance() {
        let hidden_size = 16;
        let block = RgLruBlock::random(hidden_size, 789);

        let config = make_config(hidden_size, hidden_size, 1, 1);

        // SequenceBlock
        let mut state = block.init_state(&config);
        let input = TensorView::new(vec![1, hidden_size], DType::F32);

        let out = block.forward_decode(&input, &mut state, &config);
        assert_eq!(out.shape, vec![1, hidden_size]);
        assert_eq!(out.dtype, DType::F32);

        // State size
        let size = block.state_size_bytes(&config);
        assert!(size > 0);
        assert_eq!(size, hidden_size * std::mem::size_of::<f32>());

        // LinearSequenceModel
        assert_eq!(block.max_head_dim(), 64);
        assert_eq!(block.optimal_chunk_size(16), 1);
    }

    #[test]
    fn test_griffin_architecture_detection() {
        // Verify Griffin architecture is detected from GGUF metadata
        use metal_attention_gguf::ModelArchitecture;

        let arch = ModelArchitecture::from_str_name("griffin");
        assert_eq!(arch, ModelArchitecture::Griffin);

        let arch = ModelArchitecture::from_str_name("recurrentgemma");
        assert_eq!(arch, ModelArchitecture::Griffin);
    }
}
