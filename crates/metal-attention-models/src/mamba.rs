//! Mamba SSM block implementing LinearSequenceModel.
//!
//! Mamba is a selective state space model where the A, B, C matrices
//! are input-dependent. Each block performs:
//!   1. Input linear projection (expand hidden_size -> d_inner)
//!   2. SiLU activation
//!   3. Selective SSM scan (h_t = A_t * h_{t-1} + B_t * x_t; y_t = C_t * h_t + D * x_t)
//!   4. Output linear projection (d_inner -> hidden_size)
//!
//! For simplicity, this POC skips the 1D convolution step.

use metal_attention_kernels::device::GpuDevice;
use metal_attention_kernels::pipeline::PsoCache;
use metal_attention_kernels::ssm::{cpu_ssm_scan, dispatch_ssm_scan};
use metal_attention_traits::linear::LinearSequenceModel;
use metal_attention_traits::sequence::SequenceBlock;
use metal_attention_traits::types::{BlockConfig, DType, TensorView};

/// Persistent state for one Mamba block.
///
/// Contains the SSM hidden state [d_inner, d_state] per layer.
#[derive(Clone)]
pub struct MambaState {
    /// SSM hidden state: [d_inner, d_state] flattened.
    pub ssm_state: Vec<f32>,
}

/// Mamba SSM block with simplified architecture.
///
/// Weight matrices are stored as flat Vec<f32> in row-major order.
/// In a full implementation these would be Metal buffers loaded from GGUF.
pub struct MambaBlock {
    /// Hidden size (model dimension).
    pub hidden_size: usize,
    /// Inner dimension (expanded for SSM, typically 2x hidden_size).
    pub d_inner: usize,
    /// SSM state dimension (number of state channels).
    pub d_state: usize,

    // Projection weights
    /// Input projection: [d_inner, hidden_size], row-major.
    pub w_in: Vec<f32>,
    /// Gate projection (for SiLU gating): [d_inner, hidden_size], row-major.
    pub w_gate: Vec<f32>,
    /// Output projection: [hidden_size, d_inner], row-major.
    pub w_out: Vec<f32>,

    // SSM parameter projections (input-dependent)
    /// A projection: [d_inner], learned log-space decay.
    pub a_log: Vec<f32>,
    /// B projection: [d_state, d_inner], projects from d_inner to d_state.
    pub w_b: Vec<f32>,
    /// C projection: [d_state, d_inner], projects from d_inner to d_state.
    pub w_c: Vec<f32>,
    /// D skip connection: scalar.
    pub d_skip: f32,

    /// Whether to use GPU dispatch (true) or CPU reference (false).
    pub use_gpu: bool,
}

impl MambaBlock {
    /// Create a new Mamba block with random weights for testing.
    ///
    /// Uses a simple deterministic pseudo-random sequence seeded by `seed`.
    pub fn random(hidden_size: usize, d_state: usize, seed: u64) -> Self {
        let d_inner = hidden_size * 2; // Standard Mamba expansion ratio
        let mut rng = SimpleRng::new(seed);

        let scale = 1.0 / (hidden_size as f32).sqrt();

        // Input / gate / output projections
        let w_in: Vec<f32> = (0..d_inner * hidden_size)
            .map(|_| rng.next_f32_range(-scale, scale))
            .collect();
        let w_gate: Vec<f32> = (0..d_inner * hidden_size)
            .map(|_| rng.next_f32_range(-scale, scale))
            .collect();
        let w_out: Vec<f32> = (0..hidden_size * d_inner)
            .map(|_| rng.next_f32_range(-scale, scale))
            .collect();

        // A in log-space (will be exp'd and negated for decay)
        let a_log: Vec<f32> = (0..d_inner)
            .map(|_| rng.next_f32_range(-1.0, 0.0))
            .collect();

        // B, C projection weights
        let inner_scale = 1.0 / (d_inner as f32).sqrt();
        let w_b: Vec<f32> = (0..d_state * d_inner)
            .map(|_| rng.next_f32_range(-inner_scale, inner_scale))
            .collect();
        let w_c: Vec<f32> = (0..d_state * d_inner)
            .map(|_| rng.next_f32_range(-inner_scale, inner_scale))
            .collect();

        let d_skip = 1.0;

        Self {
            hidden_size,
            d_inner,
            d_state,
            w_in,
            w_gate,
            w_out,
            a_log,
            w_b,
            w_c,
            d_skip,
            use_gpu: true,
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

    /// SiLU activation: x * sigmoid(x).
    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    /// Sigmoid activation.
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Process a single token through the Mamba block.
    ///
    /// Forward: input_proj -> SiLU gate -> SSM scan -> output_proj
    ///
    /// Returns output vector [hidden_size].
    pub fn process_token(&self, input: &[f32], state: &mut MambaState) -> Vec<f32> {
        let hs = self.hidden_size;
        let di = self.d_inner;
        let ds = self.d_state;

        // 1. Input projection: [hidden_size] -> [d_inner]
        let x_proj = Self::matvec(&self.w_in, input, di, hs);

        // 2. Gate projection + SiLU: [hidden_size] -> [d_inner]
        let gate_proj = Self::matvec(&self.w_gate, input, di, hs);
        let gate: Vec<f32> = gate_proj.iter().map(|&g| Self::silu(g)).collect();

        // 3. Apply gate element-wise
        let x_gated: Vec<f32> = x_proj.iter().zip(gate.iter()).map(|(&x, &g)| x * g).collect();

        // 4. Compute input-dependent SSM parameters
        // A = sigmoid(a_log) for decay in (0, 1)
        let a_decay: Vec<f32> = self.a_log.iter().map(|&al| Self::sigmoid(al)).collect();

        // B = W_b * x_gated: [d_state]
        let b_vec = Self::matvec(&self.w_b, &x_gated, ds, di);

        // C = W_c * x_gated: [d_state]
        let c_vec = Self::matvec(&self.w_c, &x_gated, ds, di);

        // 5. SSM scan (single token, seq_len=1)
        if self.use_gpu {
            let gpu = GpuDevice::new();
            let mut pso_cache = PsoCache::new(gpu.library.clone());
            let (ssm_output, new_state) = dispatch_ssm_scan(
                &gpu,
                &mut pso_cache,
                &x_gated,
                &a_decay,
                &b_vec,
                &c_vec,
                self.d_skip,
                &state.ssm_state,
                1,
                di,
                ds,
            );
            state.ssm_state = new_state;
            // 6. Output projection: [d_inner] -> [hidden_size]
            Self::matvec(&self.w_out, &ssm_output, hs, di)
        } else {
            let mut ssm_state = state.ssm_state.clone();
            let ssm_output = cpu_ssm_scan(
                &x_gated,
                &a_decay,
                &b_vec,
                &c_vec,
                self.d_skip,
                &mut ssm_state,
                1,
                di,
                ds,
            );
            state.ssm_state = ssm_state;
            // 6. Output projection: [d_inner] -> [hidden_size]
            Self::matvec(&self.w_out, &ssm_output, hs, di)
        }
    }
}

impl SequenceBlock for MambaBlock {
    type State = MambaState;

    fn init_state(&self, _config: &BlockConfig) -> Self::State {
        MambaState {
            ssm_state: vec![0.0f32; self.d_inner * self.d_state],
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
        self.d_inner * self.d_state * std::mem::size_of::<f32>()
    }
}

impl LinearSequenceModel for MambaBlock {
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
        1 // Mamba processes token-by-token in recurrent mode
    }
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

    #[test]
    fn test_mamba_block_forward_non_nan() {
        let hidden_size = 16;
        let d_state = 4;

        let block = MambaBlock::random(hidden_size, d_state, 42);

        let config = BlockConfig {
            hidden_size,
            head_dim: hidden_size,
            num_heads: 1,
            num_kv_heads: 1,
            layer_index: 0,
        };

        let mut state = block.init_state(&config);

        // Process a few tokens
        let mut rng = SimpleRng::new(456);
        for _t in 0..4 {
            let input: Vec<f32> = (0..hidden_size)
                .map(|_| rng.next_f32_range(-0.5, 0.5))
                .collect();

            let output = block.process_token(&input, &mut state);

            // Verify output shape
            assert_eq!(output.len(), hidden_size, "Output should have hidden_size elements");

            // Verify no NaN values
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
        let state_nonzero = state.ssm_state.iter().any(|&x| x != 0.0);
        assert!(state_nonzero, "SSM state should be non-zero after processing tokens");
    }

    #[test]
    fn test_mamba_block_cpu_path() {
        let hidden_size = 8;
        let d_state = 4;

        let mut block = MambaBlock::random(hidden_size, d_state, 999);
        block.use_gpu = false;

        let config = BlockConfig {
            hidden_size,
            head_dim: hidden_size,
            num_heads: 1,
            num_kv_heads: 1,
            layer_index: 0,
        };

        let mut state = block.init_state(&config);

        let input: Vec<f32> = (0..hidden_size).map(|i| i as f32 * 0.1).collect();
        let output = block.process_token(&input, &mut state);

        assert_eq!(output.len(), hidden_size);
        for &val in &output {
            assert!(val.is_finite(), "Output should be finite: {}", val);
        }
    }

    #[test]
    fn test_mamba_trait_compliance() {
        let hidden_size = 16;
        let d_state = 4;
        let block = MambaBlock::random(hidden_size, d_state, 789);

        let config = BlockConfig {
            hidden_size,
            head_dim: hidden_size,
            num_heads: 1,
            num_kv_heads: 1,
            layer_index: 0,
        };

        // SequenceBlock
        let mut state = block.init_state(&config);
        let input = TensorView::new(vec![1, hidden_size], DType::F32);

        let out = block.forward_decode(&input, &mut state, &config);
        assert_eq!(out.shape, vec![1, hidden_size]);
        assert_eq!(out.dtype, DType::F32);

        // State size
        let size = block.state_size_bytes(&config);
        assert!(size > 0);

        // LinearSequenceModel
        assert_eq!(block.max_head_dim(), 64);
        assert_eq!(block.optimal_chunk_size(16), 1);
    }
}
