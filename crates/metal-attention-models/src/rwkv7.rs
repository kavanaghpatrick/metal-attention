//! RWKV-7 model block implementing LinearSequenceModel.
//!
//! RWKV-7 is a pure linear recurrent model (no softmax attention).
//! Each block performs:
//!   1. Token shift (mix current token with previous)
//!   2. Linear projections for receptance (r), key (k), value (v), decay (w)
//!   3. WKV operator: state update + output computation on GPU
//!   4. Output projection
//!
//! This is a simplified version suitable for end-to-end validation.
//! Full RWKV-7 features (bonus terms, data-dependent decay, GroupNorm) can
//! be added later.

use metal_attention_kernels::device::GpuDevice;
use metal_attention_kernels::pipeline::PsoCache;
use metal_attention_kernels::rwkv::{cpu_rwkv_wkv, dispatch_rwkv_wkv};
use metal_attention_traits::linear::LinearSequenceModel;
use metal_attention_traits::sequence::SequenceBlock;
use metal_attention_traits::types::{BlockConfig, DType, TensorView};

/// Persistent state for one RWKV-7 block.
///
/// Contains the WKV hidden state (head_dim x head_dim matrix)
/// and the previous token embedding for token shift.
#[derive(Clone)]
pub struct Rwkv7State {
    /// WKV hidden state: [head_dim, head_dim] flattened.
    pub wkv_state: Vec<f32>,
    /// Previous token embedding for token shift: [hidden_size].
    pub prev_token: Vec<f32>,
}

/// RWKV-7 block with simplified architecture.
///
/// Weight matrices are stored as flat Vec<f32> in row-major order.
/// In a full implementation these would be Metal buffers loaded from GGUF.
pub struct Rwkv7Block {
    /// Hidden size (model dimension).
    pub hidden_size: usize,
    /// Head dimension for WKV state.
    pub head_dim: usize,
    /// Number of heads.
    pub num_heads: usize,

    // Token shift mixing factors [hidden_size]
    pub mix_r: Vec<f32>,
    pub mix_k: Vec<f32>,
    pub mix_v: Vec<f32>,
    pub mix_w: Vec<f32>,

    // Projection weights [hidden_size, hidden_size] (simplified: same dim in/out)
    pub w_r: Vec<f32>, // receptance projection
    pub w_k: Vec<f32>, // key projection
    pub w_v: Vec<f32>, // value projection
    pub w_w: Vec<f32>, // decay projection (output passed through sigmoid)
    pub w_o: Vec<f32>, // output projection

    /// Whether to use GPU dispatch (true) or CPU reference (false).
    pub use_gpu: bool,
}

impl Rwkv7Block {
    /// Create a new RWKV-7 block with random weights for testing.
    ///
    /// Uses a simple deterministic pseudo-random sequence seeded by `seed`.
    pub fn random(hidden_size: usize, head_dim: usize, num_heads: usize, seed: u64) -> Self {
        let mut rng = SimpleRng::new(seed);

        let hs = hidden_size;

        // Token shift mixing factors in [0, 1]
        let mix_r = (0..hs).map(|_| rng.next_f32()).collect();
        let mix_k = (0..hs).map(|_| rng.next_f32()).collect();
        let mix_v = (0..hs).map(|_| rng.next_f32()).collect();
        let mix_w = (0..hs).map(|_| rng.next_f32()).collect();

        // Projection weights: small random values for stability
        let scale = 1.0 / (hs as f32).sqrt();
        let w_r = (0..hs * hs)
            .map(|_| rng.next_f32_range(-scale, scale))
            .collect();
        let w_k = (0..hs * hs)
            .map(|_| rng.next_f32_range(-scale, scale))
            .collect();
        let w_v = (0..hs * hs)
            .map(|_| rng.next_f32_range(-scale, scale))
            .collect();
        let w_w = (0..hs * hs)
            .map(|_| rng.next_f32_range(-scale, scale))
            .collect();
        let w_o = (0..hs * hs)
            .map(|_| rng.next_f32_range(-scale, scale))
            .collect();

        Self {
            hidden_size: hs,
            head_dim,
            num_heads,
            mix_r,
            mix_k,
            mix_v,
            mix_w,
            w_r,
            w_k,
            w_v,
            w_w,
            w_o,
            use_gpu: true,
        }
    }

    /// Apply token shift: mix = factor * current + (1 - factor) * previous.
    pub fn token_shift(&self, current: &[f32], previous: &[f32], mix: &[f32]) -> Vec<f32> {
        current
            .iter()
            .zip(previous.iter())
            .zip(mix.iter())
            .map(|((c, p), m)| m * c + (1.0 - m) * p)
            .collect()
    }

    /// Matrix-vector multiply: y = W * x, W is [out_dim, in_dim], x is [in_dim].
    pub fn matvec(w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
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
    pub fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Process a single token through the RWKV-7 block.
    ///
    /// Returns output vector.
    pub fn process_token(&self, input: &[f32], state: &mut Rwkv7State) -> Vec<f32> {
        let hs = self.hidden_size;
        let hd = self.head_dim;

        // 1. Token shift
        let shifted_r = self.token_shift(input, &state.prev_token, &self.mix_r);
        let shifted_k = self.token_shift(input, &state.prev_token, &self.mix_k);
        let shifted_v = self.token_shift(input, &state.prev_token, &self.mix_v);
        let shifted_w = self.token_shift(input, &state.prev_token, &self.mix_w);

        // Update prev_token for next call
        state.prev_token = input.to_vec();

        // 2. Linear projections
        let r = Self::matvec(&self.w_r, &shifted_r, hs, hs);
        let k = Self::matvec(&self.w_k, &shifted_k, hs, hs);
        let v = Self::matvec(&self.w_v, &shifted_v, hs, hs);
        let w_raw = Self::matvec(&self.w_w, &shifted_w, hs, hs);

        // Apply sigmoid to decay (ensures values in (0, 1))
        let w: Vec<f32> = w_raw.iter().map(|&x| Self::sigmoid(x)).collect();

        // 3. WKV operator
        // If head_dim == hidden_size, single-head fast path.
        // Otherwise, process each head separately with per-head state slices.
        let wkv_output = if hd == hs {
            // Single-head fast path
            let (output, new_state) = if self.use_gpu {
                let gpu = GpuDevice::new();
                let mut pso_cache = PsoCache::new(gpu.library.clone());
                dispatch_rwkv_wkv(
                    &gpu,
                    &mut pso_cache,
                    &r,
                    &k,
                    &v,
                    &w,
                    &state.wkv_state,
                    1,
                    hd,
                )
            } else {
                let mut wkv_state = state.wkv_state.clone();
                let output = cpu_rwkv_wkv(&r, &k, &v, &w, &mut wkv_state, 1, hd);
                (output, wkv_state)
            };
            state.wkv_state = new_state;
            output
        } else {
            // Multi-head: process each head separately
            let mut output = Vec::with_capacity(hs);
            for h in 0..self.num_heads {
                let head_start = h * hd;
                let head_end = head_start + hd;
                let r_head = &r[head_start..head_end];
                let k_head = &k[head_start..head_end];
                let v_head = &v[head_start..head_end];
                let w_head = &w[head_start..head_end];

                let state_start = h * hd * hd;
                let state_end = state_start + hd * hd;
                let mut head_state = state.wkv_state[state_start..state_end].to_vec();

                let head_output =
                    cpu_rwkv_wkv(r_head, k_head, v_head, w_head, &mut head_state, 1, hd);

                output.extend_from_slice(&head_output);
                state.wkv_state[state_start..state_end].copy_from_slice(&head_state);
            }
            output
        };

        // 4. Output projection
        Self::matvec(&self.w_o, &wkv_output, hs, hs)
    }
}

impl SequenceBlock for Rwkv7Block {
    type State = Rwkv7State;

    fn init_state(&self, _config: &BlockConfig) -> Self::State {
        Rwkv7State {
            wkv_state: vec![0.0f32; self.num_heads * self.head_dim * self.head_dim],
            prev_token: vec![0.0f32; self.hidden_size],
        }
    }

    fn forward_prefill(
        &self,
        input: &TensorView,
        state: &mut Self::State,
        config: &BlockConfig,
    ) -> TensorView {
        // For prefill, process token-by-token (recurrent).
        // A real implementation would use chunked processing.
        self.forward_decode(input, state, config)
    }

    fn forward_decode(
        &self,
        input: &TensorView,
        _state: &mut Self::State,
        _config: &BlockConfig,
    ) -> TensorView {
        // For this POC, input/output are just shape descriptors.
        // The actual data flow happens through the process_token method.
        // Return a TensorView with the same shape as input.
        TensorView::new(input.shape.clone(), DType::F32)
    }

    fn state_size_bytes(&self, _config: &BlockConfig) -> usize {
        // WKV state (num_heads * head_dim * head_dim) + prev_token (hidden_size)
        (self.num_heads * self.head_dim * self.head_dim + self.hidden_size)
            * std::mem::size_of::<f32>()
    }
}

impl LinearSequenceModel for Rwkv7Block {
    fn prefill_chunked(
        &self,
        input: &TensorView,
        state: &mut Self::State,
        config: &BlockConfig,
        _chunk_size: usize,
    ) -> TensorView {
        // Simplified: just delegate to forward_prefill (token-by-token)
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
        64 // Conservative limit for 32KB threadgroup memory
    }

    fn optimal_chunk_size(&self, _head_dim: usize) -> usize {
        1 // RWKV processes token-by-token in recurrent mode
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
        // xorshift64
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
    use metal_attention_kernels::device::GpuDevice;
    use metal_attention_kernels::pipeline::PsoCache;
    use metal_attention_kernels::rwkv::{cpu_rwkv_wkv, dispatch_rwkv_wkv};

    /// Test: GPU WKV kernel matches CPU reference implementation.
    #[test]
    fn test_rwkv_wkv_gpu_vs_cpu() {
        let head_dim = 16; // Small for fast testing
        let seq_len = 4;
        let mut rng = SimpleRng::new(42);

        // Generate random inputs
        let n = seq_len * head_dim;
        let r: Vec<f32> = (0..n).map(|_| rng.next_f32_range(-0.5, 0.5)).collect();
        let k: Vec<f32> = (0..n).map(|_| rng.next_f32_range(-0.5, 0.5)).collect();
        let v: Vec<f32> = (0..n).map(|_| rng.next_f32_range(-0.5, 0.5)).collect();
        // Decay in (0, 1) -- use sigmoid of random values
        let w: Vec<f32> = (0..n)
            .map(|_| {
                let x = rng.next_f32_range(-2.0, 2.0);
                1.0 / (1.0 + (-x).exp())
            })
            .collect();

        let state = vec![0.0f32; head_dim * head_dim];

        // CPU reference
        let mut cpu_state = state.clone();
        let cpu_output = cpu_rwkv_wkv(&r, &k, &v, &w, &mut cpu_state, seq_len, head_dim);

        // GPU dispatch
        let gpu = GpuDevice::new();
        let mut pso_cache = PsoCache::new(gpu.library.clone());
        let (gpu_output, gpu_state) = dispatch_rwkv_wkv(
            &gpu,
            &mut pso_cache,
            &r,
            &k,
            &v,
            &w,
            &state,
            seq_len,
            head_dim,
        );

        // Compare outputs
        assert_eq!(cpu_output.len(), gpu_output.len());
        let atol = 1e-3;
        for i in 0..cpu_output.len() {
            let diff = (cpu_output[i] - gpu_output[i]).abs();
            assert!(
                diff < atol,
                "Output mismatch at index {}: CPU={}, GPU={}, diff={}",
                i,
                cpu_output[i],
                gpu_output[i],
                diff
            );
        }

        // Compare final states
        assert_eq!(cpu_state.len(), gpu_state.len());
        for i in 0..cpu_state.len() {
            let diff = (cpu_state[i] - gpu_state[i]).abs();
            assert!(
                diff < atol,
                "State mismatch at index {}: CPU={}, GPU={}, diff={}",
                i,
                cpu_state[i],
                gpu_state[i],
                diff
            );
        }
    }

    /// Test: Create Rwkv7Block, run forward_decode, verify output shape and non-NaN.
    #[test]
    fn test_rwkv7_block_forward_decode() {
        let hidden_size = 16;
        let head_dim = hidden_size; // Simplified: single head
        let num_heads = 1;

        let block = Rwkv7Block::random(hidden_size, head_dim, num_heads, 123);

        let config = BlockConfig {
            hidden_size,
            head_dim,
            num_heads,
            num_kv_heads: num_heads,
            layer_index: 0,
        };

        let mut state = block.init_state(&config);

        // Process a few tokens
        let mut rng = SimpleRng::new(456);
        for _t in 0..4 {
            let input: Vec<f32> = (0..hidden_size)
                .map(|_| rng.next_f32_range(-1.0, 1.0))
                .collect();

            let output = block.process_token(&input, &mut state);

            // Verify output shape
            assert_eq!(
                output.len(),
                hidden_size,
                "Output should have hidden_size elements"
            );

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
        let state_nonzero = state.wkv_state.iter().any(|&x| x != 0.0);
        assert!(
            state_nonzero,
            "WKV state should be non-zero after processing tokens"
        );
    }

    /// Test: Rwkv7Block implements SequenceBlock and LinearSequenceModel traits.
    #[test]
    fn test_rwkv7_trait_compliance() {
        let block = Rwkv7Block::random(16, 16, 1, 789);

        let config = BlockConfig {
            hidden_size: 16,
            head_dim: 16,
            num_heads: 1,
            num_kv_heads: 1,
            layer_index: 0,
        };

        // SequenceBlock
        let mut state = block.init_state(&config);
        let input = TensorView::new(vec![1, 16], DType::F32);

        let out = block.forward_decode(&input, &mut state, &config);
        assert_eq!(out.shape, vec![1, 16]);
        assert_eq!(out.dtype, DType::F32);

        let prefill_input = TensorView::new(vec![4, 16], DType::F32);
        let out2 = block.forward_prefill(&prefill_input, &mut state, &config);
        assert_eq!(out2.shape, vec![4, 16]);

        // State size
        let size = block.state_size_bytes(&config);
        assert!(size > 0);

        // LinearSequenceModel
        let chunk_out = block.prefill_chunked(&prefill_input, &mut state, &config, 2);
        assert_eq!(chunk_out.shape, vec![4, 16]);

        let step_out = block.decode_step(&input, &mut state, &config);
        assert_eq!(step_out.shape, vec![1, 16]);

        assert_eq!(block.max_head_dim(), 64);
        assert_eq!(block.optimal_chunk_size(16), 1);
    }

    /// Test: CPU-only path (no GPU dispatch).
    #[test]
    fn test_rwkv7_block_cpu_path() {
        let hidden_size = 8;
        let mut block = Rwkv7Block::random(hidden_size, hidden_size, 1, 999);
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
}
