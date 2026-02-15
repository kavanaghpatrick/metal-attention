//! Flash Attention model layer implementing SoftmaxAttention.
//!
//! FlashAttentionLayer is a standard transformer attention block with:
//!   1. Q/K/V/O linear projections (CPU matvec, simplified POC)
//!   2. Optional RoPE position encoding
//!   3. Optional GQA head expansion
//!   4. Flash Attention kernel dispatch for scaled dot-product attention
//!   5. Dense KV cache for incremental decode
//!
//! Weight layout follows Llama/Mistral conventions.

use metal_attention_kernels::device::GpuDevice;
use metal_attention_kernels::flash::dispatch_flash_attention;
use metal_attention_kernels::kv_cache::DenseKVCache;
use metal_attention_kernels::pipeline::PsoCache;
use metal_attention_traits::attention::{
    GQAConfig, KVCacheMode, PositionEncoding, SoftmaxAttention,
};
use metal_attention_traits::sequence::SequenceBlock;
use metal_attention_traits::types::{BlockConfig, DType, TensorView};

/// Maximum sequence length for KV cache allocation.
const DEFAULT_MAX_SEQ: usize = 2048;

/// Persistent state for one FlashAttention layer.
///
/// Contains the dense KV cache for this layer.
#[derive(Clone)]
pub struct FlashAttentionState {
    /// Dense KV cache storing projected K and V vectors.
    pub kv_cache: DenseKVCache,
}

/// Flash Attention layer implementing the SoftmaxAttention trait.
///
/// Performs multi-head (or grouped-query) scaled dot-product attention
/// using the Flash Attention Metal kernel for O(N^2) attention with tiling.
///
/// Weight matrices stored as flat Vec<f32> in row-major order.
/// In a full implementation these would be Metal buffers loaded from GGUF.
pub struct FlashAttentionLayer {
    /// Hidden size (model dimension).
    pub hidden_size: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Number of Q heads.
    pub num_heads: usize,
    /// Number of KV heads (== num_heads for MHA, < for GQA).
    pub num_kv_heads: usize,
    /// Maximum sequence length for KV cache.
    pub max_seq_len: usize,

    // Projection weights
    /// Q projection: [num_heads * head_dim, hidden_size] row-major.
    pub w_q: Vec<f32>,
    /// K projection: [num_kv_heads * head_dim, hidden_size] row-major.
    pub w_k: Vec<f32>,
    /// V projection: [num_kv_heads * head_dim, hidden_size] row-major.
    pub w_v: Vec<f32>,
    /// O projection: [hidden_size, num_heads * head_dim] row-major.
    pub w_o: Vec<f32>,

    /// Position encoding configuration.
    pub pos_encoding: PositionEncoding,
}

impl FlashAttentionLayer {
    /// Create a new FlashAttention layer with random weights for testing.
    ///
    /// Uses a simple deterministic pseudo-random sequence seeded by `seed`.
    pub fn random(
        hidden_size: usize,
        head_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        seed: u64,
    ) -> Self {
        let mut rng = SimpleRng::new(seed);
        let scale = 1.0 / (hidden_size as f32).sqrt();

        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let w_q = (0..q_dim * hidden_size)
            .map(|_| rng.next_f32_range(-scale, scale))
            .collect();
        let w_k = (0..kv_dim * hidden_size)
            .map(|_| rng.next_f32_range(-scale, scale))
            .collect();
        let w_v = (0..kv_dim * hidden_size)
            .map(|_| rng.next_f32_range(-scale, scale))
            .collect();
        let w_o = (0..hidden_size * q_dim)
            .map(|_| rng.next_f32_range(-scale, scale))
            .collect();

        Self {
            hidden_size,
            head_dim,
            num_heads,
            num_kv_heads,
            max_seq_len: DEFAULT_MAX_SEQ,
            w_q,
            w_k,
            w_v,
            w_o,
            pos_encoding: PositionEncoding::None,
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

    /// Project Q/K/V from a single token embedding.
    fn project_token(&self, input: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;
        let q = Self::matvec(&self.w_q, input, q_dim, self.hidden_size);
        let k = Self::matvec(&self.w_k, input, kv_dim, self.hidden_size);
        let v = Self::matvec(&self.w_v, input, kv_dim, self.hidden_size);
        (q, k, v)
    }

    /// Project Q/K/V from a sequence of token embeddings.
    fn project_sequence(&self, input: &[f32], seq_len: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;
        let mut q_all = Vec::with_capacity(seq_len * q_dim);
        let mut k_all = Vec::with_capacity(seq_len * kv_dim);
        let mut v_all = Vec::with_capacity(seq_len * kv_dim);

        for t in 0..seq_len {
            let token = &input[t * self.hidden_size..(t + 1) * self.hidden_size];
            let (q, k, v) = self.project_token(token);
            q_all.extend_from_slice(&q);
            k_all.extend_from_slice(&k);
            v_all.extend_from_slice(&v);
        }

        (q_all, k_all, v_all)
    }

    /// Split token-major tensor into per-head slices.
    ///
    /// Input:  `[seq_len, total_heads * head_dim]` (token-major)
    /// Output: Vec of per-head `[seq_len, head_dim]` slices
    fn split_heads(&self, data: &[f32], seq_len: usize, total_heads: usize) -> Vec<Vec<f32>> {
        let hd = self.head_dim;
        let mut heads = vec![vec![0.0f32; seq_len * hd]; total_heads];
        for t in 0..seq_len {
            for (h, head) in heads.iter_mut().enumerate().take(total_heads) {
                let src_start = t * total_heads * hd + h * hd;
                let dst_start = t * hd;
                head[dst_start..dst_start + hd]
                    .copy_from_slice(&data[src_start..src_start + hd]);
            }
        }
        heads
    }

    /// Expand KV per-head slices for GQA: repeat each KV head group_size times.
    fn expand_kv_per_head(&self, kv_heads: &[Vec<f32>]) -> Vec<Vec<f32>> {
        if self.num_kv_heads == self.num_heads {
            return kv_heads.to_vec();
        }
        let group_size = self.num_heads / self.num_kv_heads;
        let mut expanded = Vec::with_capacity(self.num_heads);
        for q_head in 0..self.num_heads {
            let kv_head = q_head / group_size;
            expanded.push(kv_heads[kv_head].clone());
        }
        expanded
    }

    /// Run attention using GPU flash kernel (prefill path, Q/K/V same seq_len).
    ///
    /// Dispatches per-head since `dispatch_flash_attention` expects single-head
    /// layout `[seq_len, head_dim]`. Returns token-major `[seq_len, num_heads * head_dim]`.
    fn run_attention_gpu(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
    ) -> Vec<f32> {
        let gpu = GpuDevice::new();
        let mut pso_cache = PsoCache::new(gpu.library.clone());

        let hd = self.head_dim;
        let nh = self.num_heads;

        // Split token-major to per-head slices
        let q_heads = self.split_heads(q, seq_len, nh);
        let k_heads_raw = self.split_heads(k, seq_len, self.num_kv_heads);
        let v_heads_raw = self.split_heads(v, seq_len, self.num_kv_heads);

        // Expand KV heads for GQA
        let k_heads = self.expand_kv_per_head(&k_heads_raw);
        let v_heads = self.expand_kv_per_head(&v_heads_raw);

        // Dispatch per-head and collect outputs
        let mut head_outputs: Vec<Vec<f32>> = Vec::with_capacity(nh);
        for h in 0..nh {
            let head_out = dispatch_flash_attention(
                &gpu,
                &mut pso_cache,
                &q_heads[h],
                &k_heads[h],
                &v_heads[h],
                seq_len,
                hd,
                1,
            );
            head_outputs.push(head_out);
        }

        // Merge per-head outputs to token-major:
        // [num_heads][seq_len, head_dim] -> [seq_len, num_heads * head_dim]
        let mut result = vec![0.0f32; seq_len * nh * hd];
        for (h, head_out) in head_outputs.iter().enumerate().take(nh) {
            for t in 0..seq_len {
                let src_start = t * hd;
                let dst_start = t * nh * hd + h * hd;
                result[dst_start..dst_start + hd]
                    .copy_from_slice(&head_out[src_start..src_start + hd]);
            }
        }

        result
    }

    /// Run attention using CPU softmax (decode path, Q_len=1, K/V_len=cached).
    ///
    /// Q layout: `[num_heads * head_dim]` (token-major, single token)
    /// K/V layout: `[kv_seq_len, num_kv_heads * head_dim]` (token-major from KV cache)
    ///
    /// Standard scaled dot-product attention per head:
    ///   score = Q * K^T / sqrt(d)
    ///   P = softmax(score)
    ///   O = P * V
    ///
    /// Returns: `[num_heads * head_dim]` (token-major, single token)
    fn run_attention_cpu(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        kv_seq_len: usize,
    ) -> Vec<f32> {
        let hd = self.head_dim;
        let nkv = self.num_kv_heads;
        let group_size = self.num_heads / nkv;
        let scale = 1.0 / (hd as f32).sqrt();
        let mut output = vec![0.0f32; self.num_heads * hd];

        for h in 0..self.num_heads {
            let kv_head = h / group_size;
            let q_off = h * hd;

            // Compute scores: Q[1, D] * K[N, D]^T -> [1, N]
            // K is token-major: K[n, kv_head] at offset n * nkv * hd + kv_head * hd
            let mut scores = vec![0.0f32; kv_seq_len];
            let mut max_score = f32::NEG_INFINITY;
            for (n, score) in scores.iter_mut().enumerate().take(kv_seq_len) {
                let k_off = n * nkv * hd + kv_head * hd;
                let mut dot = 0.0f32;
                for d in 0..hd {
                    dot += q[q_off + d] * k[k_off + d];
                }
                *score = dot * scale;
                if *score > max_score {
                    max_score = *score;
                }
            }

            // Softmax
            let mut sum_exp = 0.0f32;
            for score in scores.iter_mut() {
                *score = (*score - max_score).exp();
                sum_exp += *score;
            }
            if sum_exp > 0.0 {
                for s in &mut scores {
                    *s /= sum_exp;
                }
            }

            // Output: P * V -> [1, D]
            for d in 0..hd {
                let mut acc = 0.0f32;
                for (n, &score) in scores.iter().enumerate().take(kv_seq_len) {
                    let v_off = n * nkv * hd + kv_head * hd;
                    acc += score * v[v_off + d];
                }
                output[q_off + d] = acc;
            }
        }

        output
    }

    /// Run attention: GPU for prefill (same seq_len), CPU for decode (Q_len=1).
    fn run_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        q_seq_len: usize,
        kv_seq_len: usize,
    ) -> Vec<f32> {
        if q_seq_len == kv_seq_len {
            // Prefill: use GPU flash attention kernel
            self.run_attention_gpu(q, k, v, q_seq_len)
        } else {
            // Decode: use CPU softmax attention (Q_len=1)
            assert_eq!(q_seq_len, 1, "Asymmetric attention only supports Q_len=1 (decode)");
            self.run_attention_cpu(q, k, v, kv_seq_len)
        }
    }

    /// Apply output projection: project attention output back to hidden_size.
    ///
    /// Input layout: [seq_len, num_heads * head_dim] (token-major, heads concatenated).
    /// Projects each token's concatenated head output to hidden_size.
    fn output_projection(&self, attn_output: &[f32], seq_len: usize) -> Vec<f32> {
        let q_dim = self.num_heads * self.head_dim;
        let mut output = Vec::with_capacity(seq_len * self.hidden_size);
        for t in 0..seq_len {
            let token_attn = &attn_output[t * q_dim..(t + 1) * q_dim];
            let projected = Self::matvec(&self.w_o, token_attn, self.hidden_size, q_dim);
            output.extend_from_slice(&projected);
        }
        output
    }
}

impl SequenceBlock for FlashAttentionLayer {
    type State = FlashAttentionState;

    fn init_state(&self, _config: &BlockConfig) -> Self::State {
        FlashAttentionState {
            kv_cache: DenseKVCache::new(self.max_seq_len, self.num_kv_heads * self.head_dim),
        }
    }

    fn forward_prefill(
        &self,
        input: &TensorView,
        _state: &mut Self::State,
        _config: &BlockConfig,
    ) -> TensorView {
        // Return shape descriptor; actual computation happens through process methods
        TensorView::new(input.shape.clone(), DType::F32)
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
        // KV cache: 2 * max_seq * num_kv_heads * head_dim * sizeof(f32)
        2 * self.max_seq_len * self.num_kv_heads * self.head_dim * std::mem::size_of::<f32>()
    }
}

impl SoftmaxAttention for FlashAttentionLayer {
    fn cached_length(&self, state: &Self::State) -> usize {
        state.kv_cache.len()
    }

    fn max_length(&self, state: &Self::State) -> usize {
        state.kv_cache.max_len()
    }

    fn position_encoding(&self) -> PositionEncoding {
        self.pos_encoding
    }

    fn cache_mode(&self) -> KVCacheMode {
        KVCacheMode::Dense
    }

    fn gqa_config(&self) -> GQAConfig {
        GQAConfig {
            group_size: self.num_heads / self.num_kv_heads,
        }
    }

    fn prefill_attention(
        &self,
        q: &TensorView,
        _k: &TensorView,
        _v: &TensorView,
        _state: &mut Self::State,
        _config: &BlockConfig,
    ) -> TensorView {
        // Shape-only pass-through for trait compliance.
        // Actual data flow uses process_prefill.
        let seq_len = q.shape[0];
        TensorView::new(vec![seq_len, self.hidden_size], DType::F32)
    }

    fn decode_attention(
        &self,
        _q: &TensorView,
        _k: &TensorView,
        _v: &TensorView,
        _state: &mut Self::State,
        _config: &BlockConfig,
    ) -> TensorView {
        // Shape-only pass-through for trait compliance.
        // Actual data flow uses process_decode.
        TensorView::new(vec![1, self.hidden_size], DType::F32)
    }
}

impl FlashAttentionLayer {
    /// Process a full sequence through the attention layer (prefill path).
    ///
    /// 1. Project Q/K/V from input embeddings
    /// 2. Populate KV cache with projected K/V
    /// 3. Run flash attention (Q against full K/V)
    /// 4. Apply output projection
    ///
    /// Input: `[seq_len * hidden_size]` flat f32 slice.
    /// Returns: `[seq_len * hidden_size]` flat f32 output.
    pub fn process_prefill(
        &self,
        input: &[f32],
        state: &mut FlashAttentionState,
        seq_len: usize,
    ) -> Vec<f32> {
        assert_eq!(input.len(), seq_len * self.hidden_size);

        // 1. Project Q/K/V
        let (q, k, v) = self.project_sequence(input, seq_len);

        // 2. Populate KV cache
        state.kv_cache.append(&k, &v);

        // 3. Run flash attention
        let attn_out = self.run_attention(&q, &k, &v, seq_len, seq_len);

        // 4. Output projection (per-token: concat heads then project)
        self.output_projection(&attn_out, seq_len)
    }

    /// Process a single token through the attention layer (decode path).
    ///
    /// 1. Project Q/K/V from single token
    /// 2. Append new K/V to cache
    /// 3. Run flash attention (single Q against full KV cache)
    /// 4. Apply output projection
    ///
    /// Input: `[hidden_size]` flat f32 slice.
    /// Returns: `[hidden_size]` flat f32 output.
    pub fn process_decode(
        &self,
        input: &[f32],
        state: &mut FlashAttentionState,
    ) -> Vec<f32> {
        assert_eq!(input.len(), self.hidden_size);

        // 1. Project Q/K/V
        let (q, k_new, v_new) = self.project_token(input);

        // 2. Append to KV cache
        state.kv_cache.append(&k_new, &v_new);

        // 3. Run flash attention: Q=[1, head_dim], K/V=[cached_len, head_dim]
        let cached_len = state.kv_cache.len();
        let full_k = state.kv_cache.k_slice();
        let full_v = state.kv_cache.v_slice();

        let attn_out = self.run_attention(&q, full_k, full_v, 1, cached_len);

        // 4. Output projection
        self.output_projection(&attn_out, 1)
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
    fn test_flash_attn_trait_compliance() {
        let layer = FlashAttentionLayer::random(32, 16, 2, 2, 42);
        let config = make_config(32, 16, 2, 2);
        let state = layer.init_state(&config);

        // SoftmaxAttention trait methods
        assert_eq!(layer.cached_length(&state), 0);
        assert_eq!(layer.max_length(&state), DEFAULT_MAX_SEQ);
        assert_eq!(layer.position_encoding(), PositionEncoding::None);
        assert_eq!(layer.cache_mode(), KVCacheMode::Dense);
        assert_eq!(layer.gqa_config().group_size, 1);

        // State size
        let size = layer.state_size_bytes(&config);
        assert!(size > 0);
    }

    #[test]
    fn test_flash_attn_prefill_output_shape_and_finite() {
        let hidden_size = 32;
        let head_dim = 16;
        let num_heads = 2;
        let num_kv_heads = 2;
        let seq_len = 8;

        let layer = FlashAttentionLayer::random(hidden_size, head_dim, num_heads, num_kv_heads, 123);
        let config = make_config(hidden_size, head_dim, num_heads, num_kv_heads);
        let mut state = layer.init_state(&config);

        // Generate random input
        let mut rng = SimpleRng::new(456);
        let input: Vec<f32> = (0..seq_len * hidden_size)
            .map(|_| rng.next_f32_range(-0.5, 0.5))
            .collect();

        let output = layer.process_prefill(&input, &mut state, seq_len);

        // Verify output shape
        assert_eq!(
            output.len(),
            seq_len * hidden_size,
            "Output should be [seq_len, hidden_size]"
        );

        // Verify no NaN/Inf
        for (i, &val) in output.iter().enumerate() {
            assert!(
                val.is_finite(),
                "Output element {} is not finite: {}",
                i, val
            );
        }

        // Verify KV cache was populated
        assert_eq!(layer.cached_length(&state), seq_len);
    }

    #[test]
    fn test_flash_attn_decode_after_prefill() {
        let hidden_size = 32;
        let head_dim = 16;
        let num_heads = 2;
        let num_kv_heads = 2;
        let prefill_len = 8;
        let decode_steps = 4;

        let layer = FlashAttentionLayer::random(hidden_size, head_dim, num_heads, num_kv_heads, 789);
        let config = make_config(hidden_size, head_dim, num_heads, num_kv_heads);
        let mut state = layer.init_state(&config);

        let mut rng = SimpleRng::new(101);

        // Prefill
        let prefill_input: Vec<f32> = (0..prefill_len * hidden_size)
            .map(|_| rng.next_f32_range(-0.5, 0.5))
            .collect();
        let prefill_out = layer.process_prefill(&prefill_input, &mut state, prefill_len);
        assert_eq!(prefill_out.len(), prefill_len * hidden_size);
        assert_eq!(layer.cached_length(&state), prefill_len);

        // Decode tokens one at a time
        for step in 0..decode_steps {
            let token_input: Vec<f32> = (0..hidden_size)
                .map(|_| rng.next_f32_range(-0.5, 0.5))
                .collect();

            let decode_out = layer.process_decode(&token_input, &mut state);

            assert_eq!(
                decode_out.len(),
                hidden_size,
                "Decode output should be [hidden_size]"
            );

            // Verify no NaN/Inf
            for (i, &val) in decode_out.iter().enumerate() {
                assert!(
                    val.is_finite(),
                    "Decode step {} output[{}] is not finite: {}",
                    step, i, val
                );
            }

            // Verify cache grew
            assert_eq!(
                layer.cached_length(&state),
                prefill_len + step + 1,
                "Cache length should grow by 1 each decode step"
            );
        }
    }

    #[test]
    fn test_flash_attn_gqa_config() {
        // GQA: 8 Q heads, 2 KV heads -> group_size = 4
        let layer = FlashAttentionLayer::random(64, 16, 8, 2, 999);
        assert_eq!(layer.gqa_config().group_size, 4);
    }

    #[test]
    fn test_flash_attn_integration_prefill_then_decode() {
        // Integration test: prefill 64 tokens, decode 10 tokens, verify no NaN
        let hidden_size = 32;
        let head_dim = 16;
        let num_heads = 2;
        let num_kv_heads = 2;
        let prefill_len = 64;
        let decode_count = 10;

        let layer = FlashAttentionLayer::random(hidden_size, head_dim, num_heads, num_kv_heads, 42);
        let config = make_config(hidden_size, head_dim, num_heads, num_kv_heads);
        let mut state = layer.init_state(&config);

        let mut rng = SimpleRng::new(7777);

        // Prefill 64 tokens
        let prefill_input: Vec<f32> = (0..prefill_len * hidden_size)
            .map(|_| rng.next_f32_range(-0.3, 0.3))
            .collect();
        let prefill_out = layer.process_prefill(&prefill_input, &mut state, prefill_len);
        assert_eq!(prefill_out.len(), prefill_len * hidden_size);
        for &val in &prefill_out {
            assert!(val.is_finite(), "Prefill output contains non-finite: {}", val);
        }

        // Decode 10 tokens
        for step in 0..decode_count {
            let token_input: Vec<f32> = (0..hidden_size)
                .map(|_| rng.next_f32_range(-0.3, 0.3))
                .collect();
            let decode_out = layer.process_decode(&token_input, &mut state);
            assert_eq!(decode_out.len(), hidden_size);
            for &val in &decode_out {
                assert!(
                    val.is_finite(),
                    "Decode step {} output contains non-finite: {}",
                    step, val
                );
            }
        }

        // Final cache length: prefill + decode
        assert_eq!(layer.cached_length(&state), prefill_len + decode_count);
    }
}
