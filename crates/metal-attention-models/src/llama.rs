//! Llama/Mistral model layer: pure transformer with FlashAttention.
//!
//! LlamaLayer is a standard Llama-style transformer block with:
//!   1. Pre-attention RMSNorm
//!   2. FlashAttention (Q/K/V/O projections + scaled dot-product)
//!   3. Residual connection
//!   4. Pre-FFN RMSNorm
//!   5. SwiGLU FFN (gate/up/down projections)
//!   6. Residual connection
//!
//! Weight layout follows Llama/Mistral GGUF conventions:
//!   - `blk.{N}.attn_q.weight` -> Q projection
//!   - `blk.{N}.attn_k.weight` -> K projection
//!   - `blk.{N}.attn_v.weight` -> V projection
//!   - `blk.{N}.attn_output.weight` -> output projection
//!   - `blk.{N}.ffn_gate.weight` -> SwiGLU gate
//!   - `blk.{N}.ffn_up.weight` -> SwiGLU up
//!   - `blk.{N}.ffn_down.weight` -> SwiGLU down
//!   - `blk.{N}.attn_norm.weight` -> attention RMSNorm
//!   - `blk.{N}.ffn_norm.weight` -> FFN RMSNorm

use crate::flash_attn::{FlashAttentionLayer, FlashAttentionState};
use crate::registry::ModelConfig;
use metal_attention_gguf::{GgufFile, GgufType};
use metal_attention_kernels::dequant::{dispatch_dequantize_q4_0, dispatch_dequantize_q8_0};
use metal_attention_kernels::device::GpuDevice;
use metal_attention_kernels::pipeline::PsoCache;
use metal_attention_traits::attention::PositionEncoding;
use metal_attention_traits::sequence::SequenceBlock;
use metal_attention_traits::types::{BlockConfig, DType, TensorView};

/// Dequantize a named tensor from a GGUF file to f32 values.
///
/// Local helper that dispatches based on quantization type.
fn dequant_tensor(
    gguf_file: &GgufFile,
    tensor_name: &str,
    device: Option<&GpuDevice>,
    pso_cache: Option<&mut PsoCache>,
) -> Result<Vec<f32>, String> {
    let tensor_info = gguf_file
        .find_tensor(tensor_name)
        .ok_or_else(|| format!("Tensor not found: {tensor_name}"))?;

    let bytes = gguf_file.tensor_data(tensor_info);
    let n_elements = tensor_info.n_elements() as usize;

    match tensor_info.gguf_type {
        GgufType::F32 => {
            let floats: &[f32] = bytemuck::cast_slice(bytes);
            Ok(floats.to_vec())
        }
        GgufType::F16 => {
            let mut result = Vec::with_capacity(n_elements);
            for i in 0..n_elements {
                let lo = bytes[i * 2];
                let hi = bytes[i * 2 + 1];
                let f16_val = half::f16::from_le_bytes([lo, hi]);
                result.push(f16_val.to_f32());
            }
            Ok(result)
        }
        GgufType::Q4_0 => {
            let device = device.ok_or("GPU device required for Q4_0 dequantization")?;
            let pso_cache = pso_cache.ok_or("PSO cache required for Q4_0 dequantization")?;
            let block_size = tensor_info.gguf_type.block_size();
            let n_blocks = n_elements / block_size;
            Ok(dispatch_dequantize_q4_0(device, pso_cache, bytes, n_blocks))
        }
        GgufType::Q8_0 => {
            let device = device.ok_or("GPU device required for Q8_0 dequantization")?;
            let pso_cache = pso_cache.ok_or("PSO cache required for Q8_0 dequantization")?;
            let block_size = tensor_info.gguf_type.block_size();
            let n_blocks = n_elements / block_size;
            Ok(dispatch_dequantize_q8_0(device, pso_cache, bytes, n_blocks))
        }
        other => Err(format!("Unsupported quantization type: {other:?}")),
    }
}

/// Persistent state for one LlamaLayer.
///
/// Wraps the FlashAttentionState (KV cache) from the inner attention layer.
#[derive(Clone)]
pub struct LlamaState {
    /// Attention layer state (KV cache).
    pub attn_state: FlashAttentionState,
}

/// Llama-style transformer block.
///
/// Forward pass: RMSNorm -> Attention -> Residual -> RMSNorm -> SwiGLU FFN -> Residual
pub struct LlamaLayer {
    /// FlashAttention sub-layer handling Q/K/V projections and attention.
    pub attention: FlashAttentionLayer,

    /// Attention RMSNorm weight: [hidden_size].
    pub attn_norm_weight: Vec<f32>,
    /// FFN RMSNorm weight: [hidden_size].
    pub ffn_norm_weight: Vec<f32>,

    /// SwiGLU gate projection: [intermediate_size, hidden_size], row-major.
    pub w_gate: Vec<f32>,
    /// SwiGLU up projection: [intermediate_size, hidden_size], row-major.
    pub w_up: Vec<f32>,
    /// SwiGLU down projection: [hidden_size, intermediate_size], row-major.
    pub w_down: Vec<f32>,

    /// Hidden size (model dimension).
    pub hidden_size: usize,
    /// Intermediate size for FFN.
    pub intermediate_size: usize,
}

impl LlamaLayer {
    /// Create a new LlamaLayer with random weights for testing.
    ///
    /// Uses a deterministic pseudo-random sequence seeded by `seed`.
    /// The `intermediate_size` defaults to `hidden_size * 4` (standard Llama ratio).
    pub fn random(
        hidden_size: usize,
        head_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        seed: u64,
    ) -> Self {
        let intermediate_size = hidden_size * 4;
        let mut rng = SimpleRng::new(seed);

        // Attention sub-layer with its own seed
        let attention =
            FlashAttentionLayer::random(hidden_size, head_dim, num_heads, num_kv_heads, seed + 1);

        // RMSNorm weights: initialize to 1.0 (identity)
        let attn_norm_weight = vec![1.0f32; hidden_size];
        let ffn_norm_weight = vec![1.0f32; hidden_size];

        // SwiGLU FFN weights
        let scale = 1.0 / (hidden_size as f32).sqrt();
        let w_gate: Vec<f32> = (0..intermediate_size * hidden_size)
            .map(|_| rng.next_f32_range(-scale, scale))
            .collect();
        let w_up: Vec<f32> = (0..intermediate_size * hidden_size)
            .map(|_| rng.next_f32_range(-scale, scale))
            .collect();
        let w_down: Vec<f32> = (0..hidden_size * intermediate_size)
            .map(|_| rng.next_f32_range(-scale, scale))
            .collect();

        Self {
            attention,
            attn_norm_weight,
            ffn_norm_weight,
            w_gate,
            w_up,
            w_down,
            hidden_size,
            intermediate_size,
        }
    }

    /// Load a LlamaLayer from GGUF tensors.
    ///
    /// Loads 9 tensors for this layer: attn_norm, Q/K/V/O projections,
    /// ffn_norm, and SwiGLU gate/up/down weights. Uses GPU dequantization
    /// for quantized weight types.
    ///
    /// # Arguments
    /// - `gguf_file`: The parsed GGUF file.
    /// - `layer_idx`: Layer index (0-based) for tensor name lookup.
    /// - `config`: Model configuration with dimensions.
    /// - `device`: GPU device for quantized tensor dispatch.
    /// - `pso_cache`: Pipeline state cache for GPU kernels.
    #[allow(clippy::needless_option_as_deref)]
    pub fn from_gguf(
        gguf_file: &GgufFile,
        layer_idx: usize,
        config: &ModelConfig,
        intermediate_size: usize,
        device: Option<&GpuDevice>,
        pso_cache: Option<&mut PsoCache>,
    ) -> Result<Self, String> {
        let mut pso = pso_cache;

        let n = layer_idx;

        // Norm weights (usually F32, no GPU needed)
        let attn_norm_weight = dequant_tensor(
            gguf_file,
            &format!("blk.{n}.attn_norm.weight"),
            device,
            None,
        )?;
        let ffn_norm_weight =
            dequant_tensor(gguf_file, &format!("blk.{n}.ffn_norm.weight"), device, None)?;

        // Attention weights (may be quantized)
        let w_q = dequant_tensor(
            gguf_file,
            &format!("blk.{n}.attn_q.weight"),
            device,
            pso.as_deref_mut(),
        )?;
        let w_k = dequant_tensor(
            gguf_file,
            &format!("blk.{n}.attn_k.weight"),
            device,
            pso.as_deref_mut(),
        )?;
        let w_v = dequant_tensor(
            gguf_file,
            &format!("blk.{n}.attn_v.weight"),
            device,
            pso.as_deref_mut(),
        )?;
        let w_o = dequant_tensor(
            gguf_file,
            &format!("blk.{n}.attn_output.weight"),
            device,
            pso.as_deref_mut(),
        )?;

        // FFN weights (may be quantized)
        let w_gate = dequant_tensor(
            gguf_file,
            &format!("blk.{n}.ffn_gate.weight"),
            device,
            pso.as_deref_mut(),
        )?;
        let w_up = dequant_tensor(
            gguf_file,
            &format!("blk.{n}.ffn_up.weight"),
            device,
            pso.as_deref_mut(),
        )?;
        let w_down = dequant_tensor(
            gguf_file,
            &format!("blk.{n}.ffn_down.weight"),
            device,
            pso.as_deref_mut(),
        )?;

        // Build FlashAttentionLayer with loaded weights
        let attention = FlashAttentionLayer {
            hidden_size: config.hidden_size,
            head_dim: config.head_dim,
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            max_seq_len: 2048,
            w_q,
            w_k,
            w_v,
            w_o,
            pos_encoding: PositionEncoding::None,
        };

        Ok(Self {
            attention,
            attn_norm_weight,
            ffn_norm_weight,
            w_gate,
            w_up,
            w_down,
            hidden_size: config.hidden_size,
            intermediate_size,
        })
    }

    /// Apply RMSNorm: y = x * weight / rms(x).
    fn rmsnorm(x: &[f32], weight: &[f32]) -> Vec<f32> {
        let n = x.len();
        let eps = 1e-5f32;
        let rms = (x.iter().map(|&v| v * v).sum::<f32>() / n as f32 + eps).sqrt();
        x.iter()
            .zip(weight.iter())
            .map(|(&xi, &wi)| xi / rms * wi)
            .collect()
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

    /// SwiGLU FFN forward: down(silu(gate(x)) * up(x)).
    fn ffn_forward(&self, x: &[f32]) -> Vec<f32> {
        let hs = self.hidden_size;
        let is = self.intermediate_size;

        let gate = Self::matvec(&self.w_gate, x, is, hs);
        let up = Self::matvec(&self.w_up, x, is, hs);

        // SwiGLU: silu(gate) * up
        let hidden: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(&g, &u)| Self::silu(g) * u)
            .collect();

        // Down projection: [hidden_size, intermediate_size]
        Self::matvec(&self.w_down, &hidden, hs, is)
    }

    /// Element-wise addition: a + b.
    fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
    }

    /// Process a single token through the full Llama block (decode path).
    ///
    /// Forward: RMSNorm -> Attention -> Residual -> RMSNorm -> FFN -> Residual
    ///
    /// Input: `[hidden_size]` flat f32 slice.
    /// Returns: `[hidden_size]` flat f32 output.
    pub fn process_token(&self, input: &[f32], state: &mut LlamaState) -> Vec<f32> {
        // 1. Pre-attention RMSNorm
        let normed = Self::rmsnorm(input, &self.attn_norm_weight);

        // 2. Attention
        let attn_out = self
            .attention
            .process_decode(&normed, &mut state.attn_state);

        // 3. Residual connection
        let hidden = Self::add(input, &attn_out);

        // 4. Pre-FFN RMSNorm
        let normed2 = Self::rmsnorm(&hidden, &self.ffn_norm_weight);

        // 5. SwiGLU FFN
        let ffn_out = self.ffn_forward(&normed2);

        // 6. Residual connection
        Self::add(&hidden, &ffn_out)
    }

    /// Process a sequence through the full Llama block (prefill path).
    ///
    /// Processes each token through RMSNorm -> Attention (prefill) -> Residual -> RMSNorm -> FFN -> Residual.
    ///
    /// Input: `[seq_len * hidden_size]` flat f32 slice.
    /// Returns: `[seq_len * hidden_size]` flat f32 output.
    pub fn process_prefill(
        &self,
        input: &[f32],
        state: &mut LlamaState,
        seq_len: usize,
    ) -> Vec<f32> {
        let hs = self.hidden_size;
        assert_eq!(input.len(), seq_len * hs);

        // 1. Pre-attention RMSNorm (per token)
        let mut normed = Vec::with_capacity(seq_len * hs);
        for t in 0..seq_len {
            let token = &input[t * hs..(t + 1) * hs];
            normed.extend(Self::rmsnorm(token, &self.attn_norm_weight));
        }

        // 2. Attention (prefill: full sequence)
        let attn_out = self
            .attention
            .process_prefill(&normed, &mut state.attn_state, seq_len);

        // 3. Residual connection (per token)
        let mut hidden = Vec::with_capacity(seq_len * hs);
        for t in 0..seq_len {
            let inp = &input[t * hs..(t + 1) * hs];
            let attn = &attn_out[t * hs..(t + 1) * hs];
            hidden.extend(Self::add(inp, attn));
        }

        // 4. Pre-FFN RMSNorm + FFN + Residual (per token)
        let mut output = Vec::with_capacity(seq_len * hs);
        for t in 0..seq_len {
            let h = &hidden[t * hs..(t + 1) * hs];
            let normed2 = Self::rmsnorm(h, &self.ffn_norm_weight);
            let ffn_out = self.ffn_forward(&normed2);
            output.extend(Self::add(h, &ffn_out));
        }

        output
    }
}

impl SequenceBlock for LlamaLayer {
    type State = LlamaState;

    fn init_state(&self, config: &BlockConfig) -> Self::State {
        LlamaState {
            attn_state: self.attention.init_state(config),
        }
    }

    fn forward_prefill(
        &self,
        input: &TensorView,
        _state: &mut Self::State,
        _config: &BlockConfig,
    ) -> TensorView {
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

    fn state_size_bytes(&self, config: &BlockConfig) -> usize {
        self.attention.state_size_bytes(config)
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
    use metal_attention_gguf::architectures::{map_tensor_name, WeightRole};
    use metal_attention_gguf::ModelArchitecture;

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
    fn test_llama_layer_forward_decode_non_nan() {
        let hidden_size = 32;
        let head_dim = 16;
        let num_heads = 2;
        let num_kv_heads = 2;

        let layer = LlamaLayer::random(hidden_size, head_dim, num_heads, num_kv_heads, 42);
        let config = make_config(hidden_size, head_dim, num_heads, num_kv_heads);
        let mut state = layer.init_state(&config);

        // Need at least one prefill token to populate KV cache before decode
        let mut rng = SimpleRng::new(100);
        let prefill_input: Vec<f32> = (0..hidden_size)
            .map(|_| rng.next_f32_range(-0.5, 0.5))
            .collect();
        let prefill_out = layer.process_prefill(&prefill_input, &mut state, 1);
        assert_eq!(prefill_out.len(), hidden_size);
        for &v in &prefill_out {
            assert!(v.is_finite(), "Prefill output not finite: {}", v);
        }

        // Decode step
        let decode_input: Vec<f32> = (0..hidden_size)
            .map(|_| rng.next_f32_range(-0.5, 0.5))
            .collect();
        let output = layer.process_token(&decode_input, &mut state);

        assert_eq!(output.len(), hidden_size, "Output should be [hidden_size]");
        for (i, &val) in output.iter().enumerate() {
            assert!(
                val.is_finite(),
                "Output element {} is not finite: {}",
                i,
                val
            );
        }
    }

    #[test]
    fn test_llama_layer_prefill_non_nan() {
        let hidden_size = 32;
        let head_dim = 16;
        let num_heads = 2;
        let num_kv_heads = 2;
        let seq_len = 4;

        let layer = LlamaLayer::random(hidden_size, head_dim, num_heads, num_kv_heads, 123);
        let config = make_config(hidden_size, head_dim, num_heads, num_kv_heads);
        let mut state = layer.init_state(&config);

        let mut rng = SimpleRng::new(456);
        let input: Vec<f32> = (0..seq_len * hidden_size)
            .map(|_| rng.next_f32_range(-0.5, 0.5))
            .collect();

        let output = layer.process_prefill(&input, &mut state, seq_len);

        assert_eq!(
            output.len(),
            seq_len * hidden_size,
            "Output should be [seq_len, hidden_size]"
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

    #[test]
    fn test_llama_weight_name_mapping() {
        let arch = ModelArchitecture::Llama;

        // Attention weights
        assert_eq!(
            map_tensor_name(arch, "blk.0.attn_q.weight"),
            Some((Some(0), WeightRole::QProj))
        );
        assert_eq!(
            map_tensor_name(arch, "blk.0.attn_k.weight"),
            Some((Some(0), WeightRole::KProj))
        );
        assert_eq!(
            map_tensor_name(arch, "blk.0.attn_v.weight"),
            Some((Some(0), WeightRole::VProj))
        );
        assert_eq!(
            map_tensor_name(arch, "blk.0.attn_output.weight"),
            Some((Some(0), WeightRole::OutProj))
        );

        // FFN weights
        assert_eq!(
            map_tensor_name(arch, "blk.5.ffn_gate.weight"),
            Some((Some(5), WeightRole::GateProj))
        );
        assert_eq!(
            map_tensor_name(arch, "blk.5.ffn_up.weight"),
            Some((Some(5), WeightRole::UpProj))
        );
        assert_eq!(
            map_tensor_name(arch, "blk.5.ffn_down.weight"),
            Some((Some(5), WeightRole::DownProj))
        );

        // Norms
        assert_eq!(
            map_tensor_name(arch, "blk.0.attn_norm.weight"),
            Some((Some(0), WeightRole::AttnNorm))
        );
        assert_eq!(
            map_tensor_name(arch, "blk.0.ffn_norm.weight"),
            Some((Some(0), WeightRole::FfnNorm))
        );
    }

    #[test]
    fn test_llama_swiglu_ffn() {
        let hidden_size = 16;
        let layer = LlamaLayer::random(hidden_size, 8, 2, 2, 999);

        let mut rng = SimpleRng::new(777);
        let input: Vec<f32> = (0..hidden_size)
            .map(|_| rng.next_f32_range(-1.0, 1.0))
            .collect();

        let output = layer.ffn_forward(&input);
        assert_eq!(
            output.len(),
            hidden_size,
            "FFN output should be [hidden_size]"
        );
        for (i, &val) in output.iter().enumerate() {
            assert!(val.is_finite(), "FFN output[{}] not finite: {}", i, val);
        }
    }

    #[test]
    fn test_llama_trait_compliance() {
        let hidden_size = 32;
        let head_dim = 16;
        let num_heads = 2;
        let num_kv_heads = 2;

        let layer = LlamaLayer::random(hidden_size, head_dim, num_heads, num_kv_heads, 42);
        let config = make_config(hidden_size, head_dim, num_heads, num_kv_heads);

        let state = layer.init_state(&config);
        assert!(
            layer.state_size_bytes(&config) > 0,
            "State size should be positive"
        );

        // Verify state was properly initialized
        assert_eq!(state.attn_state.kv_cache.len(), 0);
    }

    #[test]
    fn test_llama_residual_connections() {
        // Verify that residual connections preserve signal:
        // output should differ from input (FFN + attention changed it)
        // but should not be all zeros (residual preserved some signal)
        let hidden_size = 32;
        let head_dim = 16;
        let num_heads = 2;
        let num_kv_heads = 2;

        let layer = LlamaLayer::random(hidden_size, head_dim, num_heads, num_kv_heads, 42);
        let config = make_config(hidden_size, head_dim, num_heads, num_kv_heads);
        let mut state = layer.init_state(&config);

        let input: Vec<f32> = (0..hidden_size).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let output = layer.process_prefill(&input, &mut state, 1);

        // Output should not be all zeros (residual preserves input)
        let has_nonzero = output.iter().any(|&v| v.abs() > 1e-10);
        assert!(
            has_nonzero,
            "Output should not be all zeros due to residual connections"
        );

        // Output should differ from input (attention + FFN changed it)
        let differs = input
            .iter()
            .zip(output.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-10);
        assert!(differs, "Output should differ from input");
    }
}
