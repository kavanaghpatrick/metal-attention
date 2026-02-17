//! EAGLE-3 draft head: lightweight prediction network for speculative decoding.
//!
//! `EagleHead` implements the EAGLE-3 draft head that fuses multi-layer hidden
//! states from the target model and produces draft token predictions via an
//! autoregressive chain. The head consists of:
//!   - FC fusion layer: projects concatenated [feat_low, feat_mid, feat_high] (3*hidden)
//!     through a linear layer to hidden_size
//!   - FC concat layer: projects [hidden_state, prev_token_embed] (2*hidden)
//!     through a linear layer to hidden_size
//!   - Single decoder layer: RMSNorm -> Attention -> FFN (same architecture as target)
//!   - Shared lm_head from target model for final logit projection
//!
//! POC uses F32 random weights. Real weights loaded from SafeTensors in Phase 2.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;

use metal_attention_kernels::buffer::{alloc_buffer, alloc_buffer_private};
use metal_attention_kernels::device::GpuDevice;
use metal_attention_kernels::pipeline::{PsoCache, PsoKey};

use crate::gpu_kv_cache::GpuKVCache;
use crate::gpu_weight_store::{AttnProjBuffers, FfnBuffers, WeightBuffer};

/// EAGLE-3 draft head for speculative token prediction.
///
/// Contains FC fusion/concat layers, a single decoder transformer layer,
/// an independent KV cache (small, reset per speculation round), and all
/// scratch buffers needed for the forward pass.
pub struct EagleHead {
    // --- FC layers (F32 for POC) ---
    /// FC fusion weight: projects 3*hidden_size -> hidden_size [hidden_size, 3*hidden_size].
    pub fc_fuse_weight: WeightBuffer,
    /// FC concat weight: projects 2*hidden_size -> hidden_size [hidden_size, 2*hidden_size].
    pub fc_concat_weight: WeightBuffer,

    // --- Single decoder layer ---
    /// Attention RMSNorm weight (F32, hidden_size elements).
    pub decoder_attn_norm: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Attention Q/K/V/O projection weights.
    pub decoder_attn: AttnProjBuffers,
    /// FFN RMSNorm weight (F32, hidden_size elements).
    pub decoder_ffn_norm: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// FFN gate/up/down weights.
    pub decoder_ffn: FfnBuffers,

    // --- Eagle-specific KV cache (1 layer, small max_len) ---
    /// KV cache for the eagle decoder layer. Reset per speculation round.
    pub eagle_kv_cache: GpuKVCache,

    // --- Scratch buffers ---
    /// Ping-pong hidden state buffer A (hidden_size * 4 bytes).
    pub hidden_a: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Ping-pong hidden state buffer B (hidden_size * 4 bytes).
    pub hidden_b: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Scratch Q projection (num_heads * head_dim * 4 bytes).
    pub scratch_q: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Scratch K projection (num_kv_heads * head_dim * 4 bytes).
    pub scratch_k: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Scratch V projection (num_kv_heads * head_dim * 4 bytes).
    pub scratch_v: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Scratch attention output (num_heads * head_dim * 4 bytes).
    pub scratch_attn_out: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Scratch gate buffer (intermediate_size * 4 bytes).
    pub scratch_gate: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Scratch up buffer (intermediate_size * 4 bytes).
    pub scratch_up: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Scratch silu buffer (intermediate_size * 4 bytes).
    pub scratch_silu: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Fused features buffer: 3 * hidden_size floats (for concat_buffers_3 output).
    pub fused_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Combined buffer: 2 * hidden_size floats (for concat_buffers_2 output).
    pub combined_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Logits buffer (vocab_size * 4 bytes).
    pub logits_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Argmax partial values buffer.
    pub argmax_partial_vals: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Argmax partial indices buffer.
    pub argmax_partial_idxs: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Argmax result buffer (single u32, shared storage for CPU readback).
    pub argmax_result: Retained<ProtocolObject<dyn MTLBuffer>>,

    // --- PSO cache ---
    /// Pipeline state object cache for all eagle head kernels.
    pub pso_cache: PsoCache,

    // --- Model dimensions ---
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,

    /// Current RoPE position for the eagle decoder (increments per draft token).
    pub position: usize,
}

/// Simple LCG PRNG for reproducible random weight initialization.
struct LcgRng {
    state: u64,
}

impl LcgRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Generate next random f32 in range [-0.01, 0.01].
    fn next_f32(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let val = ((self.state >> 33) as f32) / (u32::MAX as f32) * 0.02 - 0.01;
        val
    }

    /// Fill a shared Metal buffer with random F32 data.
    fn fill_buffer(
        &mut self,
        buf: &Retained<ProtocolObject<dyn MTLBuffer>>,
        num_elements: usize,
    ) {
        let ptr = buf.contents().as_ptr() as *mut f32;
        unsafe {
            for i in 0..num_elements {
                *ptr.add(i) = self.next_f32();
            }
        }
    }
}

/// Allocate a shared Metal buffer filled with random F32 data.
fn alloc_random_f32(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    num_elements: usize,
    rng: &mut LcgRng,
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    let buf = alloc_buffer(device, num_elements * std::mem::size_of::<f32>());
    rng.fill_buffer(&buf, num_elements);
    buf
}

impl EagleHead {
    /// Construct an EagleHead with random F32 weights (POC).
    ///
    /// All weight buffers are initialized with small random values from a
    /// deterministic LCG PRNG (seed=42). This allows testing the full EAGLE
    /// pipeline without requiring real trained weights.
    ///
    /// # Arguments
    /// - `device`: Static reference to the shared GPU device.
    /// - `hidden_size`: Target model hidden dimension (e.g., 4096 for Mistral-7B).
    /// - `num_heads`: Number of attention heads (e.g., 32).
    /// - `num_kv_heads`: Number of KV heads for GQA (e.g., 8).
    /// - `head_dim`: Dimension per head (e.g., 128).
    /// - `intermediate_size`: FFN intermediate dimension (e.g., 14336).
    /// - `vocab_size`: Vocabulary size (e.g., 32000).
    pub fn new_random(
        device: &'static GpuDevice,
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        vocab_size: usize,
    ) -> Self {
        let mut rng = LcgRng::new(42);
        let dev = &*device.device;
        let kv_dim = num_kv_heads * head_dim;

        // --- FC layers (F32) ---
        // fc_fuse: [hidden_size, 3 * hidden_size]
        let fuse_elements = hidden_size * 3 * hidden_size;
        let fc_fuse_buf = alloc_random_f32(dev, fuse_elements, &mut rng);
        let fc_fuse_weight = WeightBuffer::zero_offset(fc_fuse_buf);

        // fc_concat: [hidden_size, 2 * hidden_size]
        let concat_elements = hidden_size * 2 * hidden_size;
        let fc_concat_buf = alloc_random_f32(dev, concat_elements, &mut rng);
        let fc_concat_weight = WeightBuffer::zero_offset(fc_concat_buf);

        // --- Decoder layer attention weights (F32 for POC) ---
        // Q: [hidden_size, hidden_size]  (num_heads * head_dim = hidden_size)
        let q_buf = alloc_random_f32(dev, hidden_size * hidden_size, &mut rng);
        // K: [kv_dim, hidden_size]
        let k_buf = alloc_random_f32(dev, kv_dim * hidden_size, &mut rng);
        // V: [kv_dim, hidden_size]
        let v_buf = alloc_random_f32(dev, kv_dim * hidden_size, &mut rng);
        // O: [hidden_size, hidden_size]
        let o_buf = alloc_random_f32(dev, hidden_size * hidden_size, &mut rng);

        let decoder_attn = AttnProjBuffers {
            q: WeightBuffer::zero_offset(q_buf),
            k: WeightBuffer::zero_offset(k_buf),
            v: WeightBuffer::zero_offset(v_buf),
            o: WeightBuffer::zero_offset(o_buf),
        };

        // --- Decoder layer FFN weights (F32 for POC) ---
        // gate: [intermediate_size, hidden_size]
        let gate_buf = alloc_random_f32(dev, intermediate_size * hidden_size, &mut rng);
        // up: [intermediate_size, hidden_size]
        let up_buf = alloc_random_f32(dev, intermediate_size * hidden_size, &mut rng);
        // down: [hidden_size, intermediate_size]
        let down_buf = alloc_random_f32(dev, hidden_size * intermediate_size, &mut rng);

        let decoder_ffn = FfnBuffers {
            gate: WeightBuffer::zero_offset(gate_buf),
            up: WeightBuffer::zero_offset(up_buf),
            down: WeightBuffer::zero_offset(down_buf),
        };

        // --- Norm weights (F32, initialized to ~1.0 with small noise) ---
        let decoder_attn_norm = alloc_buffer(dev, hidden_size * std::mem::size_of::<f32>());
        {
            let ptr = decoder_attn_norm.contents().as_ptr() as *mut f32;
            unsafe {
                for i in 0..hidden_size {
                    *ptr.add(i) = 1.0 + rng.next_f32();
                }
            }
        }

        let decoder_ffn_norm = alloc_buffer(dev, hidden_size * std::mem::size_of::<f32>());
        {
            let ptr = decoder_ffn_norm.contents().as_ptr() as *mut f32;
            unsafe {
                for i in 0..hidden_size {
                    *ptr.add(i) = 1.0 + rng.next_f32();
                }
            }
        }

        // --- Eagle KV cache (1 layer, small capacity) ---
        let eagle_kv_cache = GpuKVCache::new(dev, 64, kv_dim);

        // --- Scratch buffers (private storage for GPU-only intermediates) ---
        let f32_size = std::mem::size_of::<f32>();
        let hidden_a = alloc_buffer_private(dev, hidden_size * f32_size);
        let hidden_b = alloc_buffer_private(dev, hidden_size * f32_size);
        let scratch_q = alloc_buffer_private(dev, num_heads * head_dim * f32_size);
        let scratch_k = alloc_buffer_private(dev, kv_dim * f32_size);
        let scratch_v = alloc_buffer_private(dev, kv_dim * f32_size);
        let scratch_attn_out = alloc_buffer_private(dev, num_heads * head_dim * f32_size);
        let scratch_gate = alloc_buffer_private(dev, intermediate_size * f32_size);
        let scratch_up = alloc_buffer_private(dev, intermediate_size * f32_size);
        let scratch_silu = alloc_buffer_private(dev, intermediate_size * f32_size);

        // Fusion/concat buffers (private, GPU-only)
        let fused_buf = alloc_buffer_private(dev, 3 * hidden_size * f32_size);
        let combined_buf = alloc_buffer_private(dev, 2 * hidden_size * f32_size);

        // Logits buffer (shared for CPU readback if needed, but argmax is GPU-side)
        let logits_buf = alloc_buffer_private(dev, vocab_size * f32_size);

        // Argmax buffers
        let num_argmax_groups = (vocab_size + 255) / 256;
        let argmax_partial_vals =
            alloc_buffer_private(dev, num_argmax_groups * f32_size);
        let argmax_partial_idxs =
            alloc_buffer_private(dev, num_argmax_groups * std::mem::size_of::<u32>());
        // Result buffer is Shared so CPU can read back the token id
        let argmax_result = alloc_buffer(dev, std::mem::size_of::<u32>());

        // --- PSO cache: prewarm all kernels needed by the eagle head ---
        let mut pso_cache = PsoCache::new(device.library.clone());
        let pso_keys = vec![
            PsoKey::simple("rmsnorm_optimized"),
            PsoKey::simple("matvec_f32_v2"),
            PsoKey::simple("rope_apply_dual"),
            PsoKey::simple("decode_attention_v2"),
            PsoKey::simple("ffn_silu"),
            PsoKey::simple("buffer_copy"),
            PsoKey::simple("concat_buffers_3"),
            PsoKey::simple("concat_buffers_2"),
            PsoKey::simple("argmax_reduce"),
            PsoKey::simple("argmax_final"),
            PsoKey::simple("kv_cache_copy"),
        ];
        pso_cache.prewarm(&pso_keys);

        Self {
            fc_fuse_weight,
            fc_concat_weight,
            decoder_attn_norm,
            decoder_attn,
            decoder_ffn_norm,
            decoder_ffn,
            eagle_kv_cache,
            hidden_a,
            hidden_b,
            scratch_q,
            scratch_k,
            scratch_v,
            scratch_attn_out,
            scratch_gate,
            scratch_up,
            scratch_silu,
            fused_buf,
            combined_buf,
            logits_buf,
            argmax_partial_vals,
            argmax_partial_idxs,
            argmax_result,
            pso_cache,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            position: 0,
        }
    }

    /// Reset the eagle KV cache (call at the start of each speculation round).
    ///
    /// Truncates the cache to length 0, discarding all stored KV pairs.
    /// The eagle head generates a fresh autoregressive chain each round,
    /// so the cache must be cleared between rounds.
    pub fn reset_kv_cache(&mut self) {
        self.eagle_kv_cache.truncate(0);
        self.position = 0;
    }
}
