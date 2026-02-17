#![allow(dead_code)]
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
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder,
};

use metal_attention_kernels::buffer::{alloc_buffer, alloc_buffer_private};
use metal_attention_kernels::device::GpuDevice;
use metal_attention_kernels::dispatch::{set_buffer, set_bytes};
use metal_attention_kernels::pipeline::{PsoCache, PsoKey};
use metal_attention_kernels::types::LayerParams;
use objc2_metal::MTLSize;

use crate::eagle_weights::EagleWeightStore;
use crate::gpu_forward_pass::GpuForwardPass;
use crate::gpu_kv_cache::GpuKVCache;
use crate::gpu_weight_store::{AttnProjBuffers, FfnBuffers, WeightBuffer};

/// EAGLE-3 draft head for speculative token prediction.
///
/// Contains FC fusion/concat layers, a single decoder transformer layer,
/// an independent KV cache (small, reset per speculation round), and all
/// scratch buffers needed for the forward pass.
pub struct EagleHead {
    // --- Device reference (for command buffer creation) ---
    /// Static reference to the shared GPU device.
    pub device: &'static GpuDevice,

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
    /// Embedding scratch buffer (shared, hidden_size * 4 bytes, CPU-writable for embed lookup).
    pub embed_scratch: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Final norm weight (F32, hidden_size elements) for the output projection norm.
    pub final_norm: Retained<ProtocolObject<dyn MTLBuffer>>,
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

    // --- Model parameters ---
    /// RoPE base frequency (e.g. 10000.0).
    pub rope_theta: f32,
    /// RMSNorm epsilon (e.g. 1e-5).
    pub rms_norm_eps: f32,

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
        ((self.state >> 33) as f32) / (u32::MAX as f32) * 0.02 - 0.01
    }

    /// Fill a shared Metal buffer with random F32 data.
    fn fill_buffer(&mut self, buf: &Retained<ProtocolObject<dyn MTLBuffer>>, num_elements: usize) {
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

        // Embedding scratch buffer (shared, CPU-writable for embed lookup)
        let embed_scratch = alloc_buffer(dev, hidden_size * f32_size);

        // Final norm weight (F32, initialized to ~1.0 with small noise, same as attn/ffn norms)
        let final_norm = alloc_buffer(dev, hidden_size * f32_size);
        {
            let ptr = final_norm.contents().as_ptr() as *mut f32;
            unsafe {
                for i in 0..hidden_size {
                    *ptr.add(i) = 1.0 + rng.next_f32();
                }
            }
        }

        // Logits buffer (shared for CPU readback if needed, but argmax is GPU-side)
        let logits_buf = alloc_buffer_private(dev, vocab_size * f32_size);

        // Argmax buffers
        let num_argmax_groups = vocab_size.div_ceil(256);
        let argmax_partial_vals = alloc_buffer_private(dev, num_argmax_groups * f32_size);
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
            PsoKey::simple("residual_add_inplace"),
            PsoKey::simple("matvec_q6_k"),
            PsoKey::simple("matvec_q8_0"),
        ];
        pso_cache.prewarm(&pso_keys);

        Self {
            device,
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
            embed_scratch,
            final_norm,
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
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            position: 0,
        }
    }

    /// Construct an EagleHead from real EAGLE weights loaded via `EagleWeightStore`.
    ///
    /// Moves weight buffers from the store into EagleHead fields. Scratch buffers
    /// and KV cache are freshly allocated using dimensions from the target model.
    /// Model parameters (rope_theta, rms_norm_eps) are inherited from the target.
    ///
    /// # Arguments
    /// - `weight_store`: Loaded EAGLE weights from SafeTensors.
    /// - `device`: Static reference to the shared GPU device.
    /// - `target`: Target model to derive dimensions and model params from.
    pub fn from_weights(
        weight_store: EagleWeightStore,
        device: &'static GpuDevice,
        target: &GpuForwardPass,
    ) -> Result<Self, String> {
        let dev = &*device.device;

        let hidden_size = target.hidden_size();
        let num_heads = target.num_heads();
        let num_kv_heads = target.num_kv_heads();
        let head_dim = target.head_dim();
        let intermediate_size = target.intermediate_size();
        let vocab_size = target.vocab_size();
        let kv_dim = num_kv_heads * head_dim;

        // --- Move weights from store ---
        let fc_fuse_weight = weight_store.fc_fuse_weight;
        let fc_concat_weight = weight_store.fc_concat_weight;
        let decoder_attn_norm = weight_store.decoder_attn_norm;
        let decoder_attn = weight_store.decoder_attn;
        let decoder_ffn_norm = weight_store.decoder_ffn_norm;
        let decoder_ffn = weight_store.decoder_ffn;

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

        // Embedding scratch buffer (shared, CPU-writable for embed lookup)
        let embed_scratch = alloc_buffer(dev, hidden_size * f32_size);

        // Final norm weight: initialize to 1.0 (identity RMSNorm)
        // Real EAGLE models may not ship a separate final_norm -- use identity until proven otherwise.
        let final_norm = alloc_buffer(dev, hidden_size * f32_size);
        {
            let ptr = final_norm.contents().as_ptr() as *mut f32;
            unsafe {
                for i in 0..hidden_size {
                    *ptr.add(i) = 1.0;
                }
            }
        }

        // Logits buffer (private for GPU argmax)
        let logits_buf = alloc_buffer_private(dev, vocab_size * f32_size);

        // Argmax buffers
        let num_argmax_groups = vocab_size.div_ceil(256);
        let argmax_partial_vals = alloc_buffer_private(dev, num_argmax_groups * f32_size);
        let argmax_partial_idxs =
            alloc_buffer_private(dev, num_argmax_groups * std::mem::size_of::<u32>());
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
            PsoKey::simple("residual_add_inplace"),
            PsoKey::simple("matvec_q6_k"),
            PsoKey::simple("matvec_q8_0"),
        ];
        pso_cache.prewarm(&pso_keys);

        Ok(Self {
            device,
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
            embed_scratch,
            final_norm,
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
            rope_theta: target.rope_theta(),
            rms_norm_eps: target.rms_norm_eps(),
            position: 0,
        })
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

    /// Run the full EAGLE draft head forward pass and return the predicted token ID.
    ///
    /// Pipeline (all in a single command buffer + encoder):
    ///   1. concat_buffers_3: feat_low + feat_mid + feat_high -> fused_buf (3*hidden)
    ///   2. matvec_f32_v2: fc_fuse_weight * fused_buf -> hidden_a (hidden)
    ///   3. CPU embed lookup: copy prev_token embedding -> embed_scratch
    ///   4. concat_buffers_2: hidden_a + embed_scratch -> combined_buf (2*hidden)
    ///   5. matvec_f32_v2: fc_concat_weight * combined_buf -> hidden_b (hidden)
    ///   6. buffer_copy: hidden_b -> hidden_a (set up residual stream)
    ///   7. Decoder layer: rmsnorm -> Q/K/V matvec -> RoPE -> KV append -> attention
    ///      -> O proj + residual -> FFN rmsnorm -> gate/up -> silu -> down + residual
    ///   8. Final norm -> lm_head matvec -> logits
    ///   9. argmax_reduce + argmax_final -> token_id
    ///
    /// The eagle head maintains its own RoPE position, incrementing each call.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_draft_token(
        &mut self,
        feat_low: &ProtocolObject<dyn MTLBuffer>,
        feat_mid: &ProtocolObject<dyn MTLBuffer>,
        feat_high: &ProtocolObject<dyn MTLBuffer>,
        prev_token: u32,
        target_embed_buf: &ProtocolObject<dyn MTLBuffer>,
        target_lm_head: &WeightBuffer,
        _target_lm_head_is_f32: bool,
        target_lm_head_q6k: Option<&WeightBuffer>,
        target_lm_head_q8: Option<&WeightBuffer>,
    ) -> Result<u32, String> {
        // --- CPU embed lookup: copy prev_token embedding to embed_scratch ---
        {
            let offset_bytes =
                (prev_token as usize) * self.hidden_size * std::mem::size_of::<f32>();
            let row_bytes = self.hidden_size * std::mem::size_of::<f32>();
            unsafe {
                let src = (target_embed_buf.contents().as_ptr() as *const u8).add(offset_bytes);
                let dst = self.embed_scratch.contents().as_ptr() as *mut u8;
                std::ptr::copy_nonoverlapping(src, dst, row_bytes);
            }
        }

        // Pre-look up kv_cache_copy PSO before encoding (avoids borrow conflict).
        let kv_copy_pso = self
            .pso_cache
            .get(&PsoKey::simple("kv_cache_copy"))
            .expect("kv_cache_copy PSO not prewarmed");

        // Create single command buffer + encoder for the entire pipeline.
        let cmd_buf = self
            .device
            .command_queue
            .commandBuffer()
            .expect("Failed to create command buffer");
        let encoder = cmd_buf
            .computeCommandEncoder()
            .expect("Failed to create compute encoder");

        // --- Step 1: concat_buffers_3: feat_low + feat_mid + feat_high -> fused_buf ---
        self.encode_concat_3(
            &encoder,
            feat_low,
            feat_mid,
            feat_high,
            &self.fused_buf,
            self.hidden_size,
            self.hidden_size,
            self.hidden_size,
        );

        // --- Step 2: matvec_f32_v2: fc_fuse_weight * fused_buf -> hidden_a ---
        self.encode_matvec_f32(
            &encoder,
            &self.fc_fuse_weight.buffer,
            self.fc_fuse_weight.offset,
            &self.fused_buf,
            &self.hidden_a,
            self.hidden_size,
            3 * self.hidden_size,
        );

        // --- Step 3: concat_buffers_2: hidden_a + embed_scratch -> combined_buf ---
        self.encode_concat_2(
            &encoder,
            &self.hidden_a,
            &self.embed_scratch,
            &self.combined_buf,
            self.hidden_size,
            self.hidden_size,
        );

        // --- Step 4: matvec_f32_v2: fc_concat_weight * combined_buf -> hidden_b ---
        self.encode_matvec_f32(
            &encoder,
            &self.fc_concat_weight.buffer,
            self.fc_concat_weight.offset,
            &self.combined_buf,
            &self.hidden_b,
            self.hidden_size,
            2 * self.hidden_size,
        );

        // --- Step 5: Copy hidden_b -> hidden_a (set up residual stream) ---
        self.encode_buffer_copy(&encoder, &self.hidden_b, &self.hidden_a, self.hidden_size);

        // ==== Decoder layer ====

        // --- Attention: rmsnorm -> Q/K/V matvec -> RoPE -> KV append -> attention -> O proj + residual ---

        // RMSNorm: hidden_a -> hidden_b
        self.encode_rmsnorm(
            &encoder,
            &self.hidden_a,
            &self.decoder_attn_norm,
            &self.hidden_b,
        );

        // Q projection: hidden_b -> scratch_q [hidden_size -> hidden_size]
        self.encode_matvec_f32(
            &encoder,
            &self.decoder_attn.q.buffer,
            self.decoder_attn.q.offset,
            &self.hidden_b,
            &self.scratch_q,
            self.num_heads * self.head_dim,
            self.hidden_size,
        );

        // K projection: hidden_b -> scratch_k [hidden_size -> kv_dim]
        let kv_dim = self.num_kv_heads * self.head_dim;
        self.encode_matvec_f32(
            &encoder,
            &self.decoder_attn.k.buffer,
            self.decoder_attn.k.offset,
            &self.hidden_b,
            &self.scratch_k,
            kv_dim,
            self.hidden_size,
        );

        // V projection: hidden_b -> scratch_v [hidden_size -> kv_dim]
        self.encode_matvec_f32(
            &encoder,
            &self.decoder_attn.v.buffer,
            self.decoder_attn.v.offset,
            &self.hidden_b,
            &self.scratch_v,
            kv_dim,
            self.hidden_size,
        );

        // RoPE on Q and K (dual dispatch)
        self.encode_rope_dual(&encoder, &self.scratch_q, &self.scratch_k);

        // KV cache append (GPU-side)
        self.eagle_kv_cache.encode_kv_append(
            &encoder,
            kv_copy_pso,
            &self.scratch_k,
            &self.scratch_v,
        );

        // Decode attention: Q + KV cache -> scratch_attn_out
        let kv_len = self.eagle_kv_cache.current_len() as u32;
        self.encode_decode_attention(
            &encoder,
            &self.scratch_q,
            self.eagle_kv_cache.k_buffer(),
            self.eagle_kv_cache.v_buffer(),
            &self.scratch_attn_out,
            kv_len,
        );

        // O projection: scratch_attn_out -> hidden_b [hidden_size -> hidden_size]
        self.encode_matvec_f32(
            &encoder,
            &self.decoder_attn.o.buffer,
            self.decoder_attn.o.offset,
            &self.scratch_attn_out,
            &self.hidden_b,
            self.hidden_size,
            self.num_heads * self.head_dim,
        );

        // Residual: hidden_a += hidden_b
        self.encode_residual_add_inplace(&encoder, &self.hidden_a, &self.hidden_b);

        // --- FFN: rmsnorm -> gate/up matvec -> silu -> down + residual ---

        // RMSNorm: hidden_a -> hidden_b
        self.encode_rmsnorm(
            &encoder,
            &self.hidden_a,
            &self.decoder_ffn_norm,
            &self.hidden_b,
        );

        // Gate projection: hidden_b -> scratch_gate [hidden_size -> intermediate_size]
        self.encode_matvec_f32(
            &encoder,
            &self.decoder_ffn.gate.buffer,
            self.decoder_ffn.gate.offset,
            &self.hidden_b,
            &self.scratch_gate,
            self.intermediate_size,
            self.hidden_size,
        );

        // Up projection: hidden_b -> scratch_up [hidden_size -> intermediate_size]
        self.encode_matvec_f32(
            &encoder,
            &self.decoder_ffn.up.buffer,
            self.decoder_ffn.up.offset,
            &self.hidden_b,
            &self.scratch_up,
            self.intermediate_size,
            self.hidden_size,
        );

        // SwiGLU: silu(scratch_gate) * scratch_up -> scratch_silu
        self.encode_ffn_silu(&encoder);

        // Down projection: scratch_silu -> hidden_b [intermediate_size -> hidden_size]
        self.encode_matvec_f32(
            &encoder,
            &self.decoder_ffn.down.buffer,
            self.decoder_ffn.down.offset,
            &self.scratch_silu,
            &self.hidden_b,
            self.hidden_size,
            self.intermediate_size,
        );

        // Residual: hidden_a += hidden_b
        self.encode_residual_add_inplace(&encoder, &self.hidden_a, &self.hidden_b);

        // ==== Final projection ====

        // Final norm: hidden_a -> hidden_b
        self.encode_rmsnorm(&encoder, &self.hidden_a, &self.final_norm, &self.hidden_b);

        // lm_head matvec: hidden_b -> logits_buf
        // Prefer Q6_K > Q8_0 > F32 > fallback F32 for lm_head
        if let Some(wb) = target_lm_head_q6k {
            self.encode_matvec_q6_k(
                &encoder,
                &wb.buffer,
                wb.offset,
                &self.hidden_b,
                &self.logits_buf,
                self.vocab_size,
                self.hidden_size,
            );
        } else if let Some(wb) = target_lm_head_q8 {
            self.encode_matvec_q8_0(
                &encoder,
                &wb.buffer,
                wb.offset,
                &self.hidden_b,
                &self.logits_buf,
                self.vocab_size,
                self.hidden_size,
            );
        } else {
            // F32 lm_head (tied embeddings or explicit F32)
            self.encode_matvec_f32(
                &encoder,
                &target_lm_head.buffer,
                target_lm_head.offset,
                &self.hidden_b,
                &self.logits_buf,
                self.vocab_size,
                self.hidden_size,
            );
        }

        // Argmax: logits_buf -> argmax_result
        self.encode_argmax(&encoder, &self.logits_buf);

        // Submit + wait
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        validate_eagle_cmd_buf(&cmd_buf, "forward_draft_token")?;

        // Read back argmax result (single u32)
        let token_id = unsafe {
            let ptr = self.argmax_result.contents().as_ptr() as *const u32;
            *ptr
        };

        // Increment RoPE position for next draft token
        self.position += 1;

        Ok(token_id)
    }

    // ========================================================================
    // Encode helpers: dispatch Metal compute kernels.
    // Same patterns as GpuForwardPass but operating on EagleHead's own PSO cache
    // and scratch buffers.
    // ========================================================================

    /// Encode RMSNorm: input * weight -> output. Single threadgroup of 32 threads.
    fn encode_rmsnorm(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        input_buf: &ProtocolObject<dyn MTLBuffer>,
        weight_buf: &ProtocolObject<dyn MTLBuffer>,
        output_buf: &ProtocolObject<dyn MTLBuffer>,
    ) {
        let pso = self
            .pso_cache
            .get(&PsoKey::simple("rmsnorm_optimized"))
            .expect("rmsnorm_optimized PSO not prewarmed");

        encoder.setComputePipelineState(pso);
        set_buffer(encoder, input_buf, 0, 0);
        set_buffer(encoder, weight_buf, 0, 1);
        set_buffer(encoder, output_buf, 0, 2);

        let hidden_dim_u32 = self.hidden_size as u32;
        set_bytes(encoder, &hidden_dim_u32, 3);
        set_bytes(encoder, &self.rms_norm_eps, 4);

        let grid = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    }

    /// Encode F32 matvec v2: weight * input -> output. 8 rows/TG, 256 threads.
    #[allow(clippy::too_many_arguments)]
    fn encode_matvec_f32(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        weight_buf: &ProtocolObject<dyn MTLBuffer>,
        weight_offset: usize,
        input_buf: &ProtocolObject<dyn MTLBuffer>,
        output_buf: &ProtocolObject<dyn MTLBuffer>,
        out_dim: usize,
        in_dim: usize,
    ) {
        let pso = self
            .pso_cache
            .get(&PsoKey::simple("matvec_f32_v2"))
            .expect("matvec_f32_v2 PSO not prewarmed");

        encoder.setComputePipelineState(pso);
        set_buffer(encoder, weight_buf, weight_offset, 0);
        set_buffer(encoder, input_buf, 0, 1);
        set_buffer(encoder, output_buf, 0, 2);

        let out_dim_u32 = out_dim as u32;
        let in_dim_u32 = in_dim as u32;
        set_bytes(encoder, &out_dim_u32, 3);
        set_bytes(encoder, &in_dim_u32, 4);

        const ROWS_PER_TG: usize = 8;
        let grid = MTLSize {
            width: out_dim.div_ceil(ROWS_PER_TG),
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    }

    /// Encode Q6_K matvec: weight * input -> output. 256 threads, 8 rows/TG.
    #[allow(clippy::too_many_arguments)]
    fn encode_matvec_q6_k(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        weight_buf: &ProtocolObject<dyn MTLBuffer>,
        weight_offset: usize,
        input_buf: &ProtocolObject<dyn MTLBuffer>,
        output_buf: &ProtocolObject<dyn MTLBuffer>,
        out_dim: usize,
        in_dim: usize,
    ) {
        let pso = self
            .pso_cache
            .get(&PsoKey::simple("matvec_q6_k"))
            .expect("matvec_q6_k PSO not prewarmed");

        encoder.setComputePipelineState(pso);
        set_buffer(encoder, weight_buf, weight_offset, 0);
        set_buffer(encoder, input_buf, 0, 1);
        set_buffer(encoder, output_buf, 0, 2);

        let out_dim_u32 = out_dim as u32;
        let in_dim_u32 = in_dim as u32;
        set_bytes(encoder, &out_dim_u32, 3);
        set_bytes(encoder, &in_dim_u32, 4);

        const ROWS_PER_TG: usize = 8;
        let grid = MTLSize {
            width: out_dim.div_ceil(ROWS_PER_TG),
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    }

    /// Encode Q8_0 matvec: weight * input -> output. 256 threads, 8 rows/TG.
    #[allow(clippy::too_many_arguments)]
    fn encode_matvec_q8_0(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        weight_buf: &ProtocolObject<dyn MTLBuffer>,
        weight_offset: usize,
        input_buf: &ProtocolObject<dyn MTLBuffer>,
        output_buf: &ProtocolObject<dyn MTLBuffer>,
        out_dim: usize,
        in_dim: usize,
    ) {
        let pso = self
            .pso_cache
            .get(&PsoKey::simple("matvec_q8_0"))
            .expect("matvec_q8_0 PSO not prewarmed");

        encoder.setComputePipelineState(pso);
        set_buffer(encoder, weight_buf, weight_offset, 0);
        set_buffer(encoder, input_buf, 0, 1);
        set_buffer(encoder, output_buf, 0, 2);

        let out_dim_u32 = out_dim as u32;
        let in_dim_u32 = in_dim as u32;
        set_bytes(encoder, &out_dim_u32, 3);
        set_bytes(encoder, &in_dim_u32, 4);

        const ROWS_PER_TG: usize = 8;
        let grid = MTLSize {
            width: out_dim.div_ceil(ROWS_PER_TG),
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    }

    /// Encode dual-buffer RoPE: apply RoPE to both Q and K in a single dispatch.
    fn encode_rope_dual(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        q_buf: &ProtocolObject<dyn MTLBuffer>,
        k_buf: &ProtocolObject<dyn MTLBuffer>,
    ) {
        let pso = self
            .pso_cache
            .get(&PsoKey::simple("rope_apply_dual"))
            .expect("rope_apply_dual PSO not prewarmed");

        encoder.setComputePipelineState(pso);
        set_buffer(encoder, q_buf, 0, 0);
        set_buffer(encoder, k_buf, 0, 1);

        let num_q_heads_u32 = self.num_heads as u32;
        let num_k_heads_u32 = self.num_kv_heads as u32;
        let head_dim_u32 = self.head_dim as u32;
        let position_u32 = self.position as u32;
        set_bytes(encoder, &num_q_heads_u32, 2);
        set_bytes(encoder, &num_k_heads_u32, 3);
        set_bytes(encoder, &head_dim_u32, 4);
        set_bytes(encoder, &position_u32, 5);
        set_bytes(encoder, &self.rope_theta, 6);

        let total_pairs = (self.num_heads + self.num_kv_heads) * self.head_dim / 2;
        let grid = MTLSize {
            width: total_pairs,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreads_threadsPerThreadgroup(grid, tg);
    }

    /// Encode decode attention v2: Q + KV cache -> output. 256 threads per head.
    fn encode_decode_attention(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        q_buf: &ProtocolObject<dyn MTLBuffer>,
        k_cache_buf: &ProtocolObject<dyn MTLBuffer>,
        v_cache_buf: &ProtocolObject<dyn MTLBuffer>,
        output_buf: &ProtocolObject<dyn MTLBuffer>,
        kv_len: u32,
    ) {
        let pso = self
            .pso_cache
            .get(&PsoKey::simple("decode_attention_v2"))
            .expect("decode_attention_v2 PSO not prewarmed");

        encoder.setComputePipelineState(pso);
        set_buffer(encoder, q_buf, 0, 0);
        set_buffer(encoder, k_cache_buf, 0, 1);
        set_buffer(encoder, v_cache_buf, 0, 2);
        set_buffer(encoder, output_buf, 0, 3);

        let num_heads_u32 = self.num_heads as u32;
        let num_kv_heads_u32 = self.num_kv_heads as u32;
        let head_dim_u32 = self.head_dim as u32;
        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        set_bytes(encoder, &num_heads_u32, 4);
        set_bytes(encoder, &num_kv_heads_u32, 5);
        set_bytes(encoder, &head_dim_u32, 6);
        set_bytes(encoder, &kv_len, 7);
        set_bytes(encoder, &scale, 8);

        let grid = MTLSize {
            width: self.num_heads,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    }

    /// Encode in-place residual addition: a[i] += b[i].
    fn encode_residual_add_inplace(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a_buf: &ProtocolObject<dyn MTLBuffer>,
        b_buf: &ProtocolObject<dyn MTLBuffer>,
    ) {
        let pso = self
            .pso_cache
            .get(&PsoKey::simple("residual_add_inplace"))
            .expect("residual_add_inplace PSO not prewarmed");

        encoder.setComputePipelineState(pso);
        set_buffer(encoder, a_buf, 0, 0);
        set_buffer(encoder, b_buf, 0, 1);

        let dim_u32 = self.hidden_size as u32;
        set_bytes(encoder, &dim_u32, 2);

        let grid = MTLSize {
            width: self.hidden_size,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreads_threadsPerThreadgroup(grid, tg);
    }

    /// Encode SwiGLU activation: silu(scratch_gate) * scratch_up -> scratch_silu.
    fn encode_ffn_silu(&self, encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>) {
        let pso = self
            .pso_cache
            .get(&PsoKey::simple("ffn_silu"))
            .expect("ffn_silu PSO not prewarmed");

        encoder.setComputePipelineState(pso);

        // ffn_silu kernel signature: input(0), gate(1), up(2), output(3), params(4)
        // input(0) is unused but required in the binding
        set_buffer(encoder, &self.hidden_b, 0, 0); // dummy, unused
        set_buffer(encoder, &self.scratch_gate, 0, 1);
        set_buffer(encoder, &self.scratch_up, 0, 2);
        set_buffer(encoder, &self.scratch_silu, 0, 3);

        let params = LayerParams {
            intermediate_dim: self.intermediate_size as u32,
            seq_len: 1,
            ..Default::default()
        };
        set_bytes(encoder, &params, 4);

        let grid = MTLSize {
            width: self.intermediate_size,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreads_threadsPerThreadgroup(grid, tg);
    }

    /// Encode GPU-side buffer copy: src -> dst for `count` f32 elements.
    fn encode_buffer_copy(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        src: &ProtocolObject<dyn MTLBuffer>,
        dst: &ProtocolObject<dyn MTLBuffer>,
        count: usize,
    ) {
        let pso = self
            .pso_cache
            .get(&PsoKey::simple("buffer_copy"))
            .expect("buffer_copy PSO not prewarmed");

        encoder.setComputePipelineState(pso);
        set_buffer(encoder, src, 0, 0);
        set_buffer(encoder, dst, 0, 1);

        let count_u32 = count as u32;
        set_bytes(encoder, &count_u32, 2);

        let grid = MTLSize {
            width: count,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreads_threadsPerThreadgroup(grid, tg);
    }

    /// Encode concat_buffers_3: a + b + c -> output.
    #[allow(clippy::too_many_arguments)]
    fn encode_concat_3(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &ProtocolObject<dyn MTLBuffer>,
        b: &ProtocolObject<dyn MTLBuffer>,
        c: &ProtocolObject<dyn MTLBuffer>,
        output: &ProtocolObject<dyn MTLBuffer>,
        dim_a: usize,
        dim_b: usize,
        dim_c: usize,
    ) {
        let pso = self
            .pso_cache
            .get(&PsoKey::simple("concat_buffers_3"))
            .expect("concat_buffers_3 PSO not prewarmed");

        encoder.setComputePipelineState(pso);
        set_buffer(encoder, a, 0, 0);
        set_buffer(encoder, b, 0, 1);
        set_buffer(encoder, c, 0, 2);
        set_buffer(encoder, output, 0, 3);

        let dim_a_u32 = dim_a as u32;
        let dim_b_u32 = dim_b as u32;
        let dim_c_u32 = dim_c as u32;
        set_bytes(encoder, &dim_a_u32, 4);
        set_bytes(encoder, &dim_b_u32, 5);
        set_bytes(encoder, &dim_c_u32, 6);

        let total = dim_a + dim_b + dim_c;
        let grid = MTLSize {
            width: total,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreads_threadsPerThreadgroup(grid, tg);
    }

    /// Encode concat_buffers_2: a + b -> output.
    fn encode_concat_2(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &ProtocolObject<dyn MTLBuffer>,
        b: &ProtocolObject<dyn MTLBuffer>,
        output: &ProtocolObject<dyn MTLBuffer>,
        dim_a: usize,
        dim_b: usize,
    ) {
        let pso = self
            .pso_cache
            .get(&PsoKey::simple("concat_buffers_2"))
            .expect("concat_buffers_2 PSO not prewarmed");

        encoder.setComputePipelineState(pso);
        set_buffer(encoder, a, 0, 0);
        set_buffer(encoder, b, 0, 1);
        set_buffer(encoder, output, 0, 2);

        let dim_a_u32 = dim_a as u32;
        let dim_b_u32 = dim_b as u32;
        set_bytes(encoder, &dim_a_u32, 3);
        set_bytes(encoder, &dim_b_u32, 4);

        let total = dim_a + dim_b;
        let grid = MTLSize {
            width: total,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreads_threadsPerThreadgroup(grid, tg);
    }

    /// Encode GPU-side argmax: two-stage parallel reduction on logits buffer.
    fn encode_argmax(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        logits_buf: &ProtocolObject<dyn MTLBuffer>,
    ) {
        let num_groups = self.vocab_size.div_ceil(256 * 4);
        let num_groups_u32 = num_groups as u32;
        let vocab_size_u32 = self.vocab_size as u32;

        // Stage 1: argmax_reduce
        let pso_reduce = self
            .pso_cache
            .get(&PsoKey::simple("argmax_reduce"))
            .expect("argmax_reduce PSO not prewarmed");

        encoder.setComputePipelineState(pso_reduce);
        set_buffer(encoder, logits_buf, 0, 0);
        set_bytes(encoder, &vocab_size_u32, 1);
        set_buffer(encoder, &self.argmax_partial_vals, 0, 2);
        set_buffer(encoder, &self.argmax_partial_idxs, 0, 3);

        let grid = MTLSize {
            width: num_groups,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);

        // Stage 2: argmax_final
        let pso_final = self
            .pso_cache
            .get(&PsoKey::simple("argmax_final"))
            .expect("argmax_final PSO not prewarmed");

        encoder.setComputePipelineState(pso_final);
        set_buffer(encoder, &self.argmax_partial_vals, 0, 0);
        set_buffer(encoder, &self.argmax_partial_idxs, 0, 1);
        set_bytes(encoder, &num_groups_u32, 2);
        set_buffer(encoder, &self.argmax_result, 0, 3);

        let grid_final = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_final, tg);
    }
}

/// Validate that a command buffer completed successfully.
fn validate_eagle_cmd_buf(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    stage: &str,
) -> Result<(), String> {
    let status = cmd_buf.status();
    if status == MTLCommandBufferStatus::Completed {
        Ok(())
    } else {
        let error_desc = cmd_buf
            .error()
            .map(|e| e.localizedDescription().to_string())
            .unwrap_or_else(|| "unknown error".to_string());
        Err(format!(
            "GPU command buffer failed at {stage}: status={status:?}, error={error_desc}",
        ))
    }
}
