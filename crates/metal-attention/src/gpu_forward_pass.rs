//! Full-GPU inference pipeline: single forward pass through all layers.
//!
//! `GpuForwardPass` encodes all transformer operations (RMSNorm, Q4_0 matvec,
//! RoPE, decode attention, SwiGLU FFN, residual add) on the GPU with minimal
//! CPU-GPU synchronization. Only two CPU-GPU transfers per token:
//!   1. Embedding lookup (CPU) -> hidden_a buffer write (2.3KB)
//!   2. Logits buffer readback (192KB) -> CPU argmax
//!
//! Uses a single command buffer and compute encoder for the entire forward pass
//! (all 30 layers). KV cache append and hidden-state ping-pong copies are
//! performed GPU-side via `kv_cache_copy` and `buffer_copy` Metal kernels,
//! eliminating all CPU-GPU sync points from the decode hot path.
//!
//! A separate `forward_token_debug` path retains the multi-command-buffer
//! architecture for per-layer debug readback (activated via GPU_DEBUG=1).

use std::path::Path;
use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLGPUFamily, MTLSize,
};

use metal_attention_gguf::GgufFile;
use metal_attention_kernels::buffer::{alloc_buffer, alloc_buffer_private, read_buffer_slice};
use metal_attention_kernels::device::GpuDevice;
use metal_attention_kernels::dispatch::{set_buffer, set_bytes};
use metal_attention_kernels::pipeline::{PsoCache, PsoKey};
use metal_attention_kernels::types::LayerParams;

use crate::gpu_kv_cache::GpuKVCacheSet;
use crate::gpu_weight_store::GpuWeightStore;

/// Cached batch buffers for `forward_prompt()` to avoid per-call allocation.
struct BatchBuffers {
    hidden_a: Retained<ProtocolObject<dyn MTLBuffer>>,
    hidden_b: Retained<ProtocolObject<dyn MTLBuffer>>,
    q: Retained<ProtocolObject<dyn MTLBuffer>>,
    k: Retained<ProtocolObject<dyn MTLBuffer>>,
    v: Retained<ProtocolObject<dyn MTLBuffer>>,
    attn_out: Retained<ProtocolObject<dyn MTLBuffer>>,
    gate: Retained<ProtocolObject<dyn MTLBuffer>>,
    up: Retained<ProtocolObject<dyn MTLBuffer>>,
    silu: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Maximum batch size these buffers can accommodate.
    max_batch_size: usize,
}

/// Full-GPU forward pass pipeline for autoregressive decode.
///
/// Encodes all transformer operations on the GPU. Scratch buffers are
/// allocated once and reused across `forward_token()` calls.
pub struct GpuForwardPass {
    /// Shared GPU device (device, command queue, shader library).
    device: &'static GpuDevice,
    /// Pipeline state object cache for all kernels.
    pso_cache: PsoCache,
    /// Model weights as Metal buffers (zero-copy from GGUF mmap).
    weight_store: GpuWeightStore,
    /// Per-layer KV caches.
    kv_caches: GpuKVCacheSet,

    // Ping-pong hidden state buffers (hidden_size * 4 bytes each).
    hidden_a: Retained<ProtocolObject<dyn MTLBuffer>>,
    hidden_b: Retained<ProtocolObject<dyn MTLBuffer>>,

    // Scratch buffers for intermediate results (allocated once, reused).
    scratch_q: Retained<ProtocolObject<dyn MTLBuffer>>,
    scratch_k: Retained<ProtocolObject<dyn MTLBuffer>>,
    scratch_v: Retained<ProtocolObject<dyn MTLBuffer>>,
    scratch_attn_out: Retained<ProtocolObject<dyn MTLBuffer>>,
    scratch_o: Retained<ProtocolObject<dyn MTLBuffer>>,
    scratch_gate: Retained<ProtocolObject<dyn MTLBuffer>>,
    scratch_up: Retained<ProtocolObject<dyn MTLBuffer>>,
    scratch_silu: Retained<ProtocolObject<dyn MTLBuffer>>,
    scratch_ffn: Retained<ProtocolObject<dyn MTLBuffer>>,
    scratch_residual: Retained<ProtocolObject<dyn MTLBuffer>>,
    logits_buf: Retained<ProtocolObject<dyn MTLBuffer>>,

    // Argmax buffers (GPU-side argmax to avoid logits readback).
    argmax_partial_vals: Retained<ProtocolObject<dyn MTLBuffer>>,
    argmax_partial_idxs: Retained<ProtocolObject<dyn MTLBuffer>>,
    argmax_result: Retained<ProtocolObject<dyn MTLBuffer>>,

    // Repetition penalty: GPU-side token history buffer (shared storage for CPU writes).
    // Stores token IDs generated so far; the repetition_penalty kernel penalizes
    // these tokens' logits before argmax. Max capacity = max_tokens we'd ever generate.
    token_history_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Number of tokens currently in the history buffer.
    token_history_len: usize,
    /// Repetition penalty factor (1.0 = disabled).
    repetition_penalty: f32,

    // Model dimensions.
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    num_layers: usize,
    intermediate_size: usize,
    vocab_size: usize,
    rope_theta: f32,
    rms_norm_eps: f32,

    /// Current decode position (incremented after each forward_token call).
    position: usize,

    /// Cached batch buffers for forward_prompt (reused across calls).
    batch_bufs: Option<BatchBuffers>,
}

impl GpuForwardPass {
    /// Construct a GpuForwardPass from a GGUF model file.
    ///
    /// Opens the GGUF file, extracts model configuration from metadata,
    /// builds the GpuWeightStore with zero-copy buffers, initializes
    /// KV caches, and prewarms PSOs.
    pub fn from_gguf(path: &Path) -> Result<Self, String> {
        let device = GpuDevice::shared();

        // Verify GPU supports Apple Family 7+ (M1 and later) for simd_sum
        if !device.device.supportsFamily(MTLGPUFamily::Apple7) {
            return Err(
                "GPU does not support Apple Family 7 (M1+). simd_sum requires Apple7 or later."
                    .to_string(),
            );
        }

        // Open GGUF and extract config
        let gguf = Arc::new(GgufFile::open(path).map_err(|e| format!("Failed to open GGUF: {e}"))?);

        let hidden_size = gguf
            .metadata
            .get_u32("llama.embedding_length")
            .unwrap_or(576) as usize;
        let num_heads = gguf
            .metadata
            .get_u32("llama.attention.head_count")
            .unwrap_or(9) as usize;
        let head_dim = if num_heads > 0 {
            hidden_size / num_heads
        } else {
            hidden_size
        };
        let num_kv_heads = gguf
            .metadata
            .get_u32("llama.attention.head_count_kv")
            .unwrap_or(num_heads as u32) as usize;
        let num_layers = gguf.metadata.get_u32("llama.block_count").unwrap_or(30) as usize;
        let intermediate_size = gguf
            .metadata
            .get_u32("llama.feed_forward_length")
            .unwrap_or(1536) as usize;

        // Vocab size: try metadata, fall back to embed tensor shape
        let vocab_size = if let Some(v) = gguf.metadata.get_u32("llama.vocab_size") {
            v as usize
        } else if let Some(embed_info) = gguf.find_tensor("token_embd.weight") {
            embed_info.shape[0] as usize
        } else {
            49152 // SmolLM-135M default
        };

        let rope_theta = gguf
            .metadata
            .get_f32("llama.rope.freq_base")
            .unwrap_or(10000.0);
        let rms_norm_eps = gguf
            .metadata
            .get_f32("llama.attention.layer_norm_rms_epsilon")
            .unwrap_or(1e-5);

        eprintln!(
            "GpuForwardPass: {}L {}H {}D (kv_heads={}, ffn={}, vocab={}, rope_theta={}, eps={})",
            num_layers,
            num_heads,
            hidden_size,
            num_kv_heads,
            intermediate_size,
            vocab_size,
            rope_theta,
            rms_norm_eps
        );

        // Build ModelConfig for GpuWeightStore
        let config = metal_attention_models::registry::ModelConfig {
            architecture: gguf.architecture,
            hidden_size,
            head_dim,
            num_heads,
            num_kv_heads,
            num_layers,
        };

        // Build weight store
        let weight_store = GpuWeightStore::from_gguf(gguf, &config, &device.device)?;

        // KV caches: one per layer
        let kv_dim = num_kv_heads * head_dim;
        let kv_caches = GpuKVCacheSet::new(&device.device, num_layers, 2048, kv_dim);

        // Check GPU_DEBUG: when set, keep all buffers Shared for CPU readback.
        // When not set, use StorageModePrivate for GPU-only scratch buffers.
        let debug = std::env::var("GPU_DEBUG").is_ok();

        // Allocate ping-pong hidden state buffers
        // hidden_a MUST stay Shared: embed_lookup() uses contents() for CPU memcpy
        let hidden_bytes = hidden_size * std::mem::size_of::<f32>();
        let hidden_a = alloc_buffer(&device.device, hidden_bytes);
        // hidden_b: only GPU reads/writes (rmsnorm output, matvec input)
        let hidden_b = if debug {
            alloc_buffer(&device.device, hidden_bytes)
        } else {
            alloc_buffer_private(&device.device, hidden_bytes)
        };

        // Allocate scratch buffers (reused across all layers and tokens)
        // In non-debug mode, use Private for GPU-only buffers (no CPU readback)
        let q_bytes = num_heads * head_dim * std::mem::size_of::<f32>();
        let kv_bytes = kv_dim * std::mem::size_of::<f32>();
        let ffn_bytes = intermediate_size * std::mem::size_of::<f32>();
        let logits_bytes = vocab_size * std::mem::size_of::<f32>();

        // Helper closure: alloc_buffer (Shared) when debug, alloc_buffer_private otherwise
        let alloc_scratch = |size: usize| -> Retained<ProtocolObject<dyn MTLBuffer>> {
            if debug {
                alloc_buffer(&device.device, size)
            } else {
                alloc_buffer_private(&device.device, size)
            }
        };

        let scratch_q = alloc_scratch(q_bytes);
        let scratch_k = alloc_scratch(kv_bytes);
        let scratch_v = alloc_scratch(kv_bytes);
        let scratch_attn_out = alloc_scratch(q_bytes);
        let scratch_o = alloc_scratch(hidden_bytes);
        let scratch_gate = alloc_scratch(ffn_bytes);
        let scratch_up = alloc_scratch(ffn_bytes);
        let scratch_silu = alloc_scratch(ffn_bytes);
        let scratch_ffn = alloc_scratch(hidden_bytes);
        let scratch_residual = alloc_scratch(hidden_bytes);
        // logits_buf stays Shared: forward_token() reads it back via read_buffer_slice()
        let logits_buf = alloc_buffer(&device.device, logits_bytes);

        // Argmax buffers: 48 threadgroups for vocab=49152 (ceil(49152 / (256*4)))
        let num_argmax_groups = vocab_size.div_ceil(256 * 4);
        let argmax_partial_vals = alloc_buffer_private(
            &device.device,
            num_argmax_groups * std::mem::size_of::<f32>(),
        );
        let argmax_partial_idxs = alloc_buffer_private(
            &device.device,
            num_argmax_groups * std::mem::size_of::<u32>(),
        );
        // Result buffer is Shared so CPU can read back the token id
        let argmax_result = alloc_buffer(&device.device, std::mem::size_of::<u32>());

        // Token history buffer for repetition penalty (shared storage for CPU writes).
        // Pre-allocate for up to 4096 tokens; grows if needed.
        let token_history_buf = alloc_buffer(&device.device, 4096 * std::mem::size_of::<u32>());

        // Build PSO cache and prewarm all kernels
        let mut pso_cache = PsoCache::new(device.library.clone());
        let mut pso_keys = vec![
            PsoKey::simple("matvec_q4_0_v5_coalesced"),
            PsoKey::simple("matvec_q4_0_batched"),
            PsoKey::simple("matvec_q8_0"),
            PsoKey::simple("matvec_q4_0_accumulate"),
            PsoKey::simple("silu_matvec_q4_0_accumulate"),
            PsoKey::simple("rmsnorm_optimized"),
            PsoKey::simple("residual_add"),
            PsoKey::simple("residual_add_inplace"),
            PsoKey::simple("decode_attention"),
            PsoKey::simple("decode_attention_v2"),
            PsoKey::simple("rope_apply"),
            PsoKey::simple("rope_apply_dual"),
            PsoKey::simple("ffn_silu"),
            PsoKey::simple("kv_cache_copy"),
            PsoKey::simple("buffer_copy"),
            PsoKey::simple("argmax_reduce"),
            PsoKey::simple("argmax_final"),
            PsoKey::simple("repetition_penalty"),
            PsoKey::simple("rmsnorm_matvec_q4_0"),
            PsoKey::simple("multi_token_matvec_q4_0"),
            PsoKey::simple("multi_token_matvec_q4_0_accumulate"),
        ];
        if weight_store.lm_head_is_f32() {
            pso_keys.push(PsoKey::simple("matvec_f32_v2"));
            pso_keys.push(PsoKey::simple("rmsnorm_matvec_f32"));
        }
        pso_cache.prewarm(&pso_keys);

        Ok(Self {
            device,
            pso_cache,
            weight_store,
            kv_caches,
            hidden_a,
            hidden_b,
            scratch_q,
            scratch_k,
            scratch_v,
            scratch_attn_out,
            scratch_o,
            scratch_gate,
            scratch_up,
            scratch_silu,
            scratch_ffn,
            scratch_residual,
            logits_buf,
            argmax_partial_vals,
            argmax_partial_idxs,
            argmax_result,
            token_history_buf,
            token_history_len: 0,
            repetition_penalty: 1.0,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            num_layers,
            intermediate_size,
            vocab_size,
            rope_theta,
            rms_norm_eps,
            position: 0,
            batch_bufs: None,
        })
    }

    /// Run a single-token forward pass through all layers.
    ///
    /// Returns logits [vocab_size] as `Result<Vec<f32>, String>`.
    ///
    /// Uses a single command buffer with one compute encoder for all layers.
    /// GPU-side kv_cache_copy and buffer_copy kernels eliminate CPU sync points.
    /// When `GPU_DEBUG` is set, falls back to the multi-command-buffer debug path
    /// that allows per-layer readback.
    pub fn forward_token(&mut self, token_id: u32) -> Result<Vec<f32>, String> {
        let debug = std::env::var("GPU_DEBUG").is_ok();

        // Validate token_id is within vocab range
        if token_id as usize >= self.vocab_size {
            return Err(format!(
                "token_id {} out of range (vocab_size={})",
                token_id, self.vocab_size
            ));
        }

        // 1. CPU embedding lookup -> write to hidden_a via contents() memcpy
        self.embed_lookup(token_id);

        if debug {
            let h = unsafe { read_buffer_slice(&self.hidden_a, self.hidden_size) };
            let (has_nan, min, max) = buf_stats(&h);
            eprintln!("  [embed] hidden_a: nan={has_nan} min={min:.6} max={max:.6}");
            return self.forward_token_debug(token_id);
        }

        // Pre-look up kv_cache_copy PSO before the layer loop to avoid
        // borrow conflicts between pso_cache and kv_caches.
        let kv_copy_pso = self
            .pso_cache
            .get(&PsoKey::simple("kv_cache_copy"))
            .expect("kv_cache_copy PSO not prewarmed");

        // 2. Create single command buffer + encoder for all layers
        let cmd_buf = self
            .device
            .command_queue
            .commandBuffer()
            .expect("Failed to create command buffer");
        let encoder = cmd_buf
            .computeCommandEncoder()
            .expect("Failed to create compute encoder");

        // 3. Per-layer forward (all encoded into the single encoder)
        for layer_idx in 0..self.num_layers {
            // --- Attention projections: rmsnorm -> Q/K/V matvec + RoPE ---
            let norms = self.weight_store.norm(layer_idx);
            let attn = self.weight_store.attn_proj(layer_idx);

            // RMSNorm: hidden_a -> hidden_b (compute once, reuse for Q/K/V)
            self.encode_rmsnorm(&encoder, &self.hidden_a, &norms.attn_norm, &self.hidden_b);

            // Batched Q/K/V projection: hidden_b -> scratch_q, scratch_k, scratch_v
            // Single dispatch: 576+192+192 = 960 rows (was 3 separate dispatches)
            self.encode_batched_matvec_q4_0(
                &encoder,
                &attn.q,
                &attn.k,
                &attn.v,
                &self.hidden_b,
                &self.scratch_q,
                &self.scratch_k,
                &self.scratch_v,
                self.num_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
                self.hidden_size,
            );

            // RoPE on Q and K (single dual-buffer dispatch)
            self.encode_rope_dual(&encoder, &self.scratch_q, &self.scratch_k);

            // --- GPU-side KV cache append ---
            // Dispatches kv_cache_copy kernel, increments cache len on CPU.
            // Field-level borrow: &mut self.kv_caches is disjoint from
            // &self.scratch_k / &self.scratch_v.
            self.kv_caches.cache_mut(layer_idx).encode_kv_append(
                &encoder,
                kv_copy_pso,
                &self.scratch_k,
                &self.scratch_v,
            );

            // --- Decode attention + O projection + residual ---
            let kv_cache = self.kv_caches.cache(layer_idx);
            let kv_len = kv_cache.current_len() as u32;

            self.encode_decode_attention(
                &encoder,
                &self.scratch_q,
                kv_cache.k_buffer(),
                kv_cache.v_buffer(),
                &self.scratch_attn_out,
                kv_len,
            );

            // Re-fetch attn weights for O projection
            let attn = self.weight_store.attn_proj(layer_idx);

            // O projection + residual (fused): hidden_a += O_proj(attn_out)
            self.encode_matvec_q4_0_accumulate(
                &encoder,
                &attn.o,
                &self.scratch_attn_out,
                &self.hidden_a,
                self.hidden_size,
                self.num_heads * self.head_dim,
            );

            // --- FFN: rmsnorm -> gate/up matvec + fused SiLU+down+residual ---
            let norms = self.weight_store.norm(layer_idx);
            let ffn = self.weight_store.ffn(layer_idx);

            // RMSNorm: hidden_a -> hidden_b (compute once, reuse for gate/up)
            self.encode_rmsnorm(&encoder, &self.hidden_a, &norms.ffn_norm, &self.hidden_b);

            // Batched gate/up projection: hidden_b -> scratch_gate, scratch_up
            self.encode_batched_matvec_q4_0_2(
                &encoder,
                &ffn.gate,
                &ffn.up,
                &self.hidden_b,
                &self.scratch_gate,
                &self.scratch_up,
                self.intermediate_size,
                self.intermediate_size,
                self.hidden_size,
            );

            // SwiGLU: silu(gate) * up -> scratch_silu
            self.encode_ffn_silu(&encoder);

            // Down projection + residual (fused): hidden_a += down(scratch_silu)
            self.encode_matvec_q4_0_accumulate(
                &encoder,
                &ffn.down,
                &self.scratch_silu,
                &self.hidden_a,
                self.hidden_size,
                self.intermediate_size,
            );
        }

        // 4. Final norm + lm_head
        self.encode_rmsnorm(
            &encoder,
            &self.hidden_a,
            self.weight_store.final_norm(),
            &self.hidden_b,
        );

        // Prefer Q8_0 lm_head (halves bandwidth vs F32 tied embeddings)
        if let Some(q8_buf) = self.weight_store.lm_head_q8() {
            self.encode_matvec_q8_0(
                &encoder,
                q8_buf,
                &self.hidden_b,
                &self.logits_buf,
                self.vocab_size,
                self.hidden_size,
            );
        } else if self.weight_store.lm_head_is_f32() {
            self.encode_matvec_f32(
                &encoder,
                self.weight_store.lm_head(),
                &self.hidden_b,
                &self.logits_buf,
                self.vocab_size,
                self.hidden_size,
            );
        } else {
            self.encode_matvec_q4_0(
                &encoder,
                self.weight_store.lm_head(),
                &self.hidden_b,
                &self.logits_buf,
                self.vocab_size,
                self.hidden_size,
            );
        }

        // 5. Single submit: endEncoding + commit + wait
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        validate_command_buffer(&cmd_buf, "forward_token", 0)?;

        // 6. Read logits and increment position
        let logits = unsafe { read_buffer_slice(&self.logits_buf, self.vocab_size) };
        self.position += 1;
        Ok(logits)
    }

    /// Multi-token batched forward pass for prompt prefill.
    ///
    /// Processes all tokens through each layer before moving to the next layer,
    /// allowing weight matrices to stay warm in SLC cache across tokens.
    /// Matvec operations are batched (one dispatch for all tokens), while cheap
    /// per-token ops (RMSNorm, RoPE, SiLU, attention) loop individually.
    ///
    /// Returns the greedy argmax token ID for the last token in the sequence.
    /// KV cache is populated for all tokens. Position advances by token_ids.len().
    pub fn forward_prompt(&mut self, token_ids: &[u32]) -> Result<u32, String> {
        let batch_size = token_ids.len();
        if batch_size == 0 {
            return Err("forward_prompt: empty token_ids".to_string());
        }
        // Fallback to single-token path for batch=1
        if batch_size == 1 {
            return self.forward_token_greedy(token_ids[0]);
        }

        // Validate all tokens
        for &tid in token_ids {
            if tid as usize >= self.vocab_size {
                return Err(format!(
                    "token_id {} out of range (vocab_size={})",
                    tid, self.vocab_size
                ));
            }
        }

        let h = self.hidden_size;
        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;
        let ffn_dim = self.intermediate_size;
        let f32_sz = std::mem::size_of::<f32>();

        // Allocate or reuse batch buffers
        let need_alloc = match &self.batch_bufs {
            Some(bufs) => bufs.max_batch_size < batch_size,
            None => true,
        };
        if need_alloc {
            // hidden_a is Shared (CPU writes embeddings into it), rest are Private
            self.batch_bufs = Some(BatchBuffers {
                hidden_a: alloc_buffer(&self.device.device, batch_size * h * f32_sz),
                hidden_b: alloc_buffer_private(&self.device.device, batch_size * h * f32_sz),
                q: alloc_buffer_private(&self.device.device, batch_size * q_dim * f32_sz),
                k: alloc_buffer_private(&self.device.device, batch_size * kv_dim * f32_sz),
                v: alloc_buffer_private(&self.device.device, batch_size * kv_dim * f32_sz),
                attn_out: alloc_buffer_private(&self.device.device, batch_size * q_dim * f32_sz),
                gate: alloc_buffer_private(&self.device.device, batch_size * ffn_dim * f32_sz),
                up: alloc_buffer_private(&self.device.device, batch_size * ffn_dim * f32_sz),
                silu: alloc_buffer_private(&self.device.device, batch_size * ffn_dim * f32_sz),
                max_batch_size: batch_size,
            });
        }
        let bb = self.batch_bufs.as_ref().unwrap();

        // CPU embedding lookup into bb.hidden_a
        let embed_buf = self.weight_store.embed();
        for (tok, &tid) in token_ids.iter().enumerate() {
            let embed_offset = (tid as usize) * h * f32_sz;
            let dst_offset = tok * h * f32_sz;
            unsafe {
                let src = (embed_buf.contents().as_ptr() as *const u8).add(embed_offset);
                let dst = (bb.hidden_a.contents().as_ptr() as *mut u8).add(dst_offset);
                std::ptr::copy_nonoverlapping(src, dst, h * f32_sz);
            }
        }

        // Pre-lookup PSOs for per-token inline dispatches
        let kv_copy_pso = self
            .pso_cache
            .get(&PsoKey::simple("kv_cache_copy"))
            .expect("kv_cache_copy PSO not prewarmed");
        let rmsnorm_pso = self
            .pso_cache
            .get(&PsoKey::simple("rmsnorm_optimized"))
            .expect("rmsnorm_optimized PSO not prewarmed");
        let rope_pso = self
            .pso_cache
            .get(&PsoKey::simple("rope_apply"))
            .expect("rope_apply PSO not prewarmed");
        let attn_pso = self
            .pso_cache
            .get(&PsoKey::simple("decode_attention_v2"))
            .expect("decode_attention_v2 PSO not prewarmed");
        let silu_pso = self
            .pso_cache
            .get(&PsoKey::simple("ffn_silu"))
            .expect("ffn_silu PSO not prewarmed");

        // Single command buffer + encoder for entire prefill
        let cmd_buf = self
            .device
            .command_queue
            .commandBuffer()
            .expect("Failed to create command buffer");
        let encoder = cmd_buf
            .computeCommandEncoder()
            .expect("Failed to create compute encoder");

        let initial_pos = self.position;
        let hidden_dim_u32 = h as u32;
        let num_heads_u32 = self.num_heads as u32;
        let num_kv_heads_u32 = self.num_kv_heads as u32;
        let head_dim_u32 = self.head_dim as u32;
        let scale = 1.0f32 / (self.head_dim as f32).sqrt();

        let grid_1 = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        let tg_32 = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        let tg_256 = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };

        for layer_idx in 0..self.num_layers {
            let norms = self.weight_store.norm(layer_idx);
            let attn = self.weight_store.attn_proj(layer_idx);

            // ---- Per-token attn RMSNorm: bb.hidden_a[tok] -> bb.hidden_b[tok] ----
            self.encode_rmsnorm_batched(
                &encoder,
                &bb.hidden_a,
                &norms.attn_norm,
                &bb.hidden_b,
                h,
                batch_size,
                rmsnorm_pso,
            );

            // ---- Batched QKV matvec (weight cache reuse) ----
            self.encode_multi_token_matvec_q4_0(
                &encoder,
                &attn.q,
                &bb.hidden_b,
                &bb.q,
                q_dim,
                h,
                batch_size,
            );
            self.encode_multi_token_matvec_q4_0(
                &encoder,
                &attn.k,
                &bb.hidden_b,
                &bb.k,
                kv_dim,
                h,
                batch_size,
            );
            self.encode_multi_token_matvec_q4_0(
                &encoder,
                &attn.v,
                &bb.hidden_b,
                &bb.v,
                kv_dim,
                h,
                batch_size,
            );

            // ---- Per-token RoPE on Q and K ----
            let q_pairs = self.num_heads * self.head_dim / 2;
            let k_pairs = self.num_kv_heads * self.head_dim / 2;
            for tok in 0..batch_size {
                let pos = (initial_pos + tok) as u32;

                // RoPE on Q
                encoder.setComputePipelineState(rope_pso);
                set_buffer(&encoder, &bb.q, tok * q_dim * f32_sz, 0);
                set_bytes(&encoder, &num_heads_u32, 1);
                set_bytes(&encoder, &head_dim_u32, 2);
                set_bytes(&encoder, &pos, 3);
                set_bytes(&encoder, &self.rope_theta, 4);
                encoder.dispatchThreads_threadsPerThreadgroup(
                    MTLSize {
                        width: q_pairs,
                        height: 1,
                        depth: 1,
                    },
                    tg_32,
                );

                // RoPE on K
                encoder.setComputePipelineState(rope_pso);
                set_buffer(&encoder, &bb.k, tok * kv_dim * f32_sz, 0);
                set_bytes(&encoder, &num_kv_heads_u32, 1);
                set_bytes(&encoder, &head_dim_u32, 2);
                set_bytes(&encoder, &pos, 3);
                set_bytes(&encoder, &self.rope_theta, 4);
                encoder.dispatchThreads_threadsPerThreadgroup(
                    MTLSize {
                        width: k_pairs,
                        height: 1,
                        depth: 1,
                    },
                    tg_32,
                );
            }

            // ---- Per-token KV cache append + decode attention ----
            for tok in 0..batch_size {
                let kv_offset = tok * kv_dim * f32_sz;
                let q_offset = tok * q_dim * f32_sz;
                let attn_offset = tok * q_dim * f32_sz;

                // KV cache append (increments cache len)
                self.kv_caches.cache_mut(layer_idx).encode_kv_append_offset(
                    &encoder,
                    kv_copy_pso,
                    &bb.k,
                    &bb.v,
                    kv_offset,
                );

                let kv_cache = self.kv_caches.cache(layer_idx);
                let kv_len = kv_cache.current_len() as u32;

                // Decode attention: Q[tok] + KV cache -> attn_out[tok]
                encoder.setComputePipelineState(attn_pso);
                set_buffer(&encoder, &bb.q, q_offset, 0);
                set_buffer(&encoder, kv_cache.k_buffer(), 0, 1);
                set_buffer(&encoder, kv_cache.v_buffer(), 0, 2);
                set_buffer(&encoder, &bb.attn_out, attn_offset, 3);
                set_bytes(&encoder, &num_heads_u32, 4);
                set_bytes(&encoder, &num_kv_heads_u32, 5);
                set_bytes(&encoder, &head_dim_u32, 6);
                set_bytes(&encoder, &kv_len, 7);
                set_bytes(&encoder, &scale, 8);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: self.num_heads,
                        height: 1,
                        depth: 1,
                    },
                    tg_256,
                );
            }

            // ---- Batched O-projection + residual accumulate ----
            let attn = self.weight_store.attn_proj(layer_idx);
            self.encode_multi_token_matvec_q4_0_accumulate(
                &encoder,
                &attn.o,
                &bb.attn_out,
                &bb.hidden_a,
                h,
                q_dim,
                batch_size,
            );

            // ---- Per-token FFN RMSNorm: bb.hidden_a[tok] -> bb.hidden_b[tok] ----
            let norms = self.weight_store.norm(layer_idx);
            let ffn = self.weight_store.ffn(layer_idx);
            self.encode_rmsnorm_batched(
                &encoder,
                &bb.hidden_a,
                &norms.ffn_norm,
                &bb.hidden_b,
                h,
                batch_size,
                rmsnorm_pso,
            );

            // ---- Batched gate/up matvec ----
            self.encode_multi_token_matvec_q4_0(
                &encoder,
                &ffn.gate,
                &bb.hidden_b,
                &bb.gate,
                ffn_dim,
                h,
                batch_size,
            );
            self.encode_multi_token_matvec_q4_0(
                &encoder,
                &ffn.up,
                &bb.hidden_b,
                &bb.up,
                ffn_dim,
                h,
                batch_size,
            );

            // ---- Per-token SiLU: silu(gate[tok]) * up[tok] -> silu[tok] ----
            self.encode_silu_batched(
                &encoder, &bb.gate, &bb.up, &bb.silu, ffn_dim, batch_size, silu_pso,
            );

            // ---- Batched down-projection + residual accumulate ----
            self.encode_multi_token_matvec_q4_0_accumulate(
                &encoder,
                &ffn.down,
                &bb.silu,
                &bb.hidden_a,
                h,
                ffn_dim,
                batch_size,
            );
        }

        // ---- Final: RMSNorm + lm_head for LAST token only ----
        let last_offset = (batch_size - 1) * h * f32_sz;

        // RMSNorm on last token's hidden state -> hidden_b (single token)
        encoder.setComputePipelineState(rmsnorm_pso);
        set_buffer(&encoder, &bb.hidden_a, last_offset, 0);
        set_buffer(&encoder, self.weight_store.final_norm(), 0, 1);
        set_buffer(&encoder, &self.hidden_b, 0, 2);
        set_bytes(&encoder, &hidden_dim_u32, 3);
        set_bytes(&encoder, &self.rms_norm_eps, 4);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_1, tg_32);

        // lm_head: hidden_b -> logits_buf (single token, reuse existing buffers)
        if let Some(q8_buf) = self.weight_store.lm_head_q8() {
            self.encode_matvec_q8_0(
                &encoder,
                q8_buf,
                &self.hidden_b,
                &self.logits_buf,
                self.vocab_size,
                h,
            );
        } else if self.weight_store.lm_head_is_f32() {
            self.encode_matvec_f32(
                &encoder,
                self.weight_store.lm_head(),
                &self.hidden_b,
                &self.logits_buf,
                self.vocab_size,
                h,
            );
        } else {
            self.encode_matvec_q4_0(
                &encoder,
                self.weight_store.lm_head(),
                &self.hidden_b,
                &self.logits_buf,
                self.vocab_size,
                h,
            );
        }

        // GPU-side argmax
        self.encode_argmax(&encoder, &self.logits_buf);

        // Submit
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        validate_command_buffer(&cmd_buf, "forward_prompt", 0)?;

        // Read argmax result and advance position
        let result = unsafe { read_buffer_slice::<u32>(&self.argmax_result, 1) };
        self.position += batch_size;
        Ok(result[0])
    }

    /// Debug forward pass: multi-command-buffer path with per-layer readback.
    ///
    /// Uses the original encode_attention_projections / encode_attention_output /
    /// encode_ffn_block methods with commit+wait after each, allowing CPU-side
    /// buffer readback for NaN/range diagnostics. Called when `GPU_DEBUG` is set.
    ///
    /// Assumes embed_lookup has already been called and debug embed log printed.
    fn forward_token_debug(&mut self, _token_id: u32) -> Result<Vec<f32>, String> {
        // Per-layer forward with debug readbacks
        for layer_idx in 0..self.num_layers {
            // --- Attention block ---
            self.encode_attention_projections(layer_idx)?;

            if layer_idx < 2 {
                let q =
                    unsafe { read_buffer_slice(&self.scratch_q, self.num_heads * self.head_dim) };
                let k = unsafe {
                    read_buffer_slice(&self.scratch_k, self.num_kv_heads * self.head_dim)
                };
                let (qn, qmin, qmax) = buf_stats(&q);
                let (kn, kmin, kmax) = buf_stats(&k);
                eprintln!("  [L{layer_idx} attn_proj] Q: nan={qn} min={qmin:.4} max={qmax:.4}  K: nan={kn} min={kmin:.4} max={kmax:.4}");
            }

            // CPU-side KV cache append (requires GPU work completed)
            self.kv_caches
                .cache_mut(layer_idx)
                .append_kv(&self.scratch_k, &self.scratch_v);

            // Decode attention + O projection + residual add
            self.encode_attention_output(layer_idx)?;

            if layer_idx < 2 {
                let h = unsafe { read_buffer_slice(&self.hidden_a, self.hidden_size) };
                let (hn, hmin, hmax) = buf_stats(&h);
                eprintln!(
                    "  [L{layer_idx} attn_out] hidden_a: nan={hn} min={hmin:.4} max={hmax:.4}"
                );
            }

            // --- FFN block ---
            self.encode_ffn_block(layer_idx)?;

            if layer_idx < 2 {
                let h = unsafe { read_buffer_slice(&self.hidden_a, self.hidden_size) };
                let (hn, hmin, hmax) = buf_stats(&h);
                eprintln!("  [L{layer_idx} ffn] hidden_a: nan={hn} min={hmin:.4} max={hmax:.4}");
            }
        }

        // Final norm + lm_head
        self.encode_final_logits()?;

        // Read back logits and increment position
        let logits = unsafe { read_buffer_slice(&self.logits_buf, self.vocab_size) };

        let (ln, lmin, lmax) = buf_stats(&logits);
        eprintln!("  [logits] nan={ln} min={lmin:.4} max={lmax:.4}");

        self.position += 1;
        Ok(logits)
    }

    /// CPU embedding lookup: write embedding vector to hidden_a.
    fn embed_lookup(&self, token_id: u32) {
        let embed_buf = self.weight_store.embed();
        let offset_bytes = (token_id as usize) * self.hidden_size * std::mem::size_of::<f32>();
        let row_bytes = self.hidden_size * std::mem::size_of::<f32>();

        unsafe {
            let src = (embed_buf.contents().as_ptr() as *const u8).add(offset_bytes);
            let dst = self.hidden_a.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(src, dst, row_bytes);
        }
    }

    /// Encode attention projections: rmsnorm -> Q/K/V matvec -> RoPE.
    ///
    /// After this, scratch_q has RoPE'd Q, scratch_k has RoPE'd K,
    /// scratch_v has V. Commits and waits so KV cache append can happen.
    fn encode_attention_projections(&self, layer_idx: usize) -> Result<(), String> {
        let cmd_buf = self
            .device
            .command_queue
            .commandBuffer()
            .expect("Failed to create command buffer");
        let encoder = cmd_buf
            .computeCommandEncoder()
            .expect("Failed to create compute encoder");

        let norms = self.weight_store.norm(layer_idx);
        let attn = self.weight_store.attn_proj(layer_idx);

        // RMSNorm: hidden_a -> hidden_b
        self.encode_rmsnorm(&encoder, &self.hidden_a, &norms.attn_norm, &self.hidden_b);

        // Q projection: hidden_b -> scratch_q
        self.encode_matvec_q4_0(
            &encoder,
            &attn.q,
            &self.hidden_b,
            &self.scratch_q,
            self.num_heads * self.head_dim,
            self.hidden_size,
        );

        // K projection: hidden_b -> scratch_k
        self.encode_matvec_q4_0(
            &encoder,
            &attn.k,
            &self.hidden_b,
            &self.scratch_k,
            self.num_kv_heads * self.head_dim,
            self.hidden_size,
        );

        // V projection: hidden_b -> scratch_v
        self.encode_matvec_q4_0(
            &encoder,
            &attn.v,
            &self.hidden_b,
            &self.scratch_v,
            self.num_kv_heads * self.head_dim,
            self.hidden_size,
        );

        // RoPE on Q
        self.encode_rope(&encoder, &self.scratch_q, self.num_heads);

        // RoPE on K
        self.encode_rope(&encoder, &self.scratch_k, self.num_kv_heads);

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        validate_command_buffer(&cmd_buf, "attention_projections", layer_idx)
    }

    /// Encode decode attention + O projection + residual add.
    fn encode_attention_output(&self, layer_idx: usize) -> Result<(), String> {
        let cmd_buf = self
            .device
            .command_queue
            .commandBuffer()
            .expect("Failed to create command buffer");
        let encoder = cmd_buf
            .computeCommandEncoder()
            .expect("Failed to create compute encoder");

        let attn = self.weight_store.attn_proj(layer_idx);
        let kv_cache = self.kv_caches.cache(layer_idx);
        let kv_len = kv_cache.current_len() as u32;

        // Decode attention: scratch_q + kv_cache -> scratch_attn_out
        self.encode_decode_attention(
            &encoder,
            &self.scratch_q,
            kv_cache.k_buffer(),
            kv_cache.v_buffer(),
            &self.scratch_attn_out,
            kv_len,
        );

        // O projection: scratch_attn_out -> scratch_o
        self.encode_matvec_q4_0(
            &encoder,
            &attn.o,
            &self.scratch_attn_out,
            &self.scratch_o,
            self.hidden_size,
            self.num_heads * self.head_dim,
        );

        // Residual add: hidden_a + scratch_o -> scratch_residual
        self.encode_residual_add(
            &encoder,
            &self.hidden_a,
            &self.scratch_o,
            &self.scratch_residual,
        );

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        validate_command_buffer(&cmd_buf, "attention_output", layer_idx)?;

        // Copy scratch_residual -> hidden_a (both shared-mode, CPU memcpy)
        copy_buffer(&self.scratch_residual, &self.hidden_a, self.hidden_size * 4);
        Ok(())
    }

    /// Encode FFN block: rmsnorm -> gate/up matvec -> SwiGLU -> down matvec -> residual add.
    fn encode_ffn_block(&self, layer_idx: usize) -> Result<(), String> {
        let cmd_buf = self
            .device
            .command_queue
            .commandBuffer()
            .expect("Failed to create command buffer");
        let encoder = cmd_buf
            .computeCommandEncoder()
            .expect("Failed to create compute encoder");

        let norms = self.weight_store.norm(layer_idx);
        let ffn = self.weight_store.ffn(layer_idx);

        // RMSNorm: hidden_a -> hidden_b
        self.encode_rmsnorm(&encoder, &self.hidden_a, &norms.ffn_norm, &self.hidden_b);

        // Gate projection: hidden_b -> scratch_gate
        self.encode_matvec_q4_0(
            &encoder,
            &ffn.gate,
            &self.hidden_b,
            &self.scratch_gate,
            self.intermediate_size,
            self.hidden_size,
        );

        // Up projection: hidden_b -> scratch_up
        self.encode_matvec_q4_0(
            &encoder,
            &ffn.up,
            &self.hidden_b,
            &self.scratch_up,
            self.intermediate_size,
            self.hidden_size,
        );

        // SwiGLU: silu(gate) * up -> scratch_silu
        self.encode_ffn_silu(&encoder);

        // Down projection: scratch_silu -> scratch_ffn
        self.encode_matvec_q4_0(
            &encoder,
            &ffn.down,
            &self.scratch_silu,
            &self.scratch_ffn,
            self.hidden_size,
            self.intermediate_size,
        );

        // Residual add: hidden_a + scratch_ffn -> scratch_residual
        self.encode_residual_add(
            &encoder,
            &self.hidden_a,
            &self.scratch_ffn,
            &self.scratch_residual,
        );

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        validate_command_buffer(&cmd_buf, "ffn_block", layer_idx)?;

        // Copy scratch_residual -> hidden_a
        copy_buffer(&self.scratch_residual, &self.hidden_a, self.hidden_size * 4);
        Ok(())
    }

    /// Encode final RMSNorm + lm_head matvec to produce logits.
    fn encode_final_logits(&self) -> Result<(), String> {
        let cmd_buf = self
            .device
            .command_queue
            .commandBuffer()
            .expect("Failed to create command buffer");
        let encoder = cmd_buf
            .computeCommandEncoder()
            .expect("Failed to create compute encoder");

        // Final RMSNorm: hidden_a -> hidden_b
        self.encode_rmsnorm(
            &encoder,
            &self.hidden_a,
            self.weight_store.final_norm(),
            &self.hidden_b,
        );

        // LM head matvec: hidden_b -> logits_buf [vocab_size]
        if self.weight_store.lm_head_is_f32() {
            // Tied embeddings: F32 weights -> use F32 matvec kernel
            self.encode_matvec_f32(
                &encoder,
                self.weight_store.lm_head(),
                &self.hidden_b,
                &self.logits_buf,
                self.vocab_size,
                self.hidden_size,
            );
        } else {
            // Separate output.weight: Q4_0 -> use Q4_0 matvec kernel
            self.encode_matvec_q4_0(
                &encoder,
                self.weight_store.lm_head(),
                &self.hidden_b,
                &self.logits_buf,
                self.vocab_size,
                self.hidden_size,
            );
        }

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        validate_command_buffer(&cmd_buf, "final_logits", 0)
    }

    // -----------------------------------------------------------------------
    // Inline encoder helpers: encode kernel dispatch into an existing encoder
    // -----------------------------------------------------------------------

    /// Encode RMSNorm optimized: input -> output using weight.
    /// Dispatch: grid=(1,1,1), threadgroup=(32,1,1).
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

    /// Encode v5 coalesced Q4_0 dequant + matvec: weight * input -> output.
    /// Uses multi-row dispatch: 8 rows per threadgroup, 256 threads (8 simdgroups).
    /// Dispatch: grid=(ceil(out_dim/8)) threadgroups, threadgroup=(256).
    fn encode_matvec_q4_0(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        weight_buf: &ProtocolObject<dyn MTLBuffer>,
        input_buf: &ProtocolObject<dyn MTLBuffer>,
        output_buf: &ProtocolObject<dyn MTLBuffer>,
        out_dim: usize,
        in_dim: usize,
    ) {
        let pso = self
            .pso_cache
            .get(&PsoKey::simple("matvec_q4_0_v5_coalesced"))
            .expect("matvec_q4_0_v5_coalesced PSO not prewarmed");

        encoder.setComputePipelineState(pso);
        set_buffer(encoder, weight_buf, 0, 0);
        set_buffer(encoder, input_buf, 0, 1);
        set_buffer(encoder, output_buf, 0, 2);

        let out_dim_u32 = out_dim as u32;
        let in_dim_u32 = in_dim as u32;
        set_bytes(encoder, &out_dim_u32, 3);
        set_bytes(encoder, &in_dim_u32, 4);

        const ROWS_PER_TG: usize = 8;
        let grid = MTLSize {
            width: (out_dim + ROWS_PER_TG - 1) / ROWS_PER_TG,
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

    /// Encode batched Q4_0 matvec: up to 3 projections in a single dispatch.
    /// Merges Q/K/V or gate/up into one dispatch, reducing dispatch overhead.
    /// total_rows = dim_a + dim_b + dim_c, dispatched as one grid.
    #[allow(clippy::too_many_arguments)]
    fn encode_batched_matvec_q4_0(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        weight_a: &ProtocolObject<dyn MTLBuffer>,
        weight_b: &ProtocolObject<dyn MTLBuffer>,
        weight_c: &ProtocolObject<dyn MTLBuffer>,
        input_buf: &ProtocolObject<dyn MTLBuffer>,
        output_a: &ProtocolObject<dyn MTLBuffer>,
        output_b: &ProtocolObject<dyn MTLBuffer>,
        output_c: &ProtocolObject<dyn MTLBuffer>,
        dim_a: usize,
        dim_b: usize,
        dim_c: usize,
        in_dim: usize,
    ) {
        let pso = self
            .pso_cache
            .get(&PsoKey::simple("matvec_q4_0_batched"))
            .expect("matvec_q4_0_batched PSO not prewarmed");

        encoder.setComputePipelineState(pso);
        set_buffer(encoder, weight_a, 0, 0);
        set_buffer(encoder, weight_b, 0, 1);
        set_buffer(encoder, weight_c, 0, 2);
        set_buffer(encoder, input_buf, 0, 3);
        set_buffer(encoder, output_a, 0, 4);
        set_buffer(encoder, output_b, 0, 5);
        set_buffer(encoder, output_c, 0, 6);

        let dims: [u32; 4] = [dim_a as u32, dim_b as u32, dim_c as u32, in_dim as u32];
        set_bytes(encoder, &dims, 7);

        let total_rows = dim_a + dim_b + dim_c;
        const ROWS_PER_TG: usize = 8;
        let grid = MTLSize {
            width: (total_rows + ROWS_PER_TG - 1) / ROWS_PER_TG,
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

    /// Encode batched Q4_0 matvec for 2 projections (dim_c=0).
    #[allow(clippy::too_many_arguments)]
    fn encode_batched_matvec_q4_0_2(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        weight_a: &ProtocolObject<dyn MTLBuffer>,
        weight_b: &ProtocolObject<dyn MTLBuffer>,
        input_buf: &ProtocolObject<dyn MTLBuffer>,
        output_a: &ProtocolObject<dyn MTLBuffer>,
        output_b: &ProtocolObject<dyn MTLBuffer>,
        dim_a: usize,
        dim_b: usize,
        in_dim: usize,
    ) {
        // Use the same batched kernel with weight_c=weight_a (ignored since dim_c=0)
        // and output_c=output_a (ignored since dim_c=0)
        self.encode_batched_matvec_q4_0(
            encoder, weight_a, weight_b, weight_a, // dummy, unused (dim_c=0)
            input_buf, output_a, output_b, output_a, // dummy, unused (dim_c=0)
            dim_a, dim_b, 0, in_dim,
        );
    }

    /// Encode v2 multi-row F32 matvec: weight * input -> output.
    /// Uses 8 rows per threadgroup, 256 threads (8 simdgroups), float4 vectorized reads.
    /// Dispatch: grid=(ceil(out_dim/8)) threadgroups, threadgroup=(256).
    fn encode_matvec_f32(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        weight_buf: &ProtocolObject<dyn MTLBuffer>,
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
        set_buffer(encoder, weight_buf, 0, 0);
        set_buffer(encoder, input_buf, 0, 1);
        set_buffer(encoder, output_buf, 0, 2);

        let out_dim_u32 = out_dim as u32;
        let in_dim_u32 = in_dim as u32;
        set_bytes(encoder, &out_dim_u32, 3);
        set_bytes(encoder, &in_dim_u32, 4);

        const ROWS_PER_TG: usize = 8;
        let grid = MTLSize {
            width: (out_dim + ROWS_PER_TG - 1) / ROWS_PER_TG,
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

    /// Encode Q8_0 matvec: weight * input -> output.
    /// Same dispatch geometry as Q4_0 v5: 256 threads, 8 rows/TG.
    fn encode_matvec_q8_0(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        weight_buf: &ProtocolObject<dyn MTLBuffer>,
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
        set_buffer(encoder, weight_buf, 0, 0);
        set_buffer(encoder, input_buf, 0, 1);
        set_buffer(encoder, output_buf, 0, 2);

        let out_dim_u32 = out_dim as u32;
        let in_dim_u32 = in_dim as u32;
        set_bytes(encoder, &out_dim_u32, 3);
        set_bytes(encoder, &in_dim_u32, 4);

        const ROWS_PER_TG: usize = 8;
        let grid = MTLSize {
            width: (out_dim + ROWS_PER_TG - 1) / ROWS_PER_TG,
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

    /// Encode fused RMSNorm + Q4_0 Matvec: rmsnorm(input) * dequant(weight) -> output.
    ///
    /// Eliminates the intermediate normalized buffer by computing RMS inline.
    /// Each threadgroup computes one output row: Phase 1 cooperatively computes
    /// inv_rms via simd_sum, Phase 2 applies norm + dequant + dot product.
    ///
    /// Buffer bindings: input(0), norm_weight(1), weight_q4_0(2), output(3),
    /// out_dim(4), in_dim(5), eps(6).
    /// Dispatch: grid=(out_dim, 1, 1) threadgroups, threadgroup=(32, 1, 1).
    #[allow(clippy::too_many_arguments)]
    /// Encode Q4_0 matvec with accumulate: output[row] += dot(weight_row, input).
    /// Fuses matvec + residual_add_inplace into a single dispatch.
    fn encode_matvec_q4_0_accumulate(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        weight_buf: &ProtocolObject<dyn MTLBuffer>,
        input_buf: &ProtocolObject<dyn MTLBuffer>,
        output_buf: &ProtocolObject<dyn MTLBuffer>,
        out_dim: usize,
        in_dim: usize,
    ) {
        let pso = self
            .pso_cache
            .get(&PsoKey::simple("matvec_q4_0_accumulate"))
            .expect("matvec_q4_0_accumulate PSO not prewarmed");

        encoder.setComputePipelineState(pso);
        set_buffer(encoder, weight_buf, 0, 0);
        set_buffer(encoder, input_buf, 0, 1);
        set_buffer(encoder, output_buf, 0, 2);

        let out_dim_u32 = out_dim as u32;
        let in_dim_u32 = in_dim as u32;
        set_bytes(encoder, &out_dim_u32, 3);
        set_bytes(encoder, &in_dim_u32, 4);

        const ROWS_PER_TG: usize = 8;
        let grid = MTLSize {
            width: (out_dim + ROWS_PER_TG - 1) / ROWS_PER_TG,
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

    /// Encode multi-token Q4_0 matvec: weight * input[batch] -> output[batch].
    /// Processes batch_size tokens against the same weight matrix in one dispatch.
    /// 256 threads, 8 rows/TG. Output layout: [batch_size, out_dim].
    #[allow(clippy::too_many_arguments)]
    fn encode_multi_token_matvec_q4_0(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        weight_buf: &ProtocolObject<dyn MTLBuffer>,
        input_buf: &ProtocolObject<dyn MTLBuffer>,
        output_buf: &ProtocolObject<dyn MTLBuffer>,
        out_dim: usize,
        in_dim: usize,
        batch_size: usize,
    ) {
        let pso = self
            .pso_cache
            .get(&PsoKey::simple("multi_token_matvec_q4_0"))
            .expect("multi_token_matvec_q4_0 PSO not prewarmed");

        encoder.setComputePipelineState(pso);
        set_buffer(encoder, weight_buf, 0, 0);
        set_buffer(encoder, input_buf, 0, 1);
        set_buffer(encoder, output_buf, 0, 2);

        let out_dim_u32 = out_dim as u32;
        let in_dim_u32 = in_dim as u32;
        let batch_size_u32 = batch_size as u32;
        set_bytes(encoder, &out_dim_u32, 3);
        set_bytes(encoder, &in_dim_u32, 4);
        set_bytes(encoder, &batch_size_u32, 5);

        const ROWS_PER_TG: usize = 8;
        let grid = MTLSize {
            width: (out_dim + ROWS_PER_TG - 1) / ROWS_PER_TG,
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

    /// Encode multi-token Q4_0 matvec with accumulate: output[batch] += W * input[batch].
    /// Same as above but adds to output instead of overwriting.
    #[allow(clippy::too_many_arguments)]
    fn encode_multi_token_matvec_q4_0_accumulate(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        weight_buf: &ProtocolObject<dyn MTLBuffer>,
        input_buf: &ProtocolObject<dyn MTLBuffer>,
        output_buf: &ProtocolObject<dyn MTLBuffer>,
        out_dim: usize,
        in_dim: usize,
        batch_size: usize,
    ) {
        let pso = self
            .pso_cache
            .get(&PsoKey::simple("multi_token_matvec_q4_0_accumulate"))
            .expect("multi_token_matvec_q4_0_accumulate PSO not prewarmed");

        encoder.setComputePipelineState(pso);
        set_buffer(encoder, weight_buf, 0, 0);
        set_buffer(encoder, input_buf, 0, 1);
        set_buffer(encoder, output_buf, 0, 2);

        let out_dim_u32 = out_dim as u32;
        let in_dim_u32 = in_dim as u32;
        let batch_size_u32 = batch_size as u32;
        set_bytes(encoder, &out_dim_u32, 3);
        set_bytes(encoder, &in_dim_u32, 4);
        set_bytes(encoder, &batch_size_u32, 5);

        const ROWS_PER_TG: usize = 8;
        let grid = MTLSize {
            width: (out_dim + ROWS_PER_TG - 1) / ROWS_PER_TG,
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

    /// Encode per-token RMSNorm across a batch using buffer offsets.
    ///
    /// For each token in `0..batch_size`, dispatches rmsnorm_optimized with
    /// input/output buffers offset by `tok * dim * sizeof(f32)`.
    fn encode_rmsnorm_batched(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        input_buf: &ProtocolObject<dyn MTLBuffer>,
        norm_weight: &ProtocolObject<dyn MTLBuffer>,
        output_buf: &ProtocolObject<dyn MTLBuffer>,
        dim: usize,
        batch_size: usize,
        rmsnorm_pso: &ProtocolObject<dyn MTLComputePipelineState>,
    ) {
        let f32_sz = std::mem::size_of::<f32>();
        let dim_u32 = dim as u32;
        let grid_1 = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        let tg_32 = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        for tok in 0..batch_size {
            let offset = tok * dim * f32_sz;
            encoder.setComputePipelineState(rmsnorm_pso);
            set_buffer(encoder, input_buf, offset, 0);
            set_buffer(encoder, norm_weight, 0, 1);
            set_buffer(encoder, output_buf, offset, 2);
            set_bytes(encoder, &dim_u32, 3);
            set_bytes(encoder, &self.rms_norm_eps, 4);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_1, tg_32);
        }
    }

    /// Encode per-token SiLU across a batch using buffer offsets.
    ///
    /// For each token in `0..batch_size`, dispatches ffn_silu with
    /// gate/up/output buffers offset by `tok * ffn_dim * sizeof(f32)`.
    fn encode_silu_batched(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        batch_gate: &ProtocolObject<dyn MTLBuffer>,
        batch_up: &ProtocolObject<dyn MTLBuffer>,
        batch_silu: &ProtocolObject<dyn MTLBuffer>,
        ffn_dim: usize,
        batch_size: usize,
        silu_pso: &ProtocolObject<dyn MTLComputePipelineState>,
    ) {
        let f32_sz = std::mem::size_of::<f32>();
        let params = LayerParams {
            intermediate_dim: ffn_dim as u32,
            seq_len: 1,
            ..Default::default()
        };
        let tg_256 = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        for tok in 0..batch_size {
            let ffn_offset = tok * ffn_dim * f32_sz;
            encoder.setComputePipelineState(silu_pso);
            set_buffer(encoder, &self.hidden_b, 0, 0); // dummy, unused by kernel
            set_buffer(encoder, batch_gate, ffn_offset, 1);
            set_buffer(encoder, batch_up, ffn_offset, 2);
            set_buffer(encoder, batch_silu, ffn_offset, 3);
            set_bytes(encoder, &params, 4);
            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: ffn_dim,
                    height: 1,
                    depth: 1,
                },
                tg_256,
            );
        }
    }

    /// Fused SiLU + Q4_0 down-projection matvec with accumulate.
    /// output[row] += dot(weight_row, silu(gate) * up)
    /// Saves 2 dispatches per layer (SiLU + residual_add_inplace).
    fn encode_silu_matvec_q4_0_accumulate(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        weight_buf: &ProtocolObject<dyn MTLBuffer>,
        gate_buf: &ProtocolObject<dyn MTLBuffer>,
        up_buf: &ProtocolObject<dyn MTLBuffer>,
        output_buf: &ProtocolObject<dyn MTLBuffer>,
        out_dim: usize,
        in_dim: usize,
    ) {
        let pso = self
            .pso_cache
            .get(&PsoKey::simple("silu_matvec_q4_0_accumulate"))
            .expect("silu_matvec_q4_0_accumulate PSO not prewarmed");

        encoder.setComputePipelineState(pso);
        set_buffer(encoder, weight_buf, 0, 0);
        set_buffer(encoder, gate_buf, 0, 1);
        set_buffer(encoder, up_buf, 0, 2);
        set_buffer(encoder, output_buf, 0, 3);

        let out_dim_u32 = out_dim as u32;
        let in_dim_u32 = in_dim as u32;
        set_bytes(encoder, &out_dim_u32, 4);
        set_bytes(encoder, &in_dim_u32, 5);

        const ROWS_PER_TG: usize = 8;
        let grid = MTLSize {
            width: (out_dim + ROWS_PER_TG - 1) / ROWS_PER_TG,
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

    fn encode_fused_rmsnorm_matvec_q4_0(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        input_buf: &ProtocolObject<dyn MTLBuffer>,
        norm_weight_buf: &ProtocolObject<dyn MTLBuffer>,
        weight_q4_0_buf: &ProtocolObject<dyn MTLBuffer>,
        output_buf: &ProtocolObject<dyn MTLBuffer>,
        out_dim: usize,
        in_dim: usize,
    ) {
        let pso = self
            .pso_cache
            .get(&PsoKey::simple("rmsnorm_matvec_q4_0"))
            .expect("rmsnorm_matvec_q4_0 PSO not prewarmed");

        encoder.setComputePipelineState(pso);
        set_buffer(encoder, input_buf, 0, 0);
        set_buffer(encoder, norm_weight_buf, 0, 1);
        set_buffer(encoder, weight_q4_0_buf, 0, 2);
        set_buffer(encoder, output_buf, 0, 3);

        let out_dim_u32 = out_dim as u32;
        let in_dim_u32 = in_dim as u32;
        set_bytes(encoder, &out_dim_u32, 4);
        set_bytes(encoder, &in_dim_u32, 5);
        set_bytes(encoder, &self.rms_norm_eps, 6);

        let grid = MTLSize {
            width: out_dim,
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

    /// Encode fused RMSNorm + F32 Matvec: rmsnorm(input) * weight -> output.
    ///
    /// Same as Q4_0 variant but for dense F32 weights (tied embeddings).
    ///
    /// Buffer bindings: input(0), norm_weight(1), weight_f32(2), output(3),
    /// out_dim(4), in_dim(5), eps(6).
    /// Dispatch: grid=(out_dim, 1, 1) threadgroups, threadgroup=(32, 1, 1).
    #[allow(clippy::too_many_arguments, dead_code)]
    fn encode_fused_rmsnorm_matvec_f32(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        input_buf: &ProtocolObject<dyn MTLBuffer>,
        norm_weight_buf: &ProtocolObject<dyn MTLBuffer>,
        weight_f32_buf: &ProtocolObject<dyn MTLBuffer>,
        output_buf: &ProtocolObject<dyn MTLBuffer>,
        out_dim: usize,
        in_dim: usize,
    ) {
        let pso = self
            .pso_cache
            .get(&PsoKey::simple("rmsnorm_matvec_f32"))
            .expect("rmsnorm_matvec_f32 PSO not prewarmed");

        encoder.setComputePipelineState(pso);
        set_buffer(encoder, input_buf, 0, 0);
        set_buffer(encoder, norm_weight_buf, 0, 1);
        set_buffer(encoder, weight_f32_buf, 0, 2);
        set_buffer(encoder, output_buf, 0, 3);

        let out_dim_u32 = out_dim as u32;
        let in_dim_u32 = in_dim as u32;
        set_bytes(encoder, &out_dim_u32, 4);
        set_bytes(encoder, &in_dim_u32, 5);
        set_bytes(encoder, &self.rms_norm_eps, 6);

        let grid = MTLSize {
            width: out_dim,
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

    /// Encode RoPE position encoding in-place on qk buffer.
    /// Dispatch: grid=(num_heads * head_dim / 2), threadgroup=(32).
    fn encode_rope(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        qk_buf: &ProtocolObject<dyn MTLBuffer>,
        num_heads: usize,
    ) {
        let pso = self
            .pso_cache
            .get(&PsoKey::simple("rope_apply"))
            .expect("rope_apply PSO not prewarmed");

        encoder.setComputePipelineState(pso);
        set_buffer(encoder, qk_buf, 0, 0);

        let num_heads_u32 = num_heads as u32;
        let head_dim_u32 = self.head_dim as u32;
        let position_u32 = self.position as u32;
        set_bytes(encoder, &num_heads_u32, 1);
        set_bytes(encoder, &head_dim_u32, 2);
        set_bytes(encoder, &position_u32, 3);
        set_bytes(encoder, &self.rope_theta, 4);

        let total_pairs = num_heads * self.head_dim / 2;
        let grid = MTLSize {
            width: total_pairs,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreads_threadsPerThreadgroup(grid, tg);
    }

    /// Encode dual-buffer RoPE: apply RoPE to both Q and K in a single dispatch.
    /// Saves one dispatch per layer vs. two separate rope_apply calls.
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

    /// Encode decode attention v2: Q + KV cache -> output.
    /// Uses 256 threads (8 simdgroups) per head for better occupancy.
    /// Dispatch: grid=(num_heads) threadgroups, threadgroup=(256).
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

    /// Encode element-wise residual addition: a + b -> output.
    /// Dispatch: grid=(dim), threadgroup=(256).
    fn encode_residual_add(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a_buf: &ProtocolObject<dyn MTLBuffer>,
        b_buf: &ProtocolObject<dyn MTLBuffer>,
        output_buf: &ProtocolObject<dyn MTLBuffer>,
    ) {
        let pso = self
            .pso_cache
            .get(&PsoKey::simple("residual_add"))
            .expect("residual_add PSO not prewarmed");

        encoder.setComputePipelineState(pso);
        set_buffer(encoder, a_buf, 0, 0);
        set_buffer(encoder, b_buf, 0, 1);
        set_buffer(encoder, output_buf, 0, 2);

        let dim_u32 = self.hidden_size as u32;
        set_bytes(encoder, &dim_u32, 3);

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

    /// Encode in-place residual addition: a[i] += b[i].
    /// Eliminates the separate output buffer + buffer_copy dispatch.
    /// Dispatch: grid=(dim), threadgroup=(256).
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
    /// Uses LayerParams struct at buffer(4) for ffn_silu kernel.
    /// Dispatch: grid=(intermediate_size), threadgroup=(256).
    fn encode_ffn_silu(&self, encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>) {
        let pso = self
            .pso_cache
            .get(&PsoKey::simple("ffn_silu"))
            .expect("ffn_silu PSO not prewarmed");

        encoder.setComputePipelineState(pso);

        // ffn_silu kernel signature: input(0), gate(1), up(2), output(3), params(4)
        // input(0) is unused by the kernel but required in the binding
        set_buffer(encoder, &self.hidden_b, 0, 0); // dummy, unused
        set_buffer(encoder, &self.scratch_gate, 0, 1);
        set_buffer(encoder, &self.scratch_up, 0, 2);
        set_buffer(encoder, &self.scratch_silu, 0, 3);

        // LayerParams: only seq_len and intermediate_dim are used by ffn_silu
        let params = LayerParams {
            intermediate_dim: self.intermediate_size as u32,
            seq_len: 1, // single token
            ..Default::default()
        };
        set_bytes(encoder, &params, 4);

        let total = self.intermediate_size;
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

    /// Encode GPU-side buffer copy: src -> dst for `count` f32 elements.
    /// Dispatch: grid=(count, 1, 1), threadgroup=(256, 1, 1).
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

    /// Encode GPU-side repetition penalty on logits buffer.
    ///
    /// For each token in the history, penalizes the corresponding logit:
    /// positive logits are divided by penalty, negative logits are multiplied.
    /// No-op if penalty == 1.0 or history is empty.
    fn encode_repetition_penalty(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        logits_buf: &ProtocolObject<dyn MTLBuffer>,
    ) {
        if self.repetition_penalty == 1.0 || self.token_history_len == 0 {
            return;
        }

        let pso = self
            .pso_cache
            .get(&PsoKey::simple("repetition_penalty"))
            .expect("repetition_penalty PSO not prewarmed");

        let num_tokens = self.token_history_len as u32;
        let penalty = self.repetition_penalty;

        encoder.setComputePipelineState(pso);
        set_buffer(encoder, logits_buf, 0, 0);
        set_buffer(encoder, &self.token_history_buf, 0, 1);
        set_bytes(encoder, &num_tokens, 2);
        set_bytes(encoder, &penalty, 3);

        let grid = MTLSize {
            width: self.token_history_len,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 256.min(self.token_history_len),
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreads_threadsPerThreadgroup(grid, tg);
    }

    /// Encode GPU-side argmax: two-stage parallel reduction on logits buffer.
    ///
    /// Stage 1 (`argmax_reduce`): Each threadgroup reduces a chunk of logits
    /// to a single (max_val, max_idx) pair. Dispatches `num_groups` threadgroups
    /// of 256 threads each.
    ///
    /// Stage 2 (`argmax_final`): Single threadgroup of 256 threads reduces the
    /// partial results to the global argmax. Result written to `argmax_result`
    /// buffer (Shared, CPU-readable).
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

    /// Run a single-token forward pass and return the greedy argmax token ID.
    ///
    /// Identical to `forward_token()` but appends a GPU-side argmax reduction
    /// after the lm_head matvec, avoiding the 192KB logits readback. Only the
    /// 4-byte `argmax_result` buffer (a single `u32` token_id) is read back.
    pub fn forward_token_greedy(&mut self, token_id: u32) -> Result<u32, String> {
        // Validate token_id is within vocab range
        if token_id as usize >= self.vocab_size {
            return Err(format!(
                "token_id {} out of range (vocab_size={})",
                token_id, self.vocab_size
            ));
        }

        // 1. CPU embedding lookup -> write to hidden_a via contents() memcpy
        self.embed_lookup(token_id);

        // Pre-look up kv_cache_copy PSO before the layer loop
        let kv_copy_pso = self
            .pso_cache
            .get(&PsoKey::simple("kv_cache_copy"))
            .expect("kv_cache_copy PSO not prewarmed");

        // 2. Create single command buffer + encoder for all layers + argmax
        let cmd_buf = self
            .device
            .command_queue
            .commandBuffer()
            .expect("Failed to create command buffer");
        let encoder = cmd_buf
            .computeCommandEncoder()
            .expect("Failed to create compute encoder");

        // 3. Per-layer forward (all encoded into the single encoder)
        for layer_idx in 0..self.num_layers {
            // --- Attention: rmsnorm -> Q/K/V matvec + RoPE ---
            let norms = self.weight_store.norm(layer_idx);
            let attn = self.weight_store.attn_proj(layer_idx);

            // RMSNorm: hidden_a -> hidden_b (compute once, reuse for Q/K/V)
            self.encode_rmsnorm(&encoder, &self.hidden_a, &norms.attn_norm, &self.hidden_b);

            // Batched Q/K/V projection: hidden_b -> scratch_q, scratch_k, scratch_v
            self.encode_batched_matvec_q4_0(
                &encoder,
                &attn.q,
                &attn.k,
                &attn.v,
                &self.hidden_b,
                &self.scratch_q,
                &self.scratch_k,
                &self.scratch_v,
                self.num_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
                self.hidden_size,
            );

            self.encode_rope_dual(&encoder, &self.scratch_q, &self.scratch_k);

            self.kv_caches.cache_mut(layer_idx).encode_kv_append(
                &encoder,
                kv_copy_pso,
                &self.scratch_k,
                &self.scratch_v,
            );

            let kv_cache = self.kv_caches.cache(layer_idx);
            let kv_len = kv_cache.current_len() as u32;

            self.encode_decode_attention(
                &encoder,
                &self.scratch_q,
                kv_cache.k_buffer(),
                kv_cache.v_buffer(),
                &self.scratch_attn_out,
                kv_len,
            );

            // O projection + residual (fused): hidden_a += O_proj(attn_out)
            let attn = self.weight_store.attn_proj(layer_idx);

            self.encode_matvec_q4_0_accumulate(
                &encoder,
                &attn.o,
                &self.scratch_attn_out,
                &self.hidden_a,
                self.hidden_size,
                self.num_heads * self.head_dim,
            );

            // --- FFN: rmsnorm -> gate/up matvec + fused SiLU+down+residual ---
            let norms = self.weight_store.norm(layer_idx);
            let ffn = self.weight_store.ffn(layer_idx);

            // RMSNorm: hidden_a -> hidden_b (compute once, reuse for gate/up)
            self.encode_rmsnorm(&encoder, &self.hidden_a, &norms.ffn_norm, &self.hidden_b);

            // Batched gate/up projection: hidden_b -> scratch_gate, scratch_up
            self.encode_batched_matvec_q4_0_2(
                &encoder,
                &ffn.gate,
                &ffn.up,
                &self.hidden_b,
                &self.scratch_gate,
                &self.scratch_up,
                self.intermediate_size,
                self.intermediate_size,
                self.hidden_size,
            );

            // SwiGLU: silu(gate) * up -> scratch_silu
            self.encode_ffn_silu(&encoder);

            // Down projection + residual (fused): hidden_a += down(scratch_silu)
            self.encode_matvec_q4_0_accumulate(
                &encoder,
                &ffn.down,
                &self.scratch_silu,
                &self.hidden_a,
                self.hidden_size,
                self.intermediate_size,
            );
        }

        // 4. Final norm + lm_head
        self.encode_rmsnorm(
            &encoder,
            &self.hidden_a,
            self.weight_store.final_norm(),
            &self.hidden_b,
        );

        // Prefer Q8_0 lm_head (halves bandwidth vs F32 tied embeddings)
        if let Some(q8_buf) = self.weight_store.lm_head_q8() {
            self.encode_matvec_q8_0(
                &encoder,
                q8_buf,
                &self.hidden_b,
                &self.logits_buf,
                self.vocab_size,
                self.hidden_size,
            );
        } else if self.weight_store.lm_head_is_f32() {
            self.encode_matvec_f32(
                &encoder,
                self.weight_store.lm_head(),
                &self.hidden_b,
                &self.logits_buf,
                self.vocab_size,
                self.hidden_size,
            );
        } else {
            self.encode_matvec_q4_0(
                &encoder,
                self.weight_store.lm_head(),
                &self.hidden_b,
                &self.logits_buf,
                self.vocab_size,
                self.hidden_size,
            );
        }

        // 5. GPU-side repetition penalty (modifies logits in-place before argmax)
        self.encode_repetition_penalty(&encoder, &self.logits_buf);

        // 6. GPU-side argmax on logits (avoids 192KB readback)
        self.encode_argmax(&encoder, &self.logits_buf);

        // 7. Single submit: endEncoding + commit + wait
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        validate_command_buffer(&cmd_buf, "forward_token_greedy", 0)?;

        // 8. Read back only the 4-byte argmax result
        let result = unsafe { read_buffer_slice::<u32>(&self.argmax_result, 1) };
        let token_id = result[0];

        // 9. Append to token history (CPU write to shared buffer — ~4 bytes)
        self.append_token_history(token_id);

        self.position += 1;
        Ok(token_id)
    }

    /// Get the current decode position.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Reset position and KV caches (for new generation).
    pub fn reset(&mut self) {
        self.position = 0;
        self.token_history_len = 0;
        self.kv_caches.reset();
    }

    /// Set the repetition penalty factor (1.0 = disabled).
    pub fn set_repetition_penalty(&mut self, penalty: f32) {
        self.repetition_penalty = penalty;
    }

    /// Append a token ID to the history buffer (CPU write, ~4 bytes).
    fn append_token_history(&mut self, token_id: u32) {
        let capacity = self.token_history_buf.length() / std::mem::size_of::<u32>();
        if self.token_history_len >= capacity {
            // Reallocate with 2x capacity
            let new_cap = capacity * 2;
            let new_buf = alloc_buffer(&self.device.device, new_cap * std::mem::size_of::<u32>());
            // Copy old data
            unsafe {
                let src = self.token_history_buf.contents().cast::<u32>().as_ptr();
                let dst = new_buf.contents().cast::<u32>().as_ptr();
                std::ptr::copy_nonoverlapping(src, dst, self.token_history_len);
            }
            self.token_history_buf = new_buf;
        }

        unsafe {
            let ptr = self.token_history_buf.contents().cast::<u32>().as_ptr();
            *ptr.add(self.token_history_len) = token_id;
        }
        self.token_history_len += 1;
    }

    /// Get the vocab size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get the hidden size.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

/// Debug helper: compute NaN presence, min, max for a buffer.
fn buf_stats(data: &[f32]) -> (bool, f32, f32) {
    let has_nan = data.iter().any(|v| v.is_nan() || v.is_infinite());
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    (has_nan, min, max)
}

/// Validate that a command buffer completed successfully after waitUntilCompleted.
///
/// Returns `Err` with a descriptive message if the command buffer failed.
fn validate_command_buffer(
    cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
    stage: &str,
    layer_idx: usize,
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
            "GPU command buffer failed at {stage} (layer {layer_idx}): status={:?}, error={error_desc}",
            status
        ))
    }
}

/// CPU-side buffer copy between shared-mode Metal buffers.
fn copy_buffer(
    src: &ProtocolObject<dyn MTLBuffer>,
    dst: &ProtocolObject<dyn MTLBuffer>,
    bytes: usize,
) {
    unsafe {
        let src_ptr = src.contents().as_ptr() as *const u8;
        let dst_ptr = dst.contents().as_ptr() as *mut u8;
        std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, bytes);
    }
}
