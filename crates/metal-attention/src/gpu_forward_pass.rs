//! Full-GPU inference pipeline: single forward pass through all layers.
//!
//! `GpuForwardPass` encodes all transformer operations (RMSNorm, Q4_0 matvec,
//! RoPE, decode attention, SwiGLU FFN, residual add) on the GPU with minimal
//! CPU-GPU synchronization. Only two CPU-GPU transfers per token:
//!   1. Embedding lookup (CPU) -> hidden_a buffer write (2.3KB)
//!   2. Logits buffer readback (192KB) -> CPU argmax
//!
//! For POC, uses one command buffer per encoding block (attention projections,
//! attention output, FFN, final logits) to allow CPU-side KV cache append
//! between attention projection and decode attention. Within each command
//! buffer, all kernel dispatches are inline-encoded (no per-op command buffers).
//! Task 2.1 will optimize to truly single command buffer using a GPU-side
//! copy kernel for KV cache append.

use std::path::Path;
use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLSize,
};

use metal_attention_gguf::GgufFile;
use metal_attention_kernels::buffer::{alloc_buffer, read_buffer_slice};
use metal_attention_kernels::device::GpuDevice;
use metal_attention_kernels::dispatch::{set_buffer, set_bytes};
use metal_attention_kernels::pipeline::{PsoCache, PsoKey};
use metal_attention_kernels::types::LayerParams;

use crate::gpu_kv_cache::GpuKVCacheSet;
use crate::gpu_weight_store::GpuWeightStore;

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
}

impl GpuForwardPass {
    /// Construct a GpuForwardPass from a GGUF model file.
    ///
    /// Opens the GGUF file, extracts model configuration from metadata,
    /// builds the GpuWeightStore with zero-copy buffers, initializes
    /// KV caches, and prewarms PSOs.
    pub fn from_gguf(path: &Path) -> Result<Self, String> {
        let device = GpuDevice::shared();

        // Open GGUF and extract config
        let gguf = Arc::new(
            GgufFile::open(path).map_err(|e| format!("Failed to open GGUF: {e}"))?,
        );

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
        let num_layers = gguf
            .metadata
            .get_u32("llama.block_count")
            .unwrap_or(30) as usize;
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
            num_layers, num_heads, hidden_size, num_kv_heads, intermediate_size, vocab_size,
            rope_theta, rms_norm_eps
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

        // Allocate ping-pong hidden state buffers
        let hidden_bytes = hidden_size * std::mem::size_of::<f32>();
        let hidden_a = alloc_buffer(&device.device, hidden_bytes);
        let hidden_b = alloc_buffer(&device.device, hidden_bytes);

        // Allocate scratch buffers (reused across all layers and tokens)
        let q_bytes = num_heads * head_dim * std::mem::size_of::<f32>();
        let kv_bytes = kv_dim * std::mem::size_of::<f32>();
        let ffn_bytes = intermediate_size * std::mem::size_of::<f32>();
        let logits_bytes = vocab_size * std::mem::size_of::<f32>();

        let scratch_q = alloc_buffer(&device.device, q_bytes);
        let scratch_k = alloc_buffer(&device.device, kv_bytes);
        let scratch_v = alloc_buffer(&device.device, kv_bytes);
        let scratch_attn_out = alloc_buffer(&device.device, q_bytes);
        let scratch_o = alloc_buffer(&device.device, hidden_bytes);
        let scratch_gate = alloc_buffer(&device.device, ffn_bytes);
        let scratch_up = alloc_buffer(&device.device, ffn_bytes);
        let scratch_silu = alloc_buffer(&device.device, ffn_bytes);
        let scratch_ffn = alloc_buffer(&device.device, hidden_bytes);
        let scratch_residual = alloc_buffer(&device.device, hidden_bytes);
        let logits_buf = alloc_buffer(&device.device, logits_bytes);

        // Build PSO cache and prewarm all kernels
        let mut pso_cache = PsoCache::new(device.library.clone());
        pso_cache.prewarm(&[
            PsoKey::simple("matvec_q4_0"),
            PsoKey::simple("rmsnorm_optimized"),
            PsoKey::simple("residual_add"),
            PsoKey::simple("decode_attention"),
            PsoKey::simple("rope_apply"),
            PsoKey::simple("ffn_silu"),
        ]);

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
        })
    }

    /// Run a single-token forward pass through all layers.
    ///
    /// Returns logits [vocab_size] as Vec<f32>.
    ///
    /// For POC, uses one command buffer per encoding block to allow CPU-side
    /// KV cache append between attention projection and decode attention.
    /// Within each command buffer, all kernel dispatches are inline-encoded.
    pub fn forward_token(&mut self, token_id: u32) -> Vec<f32> {
        // 1. CPU embedding lookup -> write to hidden_a via contents() memcpy
        self.embed_lookup(token_id);

        // 2. Per-layer forward
        for layer_idx in 0..self.num_layers {
            // --- Attention block ---
            // Encode rmsnorm + Q/K/V projections + RoPE, commit+wait
            self.encode_attention_projections(layer_idx);

            // CPU-side KV cache append (requires GPU work completed)
            self.kv_caches
                .cache_mut(layer_idx)
                .append_kv(&self.scratch_k, &self.scratch_v);

            // Decode attention + O projection + residual add
            self.encode_attention_output(layer_idx);

            // --- FFN block ---
            self.encode_ffn_block(layer_idx);
        }

        // 3. Final norm + lm_head
        self.encode_final_logits();

        // 4. Read back logits and increment position
        let logits = unsafe { read_buffer_slice(&self.logits_buf, self.vocab_size) };
        self.position += 1;
        logits
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
    fn encode_attention_projections(&self, layer_idx: usize) {
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
        self.encode_rmsnorm(
            &encoder,
            &self.hidden_a,
            &norms.attn_norm,
            &self.hidden_b,
        );

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
    }

    /// Encode decode attention + O projection + residual add.
    fn encode_attention_output(&self, layer_idx: usize) {
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

        // Copy scratch_residual -> hidden_a (both shared-mode, CPU memcpy)
        copy_buffer(&self.scratch_residual, &self.hidden_a, self.hidden_size * 4);
    }

    /// Encode FFN block: rmsnorm -> gate/up matvec -> SwiGLU -> down matvec -> residual add.
    fn encode_ffn_block(&self, layer_idx: usize) {
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
        self.encode_rmsnorm(
            &encoder,
            &self.hidden_a,
            &norms.ffn_norm,
            &self.hidden_b,
        );

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

        // Copy scratch_residual -> hidden_a
        copy_buffer(&self.scratch_residual, &self.hidden_a, self.hidden_size * 4);
    }

    /// Encode final RMSNorm + lm_head matvec to produce logits.
    fn encode_final_logits(&self) {
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
        self.encode_matvec_q4_0(
            &encoder,
            self.weight_store.lm_head(),
            &self.hidden_b,
            &self.logits_buf,
            self.vocab_size,
            self.hidden_size,
        );

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
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

    /// Encode fused Q4_0 dequant + matvec: weight * input -> output.
    /// Dispatch: grid=(out_dim) threadgroups, threadgroup=(32).
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
            .get(&PsoKey::simple("matvec_q4_0"))
            .expect("matvec_q4_0 PSO not prewarmed");

        encoder.setComputePipelineState(pso);
        set_buffer(encoder, weight_buf, 0, 0);
        set_buffer(encoder, input_buf, 0, 1);
        set_buffer(encoder, output_buf, 0, 2);

        let out_dim_u32 = out_dim as u32;
        let in_dim_u32 = in_dim as u32;
        set_bytes(encoder, &out_dim_u32, 3);
        set_bytes(encoder, &in_dim_u32, 4);

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

    /// Encode decode attention: Q + KV cache -> output.
    /// Dispatch: grid=(num_heads) threadgroups, threadgroup=(32).
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
            .get(&PsoKey::simple("decode_attention"))
            .expect("decode_attention PSO not prewarmed");

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
            width: 32,
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

    /// Encode SwiGLU activation: silu(scratch_gate) * scratch_up -> scratch_silu.
    /// Uses LayerParams struct at buffer(4) for ffn_silu kernel.
    /// Dispatch: grid=(intermediate_size), threadgroup=(256).
    fn encode_ffn_silu(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    ) {
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

    /// Get the current decode position.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Reset position and KV caches (for new generation).
    pub fn reset(&mut self) {
        self.position = 0;
        self.kv_caches.reset();
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
