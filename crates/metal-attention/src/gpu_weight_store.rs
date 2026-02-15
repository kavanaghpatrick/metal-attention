//! GPU weight store: zero-copy Metal buffers from GGUF mmap.
//!
//! GpuWeightStore loads all model weights from a parsed GGUF file into
//! Metal buffers. Quantized tensors (Q4_0) use zero-copy mmap buffers
//! when page-aligned, falling back to copy-based allocation otherwise.
//! F32 tensors (norms, embeddings) always use copy-based allocation.
//!
//! The store holds an `Arc<GgufFile>` to keep the mmap alive while
//! Metal buffers reference it via `newBufferWithBytesNoCopy`.

use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice};

use metal_attention_gguf::GgufFile;
use metal_attention_kernels::buffer::{alloc_buffer_with_data, create_weight_buffer};
use metal_attention_models::registry::ModelConfig;

/// Per-layer attention projection buffers (Q/K/V/O weights, raw Q4_0).
pub struct AttnProjBuffers {
    pub q: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub k: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub v: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub o: Retained<ProtocolObject<dyn MTLBuffer>>,
}

/// Per-layer FFN buffers (gate/up/down weights, Q4_0).
pub struct FfnBuffers {
    pub gate: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub up: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub down: Retained<ProtocolObject<dyn MTLBuffer>>,
}

/// Per-layer norm weight buffers (F32).
pub struct NormBuffers {
    pub attn_norm: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub ffn_norm: Retained<ProtocolObject<dyn MTLBuffer>>,
}

/// GPU weight store holding all model weights as Metal buffers.
///
/// Weights are loaded from a GGUF file with zero-copy for page-aligned
/// quantized tensors and copy-based allocation for small F32 tensors.
pub struct GpuWeightStore {
    /// Attention projection weights per layer.
    attn_projs: Vec<AttnProjBuffers>,
    /// FFN weights per layer.
    ffns: Vec<FfnBuffers>,
    /// Norm weights per layer.
    norms: Vec<NormBuffers>,
    /// Token embedding buffer (F32).
    embed: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// LM head projection buffer (Q4_0).
    lm_head: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Final RMSNorm weight buffer (F32).
    final_norm: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Keep the GGUF mmap alive while zero-copy buffers reference it.
    _gguf: Arc<GgufFile>,
}

/// Get the system page size at runtime.
fn system_page_size() -> usize {
    unsafe { libc::sysconf(libc::_SC_PAGE_SIZE) as usize }
}

/// Create a Metal buffer from GGUF tensor data, using zero-copy when page-aligned.
///
/// Returns a zero-copy buffer if the data pointer is page-aligned, otherwise
/// falls back to a copy-based buffer and logs a warning.
fn make_weight_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    data: &[u8],
    tensor_name: &str,
    page_size: usize,
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    let ptr = data.as_ptr() as usize;

    if ptr.is_multiple_of(page_size) {
        // Page-aligned: try zero-copy
        if let Some(buf) = unsafe {
            create_weight_buffer(device, data.as_ptr() as *mut std::ffi::c_void, data.len())
        } {
            return buf;
        }
        // Zero-copy failed (shouldn't happen for aligned data), fall through
        eprintln!(
            "Warning: zero-copy buffer creation failed for {tensor_name}, falling back to copy"
        );
    } else {
        eprintln!(
            "Warning: tensor {tensor_name} not page-aligned (offset 0x{:x}, page_size {}), using copy",
            ptr % page_size,
            page_size
        );
    }

    alloc_buffer_with_data(device, data)
}

impl GpuWeightStore {
    /// Create a GpuWeightStore from a parsed GGUF file.
    ///
    /// Loads all model tensors into Metal buffers. Quantized tensors
    /// (Q4_0/Q8_0) use zero-copy mmap buffers when page-aligned.
    /// F32 tensors (norms, embeddings) always use copy-based allocation.
    ///
    /// # Arguments
    /// - `gguf`: Parsed GGUF file (wrapped in Arc to keep mmap alive).
    /// - `config`: Model configuration with layer count and dimensions.
    /// - `device`: Metal device for buffer allocation.
    pub fn from_gguf(
        gguf: Arc<GgufFile>,
        config: &ModelConfig,
        device: &ProtocolObject<dyn MTLDevice>,
    ) -> Result<Self, String> {
        let page_size = system_page_size();
        let num_layers = config.num_layers;

        let mut attn_projs = Vec::with_capacity(num_layers);
        let mut ffns = Vec::with_capacity(num_layers);
        let mut norms = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            // Attention projection weights (Q4_0, zero-copy when aligned)
            let q_info = gguf
                .find_tensor(&format!("blk.{i}.attn_q.weight"))
                .ok_or_else(|| format!("Tensor not found: blk.{i}.attn_q.weight"))?;
            let k_info = gguf
                .find_tensor(&format!("blk.{i}.attn_k.weight"))
                .ok_or_else(|| format!("Tensor not found: blk.{i}.attn_k.weight"))?;
            let v_info = gguf
                .find_tensor(&format!("blk.{i}.attn_v.weight"))
                .ok_or_else(|| format!("Tensor not found: blk.{i}.attn_v.weight"))?;
            let o_info = gguf
                .find_tensor(&format!("blk.{i}.attn_output.weight"))
                .ok_or_else(|| format!("Tensor not found: blk.{i}.attn_output.weight"))?;

            let q_data = gguf.tensor_data(q_info);
            let k_data = gguf.tensor_data(k_info);
            let v_data = gguf.tensor_data(v_info);
            let o_data = gguf.tensor_data(o_info);

            attn_projs.push(AttnProjBuffers {
                q: make_weight_buffer(device, q_data, &format!("blk.{i}.attn_q.weight"), page_size),
                k: make_weight_buffer(device, k_data, &format!("blk.{i}.attn_k.weight"), page_size),
                v: make_weight_buffer(device, v_data, &format!("blk.{i}.attn_v.weight"), page_size),
                o: make_weight_buffer(
                    device,
                    o_data,
                    &format!("blk.{i}.attn_output.weight"),
                    page_size,
                ),
            });

            // FFN weights (Q4_0, zero-copy when aligned)
            let gate_info = gguf
                .find_tensor(&format!("blk.{i}.ffn_gate.weight"))
                .ok_or_else(|| format!("Tensor not found: blk.{i}.ffn_gate.weight"))?;
            let up_info = gguf
                .find_tensor(&format!("blk.{i}.ffn_up.weight"))
                .ok_or_else(|| format!("Tensor not found: blk.{i}.ffn_up.weight"))?;
            let down_info = gguf
                .find_tensor(&format!("blk.{i}.ffn_down.weight"))
                .ok_or_else(|| format!("Tensor not found: blk.{i}.ffn_down.weight"))?;

            let gate_data = gguf.tensor_data(gate_info);
            let up_data = gguf.tensor_data(up_info);
            let down_data = gguf.tensor_data(down_info);

            ffns.push(FfnBuffers {
                gate: make_weight_buffer(
                    device,
                    gate_data,
                    &format!("blk.{i}.ffn_gate.weight"),
                    page_size,
                ),
                up: make_weight_buffer(
                    device,
                    up_data,
                    &format!("blk.{i}.ffn_up.weight"),
                    page_size,
                ),
                down: make_weight_buffer(
                    device,
                    down_data,
                    &format!("blk.{i}.ffn_down.weight"),
                    page_size,
                ),
            });

            // Norm weights (F32, always copy -- small tensors)
            let attn_norm_info = gguf
                .find_tensor(&format!("blk.{i}.attn_norm.weight"))
                .ok_or_else(|| format!("Tensor not found: blk.{i}.attn_norm.weight"))?;
            let ffn_norm_info = gguf
                .find_tensor(&format!("blk.{i}.ffn_norm.weight"))
                .ok_or_else(|| format!("Tensor not found: blk.{i}.ffn_norm.weight"))?;

            let attn_norm_data = gguf.tensor_data(attn_norm_info);
            let ffn_norm_data = gguf.tensor_data(ffn_norm_info);

            norms.push(NormBuffers {
                attn_norm: alloc_buffer_with_data(device, attn_norm_data),
                ffn_norm: alloc_buffer_with_data(device, ffn_norm_data),
            });
        }

        // Embedding (F32, always copy)
        let embed_info = gguf
            .find_tensor("token_embd.weight")
            .ok_or_else(|| "Tensor not found: token_embd.weight".to_string())?;
        let embed_data = gguf.tensor_data(embed_info);
        let embed = alloc_buffer_with_data(device, embed_data);

        // LM head (may be Q4_0 or F32, use zero-copy when aligned)
        // Try output.weight first, fall back to token_embd.weight (tied embeddings)
        let lm_head = if let Some(lm_info) = gguf.find_tensor("output.weight") {
            let lm_data = gguf.tensor_data(lm_info);
            make_weight_buffer(device, lm_data, "output.weight", page_size)
        } else {
            eprintln!("Warning: output.weight not found, using tied token_embd.weight for lm_head");
            alloc_buffer_with_data(device, embed_data)
        };

        // Final norm (F32, always copy)
        let final_norm_info = gguf
            .find_tensor("output_norm.weight")
            .ok_or_else(|| "Tensor not found: output_norm.weight".to_string())?;
        let final_norm_data = gguf.tensor_data(final_norm_info);
        let final_norm = alloc_buffer_with_data(device, final_norm_data);

        Ok(Self {
            attn_projs,
            ffns,
            norms,
            embed,
            lm_head,
            final_norm,
            _gguf: gguf,
        })
    }

    /// Get attention projection buffers for a given layer.
    pub fn attn_proj(&self, layer: usize) -> &AttnProjBuffers {
        &self.attn_projs[layer]
    }

    /// Get FFN buffers for a given layer.
    pub fn ffn(&self, layer: usize) -> &FfnBuffers {
        &self.ffns[layer]
    }

    /// Get norm buffers for a given layer.
    pub fn norm(&self, layer: usize) -> &NormBuffers {
        &self.norms[layer]
    }

    /// Get the token embedding buffer.
    pub fn embed(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.embed
    }

    /// Get the LM head projection buffer.
    pub fn lm_head(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.lm_head
    }

    /// Get the final RMSNorm weight buffer.
    pub fn final_norm(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.final_norm
    }

    /// Number of layers in the weight store.
    pub fn num_layers(&self) -> usize {
        self.attn_projs.len()
    }
}
