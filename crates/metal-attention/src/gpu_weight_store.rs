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

use metal_attention_gguf::quantize::GgufType;
use metal_attention_gguf::GgufFile;
use metal_attention_kernels::buffer::{alloc_buffer_with_data, create_weight_buffer};
use metal_attention_models::registry::ModelConfig;

/// Q4_0 block size in bytes: 2 byte fp16 scale + 16 byte nibbles = 18 bytes per 32 elements.
const Q4_0_BLOCK_BYTES: usize = 18;

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
    /// LM head projection buffer (Q4_0 or F32 for tied embeddings).
    lm_head: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Whether the lm_head buffer contains F32 data (true for tied embeddings).
    lm_head_is_f32: bool,
    /// Final RMSNorm weight buffer (F32).
    final_norm: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Keep the GGUF mmap alive while zero-copy buffers reference it.
    _gguf: Arc<GgufFile>,
}

/// Get the system page size at runtime.
fn system_page_size() -> usize {
    unsafe { libc::sysconf(libc::_SC_PAGE_SIZE) as usize }
}

/// Validate that a Q4_0 tensor has the expected byte count for its element count.
///
/// Each Q4_0 block encodes 32 elements in 18 bytes (2-byte fp16 scale + 16-byte nibbles).
/// Returns `Err` if the byte length doesn't match the expected block count.
fn validate_q4_0_size(tensor_name: &str, data_len: usize, n_elements: u64) -> Result<(), String> {
    let expected_blocks = (n_elements as usize).div_ceil(32);
    let expected_bytes = expected_blocks * Q4_0_BLOCK_BYTES;
    if data_len != expected_bytes {
        return Err(format!(
            "Q4_0 block count mismatch for {tensor_name}: \
             data_len={data_len} but expected {expected_blocks} blocks * {Q4_0_BLOCK_BYTES} = {expected_bytes} bytes \
             (n_elements={n_elements})"
        ));
    }
    Ok(())
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
        // Silently fall back to copy for non-aligned tensors (common for GGUF files).
        // Set GPU_DEBUG=1 to see per-tensor alignment warnings.
        if std::env::var("GPU_DEBUG").is_ok() {
            eprintln!(
                "  tensor {tensor_name} not page-aligned (offset 0x{:x}), using copy",
                ptr % page_size,
            );
        }
    }

    alloc_buffer_with_data(device, data)
}

/// Dequantize Q8_0 data to F32.
///
/// Q8_0 block: 2 bytes fp16 scale (half d) + 32 bytes signed int8 values.
/// Total: 34 bytes per 32 elements.
/// Dequant: value = qs[i] * d
fn dequantize_q8_0_to_f32(data: &[u8], n_elements: usize) -> Vec<f32> {
    let n_blocks = n_elements / 32;
    let mut out = vec![0.0f32; n_elements];

    for b in 0..n_blocks {
        let block_offset = b * 34; // 34 bytes per Q8_0 block

        // Read fp16 scale (little-endian)
        let d_bits = u16::from_le_bytes([data[block_offset], data[block_offset + 1]]);
        let d = half::f16::from_bits(d_bits).to_f32();

        // Read 32 signed int8 values
        for i in 0..32 {
            let qs = data[block_offset + 2 + i] as i8;
            out[b * 32 + i] = qs as f32 * d;
        }
    }

    out
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

            // Validate Q4_0 block counts for quantized attention weights
            if q_info.gguf_type == GgufType::Q4_0 {
                let n_elem = q_info.shape.iter().product::<u64>();
                validate_q4_0_size(&format!("blk.{i}.attn_q.weight"), q_data.len(), n_elem)?;
            }
            if k_info.gguf_type == GgufType::Q4_0 {
                let n_elem = k_info.shape.iter().product::<u64>();
                validate_q4_0_size(&format!("blk.{i}.attn_k.weight"), k_data.len(), n_elem)?;
            }
            if v_info.gguf_type == GgufType::Q4_0 {
                let n_elem = v_info.shape.iter().product::<u64>();
                validate_q4_0_size(&format!("blk.{i}.attn_v.weight"), v_data.len(), n_elem)?;
            }
            if o_info.gguf_type == GgufType::Q4_0 {
                let n_elem = o_info.shape.iter().product::<u64>();
                validate_q4_0_size(&format!("blk.{i}.attn_output.weight"), o_data.len(), n_elem)?;
            }

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

            // Validate Q4_0 block counts for quantized FFN weights
            if gate_info.gguf_type == GgufType::Q4_0 {
                let n_elem = gate_info.shape.iter().product::<u64>();
                validate_q4_0_size(&format!("blk.{i}.ffn_gate.weight"), gate_data.len(), n_elem)?;
            }
            if up_info.gguf_type == GgufType::Q4_0 {
                let n_elem = up_info.shape.iter().product::<u64>();
                validate_q4_0_size(&format!("blk.{i}.ffn_up.weight"), up_data.len(), n_elem)?;
            }
            if down_info.gguf_type == GgufType::Q4_0 {
                let n_elem = down_info.shape.iter().product::<u64>();
                validate_q4_0_size(&format!("blk.{i}.ffn_down.weight"), down_data.len(), n_elem)?;
            }

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

        // Embedding: dequantize to F32 at load time for CPU-side embed_lookup.
        // This also provides the lm_head buffer for tied embeddings.
        let embed_info = gguf
            .find_tensor("token_embd.weight")
            .ok_or_else(|| "Tensor not found: token_embd.weight".to_string())?;
        let embed_data = gguf.tensor_data(embed_info);

        let embed_f32_bytes: Vec<u8> = match embed_info.gguf_type {
            GgufType::F32 => {
                eprintln!("token_embd.weight: F32 (no dequant needed)");
                embed_data.to_vec()
            }
            GgufType::Q8_0 => {
                let n_elements = embed_info.shape.iter().product::<u64>() as usize;
                eprintln!("token_embd.weight: Q8_0, dequantizing {n_elements} elements to F32");
                let f32_vec = dequantize_q8_0_to_f32(embed_data, n_elements);
                // Safety: reinterpret Vec<f32> as bytes
                let byte_len = f32_vec.len() * std::mem::size_of::<f32>();
                let ptr = f32_vec.as_ptr() as *const u8;
                unsafe { std::slice::from_raw_parts(ptr, byte_len) }.to_vec()
            }
            other => {
                return Err(format!(
                    "Unsupported embedding type: {:?}. Expected F32 or Q8_0.",
                    other
                ));
            }
        };
        let embed = alloc_buffer_with_data(device, &embed_f32_bytes);

        // LM head: try output.weight first, fall back to tied embedding (already F32)
        let (lm_head, lm_head_is_f32) = if let Some(lm_info) = gguf.find_tensor("output.weight") {
            let lm_data = gguf.tensor_data(lm_info);
            eprintln!("output.weight: type={:?}", lm_info.gguf_type);
            (
                make_weight_buffer(device, lm_data, "output.weight", page_size),
                false,
            )
        } else {
            eprintln!("output.weight not found, using tied F32 embedding for lm_head");
            (alloc_buffer_with_data(device, &embed_f32_bytes), true)
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
            lm_head_is_f32,
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

    /// Whether the lm_head contains F32 data (tied embeddings) vs Q4_0.
    pub fn lm_head_is_f32(&self) -> bool {
        self.lm_head_is_f32
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
