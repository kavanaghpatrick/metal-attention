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
    /// Q8_0 lm_head buffer (if tied embeddings, keeps original Q8_0 for bandwidth savings).
    lm_head_q8: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Q6_K lm_head buffer (raw quantized, avoids 512MB F32 dequant).
    lm_head_q6k: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
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
/// Dequantize Q5_K super-blocks (256 elements each, 176 bytes per block).
///
/// Layout per super-block (176 bytes):
///   - 2 bytes: fp16 d (scale)
///   - 2 bytes: fp16 dmin (min value)
///   - 12 bytes: scales/mins for 8 sub-blocks (6 bits each, packed)
///   - 32 bytes: qh — high bit for each of 256 values
///   - 128 bytes: qs — low 4 bits, packed in sub-block pairs
fn dequantize_q5_k_to_f32(data: &[u8], n_elements: usize) -> Vec<f32> {
    const QK: usize = 256;
    const BLOCK_BYTES: usize = 176;
    let n_blocks = n_elements / QK;
    let mut out = vec![0.0f32; n_elements];

    for b in 0..n_blocks {
        let bp = b * BLOCK_BYTES;

        let d = half::f16::from_bits(u16::from_le_bytes([data[bp], data[bp + 1]])).to_f32();
        let dmin =
            half::f16::from_bits(u16::from_le_bytes([data[bp + 2], data[bp + 3]])).to_f32();

        // Unpack 6-bit scales and mins for 8 sub-blocks from 12 bytes
        let scales_raw = &data[bp + 4..bp + 16];
        let mut sc = [0u8; 8];
        let mut mn = [0u8; 8];
        for i in 0..8 {
            if i < 4 {
                sc[i] = scales_raw[i] & 0x3F;
                mn[i] = scales_raw[i + 4] & 0x3F;
            } else {
                sc[i] = (scales_raw[i + 4] & 0x0F) | ((scales_raw[i - 4] >> 6) << 4);
                mn[i] = (scales_raw[i + 4] >> 4) | ((scales_raw[i] >> 6) << 4);
            }
        }

        let qh = &data[bp + 16..bp + 48]; // 32 bytes = 256 bits
        let qs = &data[bp + 48..bp + 176]; // 128 bytes (sub-block pair packed)

        // Process 4 pairs of sub-blocks (j=0..3), each pair = 64 elements
        for j in 0..4 {
            let sc1 = sc[2 * j] as f32;
            let mn1 = mn[2 * j] as f32;
            let sc2 = sc[2 * j + 1] as f32;
            let mn2 = mn[2 * j + 1] as f32;
            let q = &qs[32 * j..32 * (j + 1)]; // 32 bytes for this pair

            for l in 0..32 {
                let elem1 = 64 * j + l;
                let elem2 = 64 * j + 32 + l;

                // High bits: qh[l] packs 8 bits across all 4 pairs.
                // Bit (2*j)   → first sub-block, bit (2*j+1) → second
                let h1 = ((qh[l] >> (2 * j)) & 1) as u32;
                let h2 = ((qh[l] >> (2 * j + 1)) & 1) as u32;

                let q4_1 = (q[l] & 0x0F) as u32;
                let q4_2 = (q[l] >> 4) as u32;

                let q5_1 = q4_1 | (h1 << 4); // 5-bit value [0, 31]
                let q5_2 = q4_2 | (h2 << 4);

                out[b * QK + elem1] = d * sc1 * (q5_1 as f32) - dmin * mn1;
                out[b * QK + elem2] = d * sc2 * (q5_2 as f32) - dmin * mn2;
            }
        }
    }

    out
}

/// Dequantize Q6_K super-blocks (256 elements each, 210 bytes per block).
///
/// Layout per super-block (210 bytes):
///   - 128 bytes: ql — low 4 bits of 6-bit values
///   - 64 bytes: qh — upper 2 bits of 6-bit values
///   - 16 bytes: scales — signed int8 scales for 16 sub-blocks
///   - 2 bytes: d — fp16 super-block scale
fn dequantize_q6_k_to_f32(data: &[u8], n_elements: usize) -> Vec<f32> {
    const QK: usize = 256;
    const BLOCK_BYTES: usize = 210;
    let n_blocks = n_elements / QK;
    let mut out = vec![0.0f32; n_elements];

    for b in 0..n_blocks {
        let bp = b * BLOCK_BYTES;
        let ql = &data[bp..bp + 128];
        let qh = &data[bp + 128..bp + 192];
        let scales = &data[bp + 192..bp + 208];
        let d = half::f16::from_bits(u16::from_le_bytes([data[bp + 208], data[bp + 209]]))
            .to_f32();

        // Process two 128-element chunks (n=0, n=128)
        for chunk in 0..2 {
            let ql_off = chunk * 64;
            let qh_off = chunk * 32;
            let sc_off = chunk * 8;
            let out_off = b * QK + chunk * 128;

            for l in 0..32 {
                let is = l / 16; // 0 or 1

                // Reconstruct 6-bit values: 4 low bits from ql + 2 high bits from qh
                let q1 = ((ql[ql_off + l] & 0xF) | (((qh[qh_off + l] >> 0) & 3) << 4)) as i32
                    - 32;
                let q2 = ((ql[ql_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4))
                    as i32
                    - 32;
                let q3 = ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i32
                    - 32;
                let q4 = ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4))
                    as i32
                    - 32;

                let sc0 = scales[sc_off + is] as i8 as f32;
                let sc1 = scales[sc_off + is + 2] as i8 as f32;
                let sc2 = scales[sc_off + is + 4] as i8 as f32;
                let sc3 = scales[sc_off + is + 6] as i8 as f32;

                out[out_off + l] = d * sc0 * q1 as f32;
                out[out_off + l + 32] = d * sc1 * q2 as f32;
                out[out_off + l + 64] = d * sc2 * q3 as f32;
                out[out_off + l + 96] = d * sc3 * q4 as f32;
            }
        }
    }

    out
}

fn dequantize_q4_0_to_f32(data: &[u8], n_elements: usize) -> Vec<f32> {
    let n_blocks = n_elements / 32;
    let mut out = vec![0.0f32; n_elements];

    for b in 0..n_blocks {
        let block_offset = b * 18; // 18 bytes per Q4_0 block: 2 (fp16 scale) + 16 (32 nibbles)

        // Read fp16 scale (little-endian)
        let d_bits = u16::from_le_bytes([data[block_offset], data[block_offset + 1]]);
        let d = half::f16::from_bits(d_bits).to_f32();

        // Q4_0 layout: bytes 0-15, low nibble → elements 0-15
        //              bytes 0-15, high nibble → elements 16-31
        for i in 0..16 {
            let byte = data[block_offset + 2 + i];
            let lo = (byte & 0x0F) as i32 - 8;
            let hi = ((byte >> 4) & 0x0F) as i32 - 8;
            out[b * 32 + i] = lo as f32 * d;
            out[b * 32 + i + 16] = hi as f32 * d;
        }
    }

    out
}

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
                let byte_len = f32_vec.len() * std::mem::size_of::<f32>();
                let ptr = f32_vec.as_ptr() as *const u8;
                unsafe { std::slice::from_raw_parts(ptr, byte_len) }.to_vec()
            }
            GgufType::Q4_0 => {
                let n_elements = embed_info.shape.iter().product::<u64>() as usize;
                eprintln!("token_embd.weight: Q4_0, dequantizing {n_elements} elements to F32");
                let f32_vec = dequantize_q4_0_to_f32(embed_data, n_elements);
                let byte_len = f32_vec.len() * std::mem::size_of::<f32>();
                let ptr = f32_vec.as_ptr() as *const u8;
                unsafe { std::slice::from_raw_parts(ptr, byte_len) }.to_vec()
            }
            other => {
                return Err(format!(
                    "Unsupported embedding type: {:?}. Expected F32, Q8_0, or Q4_0.",
                    other
                ));
            }
        };
        let embed = alloc_buffer_with_data(device, &embed_f32_bytes);

        // LM head: try output.weight first, fall back to tied embedding.
        // Supported native types: Q4_0, Q8_0, Q6_K (native kernel), F32.
        // Q5_K still dequantized to F32 at load time.
        let (lm_head, lm_head_is_f32, lm_head_q8, lm_head_q6k) = if let Some(lm_info) =
            gguf.find_tensor("output.weight")
        {
            let lm_data = gguf.tensor_data(lm_info);
            eprintln!("output.weight: type={:?}", lm_info.gguf_type);
            match lm_info.gguf_type {
                GgufType::Q4_0 => (
                    make_weight_buffer(device, lm_data, "output.weight", page_size),
                    false,
                    None,
                    None,
                ),
                GgufType::Q8_0 => (
                    make_weight_buffer(device, lm_data, "output.weight", page_size),
                    false,
                    Some(make_weight_buffer(device, lm_data, "output.weight(q8)", page_size)),
                    None,
                ),
                GgufType::F32 => (
                    alloc_buffer_with_data(device, lm_data),
                    true,
                    None,
                    None,
                ),
                GgufType::Q6_K => {
                    // Native Q6_K kernel: store raw Q6_K buffer (108 MB vs 512 MB F32)
                    let q6k_buf = make_weight_buffer(device, lm_data, "output.weight(q6k)", page_size);
                    // Also dequantize to F32 as fallback
                    let n_elements = lm_info.shape.iter().product::<u64>() as usize;
                    let f32_vec = dequantize_q6_k_to_f32(lm_data, n_elements);
                    let byte_len = f32_vec.len() * std::mem::size_of::<f32>();
                    let ptr = f32_vec.as_ptr() as *const u8;
                    let bytes = unsafe { std::slice::from_raw_parts(ptr, byte_len) };
                    eprintln!(
                        "  Q6_K output.weight: native kernel ({:.1} MB) + F32 fallback ({:.1} MB)",
                        lm_data.len() as f64 / 1_048_576.0,
                        byte_len as f64 / 1_048_576.0
                    );
                    (alloc_buffer_with_data(device, bytes), true, None, Some(q6k_buf))
                }
                _ => {
                    // Unsupported quant type — dequantize to F32
                    let n_elements = lm_info.shape.iter().product::<u64>() as usize;
                    eprintln!(
                        "  Dequantizing {:?} output.weight ({} elements) to F32",
                        lm_info.gguf_type, n_elements
                    );
                    let f32_vec = match lm_info.gguf_type {
                        GgufType::Q5_K_S | GgufType::Q5_K_M => dequantize_q5_k_to_f32(lm_data, n_elements),
                        _ => {
                            return Err(format!(
                                "Unsupported output.weight type: {:?}. Cannot dequantize.",
                                lm_info.gguf_type
                            ));
                        }
                    };
                    let byte_len = f32_vec.len() * std::mem::size_of::<f32>();
                    let ptr = f32_vec.as_ptr() as *const u8;
                    let bytes = unsafe { std::slice::from_raw_parts(ptr, byte_len) };
                    (alloc_buffer_with_data(device, bytes), true, None, None)
                }
            }
        } else {
            // Tied embeddings: keep Q8_0/Q4_0 raw buffer for bandwidth-optimized lm_head
            let q8_buf = match embed_info.gguf_type {
                GgufType::Q8_0 => {
                    eprintln!(
                        "output.weight not found, using tied Q8_0 embedding for lm_head (bandwidth optimized)"
                    );
                    Some(make_weight_buffer(
                        device,
                        embed_data,
                        "token_embd.weight(q8_lm_head)",
                        page_size,
                    ))
                }
                _ => {
                    eprintln!("output.weight not found, using tied F32 embedding for lm_head");
                    None
                }
            };
            (
                alloc_buffer_with_data(device, &embed_f32_bytes),
                true,
                q8_buf,
                None,
            )
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
            lm_head_q8,
            lm_head_q6k,
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

    /// Get the Q8_0 lm_head buffer (if available, for bandwidth-optimized lm_head).
    pub fn lm_head_q8(&self) -> Option<&ProtocolObject<dyn MTLBuffer>> {
        self.lm_head_q8.as_deref()
    }

    /// Get the Q6_K lm_head buffer (if available, saves ~4x bandwidth vs F32 dequant).
    pub fn lm_head_q6k(&self) -> Option<&ProtocolObject<dyn MTLBuffer>> {
        self.lm_head_q6k.as_deref()
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
