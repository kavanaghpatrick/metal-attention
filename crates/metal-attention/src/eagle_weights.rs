//! EAGLE-3 weight loading from SafeTensors format.
//!
//! Provides a minimal SafeTensors parser (`SafeTensorsFile`) that memory-maps a
//! `.safetensors` file, parses the JSON header, and exposes tensor data by name.
//!
//! `EagleWeightStore` maps SafeTensors tensor names to EAGLE head weight buffers
//! (FC fusion, FC concat, single decoder layer), handling BF16->F32 conversion
//! and dimension validation.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;
use serde_json::Value;

use metal_attention_kernels::buffer::alloc_buffer_with_data;
use metal_attention_kernels::device::GpuDevice;

use crate::gpu_weight_store::{AttnProjBuffers, FfnBuffers, WeightBuffer};

// ---------------------------------------------------------------------------
// SafeTensors parser
// ---------------------------------------------------------------------------

/// Metadata for a single tensor in a SafeTensors file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Data type string (e.g. "F32", "F16", "BF16").
    pub dtype: String,
    /// Shape dimensions (e.g. [4096, 12288]).
    pub shape: Vec<usize>,
    /// Byte offsets [start, end) within the data section.
    pub data_offsets: (usize, usize),
}

/// Memory-mapped SafeTensors file with parsed header.
///
/// SafeTensors layout:
///   [8 bytes: header_size as u64 LE] [header_size bytes: JSON header] [tensor data]
pub struct SafeTensorsFile {
    /// Parsed tensor metadata keyed by tensor name.
    pub header: HashMap<String, TensorInfo>,
    /// Byte offset where tensor data begins (= 8 + header_size).
    pub data_offset: usize,
    /// Memory-mapped file contents.
    mmap: Mmap,
}

impl SafeTensorsFile {
    /// Open and parse a SafeTensors file.
    ///
    /// Reads the 8-byte header size, parses the JSON metadata header, and
    /// memory-maps the entire file for zero-copy tensor data access.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let file = File::open(path)
            .map_err(|e| format!("Failed to open SafeTensors file '{}': {}", path.display(), e))?;

        // Safety: we only read from the mmap; the file stays open for lifetime of Mmap.
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| format!("Failed to mmap '{}': {}", path.display(), e))?;

        if mmap.len() < 8 {
            return Err(format!(
                "SafeTensors file '{}' too small ({} bytes, need at least 8)",
                path.display(),
                mmap.len()
            ));
        }

        // Read header size (first 8 bytes, little-endian u64).
        let header_size = u64::from_le_bytes(
            mmap[..8]
                .try_into()
                .map_err(|_| "Failed to read header size".to_string())?,
        ) as usize;

        let data_offset = 8 + header_size;
        if mmap.len() < data_offset {
            return Err(format!(
                "SafeTensors header claims {} bytes but file is only {} bytes",
                data_offset,
                mmap.len()
            ));
        }

        // Parse JSON header.
        let header_json = std::str::from_utf8(&mmap[8..data_offset])
            .map_err(|e| format!("SafeTensors header is not valid UTF-8: {}", e))?;

        let parsed: Value = serde_json::from_str(header_json)
            .map_err(|e| format!("Failed to parse SafeTensors JSON header: {}", e))?;

        let obj = parsed
            .as_object()
            .ok_or("SafeTensors header is not a JSON object")?;

        let mut header = HashMap::new();
        for (key, value) in obj {
            // Skip the special "__metadata__" entry.
            if key == "__metadata__" {
                continue;
            }

            let tensor_obj = value
                .as_object()
                .ok_or_else(|| format!("Tensor '{}' metadata is not a JSON object", key))?;

            let dtype = tensor_obj
                .get("dtype")
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("Tensor '{}' missing 'dtype' field", key))?
                .to_string();

            let shape: Vec<usize> = tensor_obj
                .get("shape")
                .and_then(|v| v.as_array())
                .ok_or_else(|| format!("Tensor '{}' missing 'shape' array", key))?
                .iter()
                .map(|v| {
                    v.as_u64()
                        .ok_or_else(|| format!("Tensor '{}' shape element not a number", key))
                        .map(|n| n as usize)
                })
                .collect::<Result<Vec<_>, _>>()?;

            let offsets = tensor_obj
                .get("data_offsets")
                .and_then(|v| v.as_array())
                .ok_or_else(|| format!("Tensor '{}' missing 'data_offsets' array", key))?;

            if offsets.len() != 2 {
                return Err(format!(
                    "Tensor '{}' data_offsets must have exactly 2 elements",
                    key
                ));
            }

            let start = offsets[0]
                .as_u64()
                .ok_or_else(|| format!("Tensor '{}' data_offsets[0] not a number", key))?
                as usize;
            let end = offsets[1]
                .as_u64()
                .ok_or_else(|| format!("Tensor '{}' data_offsets[1] not a number", key))?
                as usize;

            header.insert(
                key.clone(),
                TensorInfo {
                    dtype,
                    shape,
                    data_offsets: (start, end),
                },
            );
        }

        Ok(Self {
            header,
            data_offset,
            mmap,
        })
    }

    /// Return the raw byte slice for a tensor's data.
    pub fn get_tensor_data(&self, name: &str) -> Result<&[u8], String> {
        let info = self
            .header
            .get(name)
            .ok_or_else(|| format!("Tensor '{}' not found in SafeTensors file", name))?;

        let abs_start = self.data_offset + info.data_offsets.0;
        let abs_end = self.data_offset + info.data_offsets.1;

        if abs_end > self.mmap.len() {
            return Err(format!(
                "Tensor '{}' data range [{}, {}) exceeds file size {}",
                name,
                abs_start,
                abs_end,
                self.mmap.len()
            ));
        }

        Ok(&self.mmap[abs_start..abs_end])
    }

    /// Return the shape of a tensor.
    pub fn get_tensor_shape(&self, name: &str) -> Result<Vec<usize>, String> {
        let info = self
            .header
            .get(name)
            .ok_or_else(|| format!("Tensor '{}' not found in SafeTensors file", name))?;
        Ok(info.shape.clone())
    }

    /// Return the dtype string for a tensor.
    pub fn get_tensor_dtype(&self, name: &str) -> Result<&str, String> {
        let info = self
            .header
            .get(name)
            .ok_or_else(|| format!("Tensor '{}' not found in SafeTensors file", name))?;
        Ok(&info.dtype)
    }

    /// List all tensor names in the file.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.header.keys().map(|s| s.as_str()).collect()
    }
}

// ---------------------------------------------------------------------------
// BF16 -> F32 conversion
// ---------------------------------------------------------------------------

/// Convert a single BF16 value (stored as u16) to F32.
///
/// BF16 is the upper 16 bits of an IEEE 754 float32, so conversion
/// is a simple left-shift by 16 bits.
#[inline]
fn bf16_to_f32(bf16_bits: u16) -> f32 {
    f32::from_bits((bf16_bits as u32) << 16)
}

/// Convert a BF16 byte slice to a Vec<f32>.
///
/// Input bytes are interpreted as little-endian u16 BF16 values.
fn bf16_bytes_to_f32(data: &[u8]) -> Result<Vec<f32>, String> {
    if !data.len().is_multiple_of(2) {
        return Err(format!(
            "BF16 data length {} is not a multiple of 2",
            data.len()
        ));
    }
    let num_elements = data.len() / 2;
    let mut out = Vec::with_capacity(num_elements);
    for i in 0..num_elements {
        let lo = data[i * 2] as u16;
        let hi = data[i * 2 + 1] as u16;
        let bits = lo | (hi << 8);
        out.push(bf16_to_f32(bits));
    }
    Ok(out)
}

/// Convert an F16 byte slice to a Vec<f32>.
///
/// Uses the `half` crate for proper F16->F32 conversion.
fn f16_bytes_to_f32(data: &[u8]) -> Result<Vec<f32>, String> {
    if !data.len().is_multiple_of(2) {
        return Err(format!(
            "F16 data length {} is not a multiple of 2",
            data.len()
        ));
    }
    let num_elements = data.len() / 2;
    let mut out = Vec::with_capacity(num_elements);
    for i in 0..num_elements {
        let lo = data[i * 2] as u16;
        let hi = data[i * 2 + 1] as u16;
        let bits = lo | (hi << 8);
        out.push(half::f16::from_bits(bits).to_f32());
    }
    Ok(out)
}

/// Load tensor data as F32, converting from BF16/F16 if needed.
///
/// Returns the F32 data as a Vec<f32> regardless of the source dtype.
fn load_tensor_as_f32(st: &SafeTensorsFile, name: &str) -> Result<Vec<f32>, String> {
    let data = st.get_tensor_data(name)?;
    let dtype = st.get_tensor_dtype(name)?;

    match dtype {
        "F32" => {
            if !data.len().is_multiple_of(4) {
                return Err(format!(
                    "F32 tensor '{}' data length {} not a multiple of 4",
                    name,
                    data.len()
                ));
            }
            let num_elements = data.len() / 4;
            let mut out = Vec::with_capacity(num_elements);
            for i in 0..num_elements {
                let bytes: [u8; 4] = data[i * 4..(i + 1) * 4]
                    .try_into()
                    .map_err(|_| format!("F32 slice error at element {}", i))?;
                out.push(f32::from_le_bytes(bytes));
            }
            Ok(out)
        }
        "BF16" => bf16_bytes_to_f32(data),
        "F16" => f16_bytes_to_f32(data),
        other => Err(format!(
            "Unsupported dtype '{}' for tensor '{}' (expected F32, BF16, or F16)",
            other, name
        )),
    }
}

// ---------------------------------------------------------------------------
// EagleWeightStore
// ---------------------------------------------------------------------------

/// EAGLE draft head weight store loaded from SafeTensors.
///
/// Maps SafeTensors tensor names to Metal GPU buffers following the EAGLE-3
/// architecture: FC fusion layer, FC concat layer, and a single decoder layer
/// (attention + FFN with norms).
///
/// Tensor name mapping:
/// | SafeTensors Key                           | Field           | Shape           |
/// |-------------------------------------------|-----------------|-----------------|
/// | `fc.weight`                               | fc_fuse         | [H, 3*H]       |
/// | `fc2.weight`                              | fc_concat       | [H, 2*H]       |
/// | `layers.0.self_attn.q_proj.weight`        | decoder attn Q  | [H, H]         |
/// | `layers.0.self_attn.k_proj.weight`        | decoder attn K  | [KV, H]        |
/// | `layers.0.self_attn.v_proj.weight`        | decoder attn V  | [KV, H]        |
/// | `layers.0.self_attn.o_proj.weight`        | decoder attn O  | [H, H]         |
/// | `layers.0.mlp.gate_proj.weight`           | decoder FFN gate| [I, H]         |
/// | `layers.0.mlp.up_proj.weight`             | decoder FFN up  | [I, H]         |
/// | `layers.0.mlp.down_proj.weight`           | decoder FFN down| [H, I]         |
/// | `layers.0.input_layernorm.weight`         | attn norm       | [H]            |
/// | `layers.0.post_attention_layernorm.weight` | ffn norm       | [H]            |
pub struct EagleWeightStore {
    /// FC fusion weight: [hidden_size, 3 * hidden_size].
    pub fc_fuse_weight: WeightBuffer,
    /// FC concat weight: [hidden_size, 2 * hidden_size].
    pub fc_concat_weight: WeightBuffer,
    /// Attention RMSNorm weight (F32, hidden_size elements).
    pub decoder_attn_norm: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Attention Q/K/V/O projection weights.
    pub decoder_attn: AttnProjBuffers,
    /// FFN RMSNorm weight (F32, hidden_size elements).
    pub decoder_ffn_norm: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// FFN gate/up/down weights.
    pub decoder_ffn: FfnBuffers,
}

/// Expected tensor name -> (field description, expected shape validation kind).
struct TensorSpec {
    name: &'static str,
    description: &'static str,
}

/// All expected EAGLE head tensor names.
const EAGLE_TENSOR_SPECS: &[TensorSpec] = &[
    TensorSpec {
        name: "fc.weight",
        description: "FC fusion",
    },
    TensorSpec {
        name: "fc2.weight",
        description: "FC concat",
    },
    TensorSpec {
        name: "layers.0.self_attn.q_proj.weight",
        description: "decoder attention Q",
    },
    TensorSpec {
        name: "layers.0.self_attn.k_proj.weight",
        description: "decoder attention K",
    },
    TensorSpec {
        name: "layers.0.self_attn.v_proj.weight",
        description: "decoder attention V",
    },
    TensorSpec {
        name: "layers.0.self_attn.o_proj.weight",
        description: "decoder attention O",
    },
    TensorSpec {
        name: "layers.0.mlp.gate_proj.weight",
        description: "decoder FFN gate",
    },
    TensorSpec {
        name: "layers.0.mlp.up_proj.weight",
        description: "decoder FFN up",
    },
    TensorSpec {
        name: "layers.0.mlp.down_proj.weight",
        description: "decoder FFN down",
    },
    TensorSpec {
        name: "layers.0.input_layernorm.weight",
        description: "attention layer norm",
    },
    TensorSpec {
        name: "layers.0.post_attention_layernorm.weight",
        description: "FFN layer norm",
    },
];

/// Load a tensor from SafeTensors as an F32 Metal buffer with a WeightBuffer wrapper.
fn load_weight_buffer(
    st: &SafeTensorsFile,
    name: &str,
    device: &GpuDevice,
) -> Result<WeightBuffer, String> {
    let f32_data = load_tensor_as_f32(st, name)?;
    let buf = alloc_buffer_with_data(&device.device, &f32_data);
    Ok(WeightBuffer::zero_offset(buf))
}

/// Load a tensor from SafeTensors as an F32 Metal buffer (raw Retained buffer).
fn load_norm_buffer(
    st: &SafeTensorsFile,
    name: &str,
    device: &GpuDevice,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, String> {
    let f32_data = load_tensor_as_f32(st, name)?;
    Ok(alloc_buffer_with_data(&device.device, &f32_data))
}

impl EagleWeightStore {
    /// Load EAGLE head weights from a SafeTensors file.
    ///
    /// Validates that all expected tensors are present and that their shapes
    /// match the target model's hidden size. Handles BF16->F32 conversion
    /// automatically (EAGLE weights on HuggingFace are typically BF16).
    ///
    /// # Arguments
    /// - `path`: Path to the `.safetensors` file.
    /// - `device`: GPU device for Metal buffer allocation.
    /// - `target_hidden_size`: Target model hidden dimension (e.g. 4096 for Mistral-7B).
    ///
    /// # Errors
    /// Returns an error if:
    /// - The file cannot be opened or parsed
    /// - Any expected tensor is missing
    /// - Tensor shapes don't match expected dimensions
    pub fn from_safetensors(
        path: impl AsRef<Path>,
        device: &'static GpuDevice,
        target_hidden_size: usize,
    ) -> Result<Self, String> {
        let st = SafeTensorsFile::open(path)?;

        // Validate all expected tensors are present.
        for spec in EAGLE_TENSOR_SPECS {
            if !st.header.contains_key(spec.name) {
                return Err(format!(
                    "Missing tensor '{}' ({}) in SafeTensors file. Available tensors: {:?}",
                    spec.name,
                    spec.description,
                    st.tensor_names()
                ));
            }
        }

        // --- Validate FC layer shapes ---
        let fc_fuse_shape = st.get_tensor_shape("fc.weight")?;
        if fc_fuse_shape.len() != 2 {
            return Err(format!(
                "fc.weight expected 2D, got {}D: {:?}",
                fc_fuse_shape.len(),
                fc_fuse_shape
            ));
        }
        if fc_fuse_shape[0] != target_hidden_size {
            return Err(format!(
                "fc.weight shape[0]={} does not match target_hidden_size={}",
                fc_fuse_shape[0], target_hidden_size
            ));
        }
        if fc_fuse_shape[1] != 3 * target_hidden_size {
            return Err(format!(
                "fc.weight shape[1]={} expected {} (3 * hidden_size)",
                fc_fuse_shape[1],
                3 * target_hidden_size
            ));
        }

        let fc_concat_shape = st.get_tensor_shape("fc2.weight")?;
        if fc_concat_shape.len() != 2 {
            return Err(format!(
                "fc2.weight expected 2D, got {}D: {:?}",
                fc_concat_shape.len(),
                fc_concat_shape
            ));
        }
        if fc_concat_shape[0] != target_hidden_size {
            return Err(format!(
                "fc2.weight shape[0]={} does not match target_hidden_size={}",
                fc_concat_shape[0], target_hidden_size
            ));
        }
        if fc_concat_shape[1] != 2 * target_hidden_size {
            return Err(format!(
                "fc2.weight shape[1]={} expected {} (2 * hidden_size)",
                fc_concat_shape[1],
                2 * target_hidden_size
            ));
        }

        // --- Validate Q/K/V/O shapes ---
        let q_shape = st.get_tensor_shape("layers.0.self_attn.q_proj.weight")?;
        if q_shape.len() != 2 || q_shape[1] != target_hidden_size {
            return Err(format!(
                "Q proj shape mismatch: expected [*, {}], got {:?}",
                target_hidden_size, q_shape
            ));
        }
        let hidden_size = q_shape[0]; // Should equal target_hidden_size for non-GQA Q.

        let k_shape = st.get_tensor_shape("layers.0.self_attn.k_proj.weight")?;
        if k_shape.len() != 2 || k_shape[1] != target_hidden_size {
            return Err(format!(
                "K proj shape mismatch: expected [*, {}], got {:?}",
                target_hidden_size, k_shape
            ));
        }
        let kv_dim = k_shape[0];

        let v_shape = st.get_tensor_shape("layers.0.self_attn.v_proj.weight")?;
        if v_shape.len() != 2 || v_shape[0] != kv_dim || v_shape[1] != target_hidden_size {
            return Err(format!(
                "V proj shape mismatch: expected [{}, {}], got {:?}",
                kv_dim, target_hidden_size, v_shape
            ));
        }

        let o_shape = st.get_tensor_shape("layers.0.self_attn.o_proj.weight")?;
        if o_shape.len() != 2 || o_shape[0] != target_hidden_size || o_shape[1] != hidden_size {
            return Err(format!(
                "O proj shape mismatch: expected [{}, {}], got {:?}",
                target_hidden_size, hidden_size, o_shape
            ));
        }

        // --- Validate FFN shapes ---
        let gate_shape = st.get_tensor_shape("layers.0.mlp.gate_proj.weight")?;
        if gate_shape.len() != 2 || gate_shape[1] != target_hidden_size {
            return Err(format!(
                "gate_proj shape mismatch: expected [*, {}], got {:?}",
                target_hidden_size, gate_shape
            ));
        }
        let intermediate_size = gate_shape[0];

        let up_shape = st.get_tensor_shape("layers.0.mlp.up_proj.weight")?;
        if up_shape.len() != 2
            || up_shape[0] != intermediate_size
            || up_shape[1] != target_hidden_size
        {
            return Err(format!(
                "up_proj shape mismatch: expected [{}, {}], got {:?}",
                intermediate_size, target_hidden_size, up_shape
            ));
        }

        let down_shape = st.get_tensor_shape("layers.0.mlp.down_proj.weight")?;
        if down_shape.len() != 2
            || down_shape[0] != target_hidden_size
            || down_shape[1] != intermediate_size
        {
            return Err(format!(
                "down_proj shape mismatch: expected [{}, {}], got {:?}",
                target_hidden_size, intermediate_size, down_shape
            ));
        }

        // --- Validate norm shapes ---
        let attn_norm_shape = st.get_tensor_shape("layers.0.input_layernorm.weight")?;
        if attn_norm_shape != [target_hidden_size] {
            return Err(format!(
                "input_layernorm shape mismatch: expected [{}], got {:?}",
                target_hidden_size, attn_norm_shape
            ));
        }

        let ffn_norm_shape = st.get_tensor_shape("layers.0.post_attention_layernorm.weight")?;
        if ffn_norm_shape != [target_hidden_size] {
            return Err(format!(
                "post_attention_layernorm shape mismatch: expected [{}], got {:?}",
                target_hidden_size, ffn_norm_shape
            ));
        }

        // --- Load all tensors into Metal buffers ---
        let fc_fuse_weight = load_weight_buffer(&st, "fc.weight", device)?;
        let fc_concat_weight = load_weight_buffer(&st, "fc2.weight", device)?;

        let decoder_attn = AttnProjBuffers {
            q: load_weight_buffer(&st, "layers.0.self_attn.q_proj.weight", device)?,
            k: load_weight_buffer(&st, "layers.0.self_attn.k_proj.weight", device)?,
            v: load_weight_buffer(&st, "layers.0.self_attn.v_proj.weight", device)?,
            o: load_weight_buffer(&st, "layers.0.self_attn.o_proj.weight", device)?,
        };

        let decoder_ffn = FfnBuffers {
            gate: load_weight_buffer(&st, "layers.0.mlp.gate_proj.weight", device)?,
            up: load_weight_buffer(&st, "layers.0.mlp.up_proj.weight", device)?,
            down: load_weight_buffer(&st, "layers.0.mlp.down_proj.weight", device)?,
        };

        let decoder_attn_norm =
            load_norm_buffer(&st, "layers.0.input_layernorm.weight", device)?;
        let decoder_ffn_norm =
            load_norm_buffer(&st, "layers.0.post_attention_layernorm.weight", device)?;

        Ok(Self {
            fc_fuse_weight,
            fc_concat_weight,
            decoder_attn_norm,
            decoder_attn,
            decoder_ffn_norm,
            decoder_ffn,
        })
    }
}
