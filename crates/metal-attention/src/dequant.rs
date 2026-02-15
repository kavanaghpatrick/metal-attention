//! Generic tensor dequantization dispatcher.
//!
//! Routes GGUF tensor data through the appropriate dequantization path
//! based on the tensor's quantization type:
//!   - F32: direct byte cast (zero-copy via bytemuck)
//!   - F16: half-precision conversion loop
//!   - Q4_0/Q8_0: GPU kernel dispatch
//!   - Other: unsupported (returns error)

use metal_attention_gguf::{GgufFile, GgufType};
use metal_attention_kernels::dequant::{dispatch_dequantize_q4_0, dispatch_dequantize_q8_0};
use metal_attention_kernels::device::GpuDevice;
use metal_attention_kernels::pipeline::PsoCache;

/// Dequantize a named tensor from a GGUF file to f32 values.
///
/// Dispatches to the appropriate dequantization path based on the tensor's
/// quantization type. GPU-quantized types (Q4_0, Q8_0) require a device
/// and PSO cache.
///
/// # Arguments
/// - `gguf_file`: The parsed GGUF file containing tensor data.
/// - `tensor_name`: Name of the tensor to dequantize (e.g. "blk.0.attn_q.weight").
/// - `device`: GPU device for quantized tensor dispatch (required for Q4_0/Q8_0).
/// - `pso_cache`: Pipeline state cache for GPU kernels (required for Q4_0/Q8_0).
///
/// # Returns
/// Dequantized f32 values, or an error string if the tensor is not found
/// or uses an unsupported quantization type.
pub fn dequantize_tensor(
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
            // Direct cast: reinterpret &[u8] as &[f32]
            let floats: &[f32] = bytemuck::cast_slice(bytes);
            Ok(floats.to_vec())
        }
        GgufType::F16 => {
            // Convert each f16 to f32
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
