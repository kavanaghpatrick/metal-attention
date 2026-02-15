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

#[cfg(test)]
mod tests {
    use super::*;
    use metal_attention_gguf::{GgufBuilder, GgufType};

    /// Test: F32 tensors pass through via bytemuck cast without modification.
    #[test]
    fn test_dequantize_f32_passthrough() {
        // Create known f32 values and encode as bytes
        let values: Vec<f32> = vec![1.0, -2.5, 3.125, 0.0, f32::MAX, f32::MIN_POSITIVE];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let data = GgufBuilder::new()
            .add_string("general.architecture", "llama")
            .add_tensor("test.weight", &[6], GgufType::F32, bytes)
            .build();

        let gguf = GgufFile::from_bytes(data).expect("parse failed");
        let result = dequantize_tensor(&gguf, "test.weight", None, None).expect("dequant failed");

        assert_eq!(result.len(), 6);
        for (i, (&expected, &actual)) in values.iter().zip(result.iter()).enumerate() {
            assert_eq!(
                expected, actual,
                "F32 passthrough mismatch at index {i}: expected {expected}, got {actual}"
            );
        }
    }

    /// Test: Q4_0 tensors are dequantized on GPU and produce finite values.
    #[test]
    fn test_dequantize_q4_0_gpu() {
        // Q4_0 block: 2 bytes f16 scale + 16 bytes packed nibbles = 18 bytes per 32 elements
        // Create one block: scale = 1.0 (f16), nibbles all set to 8 (zero-point)
        let scale_f16 = half::f16::from_f32(1.0);
        let mut block = Vec::with_capacity(18);
        block.extend_from_slice(&scale_f16.to_le_bytes());
        // 16 bytes of packed nibbles: each byte holds 2 4-bit values
        // Value 8 maps to (8 - 8) * scale = 0.0 after dequant
        block.extend_from_slice(&[0x88u8; 16]);

        let data = GgufBuilder::new()
            .add_string("general.architecture", "llama")
            .add_tensor("q4_test.weight", &[32], GgufType::Q4_0, block)
            .build();

        let gguf = GgufFile::from_bytes(data).expect("parse failed");
        let device = GpuDevice::new();
        let mut pso_cache = PsoCache::new(device.library.clone());

        let result =
            dequantize_tensor(&gguf, "q4_test.weight", Some(&device), Some(&mut pso_cache))
                .expect("dequant failed");

        assert_eq!(result.len(), 32, "Q4_0 block should produce 32 elements");
        for (i, &val) in result.iter().enumerate() {
            assert!(
                val.is_finite(),
                "Q4_0 dequant element {i} is not finite: {val}"
            );
        }
    }

    /// Test: Q4_0 dequantization requires GPU device.
    #[test]
    fn test_dequantize_q4_0_requires_device() {
        let scale_f16 = half::f16::from_f32(1.0);
        let mut block = Vec::with_capacity(18);
        block.extend_from_slice(&scale_f16.to_le_bytes());
        block.extend_from_slice(&[0x88u8; 16]);

        let data = GgufBuilder::new()
            .add_string("general.architecture", "llama")
            .add_tensor("q4_test.weight", &[32], GgufType::Q4_0, block)
            .build();

        let gguf = GgufFile::from_bytes(data).expect("parse failed");
        let result = dequantize_tensor(&gguf, "q4_test.weight", None, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("GPU device required for Q4_0"));
    }

    /// Test: unsupported quantization type returns error.
    #[test]
    fn test_dequantize_unsupported_type() {
        // Q4_K_M is not supported by dequantize_tensor
        let data = GgufBuilder::new()
            .add_string("general.architecture", "llama")
            .add_tensor_zeros("qkm_test.weight", &[256], GgufType::Q4_K_M)
            .build();

        let gguf = GgufFile::from_bytes(data).expect("parse failed");
        let result = dequantize_tensor(&gguf, "qkm_test.weight", None, None);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.contains("Unsupported quantization type"),
            "Expected unsupported type error, got: {err}"
        );
    }

    /// Test: requesting a nonexistent tensor returns error.
    #[test]
    fn test_dequantize_missing_tensor() {
        let data = GgufBuilder::new()
            .add_string("general.architecture", "llama")
            .add_tensor_zeros("exists.weight", &[16], GgufType::F32)
            .build();

        let gguf = GgufFile::from_bytes(data).expect("parse failed");
        let result = dequantize_tensor(&gguf, "nonexistent.weight", None, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Tensor not found"));
    }
}
