//! Fused Q4_0 dequantize + matrix-vector multiply dispatch: GPU kernel host code.
//!
//! Dispatches the `matvec_q4_0` Metal kernel which computes:
//!   output[row] = dot(dequant(weight_row), input)
//! where weight is stored in Q4_0 format (32 elements per 18-byte block).
//!
//! Uses 32-thread simdgroups with simd_sum for cooperative reduction.

use crate::buffer::{alloc_buffer, alloc_buffer_with_data, read_buffer_slice};
use crate::device::GpuDevice;
use crate::dispatch::{set_buffer, set_bytes};
use crate::pipeline::{PsoCache, PsoKey};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
};

/// Q4_0 block size: 32 elements per block.
pub const Q4_0_BLOCK_SIZE: usize = 32;
/// Q4_0 byte size: 2 (scale as float16) + 16 (32 nibbles) = 18 bytes.
pub const Q4_0_BYTES_PER_BLOCK: usize = 18;

/// Run fused Q4_0 dequant + matrix-vector multiply on the GPU.
///
/// Computes `output = weight * input` where weight is in Q4_0 format.
///
/// # Arguments
/// - `device`: GpuDevice with Metal device, command queue, and shader library
/// - `pso_cache`: PSO cache for kernel compilation
/// - `weight_bytes`: Raw Q4_0 packed weight data (out_dim * n_blocks_per_row * 18 bytes)
/// - `input`: Input vector [in_dim] as f32
/// - `out_dim`: Number of output rows
/// - `in_dim`: Number of input elements (must be multiple of 32)
///
/// # Returns
/// Output vector [out_dim] as Vec<f32>
///
/// # Panics
/// Panics if in_dim is not a multiple of 32, if weight_bytes length doesn't match
/// expected size, or if the command buffer fails.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_matvec_q4_0(
    device: &GpuDevice,
    pso_cache: &mut PsoCache,
    weight_bytes: &[u8],
    input: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    assert_eq!(in_dim % Q4_0_BLOCK_SIZE, 0, "in_dim must be multiple of 32");
    assert_eq!(input.len(), in_dim, "input length mismatch");

    let n_blocks_per_row = in_dim / Q4_0_BLOCK_SIZE;
    let expected_bytes = out_dim * n_blocks_per_row * Q4_0_BYTES_PER_BLOCK;
    assert_eq!(
        weight_bytes.len(),
        expected_bytes,
        "weight_bytes length mismatch: got {}, expected {} (out_dim={}, in_dim={}, blocks_per_row={})",
        weight_bytes.len(),
        expected_bytes,
        out_dim,
        in_dim,
        n_blocks_per_row
    );

    // Allocate Metal buffers
    let weight_buf = alloc_buffer_with_data(&device.device, weight_bytes);
    let input_buf = alloc_buffer_with_data(&device.device, input);
    let output_buf = alloc_buffer(&device.device, out_dim * std::mem::size_of::<f32>());

    // Compile PSO (no function constants needed)
    let pso_key = PsoKey::simple("matvec_q4_0");
    let pso = pso_cache.get_or_compile(&pso_key);

    // Create command buffer and compute encoder
    let command_buffer = device
        .command_queue
        .commandBuffer()
        .expect("Failed to create command buffer");
    let encoder = command_buffer
        .computeCommandEncoder()
        .expect("Failed to create compute encoder");

    // Set PSO and bind buffers
    encoder.setComputePipelineState(pso);
    set_buffer(&encoder, &weight_buf, 0, 0);
    set_buffer(&encoder, &input_buf, 0, 1);
    set_buffer(&encoder, &output_buf, 0, 2);

    // Set dimension constants via setBytes (constant uint& in shader)
    let out_dim_u32 = out_dim as u32;
    let in_dim_u32 = in_dim as u32;
    set_bytes(&encoder, &out_dim_u32, 3);
    set_bytes(&encoder, &in_dim_u32, 4);

    // Dispatch: one threadgroup per output row, 32 threads per threadgroup (1 simdgroup)
    let threadgroups_per_grid = MTLSize {
        width: out_dim,
        height: 1,
        depth: 1,
    };
    let threads_per_threadgroup = MTLSize {
        width: 32,
        height: 1,
        depth: 1,
    };

    encoder
        .dispatchThreadgroups_threadsPerThreadgroup(threadgroups_per_grid, threads_per_threadgroup);
    encoder.endEncoding();

    // Commit and wait
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    // Check command buffer status
    let status = command_buffer.status();
    assert_eq!(
        status,
        objc2_metal::MTLCommandBufferStatus::Completed,
        "matvec_q4_0 command buffer failed with status {:?}. Error: {:?}",
        status,
        command_buffer.error()
    );

    // Read back output
    unsafe { read_buffer_slice(&output_buf, out_dim) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::GpuDevice;
    use crate::pipeline::PsoCache;

    /// Encode a Q4_0 block from a scale (f32) and 32 dequantized values.
    ///
    /// Returns 18 bytes: 2 bytes fp16 scale + 16 bytes packed nibbles.
    fn encode_q4_0_block(scale: f32, values: &[i8; 32]) -> [u8; 18] {
        let mut block = [0u8; 18];

        // Convert scale to fp16 bytes (little-endian)
        let scale_f16 = half::f16::from_f32(scale);
        let scale_bytes = scale_f16.to_le_bytes();
        block[0] = scale_bytes[0];
        block[1] = scale_bytes[1];

        // Pack nibbles: low nibbles [0..15] and high nibbles [16..31]
        // Each byte stores: low_value at index i (lower 4 bits) | high_value at index i+16 (upper 4 bits)
        for i in 0..16 {
            let lo = (values[i] + 8) as u8 & 0x0F;
            let hi = (values[i + 16] + 8) as u8 & 0x0F;
            block[2 + i] = lo | (hi << 4);
        }

        block
    }

    /// CPU reference: dequantize Q4_0 block and compute dot product with input slice.
    fn cpu_q4_0_dot(weight_bytes: &[u8], input: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
        let n_blocks_per_row = in_dim / Q4_0_BLOCK_SIZE;
        let mut output = vec![0.0f32; out_dim];

        for row in 0..out_dim {
            let mut sum = 0.0f32;
            for b in 0..n_blocks_per_row {
                let block_offset = (row * n_blocks_per_row + b) * Q4_0_BYTES_PER_BLOCK;
                let scale_bytes = [weight_bytes[block_offset], weight_bytes[block_offset + 1]];
                let scale = half::f16::from_le_bytes(scale_bytes).to_f32();
                let base = b * Q4_0_BLOCK_SIZE;

                for i in 0..16 {
                    let byte_val = weight_bytes[block_offset + 2 + i];
                    let lo = ((byte_val & 0x0F) as i8 - 8) as f32 * scale;
                    let hi = (((byte_val >> 4) & 0x0F) as i8 - 8) as f32 * scale;
                    sum += lo * input[base + i];
                    sum += hi * input[base + i + 16];
                }
            }
            output[row] = sum;
        }

        output
    }

    #[test]
    fn test_matvec_q4_0_basic() {
        let gpu = GpuDevice::new();
        let mut pso_cache = PsoCache::new(gpu.library.clone());

        // Create a simple 2x32 weight matrix in Q4_0 format
        // Each row = 1 block (32 elements)
        // Q4_0 nibble range is 0-15, dequantized as (nibble - 8) * scale => values in [-8, 7]
        let out_dim = 2;
        let in_dim = 32;

        // Row 0: scale=1.0, values = [-8,-7,...,-1, 0,1,...,7] (full range)
        let mut values_0 = [0i8; 32];
        for i in 0..16 {
            values_0[i] = (i as i8) - 8; // low nibbles: -8..7
            values_0[i + 16] = (i as i8) - 8; // high nibbles: -8..7
        }
        let block_0 = encode_q4_0_block(1.0, &values_0);

        // Row 1: scale=2.0, all values = 3
        let values_1 = [3i8; 32];
        let block_1 = encode_q4_0_block(2.0, &values_1);

        let mut weight_bytes = Vec::new();
        weight_bytes.extend_from_slice(&block_0);
        weight_bytes.extend_from_slice(&block_1);

        // Input: all ones
        let input = vec![1.0f32; in_dim];

        // CPU reference
        let expected = cpu_q4_0_dot(&weight_bytes, &input, out_dim, in_dim);

        // GPU dispatch
        let result =
            dispatch_matvec_q4_0(&gpu, &mut pso_cache, &weight_bytes, &input, out_dim, in_dim);

        assert_eq!(result.len(), out_dim);
        for i in 0..out_dim {
            let diff = (result[i] - expected[i]).abs();
            assert!(
                diff < 1e-3,
                "Row {}: GPU={}, CPU={}, diff={}",
                i,
                result[i],
                expected[i],
                diff
            );
        }

        // Sanity check row 0: scale=1.0, values [-8..7] repeated twice, input all 1s
        // sum(-8..7) = -8, total = 2 * (-8) * 1.0 = -16.0
        assert!(
            (result[0] - (-16.0)).abs() < 1e-3,
            "Row 0 expected -16.0, got {}",
            result[0]
        );

        // Sanity check row 1: scale=2.0, all values=3, input all 1s
        // total = 32 * 3 * 2.0 = 192.0
        assert!(
            (result[1] - 192.0).abs() < 1e-3,
            "Row 1 expected 192.0, got {}",
            result[1]
        );
    }

    #[test]
    fn test_matvec_q4_0_576x576() {
        let gpu = GpuDevice::new();
        let mut pso_cache = PsoCache::new(gpu.library.clone());

        let out_dim = 576;
        let in_dim = 576;
        let n_blocks_per_row = in_dim / Q4_0_BLOCK_SIZE; // 18 blocks per row

        // Generate random Q4_0 data using a simple deterministic pattern
        let mut weight_bytes =
            Vec::with_capacity(out_dim * n_blocks_per_row * Q4_0_BYTES_PER_BLOCK);
        for row in 0..out_dim {
            for b in 0..n_blocks_per_row {
                // Use a deterministic scale based on row and block index
                let scale = 0.01 * ((row * n_blocks_per_row + b) % 100 + 1) as f32;
                // Generate deterministic nibble values
                let mut values = [0i8; 32];
                for i in 0..32 {
                    // Values in [-7, 7] range (valid Q4_0 range)
                    values[i] = ((row + b + i) % 15) as i8 - 7;
                }
                let block = encode_q4_0_block(scale, &values);
                weight_bytes.extend_from_slice(&block);
            }
        }

        // Generate deterministic input
        let input: Vec<f32> = (0..in_dim).map(|i| 0.1 * ((i % 10) as f32 - 5.0)).collect();

        // CPU reference
        let expected = cpu_q4_0_dot(&weight_bytes, &input, out_dim, in_dim);

        // GPU dispatch
        let result =
            dispatch_matvec_q4_0(&gpu, &mut pso_cache, &weight_bytes, &input, out_dim, in_dim);

        assert_eq!(result.len(), out_dim);
        let mut max_diff = 0.0f32;
        for i in 0..out_dim {
            let diff = (result[i] - expected[i]).abs();
            max_diff = max_diff.max(diff);
            assert!(
                diff < 1e-2,
                "Row {}: GPU={}, CPU={}, diff={}",
                i,
                result[i],
                expected[i],
                diff
            );
        }
        eprintln!(
            "test_matvec_q4_0_576x576: max_diff={:.6}, all {} rows within tolerance",
            max_diff, out_dim
        );
    }

    #[test]
    fn test_matvec_q4_0_1536x576() {
        // Gate/up projection dimensions (ffn intermediate size)
        let gpu = GpuDevice::new();
        let mut pso_cache = PsoCache::new(gpu.library.clone());

        let out_dim = 1536;
        let in_dim = 576;
        let n_blocks_per_row = in_dim / Q4_0_BLOCK_SIZE; // 18

        let mut weight_bytes =
            Vec::with_capacity(out_dim * n_blocks_per_row * Q4_0_BYTES_PER_BLOCK);
        for row in 0..out_dim {
            for b in 0..n_blocks_per_row {
                let scale = 0.01 * ((row * n_blocks_per_row + b) % 100 + 1) as f32;
                let mut values = [0i8; 32];
                for i in 0..32 {
                    values[i] = ((row + b + i) % 15) as i8 - 7;
                }
                let block = encode_q4_0_block(scale, &values);
                weight_bytes.extend_from_slice(&block);
            }
        }

        let input: Vec<f32> = (0..in_dim).map(|i| 0.1 * ((i % 10) as f32 - 5.0)).collect();

        let expected = cpu_q4_0_dot(&weight_bytes, &input, out_dim, in_dim);
        let result =
            dispatch_matvec_q4_0(&gpu, &mut pso_cache, &weight_bytes, &input, out_dim, in_dim);

        assert_eq!(result.len(), out_dim);
        let mut max_diff = 0.0f32;
        for i in 0..out_dim {
            let diff = (result[i] - expected[i]).abs();
            max_diff = max_diff.max(diff);
            assert!(
                diff < 1e-2,
                "Row {}: GPU={}, CPU={}, diff={}",
                i,
                result[i],
                expected[i],
                diff
            );
        }
        eprintln!(
            "test_matvec_q4_0_1536x576: max_diff={:.6}, all {} rows within tolerance",
            max_diff, out_dim
        );
    }

    #[test]
    fn test_matvec_q4_0_49152x576() {
        // lm_head dimensions (largest matvec: vocab_size x hidden_dim)
        // Use all-ones pattern for determinism and speed
        let gpu = GpuDevice::new();
        let mut pso_cache = PsoCache::new(gpu.library.clone());

        let out_dim = 49152;
        let in_dim = 576;
        let n_blocks_per_row = in_dim / Q4_0_BLOCK_SIZE; // 18

        // All blocks: scale=1.0, all values=1 => dequantized value = 1.0
        // So each row dot with input = sum(input) * 1.0
        let block_ones = encode_q4_0_block(1.0, &[1i8; 32]);

        let mut weight_bytes =
            Vec::with_capacity(out_dim * n_blocks_per_row * Q4_0_BYTES_PER_BLOCK);
        for _row in 0..out_dim {
            for _b in 0..n_blocks_per_row {
                weight_bytes.extend_from_slice(&block_ones);
            }
        }

        // Input: all 1.0
        let input = vec![1.0f32; in_dim];

        // Expected: each row = 576 * 1.0 * 1.0 = 576.0
        let expected_val = in_dim as f32;

        let result =
            dispatch_matvec_q4_0(&gpu, &mut pso_cache, &weight_bytes, &input, out_dim, in_dim);

        assert_eq!(result.len(), out_dim);
        let mut max_diff = 0.0f32;
        for i in 0..out_dim {
            let diff = (result[i] - expected_val).abs();
            max_diff = max_diff.max(diff);
            assert!(
                diff < 1e-3,
                "Row {}: GPU={}, expected={}, diff={}",
                i,
                result[i],
                expected_val,
                diff
            );
        }
        eprintln!(
            "test_matvec_q4_0_49152x576: max_diff={:.6}, all {} rows within tolerance",
            max_diff, out_dim
        );
    }

    #[test]
    fn test_matvec_q4_0_192x576() {
        // K/V projection dimensions (kv_heads=3, head_dim=64 => 192)
        let gpu = GpuDevice::new();
        let mut pso_cache = PsoCache::new(gpu.library.clone());

        let out_dim = 192;
        let in_dim = 576;
        let n_blocks_per_row = in_dim / Q4_0_BLOCK_SIZE; // 18

        let mut weight_bytes =
            Vec::with_capacity(out_dim * n_blocks_per_row * Q4_0_BYTES_PER_BLOCK);
        for row in 0..out_dim {
            for b in 0..n_blocks_per_row {
                let scale = 0.02 * ((row * n_blocks_per_row + b) % 50 + 1) as f32;
                let mut values = [0i8; 32];
                for i in 0..32 {
                    values[i] = ((row + b * 3 + i) % 15) as i8 - 7;
                }
                let block = encode_q4_0_block(scale, &values);
                weight_bytes.extend_from_slice(&block);
            }
        }

        let input: Vec<f32> = (0..in_dim)
            .map(|i| 0.05 * ((i % 20) as f32 - 10.0))
            .collect();

        let expected = cpu_q4_0_dot(&weight_bytes, &input, out_dim, in_dim);
        let result =
            dispatch_matvec_q4_0(&gpu, &mut pso_cache, &weight_bytes, &input, out_dim, in_dim);

        assert_eq!(result.len(), out_dim);
        let mut max_diff = 0.0f32;
        for i in 0..out_dim {
            let diff = (result[i] - expected[i]).abs();
            max_diff = max_diff.max(diff);
            assert!(
                diff < 1e-2,
                "Row {}: GPU={}, CPU={}, diff={}",
                i,
                result[i],
                expected[i],
                diff
            );
        }
        eprintln!(
            "test_matvec_q4_0_192x576: max_diff={:.6}, all {} rows within tolerance",
            max_diff, out_dim
        );
    }

    #[test]
    fn test_matvec_q4_0_zero_scale() {
        // Zero scale should produce zero output regardless of nibble values
        let gpu = GpuDevice::new();
        let mut pso_cache = PsoCache::new(gpu.library.clone());

        let out_dim = 4;
        let in_dim = 32;

        // All blocks have scale=0.0, random nibble values
        let mut weight_bytes = Vec::new();
        for row in 0..out_dim {
            let mut values = [0i8; 32];
            for i in 0..32 {
                values[i] = ((row + i) % 15) as i8 - 7;
            }
            let block = encode_q4_0_block(0.0, &values);
            weight_bytes.extend_from_slice(&block);
        }

        // Non-zero input
        let input: Vec<f32> = (0..in_dim).map(|i| (i as f32 + 1.0) * 0.5).collect();

        let result =
            dispatch_matvec_q4_0(&gpu, &mut pso_cache, &weight_bytes, &input, out_dim, in_dim);

        assert_eq!(result.len(), out_dim);
        for i in 0..out_dim {
            assert!(
                result[i].abs() < 1e-6,
                "Row {}: expected ~0.0 with zero scale, got {}",
                i,
                result[i]
            );
        }
    }
}
