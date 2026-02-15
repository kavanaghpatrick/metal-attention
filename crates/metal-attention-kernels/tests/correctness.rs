//! GPU vs CPU reference correctness tests for all Metal kernels.
//!
//! Runs GPU kernels and compares output against FP64 CPU reference implementations.
//! Must be run with --test-threads=1 since Metal tests can't safely run in parallel.
//!
//! Run: MTL_SHADER_VALIDATION=1 cargo test --test correctness -- --test-threads=1

use metal_attention_kernels::dequant::dispatch_dequantize_q4_0;
use metal_attention_kernels::device::GpuDevice;
use metal_attention_kernels::embed::dispatch_embedding_lookup;
use metal_attention_kernels::ffn::dispatch_ffn_silu;
use metal_attention_kernels::flash::dispatch_flash_attention;
use metal_attention_kernels::linear::dispatch_linear_attention;
use metal_attention_kernels::matmul::dispatch_matmul;
use metal_attention_kernels::norm::dispatch_rmsnorm;
use metal_attention_kernels::pipeline::PsoCache;

// ---------------------------------------------------------------------------
// CPU reference implementations (ported from proto)
// ---------------------------------------------------------------------------

/// Naive scaled dot-product attention computed entirely in FP64.
///
/// Computes: softmax(Q * K^T / sqrt(head_dim)) * V
///
/// All intermediate values use FP64 for maximum precision.
/// The final output is truncated to FP32.
fn cpu_attention_f64(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    assert_eq!(q.len(), seq_len * head_dim, "Q length mismatch");
    assert_eq!(k.len(), seq_len * head_dim, "K length mismatch");
    assert_eq!(v.len(), seq_len * head_dim, "V length mismatch");

    let scale = 1.0 / (head_dim as f64).sqrt();
    let mut output = vec![0.0f64; seq_len * head_dim];

    for i in 0..seq_len {
        // Compute attention scores: q_i * k_j^T * scale
        let mut scores = vec![0.0f64; seq_len];
        let mut max_score = f64::NEG_INFINITY;

        for j in 0..seq_len {
            let mut dot = 0.0f64;
            for d in 0..head_dim {
                dot += q[i * head_dim + d] as f64 * k[j * head_dim + d] as f64;
            }
            scores[j] = dot * scale;
            max_score = max_score.max(scores[j]);
        }

        // Safe softmax: subtract max for numerical stability, then exp and normalize
        let mut sum_exp = 0.0f64;
        for j in 0..seq_len {
            scores[j] = (scores[j] - max_score).exp();
            sum_exp += scores[j];
        }

        // Weighted sum of values
        for j in 0..seq_len {
            let weight = scores[j] / sum_exp;
            for d in 0..head_dim {
                output[i * head_dim + d] += weight * v[j * head_dim + d] as f64;
            }
        }
    }

    // Truncate FP64 accumulation to FP32 output
    output.iter().map(|&x| x as f32).collect()
}

/// Chunk-based linear attention computed entirely in FP64.
///
/// Implements the recurrence:
///   H_0 = 0 (D x D zero matrix)
///   For each chunk c of chunk_size tokens:
///     H_c = H_{c-1} + sum_{t in chunk} K[t]^T * V[t]  (outer product accumulation)
///     O_chunk = Q_chunk * H_c                          (matrix-vector products)
fn cpu_linear_attention_f64(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    chunk_size: usize,
) -> Vec<f32> {
    assert_eq!(q.len(), seq_len * head_dim, "Q length mismatch");
    assert_eq!(k.len(), seq_len * head_dim, "K length mismatch");
    assert_eq!(v.len(), seq_len * head_dim, "V length mismatch");
    assert!(
        seq_len % chunk_size == 0,
        "seq_len ({seq_len}) must be divisible by chunk_size ({chunk_size})"
    );

    let num_chunks = seq_len / chunk_size;

    // H: D x D hidden state matrix, accumulated across chunks
    let mut h = vec![0.0f64; head_dim * head_dim];

    // Output: seq_len x head_dim
    let mut output = vec![0.0f64; seq_len * head_dim];

    for c in 0..num_chunks {
        let start = c * chunk_size;

        // Update H: H += sum_{t in chunk} K[t]^T * V[t] (outer product accumulation)
        for t in 0..chunk_size {
            let token_idx = start + t;
            for i in 0..head_dim {
                let k_val = k[token_idx * head_dim + i] as f64;
                for j in 0..head_dim {
                    let v_val = v[token_idx * head_dim + j] as f64;
                    h[i * head_dim + j] += k_val * v_val;
                }
            }
        }

        // Compute output: O_chunk = Q_chunk * H
        for t in 0..chunk_size {
            let token_idx = start + t;
            for j in 0..head_dim {
                let mut sum = 0.0f64;
                for i in 0..head_dim {
                    sum += q[token_idx * head_dim + i] as f64 * h[i * head_dim + j];
                }
                output[token_idx * head_dim + j] = sum;
            }
        }
    }

    // Truncate FP64 accumulation to FP32 output
    output.iter().map(|&x| x as f32).collect()
}

// ---------------------------------------------------------------------------
// Test helper
// ---------------------------------------------------------------------------

/// Assert that two slices are element-wise close within absolute and relative tolerance.
///
/// An element passes if: |gpu - cpu| <= atol  OR  |gpu - cpu| / |cpu| <= rtol
fn assert_allclose(gpu: &[f32], cpu: &[f32], atol: f32, rtol: f32, context: &str) {
    assert_eq!(gpu.len(), cpu.len(), "{context}: length mismatch");

    let mut max_abs_err = 0.0f32;
    let mut max_rel_err = 0.0f32;
    let mut fail_count = 0;

    for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        let abs_err = (g - c).abs();
        let rel_err = if c.abs() > 1e-8 {
            abs_err / c.abs()
        } else {
            abs_err
        };
        max_abs_err = max_abs_err.max(abs_err);
        max_rel_err = max_rel_err.max(rel_err);

        if abs_err > atol && rel_err > rtol {
            fail_count += 1;
            if fail_count <= 5 {
                eprintln!(
                    "{context}[{i}]: gpu={g:.6}, cpu={c:.6}, abs_err={abs_err:.2e}, rel_err={rel_err:.2e}"
                );
            }
        }
    }

    if fail_count > 0 {
        panic!(
            "{context}: {fail_count}/{} elements exceed tolerance (max_abs={max_abs_err:.2e}, max_rel={max_rel_err:.2e})",
            gpu.len()
        );
    }

    eprintln!(
        "{context}: PASS (max_abs={max_abs_err:.2e}, max_rel={max_rel_err:.2e})"
    );
}

/// Generate deterministic test data using simple trig formula.
fn gen_data(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|i| (i as f32 * 0.1 + seed).sin() * 0.5)
        .collect()
}

// ---------------------------------------------------------------------------
// GPU correctness tests
// ---------------------------------------------------------------------------

#[test]
fn test_flash_attention_gpu_vs_cpu_small() {
    let seq_len = 64;
    let head_dim = 64;
    let num_heads = 1;

    let q = gen_data(seq_len * head_dim, 0.0);
    let k = gen_data(seq_len * head_dim, 1.0);
    let v = gen_data(seq_len * head_dim, 2.0);

    let cpu_out = cpu_attention_f64(&q, &k, &v, seq_len, head_dim);

    let device = GpuDevice::new();
    let mut pso_cache = PsoCache::new(device.library.clone());
    let gpu_out = dispatch_flash_attention(
        &device,
        &mut pso_cache,
        &q,
        &k,
        &v,
        seq_len,
        head_dim,
        num_heads,
    );

    assert_allclose(
        &gpu_out,
        &cpu_out,
        5e-3,
        1e-2,
        "flash_attention N=64 D=64",
    );
}

#[test]
fn test_flash_attention_gpu_vs_cpu_medium() {
    let seq_len = 256;
    let head_dim = 64;
    let num_heads = 1;

    let q = gen_data(seq_len * head_dim, 3.0);
    let k = gen_data(seq_len * head_dim, 4.0);
    let v = gen_data(seq_len * head_dim, 5.0);

    let cpu_out = cpu_attention_f64(&q, &k, &v, seq_len, head_dim);

    let device = GpuDevice::new();
    let mut pso_cache = PsoCache::new(device.library.clone());
    let gpu_out = dispatch_flash_attention(
        &device,
        &mut pso_cache,
        &q,
        &k,
        &v,
        seq_len,
        head_dim,
        num_heads,
    );

    assert_allclose(
        &gpu_out,
        &cpu_out,
        5e-3,
        1e-2,
        "flash_attention N=256 D=64",
    );
}

#[test]
fn test_linear_attention_gpu_vs_cpu_small() {
    let seq_len = 64;
    let head_dim = 64;
    let chunk_size = 32;

    let q = gen_data(seq_len * head_dim, 0.0);
    let k = gen_data(seq_len * head_dim, 1.0);
    let v = gen_data(seq_len * head_dim, 2.0);

    let cpu_out = cpu_linear_attention_f64(&q, &k, &v, seq_len, head_dim, chunk_size);

    let device = GpuDevice::new();
    let mut pso_cache = PsoCache::new(device.library.clone());
    let gpu_out = dispatch_linear_attention(
        &device,
        &mut pso_cache,
        &q,
        &k,
        &v,
        seq_len,
        head_dim,
        chunk_size,
    );

    assert_allclose(
        &gpu_out,
        &cpu_out,
        1e-3,
        1e-2,
        "linear_attention N=64 D=64 C=32",
    );
}

#[test]
fn test_linear_attention_gpu_vs_cpu_medium() {
    let seq_len = 256;
    let head_dim = 64;
    let chunk_size = 32;

    let q = gen_data(seq_len * head_dim, 3.0);
    let k = gen_data(seq_len * head_dim, 4.0);
    let v = gen_data(seq_len * head_dim, 5.0);

    let cpu_out = cpu_linear_attention_f64(&q, &k, &v, seq_len, head_dim, chunk_size);

    let device = GpuDevice::new();
    let mut pso_cache = PsoCache::new(device.library.clone());
    let gpu_out = dispatch_linear_attention(
        &device,
        &mut pso_cache,
        &q,
        &k,
        &v,
        seq_len,
        head_dim,
        chunk_size,
    );

    assert_allclose(
        &gpu_out,
        &cpu_out,
        1e-3,
        1e-2,
        "linear_attention N=256 D=64 C=32",
    );
}

// ---------------------------------------------------------------------------
// CPU reference unit tests
// ---------------------------------------------------------------------------

#[test]
fn test_cpu_attention_single_token() {
    let q = vec![1.0f32, 0.0, 0.0, 0.0];
    let k = vec![0.5f32, 0.5, 0.5, 0.5];
    let v = vec![3.0f32, 7.0, 11.0, 13.0];

    let output = cpu_attention_f64(&q, &k, &v, 1, 4);
    // With seq_len=1, softmax of a single score is always 1.0, so output = v
    assert_allclose(&output, &v, 1e-6, 1e-5, "single token");
}

#[test]
fn test_cpu_linear_attention_identity() {
    let head_dim = 4;
    let chunk_size = 4;
    let seq_len = 4;

    // Q = K = identity rows
    let mut q = vec![0.0f32; seq_len * head_dim];
    let mut k = vec![0.0f32; seq_len * head_dim];
    for i in 0..seq_len {
        q[i * head_dim + i] = 1.0;
        k[i * head_dim + i] = 1.0;
    }

    let v: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];

    let output = cpu_linear_attention_f64(&q, &k, &v, seq_len, head_dim, chunk_size);
    // With Q=I, K=I, single chunk: output = V
    assert_allclose(&output, &v, 1e-6, 1e-5, "linear identity");
}

#[test]
fn test_assert_allclose_passes() {
    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![1.0f32, 2.0, 3.0, 4.0];
    assert_allclose(&a, &b, 1e-6, 1e-5, "exact match");
}

#[test]
#[should_panic(expected = "elements exceed tolerance")]
fn test_assert_allclose_fails() {
    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![1.0f32, 2.0, 3.0, 5.0];
    assert_allclose(&a, &b, 1e-6, 1e-5, "mismatch");
}

// ---------------------------------------------------------------------------
// CPU references for new kernels
// ---------------------------------------------------------------------------

/// CPU reference for RMSNorm.
///
/// output[i] = (input[i] / rms) * weight[i]
/// where rms = sqrt(mean(input^2) + eps)
fn cpu_rmsnorm(input: &[f32], weight: &[f32], num_tokens: usize, hidden_dim: usize, eps: f32) -> Vec<f32> {
    let mut output = vec![0.0f32; num_tokens * hidden_dim];
    for t in 0..num_tokens {
        let offset = t * hidden_dim;
        // Compute sum of squares
        let mut ss = 0.0f64;
        for d in 0..hidden_dim {
            let v = input[offset + d] as f64;
            ss += v * v;
        }
        let rms = ((ss / hidden_dim as f64) + eps as f64).sqrt();
        for d in 0..hidden_dim {
            output[offset + d] = ((input[offset + d] as f64 / rms) * weight[d] as f64) as f32;
        }
    }
    output
}

/// CPU reference for SwiGLU activation: silu(gate) * up.
fn cpu_ffn_silu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up.iter())
        .map(|(&g, &u)| {
            let silu_g = g / (1.0 + (-g).exp()); // silu(x) = x * sigmoid(x)
            silu_g * u
        })
        .collect()
}

/// CPU reference for embedding lookup.
fn cpu_embedding_lookup(table: &[f32], token_ids: &[u32], hidden_dim: usize) -> Vec<f32> {
    let mut output = Vec::with_capacity(token_ids.len() * hidden_dim);
    for &tid in token_ids {
        let start = tid as usize * hidden_dim;
        output.extend_from_slice(&table[start..start + hidden_dim]);
    }
    output
}

/// CPU reference for matrix multiplication: C = A * B.
fn cpu_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f64;
            for i in 0..k {
                acc += a[row * k + i] as f64 * b[i * n + col] as f64;
            }
            c[row * n + col] = acc as f32;
        }
    }
    c
}

/// CPU reference for Q4_0 dequantization.
fn cpu_dequantize_q4_0(input: &[u8], num_blocks: usize) -> Vec<f32> {
    let mut output = Vec::with_capacity(num_blocks * 32);
    for block in 0..num_blocks {
        let block_offset = block * 18;
        // Read scale as f16 (stored as 2 little-endian bytes)
        let scale_bits = u16::from_le_bytes([input[block_offset], input[block_offset + 1]]);
        let scale = half::f16::from_bits(scale_bits).to_f32();
        let quants = &input[block_offset + 2..block_offset + 18];
        for i in 0..16 {
            let byte_val = quants[i];
            let lo = ((byte_val & 0x0F) as i32 - 8) as f32 * scale;
            let hi = (((byte_val >> 4) & 0x0F) as i32 - 8) as f32 * scale;
            output.push(lo);
            output.push(hi);
        }
    }
    output
}

// ---------------------------------------------------------------------------
// GPU correctness tests: RMSNorm
// ---------------------------------------------------------------------------

#[test]
fn test_rmsnorm_gpu_vs_cpu() {
    let num_tokens = 4;
    let hidden_dim = 64;
    let eps = 1e-5f32;

    let input = gen_data(num_tokens * hidden_dim, 0.0);
    let weight: Vec<f32> = (0..hidden_dim).map(|i| 0.5 + (i as f32 * 0.01).sin() * 0.3).collect();

    let cpu_out = cpu_rmsnorm(&input, &weight, num_tokens, hidden_dim, eps);

    let device = GpuDevice::new();
    let mut pso_cache = PsoCache::new(device.library.clone());
    let gpu_out = dispatch_rmsnorm(&device, &mut pso_cache, &input, &weight, num_tokens, hidden_dim, eps);

    assert_allclose(&gpu_out, &cpu_out, 1e-4, 1e-3, "rmsnorm 4x64");
}

#[test]
fn test_rmsnorm_gpu_vs_cpu_single_token() {
    let num_tokens = 1;
    let hidden_dim = 16;
    let eps = 1e-5f32;

    let input = vec![1.0f32; hidden_dim];
    let weight = vec![2.0f32; hidden_dim];

    let cpu_out = cpu_rmsnorm(&input, &weight, num_tokens, hidden_dim, eps);

    let device = GpuDevice::new();
    let mut pso_cache = PsoCache::new(device.library.clone());
    let gpu_out = dispatch_rmsnorm(&device, &mut pso_cache, &input, &weight, num_tokens, hidden_dim, eps);

    // RMS of all-1s vector with dim=16 is sqrt(1 + eps) ~= 1.0
    // So output should be ~2.0 for each element
    assert_allclose(&gpu_out, &cpu_out, 1e-5, 1e-4, "rmsnorm 1x16 all-ones");
}

// ---------------------------------------------------------------------------
// GPU correctness tests: FFN SwiGLU
// ---------------------------------------------------------------------------

#[test]
fn test_ffn_silu_gpu_vs_cpu() {
    let num_tokens = 4;
    let intermediate_dim = 64;
    let total = num_tokens * intermediate_dim;

    let gate = gen_data(total, 0.0);
    let up = gen_data(total, 1.0);

    let cpu_out = cpu_ffn_silu(&gate, &up);

    let device = GpuDevice::new();
    let mut pso_cache = PsoCache::new(device.library.clone());
    let gpu_out = dispatch_ffn_silu(&device, &mut pso_cache, &gate, &up, num_tokens, intermediate_dim);

    assert_allclose(&gpu_out, &cpu_out, 1e-5, 1e-4, "ffn_silu 4x64");
}

#[test]
fn test_ffn_silu_zeros() {
    let num_tokens = 1;
    let intermediate_dim = 8;

    let gate = vec![0.0f32; 8]; // silu(0) = 0
    let up = vec![1.0f32; 8];

    let cpu_out = cpu_ffn_silu(&gate, &up);

    let device = GpuDevice::new();
    let mut pso_cache = PsoCache::new(device.library.clone());
    let gpu_out = dispatch_ffn_silu(&device, &mut pso_cache, &gate, &up, num_tokens, intermediate_dim);

    // silu(0) * 1.0 = 0.0
    assert_allclose(&gpu_out, &cpu_out, 1e-6, 1e-5, "ffn_silu zeros");
}

// ---------------------------------------------------------------------------
// GPU correctness tests: Embedding lookup
// ---------------------------------------------------------------------------

#[test]
fn test_embedding_lookup_gpu_vs_cpu() {
    let vocab_size = 10;
    let hidden_dim = 8;
    let seq_len = 4;

    // Build a small embedding table
    let table: Vec<f32> = (0..vocab_size * hidden_dim)
        .map(|i| (i as f32 * 0.1).sin())
        .collect();
    let token_ids = vec![0u32, 3, 7, 1];

    let cpu_out = cpu_embedding_lookup(&table, &token_ids, hidden_dim);

    let device = GpuDevice::new();
    let mut pso_cache = PsoCache::new(device.library.clone());
    let gpu_out = dispatch_embedding_lookup(
        &device,
        &mut pso_cache,
        &table,
        &token_ids,
        seq_len,
        hidden_dim,
    );

    assert_allclose(&gpu_out, &cpu_out, 1e-6, 1e-5, "embedding_lookup 4 tokens");
}

#[test]
fn test_embedding_lookup_single_token() {
    let vocab_size = 5;
    let hidden_dim = 4;

    let table: Vec<f32> = (0..vocab_size * hidden_dim).map(|i| i as f32).collect();
    let token_ids = vec![2u32];

    let cpu_out = cpu_embedding_lookup(&table, &token_ids, hidden_dim);
    // Token 2 -> elements [8, 9, 10, 11]
    assert_eq!(cpu_out, vec![8.0, 9.0, 10.0, 11.0]);

    let device = GpuDevice::new();
    let mut pso_cache = PsoCache::new(device.library.clone());
    let gpu_out = dispatch_embedding_lookup(
        &device,
        &mut pso_cache,
        &table,
        &token_ids,
        1,
        hidden_dim,
    );

    assert_allclose(&gpu_out, &cpu_out, 1e-6, 1e-5, "embedding_lookup single token");
}

// ---------------------------------------------------------------------------
// GPU correctness tests: Matrix multiplication
// ---------------------------------------------------------------------------

#[test]
fn test_matmul_gpu_vs_cpu() {
    let m = 8;
    let n = 8;
    let k = 16;

    let a = gen_data(m * k, 0.0);
    let b = gen_data(k * n, 1.0);

    let cpu_out = cpu_matmul(&a, &b, m, n, k);

    let device = GpuDevice::new();
    let mut pso_cache = PsoCache::new(device.library.clone());
    let gpu_out = dispatch_matmul(&device, &mut pso_cache, &a, &b, m, n, k);

    assert_allclose(&gpu_out, &cpu_out, 1e-4, 1e-3, "matmul 8x16 * 16x8");
}

#[test]
fn test_matmul_identity() {
    let n = 4;

    // A = I (4x4 identity)
    let mut a = vec![0.0f32; n * n];
    for i in 0..n {
        a[i * n + i] = 1.0;
    }
    // B = arbitrary
    let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];

    let cpu_out = cpu_matmul(&a, &b, n, n, n);
    // I * B = B
    assert_allclose(&cpu_out, &b, 1e-6, 1e-5, "cpu matmul identity");

    let device = GpuDevice::new();
    let mut pso_cache = PsoCache::new(device.library.clone());
    let gpu_out = dispatch_matmul(&device, &mut pso_cache, &a, &b, n, n, n);

    assert_allclose(&gpu_out, &b, 1e-5, 1e-4, "gpu matmul identity");
}

#[test]
fn test_matmul_non_square() {
    let m = 3;
    let n = 5;
    let k = 4;

    let a = gen_data(m * k, 2.0);
    let b = gen_data(k * n, 3.0);

    let cpu_out = cpu_matmul(&a, &b, m, n, k);

    let device = GpuDevice::new();
    let mut pso_cache = PsoCache::new(device.library.clone());
    let gpu_out = dispatch_matmul(&device, &mut pso_cache, &a, &b, m, n, k);

    assert_allclose(&gpu_out, &cpu_out, 1e-4, 1e-3, "matmul 3x4 * 4x5");
}

// ---------------------------------------------------------------------------
// GPU correctness tests: Q4_0 Dequantization
// ---------------------------------------------------------------------------

/// Pack a Q4_0 block from a scale and 32 quantized values (pre-offset by +8).
fn pack_q4_0_block(scale: f32, values: &[i8; 32]) -> [u8; 18] {
    let mut block = [0u8; 18];
    // Write scale as f16
    let scale_f16 = half::f16::from_f32(scale);
    let scale_bytes = scale_f16.to_bits().to_le_bytes();
    block[0] = scale_bytes[0];
    block[1] = scale_bytes[1];
    // Pack nibbles (values are already in 0..15 range, representing val-8 = -8..+7)
    for i in 0..16 {
        let lo = (values[i * 2] + 8) as u8 & 0x0F;
        let hi = (values[i * 2 + 1] + 8) as u8 & 0x0F;
        block[2 + i] = lo | (hi << 4);
    }
    block
}

#[test]
fn test_dequantize_q4_0_gpu_vs_cpu() {
    // Create 2 blocks of Q4_0 data
    let scale1 = 0.5f32;
    let vals1: [i8; 32] = [
        -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7,
        -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7,
    ];
    let block1 = pack_q4_0_block(scale1, &vals1);

    let scale2 = 1.0f32;
    let vals2: [i8; 32] = [
        0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1, 7, 7, 7, 7,
        -8, -8, -8, -8, 3, 3, 3, 3, -5, -5, -5, -5, 2, 2, 2, 2,
    ];
    let block2 = pack_q4_0_block(scale2, &vals2);

    let mut input = Vec::with_capacity(36);
    input.extend_from_slice(&block1);
    input.extend_from_slice(&block2);

    let num_blocks = 2;
    let cpu_out = cpu_dequantize_q4_0(&input, num_blocks);

    let device = GpuDevice::new();
    let mut pso_cache = PsoCache::new(device.library.clone());
    let gpu_out = dispatch_dequantize_q4_0(&device, &mut pso_cache, &input, num_blocks);

    assert_allclose(&gpu_out, &cpu_out, 1e-3, 1e-2, "dequantize_q4_0 2 blocks");
}

#[test]
fn test_dequantize_q4_0_zero_scale() {
    // Zero scale should produce all zeros
    let vals: [i8; 32] = [
        -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7,
        -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7,
    ];
    let block = pack_q4_0_block(0.0, &vals);
    let input = block.to_vec();

    let cpu_out = cpu_dequantize_q4_0(&input, 1);
    // All should be 0.0 since scale is 0
    assert!(cpu_out.iter().all(|&v| v == 0.0), "CPU: zero scale should give zeros");

    let device = GpuDevice::new();
    let mut pso_cache = PsoCache::new(device.library.clone());
    let gpu_out = dispatch_dequantize_q4_0(&device, &mut pso_cache, &input, 1);

    assert_allclose(&gpu_out, &cpu_out, 1e-6, 1e-5, "dequantize_q4_0 zero scale");
}
