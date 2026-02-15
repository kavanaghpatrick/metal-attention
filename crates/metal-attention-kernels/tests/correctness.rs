//! GPU vs CPU reference correctness tests for flash and linear attention.
//!
//! Runs GPU kernels and compares output against FP64 CPU reference implementations.
//! Must be run with --test-threads=1 since Metal tests can't safely run in parallel.
//!
//! Run: MTL_SHADER_VALIDATION=1 cargo test --test correctness -- --test-threads=1

use metal_attention_kernels::device::GpuDevice;
use metal_attention_kernels::flash::dispatch_flash_attention;
use metal_attention_kernels::linear::dispatch_linear_attention;
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
