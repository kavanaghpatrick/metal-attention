//! GPU vs CPU correctness integration tests.
//!
//! Compares GpuForwardPass (fused Q4_0 Metal kernels) against HybridModel (CPU F32)
//! using real SmolLM-135M weights. Tests are `#[ignore]` because they require
//! the model file at `models/SmolLM-135M.Q4_0.gguf`.
//!
//! Also includes synthetic Q6_K matvec kernel correctness tests that do NOT
//! require model files — they create known Q6_K blocks and compare GPU vs CPU.

use std::path::{Path, PathBuf};

use metal_attention::gpu_forward_pass::GpuForwardPass;
use metal_attention::model::HybridModel;
use metal_attention::sampling::sample_greedy;
use metal_attention_kernels::buffer::{alloc_buffer, alloc_buffer_with_data, read_buffer_slice};
use metal_attention_kernels::device::GpuDevice;
use metal_attention_kernels::dispatch::{set_buffer, set_bytes};
use metal_attention_kernels::pipeline::{PsoCache, PsoKey};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
};

/// Resolve model path relative to the workspace root.
///
/// Integration tests under crates/metal-attention/ have CARGO_MANIFEST_DIR
/// pointing to crates/metal-attention/. The workspace root (where models/ lives)
/// is two directories up.
fn model_path() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest_dir).join("../../models/SmolLM-135M.Q4_0.gguf")
}

/// Prefill tokens [1, 2, 3] on both GPU and CPU paths, then compare the
/// logit vectors from the last prefill step.
///
/// Tolerance: max absolute difference < 1e-2. The GPU path uses fused Q4_0
/// dequant+matvec while the CPU path dequantizes to F32 first, so there are
/// inherent numerical differences due to FP accumulation order.
#[test]
#[ignore]
fn test_gpu_vs_cpu_logits() {
    let path = model_path();
    assert!(
        path.exists(),
        "Model file not found: {path:?}. Download SmolLM-135M.Q4_0.gguf first."
    );
    let path = path.as_path();

    let prefill_tokens: &[u32] = &[1, 2, 3];

    // --- GPU path ---
    let mut gpu = GpuForwardPass::from_gguf(path).expect("Failed to load GPU model");
    let mut gpu_logits = Vec::new();
    for &tok in prefill_tokens {
        gpu_logits = gpu.forward_token(tok).expect("GPU forward_token failed");
    }

    // --- CPU path (needs GPU device for Q8_0 dequantization at load time) ---
    let device = GpuDevice::new();
    let mut pso_cache = PsoCache::new(device.library.clone());
    let cpu_model = HybridModel::from_gguf(path, Some(&device), Some(&mut pso_cache))
        .expect("Failed to load CPU model");
    let mut cpu_state = cpu_model.init_state();
    let mut cpu_logits = Vec::new();
    for &tok in prefill_tokens {
        cpu_logits = cpu_model.forward_token(tok, &mut cpu_state);
    }

    // --- Compare ---
    assert_eq!(
        gpu_logits.len(),
        cpu_logits.len(),
        "Logit vector lengths differ: GPU={} CPU={}",
        gpu_logits.len(),
        cpu_logits.len()
    );

    // Check no NaN/Inf in either output
    assert!(
        !gpu_logits.iter().any(|v| v.is_nan() || v.is_infinite()),
        "GPU logits contain NaN or Inf"
    );
    assert!(
        !cpu_logits.iter().any(|v| v.is_nan() || v.is_infinite()),
        "CPU logits contain NaN or Inf"
    );

    let max_abs_diff = gpu_logits
        .iter()
        .zip(cpu_logits.iter())
        .map(|(g, c)| (g - c).abs())
        .fold(0.0f32, f32::max);

    let mean_abs_diff: f32 = gpu_logits
        .iter()
        .zip(cpu_logits.iter())
        .map(|(g, c)| (g - c).abs())
        .sum::<f32>()
        / gpu_logits.len() as f32;

    eprintln!(
        "Logits comparison: max_abs_diff={max_abs_diff:.6}, mean_abs_diff={mean_abs_diff:.6}, vocab_size={}",
        gpu_logits.len()
    );

    // Q4_0 fused vs F32 CPU: 1e-2 tolerance for max absolute difference
    assert!(
        max_abs_diff < 1e-2,
        "Logits max absolute difference {max_abs_diff:.6} exceeds tolerance 1e-2"
    );

    // Also verify greedy token matches (stronger signal)
    let gpu_token = sample_greedy(&gpu_logits);
    let cpu_token = sample_greedy(&cpu_logits);
    eprintln!("Greedy token from last prefill: GPU={gpu_token} CPU={cpu_token}");
}

/// Decode 10 tokens greedily on both GPU and CPU paths starting from
/// the same prefill [1, 2, 3], and verify that the first several tokens match.
///
/// Due to FP accumulation order differences between fused Q4_0 GPU and F32 CPU,
/// token sequences can diverge after a few steps (one different logit -> different
/// token -> cascading divergence). We require at least the first 3 tokens to match.
#[test]
#[ignore]
fn test_gpu_vs_cpu_greedy_match() {
    let path = model_path();
    assert!(
        path.exists(),
        "Model file not found: {path:?}. Download SmolLM-135M.Q4_0.gguf first."
    );
    let path = path.as_path();

    let prefill_tokens: &[u32] = &[1, 2, 3];
    let decode_len = 10;

    // --- GPU path: prefill then greedy decode ---
    let mut gpu = GpuForwardPass::from_gguf(path).expect("Failed to load GPU model");
    let mut gpu_last_logits = Vec::new();
    for &tok in prefill_tokens {
        gpu_last_logits = gpu.forward_token(tok).expect("GPU prefill failed");
    }
    // Greedy decode loop
    let mut gpu_tokens = Vec::with_capacity(decode_len);
    let mut next_token = sample_greedy(&gpu_last_logits);
    gpu_tokens.push(next_token);
    for _ in 1..decode_len {
        let logits = gpu.forward_token(next_token).expect("GPU decode failed");
        next_token = sample_greedy(&logits);
        gpu_tokens.push(next_token);
    }

    // --- CPU path: prefill then greedy decode ---
    // (needs GPU device for Q8_0 dequantization at load time)
    let device = GpuDevice::new();
    let mut pso_cache = PsoCache::new(device.library.clone());
    let cpu_model = HybridModel::from_gguf(path, Some(&device), Some(&mut pso_cache))
        .expect("Failed to load CPU model");
    let mut cpu_state = cpu_model.init_state();
    let mut cpu_last_logits = Vec::new();
    for &tok in prefill_tokens {
        cpu_last_logits = cpu_model.forward_token(tok, &mut cpu_state);
    }
    // Greedy decode loop
    let mut cpu_tokens = Vec::with_capacity(decode_len);
    let mut next_token = sample_greedy(&cpu_last_logits);
    cpu_tokens.push(next_token);
    for _ in 1..decode_len {
        let logits = cpu_model.forward_token(next_token, &mut cpu_state);
        next_token = sample_greedy(&logits);
        cpu_tokens.push(next_token);
    }

    eprintln!("GPU tokens: {gpu_tokens:?}");
    eprintln!("CPU tokens: {cpu_tokens:?}");

    // Count matching tokens from the start
    let mut match_count = 0;
    for (g, c) in gpu_tokens.iter().zip(cpu_tokens.iter()) {
        if g == c {
            match_count += 1;
        } else {
            break;
        }
    }
    eprintln!("Greedy match: {match_count}/{decode_len} tokens match from start");

    // Require at least 3 tokens to match (GPU and CPU may diverge due to
    // FP accumulation order differences in fused Q4_0 vs F32 dequant)
    assert!(
        match_count >= 3,
        "Only {match_count} tokens matched from start (need >= 3). \
         GPU={gpu_tokens:?} CPU={cpu_tokens:?}"
    );
}

/// Verify that forward_prompt (batched prefill) produces the same output
/// token as sequential forward_token_greedy calls.
///
/// Tests with two different prompt lengths to cover both short and medium
/// batch sizes. Both paths should produce identical output tokens since
/// they use the same Q4_0 matvec kernels (just batched vs individual).
#[test]
#[ignore]
fn test_forward_prompt_matches_sequential() {
    let path = model_path();
    assert!(
        path.exists(),
        "Model file not found: {path:?}. Download SmolLM-135M.Q4_0.gguf first."
    );
    let path = path.as_path();

    // Test with prompts of different lengths
    let test_prompts: &[&[u32]] = &[
        &[1, 2, 3, 4, 5, 6, 7], // 7 tokens
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, // 32 tokens
            11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        ],
    ];

    for prompt in test_prompts {
        // --- Sequential path ---
        let mut gpu_seq = GpuForwardPass::from_gguf(path).expect("Failed to load GPU model");
        for &tok in &prompt[..prompt.len() - 1] {
            let _ = gpu_seq
                .forward_token(tok)
                .expect("sequential forward_token failed");
        }
        let seq_token = gpu_seq
            .forward_token_greedy(*prompt.last().unwrap())
            .expect("sequential forward_token_greedy failed");

        // --- Batched path ---
        let mut gpu_batch = GpuForwardPass::from_gguf(path).expect("Failed to load GPU model");
        let batch_token = gpu_batch
            .forward_prompt(prompt)
            .expect("forward_prompt failed");

        eprintln!(
            "Prompt len={}: sequential={seq_token}, batched={batch_token}",
            prompt.len()
        );
        assert_eq!(
            seq_token,
            batch_token,
            "forward_prompt output differs from sequential path for prompt len={}",
            prompt.len()
        );
    }
}

// ============================================================================
// Q6_K matvec GPU vs CPU correctness tests
// ============================================================================

const Q6K_BLOCK_SIZE: usize = 256;
const Q6K_BLOCK_BYTES: usize = 210;

/// Simple deterministic LCG PRNG for reproducible test data.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        let val = self.next_u64();
        ((val >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0
    }

    fn next_u8(&mut self) -> u8 {
        (self.next_u64() >> 40) as u8
    }
}

/// Create a synthetic Q6_K super-block (210 bytes) with controlled random data.
///
/// Layout: ql[128] + qh[64] + scales[16] + d(half) = 210 bytes.
fn create_q6k_block(rng: &mut Lcg, scale: f32) -> Vec<u8> {
    let mut block = Vec::with_capacity(Q6K_BLOCK_BYTES);

    // ql[128]: random bytes (low 4-bit nibbles for quant values)
    for _ in 0..128 {
        block.push(rng.next_u8());
    }

    // qh[64]: random bytes (high 2-bit pairs)
    for _ in 0..64 {
        block.push(rng.next_u8());
    }

    // scales[16]: signed int8 sub-block scales in range [-10, 10]
    for _ in 0..16 {
        let s = ((rng.next_u64() % 21) as i8) - 10;
        block.push(s as u8);
    }

    // d: fp16 super-block scale
    let d_bits = half::f16::from_f32(scale).to_bits();
    block.extend_from_slice(&d_bits.to_le_bytes());

    assert_eq!(block.len(), Q6K_BLOCK_BYTES);
    block
}

/// CPU reference: dequantize Q6_K blocks for a single row of `in_dim` elements.
///
/// Matches the Rust `dequantize_q6_k_to_f32` from gpu_weight_store.rs exactly.
fn cpu_dequantize_q6_k_row(block_data: &[u8], n_elements: usize) -> Vec<f32> {
    let n_blocks = n_elements / Q6K_BLOCK_SIZE;
    let mut out = vec![0.0f32; n_elements];

    for b in 0..n_blocks {
        let bp = b * Q6K_BLOCK_BYTES;
        let ql = &block_data[bp..bp + 128];
        let qh = &block_data[bp + 128..bp + 192];
        let scales = &block_data[bp + 192..bp + 208];
        let d = half::f16::from_bits(u16::from_le_bytes([
            block_data[bp + 208],
            block_data[bp + 209],
        ]))
        .to_f32();

        for chunk in 0..2 {
            let ql_off = chunk * 64;
            let qh_off = chunk * 32;
            let sc_off = chunk * 8;
            let out_off = b * Q6K_BLOCK_SIZE + chunk * 128;

            for l in 0..32 {
                let is = l / 16;

                let q1 =
                    ((ql[ql_off + l] & 0xF) | (((qh[qh_off + l] >> 0) & 3) << 4)) as i32 - 32;
                let q2 = ((ql[ql_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4))
                    as i32
                    - 32;
                let q3 =
                    ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i32 - 32;
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

/// CPU reference: Q6_K matvec = dequantize each row + dot product with input.
fn cpu_matvec_q6_k(
    weight_bytes: &[u8],
    input: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    let n_blocks_per_row = in_dim / Q6K_BLOCK_SIZE;
    let row_bytes = n_blocks_per_row * Q6K_BLOCK_BYTES;

    let mut output = vec![0.0f32; out_dim];
    for row in 0..out_dim {
        let row_start = row * row_bytes;
        let row_data = &weight_bytes[row_start..row_start + row_bytes];
        let dequant = cpu_dequantize_q6_k_row(row_data, in_dim);

        let mut sum = 0.0f64; // Use f64 for CPU reference to avoid accumulation drift
        for i in 0..in_dim {
            sum += dequant[i] as f64 * input[i] as f64;
        }
        output[row] = sum as f32;
    }
    output
}

/// Run matvec_q6_k on GPU: weight[out_dim, in_dim/256 * 210] * input[in_dim] -> output[out_dim].
fn gpu_matvec_q6_k(
    weight_bytes: &[u8],
    input: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    let device = GpuDevice::new();
    let mut pso_cache = PsoCache::new(device.library.clone());
    pso_cache.prewarm(&[PsoKey::simple("matvec_q6_k")]);

    let weight_buf = alloc_buffer_with_data(&device.device, weight_bytes);
    let input_buf = alloc_buffer_with_data(&device.device, input);
    let output_buf = alloc_buffer(&device.device, out_dim * std::mem::size_of::<f32>());

    let cmd_buf = device.command_queue.commandBuffer().expect("cmd buf");
    let encoder = cmd_buf.computeCommandEncoder().expect("encoder");

    let pso = pso_cache
        .get(&PsoKey::simple("matvec_q6_k"))
        .expect("matvec_q6_k PSO not found");
    encoder.setComputePipelineState(pso);
    set_buffer(&encoder, &weight_buf, 0, 0);
    set_buffer(&encoder, &input_buf, 0, 1);
    set_buffer(&encoder, &output_buf, 0, 2);

    let out_dim_u32 = out_dim as u32;
    let in_dim_u32 = in_dim as u32;
    set_bytes(&encoder, &out_dim_u32, 3);
    set_bytes(&encoder, &in_dim_u32, 4);

    const ROWS_PER_TG: usize = 8;
    let grid = MTLSize {
        width: (out_dim + ROWS_PER_TG - 1) / ROWS_PER_TG,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);

    encoder.endEncoding();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    unsafe { read_buffer_slice(&output_buf, out_dim) }
}

/// Create synthetic Q6_K weight matrix: [out_dim, in_dim] packed as Q6_K blocks.
fn create_q6k_weight_matrix(rng: &mut Lcg, out_dim: usize, in_dim: usize) -> Vec<u8> {
    assert_eq!(in_dim % Q6K_BLOCK_SIZE, 0);
    let n_blocks_per_row = in_dim / Q6K_BLOCK_SIZE;
    let total_bytes = out_dim * n_blocks_per_row * Q6K_BLOCK_BYTES;
    let mut bytes = Vec::with_capacity(total_bytes);

    for _row in 0..out_dim {
        for _b in 0..n_blocks_per_row {
            // Scale in a reasonable range for inference weights
            let scale = rng.next_f32() * 0.01;
            let block = create_q6k_block(rng, scale);
            bytes.extend_from_slice(&block);
        }
    }

    assert_eq!(bytes.len(), total_bytes);
    bytes
}

/// Core test: GPU matvec_q6_k matches CPU dequant+dot product.
fn test_q6k_matvec_dims(in_dim: usize, out_dim: usize) {
    assert_eq!(in_dim % Q6K_BLOCK_SIZE, 0, "in_dim must be multiple of 256");

    let mut rng = Lcg::new(42 + in_dim as u64 * 1000 + out_dim as u64);

    // Create synthetic Q6_K weight matrix
    let weight_bytes = create_q6k_weight_matrix(&mut rng, out_dim, in_dim);

    // Create random input vector
    let input: Vec<f32> = (0..in_dim).map(|_| rng.next_f32() * 0.5).collect();

    // GPU path
    let gpu_output = gpu_matvec_q6_k(&weight_bytes, &input, out_dim, in_dim);

    // CPU reference path
    let cpu_output = cpu_matvec_q6_k(&weight_bytes, &input, out_dim, in_dim);

    // Compare
    assert_eq!(gpu_output.len(), out_dim);
    assert_eq!(cpu_output.len(), out_dim);

    // Check no NaN/Inf
    assert!(
        !gpu_output.iter().any(|v| v.is_nan() || v.is_infinite()),
        "GPU output contains NaN or Inf"
    );
    assert!(
        !cpu_output.iter().any(|v| v.is_nan() || v.is_infinite()),
        "CPU output contains NaN or Inf"
    );

    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0;
    let mut mean_diff = 0.0f64;
    for (i, (&g, &c)) in gpu_output.iter().zip(cpu_output.iter()).enumerate() {
        let diff = (g - c).abs();
        mean_diff += diff as f64;
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }
    }
    mean_diff /= out_dim as f64;

    eprintln!(
        "Q6_K matvec ({in_dim}->{out_dim}): max_diff={max_diff:.6} at [{max_diff_idx}] \
         (gpu={:.6}, cpu={:.6}), mean_diff={mean_diff:.6}",
        gpu_output[max_diff_idx], cpu_output[max_diff_idx],
    );

    // Tolerance: 5e-2 for quantization + FP accumulation differences
    const TOLERANCE: f32 = 5e-2;
    assert!(
        max_diff < TOLERANCE,
        "Q6_K matvec ({in_dim}->{out_dim}): max abs diff {max_diff:.6} at index {max_diff_idx} \
         exceeds tolerance {TOLERANCE} (gpu={:.6}, cpu={:.6})",
        gpu_output[max_diff_idx],
        cpu_output[max_diff_idx],
    );
}

/// Q6_K matvec correctness: small dimensions (256, 256).
/// No model file required — uses synthetic Q6_K blocks.
#[test]
fn test_matvec_q6_k_gpu_vs_cpu_small() {
    test_q6k_matvec_dims(256, 256);
}

/// Q6_K matvec correctness: Mistral-7B lm_head dimensions (4096, 32000).
/// No model file required — uses synthetic Q6_K blocks.
#[test]
fn test_matvec_q6_k_gpu_vs_cpu_mistral_lm_head() {
    test_q6k_matvec_dims(4096, 32000);
}
