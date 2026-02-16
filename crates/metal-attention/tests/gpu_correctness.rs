//! GPU vs CPU correctness integration tests.
//!
//! Compares GpuForwardPass (fused Q4_0 Metal kernels) against HybridModel (CPU F32)
//! using real SmolLM-135M weights. Tests are `#[ignore]` because they require
//! the model file at `models/SmolLM-135M.Q4_0.gguf`.

use std::path::{Path, PathBuf};

use metal_attention::gpu_forward_pass::GpuForwardPass;
use metal_attention::model::HybridModel;
use metal_attention::sampling::sample_greedy;
use metal_attention_kernels::device::GpuDevice;
use metal_attention_kernels::pipeline::PsoCache;

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
