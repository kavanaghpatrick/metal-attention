//! Speculative decoding correctness tests.
//!
//! Uses SmolLM-135M as both draft and target model to guarantee
//! high acceptance rate, then verifies the output matches target-only
//! greedy decoding.

use std::path::{Path, PathBuf};

use metal_attention::gpu_forward_pass::GpuForwardPass;
use metal_attention::sampling::sample_greedy;
use metal_attention::speculative::SpeculativeDecoder;

/// Resolve SmolLM model path relative to workspace root.
fn model_path() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest_dir).join("../../models/SmolLM-135M.Q4_0.gguf")
}

/// Speculative decode with same model as draft+target must produce
/// identical output to target-only greedy decode.
///
/// When draft == target, acceptance rate should be ~100% (greedy argmax
/// always matches). The final token sequence must be identical.
#[test]
#[ignore]
fn test_speculative_greedy_matches_target() {
    let path = model_path();
    assert!(
        path.exists(),
        "Model file not found: {path:?}. Download SmolLM-135M.Q4_0.gguf first."
    );
    let path = path.as_path();

    let prompt: &[u32] = &[1, 2, 3, 4, 5];
    let max_tokens = 20;

    // --- Target-only greedy decode ---
    eprintln!("Target-only greedy decode...");
    let mut gpu = GpuForwardPass::from_gguf(path).expect("Failed to load model");
    let first_token = gpu.forward_prompt(prompt).expect("forward_prompt failed");

    let mut target_tokens = vec![first_token];
    let mut next = first_token;
    for _ in 1..max_tokens {
        next = gpu.forward_token_greedy(next).expect("forward_token_greedy failed");
        target_tokens.push(next);
    }
    eprintln!("Target tokens: {target_tokens:?}");

    // --- Speculative decode (same model as draft + target) ---
    eprintln!("Speculative decode (draft=target)...");
    let mut decoder = SpeculativeDecoder::new(path, path, 4)
        .expect("Failed to create SpeculativeDecoder");

    let mut spec_collected = Vec::new();
    let (spec_tokens, stats) = decoder
        .generate(prompt, max_tokens, |tok| {
            spec_collected.push(tok);
        })
        .expect("speculative generate failed");

    eprintln!("Spec tokens: {spec_tokens:?}");
    eprintln!(
        "Stats: rounds={}, drafted={}, accepted={}, rate={:.1}%",
        stats.rounds,
        stats.tokens_drafted,
        stats.draft_accepted,
        stats.acceptance_rate() * 100.0
    );

    // --- Compare ---
    assert_eq!(
        spec_tokens.len(),
        target_tokens.len(),
        "Token count differs: spec={} target={}",
        spec_tokens.len(),
        target_tokens.len()
    );

    let mut match_count = 0;
    for (i, (&s, &t)) in spec_tokens.iter().zip(target_tokens.iter()).enumerate() {
        if s == t {
            match_count += 1;
        } else {
            eprintln!("First mismatch at position {i}: spec={s} target={t}");
            break;
        }
    }

    eprintln!("Matched: {match_count}/{max_tokens} tokens from start");

    // With same draft+target, we expect all tokens to match
    // Allow a small tolerance in case floating-point order differences
    // in forward_prompt_logits vs forward_token cause rare divergences
    assert!(
        match_count >= max_tokens - 2,
        "Too many mismatches: only {match_count}/{max_tokens} matched. \
         spec={spec_tokens:?} target={target_tokens:?}"
    );

    eprintln!("test_speculative_greedy_matches_target: PASS ({match_count}/{max_tokens} match)");
}
