//! EAGLE-3 speculative decoder integration tests.
//!
//! Requires Mistral-7B GGUF model file -- all tests are `#[ignore]`.

use std::path::PathBuf;
use std::time::Instant;

use metal_attention::eagle_head::EagleHead;
use metal_attention::gpu_forward_pass::GpuForwardPass;
use metal_attention::EagleDecoder;

use metal_attention_kernels::buffer::{alloc_buffer_with_data, read_buffer_slice};
use metal_attention_kernels::device::GpuDevice;

/// Resolve Mistral-7B model path relative to workspace root.
fn mistral_path() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    std::path::Path::new(manifest_dir).join("../../models/mistral-7b-v0.1.Q4_0.gguf")
}

/// Simple argmax helper: find index of maximum value in a logits slice.
fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap()
}

/// End-to-end POC: construct EagleDecoder with random weights, generate 20
/// tokens on Mistral-7B, and verify no crashes.
///
/// Random weights give ~0% acceptance (all draft tokens rejected), but the
/// infrastructure must not panic. Every speculation round will reject all
/// drafts and fall through to the correction token from the target model.
#[test]
#[ignore]
fn test_eagle_random_weights_runs() {
    let path = mistral_path();
    assert!(
        path.exists(),
        "Model file not found: {path:?}. Download Mistral-7B Q4_0 GGUF first."
    );

    eprintln!("Loading EagleDecoder with random weights...");
    let mut decoder =
        EagleDecoder::new_random(path.as_path(), 6).expect("Failed to create EagleDecoder");

    // Mistral tokenizer: approximate "Hello world" prompt
    let prompt: &[u32] = &[1, 733, 16044, 28747];
    let max_tokens = 20;

    eprintln!("Generating {max_tokens} tokens with EAGLE random-weight POC...");
    let mut collected = Vec::new();
    let (tokens, stats) = decoder
        .generate(prompt, max_tokens, |tok| {
            collected.push(tok);
            eprint!("{tok} ");
        })
        .expect("EagleDecoder::generate failed");

    eprintln!();
    eprintln!("Generated {} tokens: {:?}", tokens.len(), &tokens);
    eprintln!(
        "SpecStats: rounds={}, drafted={}, accepted={}, draft_accepted={}, rate={:.1}%",
        stats.rounds,
        stats.tokens_drafted,
        stats.tokens_accepted,
        stats.draft_accepted,
        stats.acceptance_rate() * 100.0
    );

    // Assertions
    assert_eq!(
        tokens.len(),
        max_tokens,
        "Expected {max_tokens} tokens, got {}",
        tokens.len()
    );

    // Callback should have received the same tokens
    assert!(!collected.is_empty(), "Callback received no tokens");

    // With random weights, acceptance rate should be very low (near 0%)
    // but we don't enforce this strictly -- just print it
    eprintln!(
        "Acceptance rate: {:.1}% (expected ~0% with random weights)",
        stats.acceptance_rate() * 100.0
    );

    eprintln!("test_eagle_random_weights_runs: PASS");
}

// ===========================================================================
// Task 3.1: Hidden state capture correctness tests
// ===========================================================================

/// Verify that hidden state capture produces valid, non-zero GPU buffers
/// with the correct number of elements and no NaN/Inf values.
#[test]
#[ignore]
fn test_hidden_state_capture() {
    let path = mistral_path();
    assert!(path.exists(), "Model file not found: {path:?}");

    eprintln!("Loading Mistral-7B with eagle capture enabled...");
    let mut model = GpuForwardPass::from_gguf(&path).expect("Failed to load model");
    let hidden_size = model.hidden_size();

    model.enable_eagle_capture(0, 16, 31);

    // Run forward_prompt to populate KV cache + capture buffers.
    // Use a small prompt, then call forward_token for a single decode step.
    let prompt: &[u32] = &[1, 733, 16044, 28747]; // <s> Hello world:
    let first_token = model.forward_prompt(prompt).expect("forward_prompt failed");
    eprintln!("Prompt produced first token: {first_token}");

    // Now run forward_token which captures hidden states at layers 0/16/31.
    let _logits = model
        .forward_token(first_token)
        .expect("forward_token failed");

    // Read back capture buffers.
    let feat_low_buf = model
        .eagle_capture_low()
        .expect("eagle_capture_low not set");
    let feat_mid_buf = model
        .eagle_capture_mid()
        .expect("eagle_capture_mid not set");
    let feat_high_buf = model
        .eagle_capture_high()
        .expect("eagle_capture_high not set");

    let feat_low: Vec<f32> = unsafe { read_buffer_slice(feat_low_buf, hidden_size) };
    let feat_mid: Vec<f32> = unsafe { read_buffer_slice(feat_mid_buf, hidden_size) };
    let feat_high: Vec<f32> = unsafe { read_buffer_slice(feat_high_buf, hidden_size) };

    // Assert correct size.
    assert_eq!(feat_low.len(), hidden_size, "feat_low wrong size");
    assert_eq!(feat_mid.len(), hidden_size, "feat_mid wrong size");
    assert_eq!(feat_high.len(), hidden_size, "feat_high wrong size");

    // Assert no NaN/Inf.
    for (name, buf) in [
        ("feat_low", &feat_low),
        ("feat_mid", &feat_mid),
        ("feat_high", &feat_high),
    ] {
        let has_nan = buf.iter().any(|v| v.is_nan());
        let has_inf = buf.iter().any(|v| v.is_infinite());
        assert!(!has_nan, "{name} contains NaN values");
        assert!(!has_inf, "{name} contains Inf values");
    }

    // Assert non-zero (at least some elements must be non-zero).
    for (name, buf) in [
        ("feat_low", &feat_low),
        ("feat_mid", &feat_mid),
        ("feat_high", &feat_high),
    ] {
        let non_zero_count = buf.iter().filter(|&&v| v != 0.0).count();
        assert!(
            non_zero_count > 0,
            "{name} is all zeros -- capture likely not working"
        );
        eprintln!(
            "{name}: {non_zero_count}/{} non-zero, min={:.6}, max={:.6}",
            buf.len(),
            buf.iter().cloned().fold(f32::INFINITY, f32::min),
            buf.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        );
    }

    eprintln!("test_hidden_state_capture: PASS");
}

/// Verify that enabling eagle capture adds minimal overhead to forward_token.
/// Measures 100 forward_token calls with and without capture, asserts < 0.5ms overhead.
#[test]
#[ignore]
fn test_hidden_state_capture_zero_overhead() {
    let path = mistral_path();
    assert!(path.exists(), "Model file not found: {path:?}");

    // --- Baseline: without eagle capture ---
    eprintln!("Loading Mistral-7B WITHOUT eagle capture...");
    let mut model_no_capture = GpuForwardPass::from_gguf(&path).expect("Failed to load model");

    let prompt: &[u32] = &[1, 733, 16044, 28747];
    let first_token = model_no_capture
        .forward_prompt(prompt)
        .expect("forward_prompt failed");

    // Warmup
    let mut token = first_token;
    for _ in 0..5 {
        let logits = model_no_capture
            .forward_token(token)
            .expect("forward_token failed");
        token = argmax(&logits);
    }

    let n_iters = 100;
    let start_no_capture = Instant::now();
    for _ in 0..n_iters {
        let logits = model_no_capture
            .forward_token(token)
            .expect("forward_token failed");
        token = argmax(&logits);
    }
    let elapsed_no_capture = start_no_capture.elapsed();
    let ms_per_token_no_capture = elapsed_no_capture.as_secs_f64() * 1000.0 / n_iters as f64;
    eprintln!(
        "Without capture: {n_iters} tokens in {:.1}ms ({:.3}ms/token)",
        elapsed_no_capture.as_secs_f64() * 1000.0,
        ms_per_token_no_capture,
    );

    // --- With eagle capture ---
    eprintln!("Loading Mistral-7B WITH eagle capture...");
    let mut model_with_capture = GpuForwardPass::from_gguf(&path).expect("Failed to load model");
    model_with_capture.enable_eagle_capture(0, 16, 31);

    let first_token2 = model_with_capture
        .forward_prompt(prompt)
        .expect("forward_prompt failed");

    // Warmup
    let mut token2 = first_token2;
    for _ in 0..5 {
        let logits = model_with_capture
            .forward_token(token2)
            .expect("forward_token failed");
        token2 = argmax(&logits);
    }

    let start_with_capture = Instant::now();
    for _ in 0..n_iters {
        let logits = model_with_capture
            .forward_token(token2)
            .expect("forward_token failed");
        token2 = argmax(&logits);
    }
    let elapsed_with_capture = start_with_capture.elapsed();
    let ms_per_token_with_capture = elapsed_with_capture.as_secs_f64() * 1000.0 / n_iters as f64;
    eprintln!(
        "With capture: {n_iters} tokens in {:.1}ms ({:.3}ms/token)",
        elapsed_with_capture.as_secs_f64() * 1000.0,
        ms_per_token_with_capture,
    );

    let overhead_ms = ms_per_token_with_capture - ms_per_token_no_capture;
    eprintln!("Overhead: {overhead_ms:.3}ms per token");

    assert!(
        overhead_ms < 0.5,
        "Eagle capture overhead too high: {overhead_ms:.3}ms (limit: 0.5ms)"
    );

    eprintln!("test_hidden_state_capture_zero_overhead: PASS");
}

// ===========================================================================
// Task 3.2: EagleHead forward_draft_token correctness tests
// ===========================================================================

/// Verify that EagleHead::forward_draft_token produces valid token IDs
/// within vocab range (< 32000 for Mistral-7B).
#[test]
#[ignore]
fn test_eagle_head_produces_valid_tokens() {
    let path = mistral_path();
    assert!(path.exists(), "Model file not found: {path:?}");

    eprintln!("Loading Mistral-7B for EagleHead test...");
    let mut model = GpuForwardPass::from_gguf(&path).expect("Failed to load model");
    model.enable_eagle_capture(0, 16, 31);

    let hidden_size = model.hidden_size();
    let vocab_size = model.vocab_size();
    let device = GpuDevice::shared();

    // Create EagleHead with random weights.
    let mut eagle_head = EagleHead::new_random(
        device,
        model.hidden_size(),
        model.num_heads(),
        model.num_kv_heads(),
        model.head_dim(),
        model.intermediate_size(),
        model.vocab_size(),
    );

    // Allocate 3 fake feature buffers with small sinusoidal F32 data (hidden_size elements each).
    let dev = &*device.device;
    let data_low: Vec<f32> = (0..hidden_size)
        .map(|j| (j as f32 * 0.001).sin() * 0.01)
        .collect();
    let data_mid: Vec<f32> = (0..hidden_size)
        .map(|j| ((hidden_size + j) as f32 * 0.001).sin() * 0.01)
        .collect();
    let data_high: Vec<f32> = (0..hidden_size)
        .map(|j| ((2 * hidden_size + j) as f32 * 0.001).sin() * 0.01)
        .collect();
    let feat_low = alloc_buffer_with_data(dev, &data_low);
    let feat_mid = alloc_buffer_with_data(dev, &data_mid);
    let feat_high = alloc_buffer_with_data(dev, &data_high);

    // Run forward_draft_token 6 times.
    let n_drafts = 6;
    let mut prev_token: u32 = 1; // BOS token
    for i in 0..n_drafts {
        let token_id = eagle_head
            .forward_draft_token(
                &feat_low,
                &feat_mid,
                &feat_high,
                prev_token,
                model.embed(),
                model.lm_head(),
                model.lm_head_is_f32(),
                model.lm_head_q6k(),
                model.lm_head_q8(),
            )
            .expect("forward_draft_token failed");

        eprintln!("Draft {i}: token_id={token_id} (vocab_size={vocab_size})");

        assert!(
            (token_id as usize) < vocab_size,
            "Draft token {token_id} >= vocab_size {vocab_size}"
        );

        prev_token = token_id;
    }

    eprintln!("test_eagle_head_produces_valid_tokens: PASS");
}

/// Verify that EagleHead KV cache reset allows multiple rounds without crash.
#[test]
#[ignore]
fn test_eagle_head_kv_cache_reset() {
    let path = mistral_path();
    assert!(path.exists(), "Model file not found: {path:?}");

    eprintln!("Loading Mistral-7B for KV cache reset test...");
    let mut model = GpuForwardPass::from_gguf(&path).expect("Failed to load model");
    model.enable_eagle_capture(0, 16, 31);

    let hidden_size = model.hidden_size();
    let device = GpuDevice::shared();

    let mut eagle_head = EagleHead::new_random(
        device,
        model.hidden_size(),
        model.num_heads(),
        model.num_kv_heads(),
        model.head_dim(),
        model.intermediate_size(),
        model.vocab_size(),
    );

    // Allocate fake feature buffers filled with small constant values.
    let dev = &*device.device;
    let data: Vec<f32> = vec![0.001; hidden_size];
    let feat_low = alloc_buffer_with_data(dev, &data);
    let feat_mid = alloc_buffer_with_data(dev, &data);
    let feat_high = alloc_buffer_with_data(dev, &data);

    // Round 1: run 6 draft tokens.
    eprintln!("Round 1: 6 draft tokens...");
    let mut prev_token: u32 = 1;
    for _ in 0..6 {
        let tok = eagle_head
            .forward_draft_token(
                &feat_low,
                &feat_mid,
                &feat_high,
                prev_token,
                model.embed(),
                model.lm_head(),
                model.lm_head_is_f32(),
                model.lm_head_q6k(),
                model.lm_head_q8(),
            )
            .expect("Round 1 forward_draft_token failed");
        prev_token = tok;
    }

    // Reset KV cache.
    eprintln!("Resetting eagle KV cache...");
    eagle_head.reset_kv_cache();

    // Round 2: run 6 more draft tokens.
    eprintln!("Round 2: 6 draft tokens after reset...");
    prev_token = 1;
    for _ in 0..6 {
        let tok = eagle_head
            .forward_draft_token(
                &feat_low,
                &feat_mid,
                &feat_high,
                prev_token,
                model.embed(),
                model.lm_head(),
                model.lm_head_is_f32(),
                model.lm_head_q6k(),
                model.lm_head_q8(),
            )
            .expect("Round 2 forward_draft_token failed");
        prev_token = tok;
    }

    eprintln!("test_eagle_head_kv_cache_reset: PASS");
}

// ===========================================================================
// Task 3.4: EagleDecoder greedy output correctness
// ===========================================================================

/// Verify that EagleDecoder with random weights (0% acceptance) produces the
/// same output as target-only greedy decode.
///
/// With random weights, every draft token is rejected. The verification step
/// always takes the target model's argmax at position 0, which is exactly
/// what target-only greedy decode would produce. This validates the
/// verification + rollback logic preserves correctness.
#[test]
#[ignore]
fn test_eagle_greedy_matches_target_only() {
    let path = mistral_path();
    assert!(path.exists(), "Model file not found: {path:?}");

    let prompt: &[u32] = &[1, 733, 16044, 28747]; // <s> Hello world:
    let max_tokens = 50;

    // --- Target-only greedy decode ---
    eprintln!("=== Target-only greedy decode ({max_tokens} tokens) ===");
    let mut target_model = GpuForwardPass::from_gguf(&path).expect("Failed to load target model");

    let first_token = target_model
        .forward_prompt(prompt)
        .expect("forward_prompt failed");
    let mut target_tokens = vec![first_token];
    eprint!("{first_token} ");

    let mut current_token = first_token;
    for _ in 1..max_tokens {
        let logits = target_model
            .forward_token(current_token)
            .expect("forward_token failed");
        let next_token = argmax(&logits);
        target_tokens.push(next_token);
        eprint!("{next_token} ");
        current_token = next_token;
    }
    eprintln!();
    eprintln!("Target-only tokens: {:?}", &target_tokens);

    // --- EagleDecoder with random weights ---
    eprintln!("=== EagleDecoder random-weight decode ({max_tokens} tokens) ===");
    let mut decoder = EagleDecoder::new_random(&path, 6).expect("Failed to create EagleDecoder");

    let (eagle_tokens, stats) = decoder
        .generate(prompt, max_tokens, |tok| {
            eprint!("{tok} ");
        })
        .expect("EagleDecoder::generate failed");
    eprintln!();
    eprintln!("Eagle tokens:       {:?}", &eagle_tokens);

    // Print speculation stats.
    eprintln!(
        "SpecStats: rounds={}, drafted={}, accepted={}, draft_accepted={}, rate={:.1}%",
        stats.rounds,
        stats.tokens_drafted,
        stats.tokens_accepted,
        stats.draft_accepted,
        stats.acceptance_rate() * 100.0,
    );

    // Assert outputs are identical.
    assert_eq!(
        eagle_tokens.len(),
        target_tokens.len(),
        "Token count mismatch: eagle={} vs target={}",
        eagle_tokens.len(),
        target_tokens.len(),
    );

    for (i, (eagle_tok, target_tok)) in eagle_tokens.iter().zip(target_tokens.iter()).enumerate() {
        assert_eq!(
            eagle_tok, target_tok,
            "Token mismatch at position {i}: eagle={eagle_tok} vs target={target_tok}"
        );
    }

    eprintln!("All {max_tokens} tokens match! EagleDecoder greedy output is correct.");
    eprintln!("test_eagle_greedy_matches_target_only: PASS");
}
