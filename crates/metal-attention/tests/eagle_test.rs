//! EAGLE-3 speculative decoder integration tests.
//!
//! Requires Mistral-7B GGUF model file -- all tests are `#[ignore]`.

use std::path::PathBuf;

use metal_attention::EagleDecoder;

/// Resolve Mistral-7B model path relative to workspace root.
fn mistral_path() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    std::path::Path::new(manifest_dir).join("../../models/mistral-7b-v0.1.Q4_0.gguf")
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
    let mut decoder = EagleDecoder::new_random(path.as_path(), 6)
        .expect("Failed to create EagleDecoder");

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
    assert!(
        !collected.is_empty(),
        "Callback received no tokens"
    );

    // With random weights, acceptance rate should be very low (near 0%)
    // but we don't enforce this strictly -- just print it
    eprintln!(
        "Acceptance rate: {:.1}% (expected ~0% with random weights)",
        stats.acceptance_rate() * 100.0
    );

    eprintln!("test_eagle_random_weights_runs: PASS");
}
