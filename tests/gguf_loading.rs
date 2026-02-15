//! Integration tests for GGUF model loading and inference.
//!
//! These tests load the real SmolLM-135M Q4_0 model and verify that
//! the full pipeline works end-to-end: GGUF parsing, weight dequantization,
//! model construction, and inference.
//!
//! Marked #[ignore] since they require the model file (87MB) at
//! models/SmolLM-135M.Q4_0.gguf.

use std::path::Path;

use metal_attention::inference::{decode_step, prefill};
use metal_attention::model::HybridModel;
use metal_attention_gguf::ModelArchitecture;
use metal_attention_kernels::device::GpuDevice;
use metal_attention_kernels::pipeline::PsoCache;

const MODEL_PATH: &str = "models/SmolLM-135M.Q4_0.gguf";

/// Test: SmolLM-135M loads from GGUF without panic and config matches expected values.
#[test]
#[ignore]
fn test_load_smollm_135m() {
    let path = Path::new(MODEL_PATH);
    assert!(path.exists(), "Model file not found: {MODEL_PATH}");

    let device = GpuDevice::new();
    let mut pso_cache = PsoCache::new(device.library.clone());

    let model = HybridModel::from_gguf(path, Some(&device), Some(&mut pso_cache))
        .expect("Failed to load SmolLM from GGUF");

    // SmolLM-135M: 30 layers, 576 hidden size, 9 heads, 3 kv_heads
    assert_eq!(model.config.num_layers, 30, "expected 30 layers");
    assert_eq!(model.config.hidden_size, 576, "expected 576 hidden size");
    assert_eq!(model.config.num_heads, 9, "expected 9 attention heads");
    assert_eq!(model.config.num_kv_heads, 3, "expected 3 KV heads");
    assert_eq!(model.config.head_dim, 64, "expected 64 head dim");
    assert_eq!(
        model.config.architecture,
        ModelArchitecture::Llama,
        "expected Llama architecture"
    );
    assert_eq!(model.layers.len(), 30, "expected 30 model layers");

    // Embedding should have vocab_size * hidden_size elements
    let vocab_size = model.vocab_size();
    assert!(vocab_size > 0, "vocab size must be positive");
    assert_eq!(
        model.embed_weight.len(),
        vocab_size * 576,
        "embed_weight size mismatch"
    );
}

/// Test: SmolLM-135M inference produces finite logits and valid tokens.
#[test]
#[ignore]
fn test_inference_smollm() {
    let path = Path::new(MODEL_PATH);
    assert!(path.exists(), "Model file not found: {MODEL_PATH}");

    let device = GpuDevice::new();
    let mut pso_cache = PsoCache::new(device.library.clone());

    let model = HybridModel::from_gguf(path, Some(&device), Some(&mut pso_cache))
        .expect("Failed to load SmolLM from GGUF");

    let vocab_size = model.vocab_size();
    let mut state = model.init_state();

    // Prefill with a short prompt (token IDs that are valid for SmolLM)
    let prompt_tokens = vec![1u32, 2, 3];
    let logits = prefill(&model, &prompt_tokens, &mut state);

    assert_eq!(
        logits.len(),
        vocab_size,
        "logits should have vocab_size elements"
    );
    for (i, &v) in logits.iter().enumerate() {
        assert!(
            v.is_finite(),
            "prefill logit at index {i} is not finite: {v}"
        );
    }

    // Decode 5 tokens
    let mut generated_tokens = Vec::new();
    let mut current_logits = logits;

    for step in 0..5 {
        // Greedy sample: argmax
        let token = current_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap();

        assert!(
            (token as usize) < vocab_size,
            "decode step {step}: token {token} out of vocab range [0, {vocab_size})"
        );

        generated_tokens.push(token);
        current_logits = decode_step(&model, token, &mut state);

        assert_eq!(current_logits.len(), vocab_size);
        for (i, &v) in current_logits.iter().enumerate() {
            assert!(
                v.is_finite(),
                "decode step {step} logit at index {i} is not finite: {v}"
            );
        }
    }

    assert_eq!(generated_tokens.len(), 5, "should have generated 5 tokens");

    // Not all tokens should be the same (basic sanity - model should produce varied output)
    // This is a soft check; with a real model the outputs should differ
    let all_same = generated_tokens.iter().all(|&t| t == generated_tokens[0]);
    if all_same {
        eprintln!(
            "Warning: all 5 generated tokens are the same ({}). Model may not be producing varied output.",
            generated_tokens[0]
        );
    }
}
