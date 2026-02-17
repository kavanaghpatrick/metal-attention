//! End-to-end text generation tests.
//!
//! Tests the full inference pipeline: model construction, prefill, decode loop,
//! sampling, and streaming generation using synthetic in-memory models.

use metal_attention::config::InferenceConfig;
use metal_attention::inference::{decode_step, generate, generate_streaming, prefill};
use metal_attention::model::{HybridModel, ModelLayer};
use metal_attention_models::llama::LlamaLayer;
use metal_attention_models::rwkv7::Rwkv7Block;

fn make_small_rwkv_model() -> HybridModel {
    // Small RWKV model: vocab=100, hidden=32, 3 layers
    HybridModel::random(100, 32, 32, 1, 3, 12345)
}

fn make_small_llama_model() -> HybridModel {
    // Construct a small Llama-style model manually
    let vocab_size = 100;
    let hidden_size = 32;
    let head_dim = 32;
    let num_heads = 1;
    let num_layers = 2;

    // Random embeddings
    let mut rng = metal_attention::sampling::SimpleRng::new(54321);
    let scale = 1.0 / (hidden_size as f32).sqrt();
    let embed_weight: Vec<f32> = (0..vocab_size * hidden_size)
        .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale)
        .collect();
    let final_norm_weight = vec![1.0f32; hidden_size];
    let lm_head_weight: Vec<f32> = (0..vocab_size * hidden_size)
        .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale)
        .collect();

    // Create Llama layers
    let mut layers = Vec::new();
    for i in 0..num_layers {
        let layer = LlamaLayer::random(
            hidden_size,
            head_dim,
            num_heads,
            num_heads,
            54321 + i as u64 * 17,
        );
        layers.push(ModelLayer::Llama(layer));
    }

    let config = metal_attention_models::registry::ModelConfig {
        architecture: metal_attention_gguf::ModelArchitecture::Llama,
        hidden_size,
        head_dim,
        num_heads,
        num_kv_heads: num_heads,
        num_layers,
    };

    let block_config = metal_attention_traits::types::BlockConfig {
        hidden_size,
        head_dim,
        num_heads,
        num_kv_heads: num_heads,
        layer_index: 0,
    };

    HybridModel {
        embed_weight,
        final_norm_weight,
        lm_head_weight,
        layers,
        config,
        block_config,
    }
}

fn make_hybrid_model() -> HybridModel {
    // Mix of RWKV and Llama layers
    let vocab_size = 100;
    let hidden_size = 32;
    let head_dim = 32;
    let num_heads = 1;

    let mut rng = metal_attention::sampling::SimpleRng::new(99999);
    let scale = 1.0 / (hidden_size as f32).sqrt();
    let embed_weight: Vec<f32> = (0..vocab_size * hidden_size)
        .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale)
        .collect();
    let final_norm_weight = vec![1.0f32; hidden_size];
    let lm_head_weight: Vec<f32> = (0..vocab_size * hidden_size)
        .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale)
        .collect();

    // 2 RWKV layers + 2 Llama layers
    let mut layers = Vec::new();
    for i in 0..2 {
        let block = Rwkv7Block::random(hidden_size, head_dim, num_heads, 77777 + i as u64 * 11);
        layers.push(ModelLayer::Rwkv7(block));
    }
    for i in 0..2 {
        let layer = LlamaLayer::random(
            hidden_size,
            head_dim,
            num_heads,
            num_heads,
            88888 + i as u64 * 13,
        );
        layers.push(ModelLayer::Llama(layer));
    }

    let config = metal_attention_models::registry::ModelConfig {
        architecture: metal_attention_gguf::ModelArchitecture::Rwkv,
        hidden_size,
        head_dim,
        num_heads,
        num_kv_heads: num_heads,
        num_layers: 4,
    };

    let block_config = metal_attention_traits::types::BlockConfig {
        hidden_size,
        head_dim,
        num_heads,
        num_kv_heads: num_heads,
        layer_index: 0,
    };

    HybridModel {
        embed_weight,
        final_norm_weight,
        lm_head_weight,
        layers,
        config,
        block_config,
    }
}

#[test]
fn test_e2e_prefill_decode_loop() {
    let model = make_small_rwkv_model();
    let mut state = model.init_state();
    let prompt = vec![1, 2, 3, 4];

    // Prefill
    let logits = prefill(&model, &prompt, &mut state);
    assert_eq!(logits.len(), 100);
    for &v in &logits {
        assert!(v.is_finite(), "logits must be finite");
    }

    // Decode 10 tokens
    let mut generated = Vec::new();
    for _ in 0..10 {
        let token = metal_attention::sampling::sample_greedy(&logits);
        assert!((token as usize) < 100, "token must be in vocab range");
        generated.push(token);

        let new_logits = decode_step(&model, token, &mut state);
        assert_eq!(new_logits.len(), 100);
    }

    assert_eq!(generated.len(), 10);
}

#[test]
fn test_e2e_generate_valid_tokens() {
    let model = make_small_rwkv_model();
    let config = InferenceConfig {
        max_tokens: 10,
        temperature: 0.0, // greedy
        seed: Some(42),
        ..InferenceConfig::default()
    };

    let tokens = generate(&model, &[1, 2, 3], &config);

    assert!(!tokens.is_empty());
    assert!(tokens.len() <= 10);

    // All tokens should be valid vocab IDs
    for &tok in &tokens {
        assert!(
            (tok as usize) < 100,
            "token {} out of vocab range [0, 100)",
            tok
        );
    }
}

#[test]
fn test_e2e_generate_deterministic() {
    let model = make_small_rwkv_model();
    let config = InferenceConfig {
        max_tokens: 10,
        temperature: 0.0,
        seed: Some(42),
        ..InferenceConfig::default()
    };

    let tokens1 = generate(&model, &[5, 6], &config);
    let tokens2 = generate(&model, &[5, 6], &config);

    assert_eq!(
        tokens1, tokens2,
        "same seed + greedy sampling should produce identical output"
    );
}

#[test]
fn test_e2e_streaming_callback() {
    let model = make_small_rwkv_model();
    let config = InferenceConfig {
        max_tokens: 10,
        temperature: 0.0,
        seed: Some(42),
        ..InferenceConfig::default()
    };

    let mut collected = Vec::new();
    generate_streaming(&model, &[1, 2], &config, |token| {
        assert!(
            (token as usize) < 100,
            "streaming token {} out of vocab",
            token
        );
        collected.push(token);
        true
    });

    assert!(!collected.is_empty());
    assert!(collected.len() <= 10);

    // Should match non-streaming output
    let batch = generate(&model, &[1, 2], &config);
    assert_eq!(collected, batch);
}

#[test]
fn test_e2e_rwkv_architecture() {
    let model = make_small_rwkv_model();
    let config = InferenceConfig {
        max_tokens: 15,
        temperature: 0.0,
        seed: Some(100),
        ..InferenceConfig::default()
    };

    let tokens = generate(&model, &[10, 20, 30], &config);

    assert!(!tokens.is_empty());
    for &tok in &tokens {
        assert!((tok as usize) < 100);
    }
}

#[test]
fn test_e2e_llama_architecture() {
    let model = make_small_llama_model();
    let config = InferenceConfig {
        max_tokens: 15,
        temperature: 0.0,
        seed: Some(200),
        ..InferenceConfig::default()
    };

    let tokens = generate(&model, &[5, 10, 15], &config);

    assert!(!tokens.is_empty());
    for &tok in &tokens {
        assert!((tok as usize) < 100);
    }
}

#[test]
fn test_e2e_hybrid_architecture() {
    let model = make_hybrid_model();
    let config = InferenceConfig {
        max_tokens: 12,
        temperature: 0.0,
        seed: Some(300),
        ..InferenceConfig::default()
    };

    let tokens = generate(&model, &[1, 2], &config);

    assert!(!tokens.is_empty());
    for &tok in &tokens {
        assert!((tok as usize) < 100);
    }
}

#[test]
fn test_e2e_memory_stability() {
    // Run inference 50 times and ensure no memory growth/crashes
    let model = make_small_rwkv_model();
    let config = InferenceConfig {
        max_tokens: 5,
        temperature: 0.0,
        seed: Some(42),
        ..InferenceConfig::default()
    };

    let mut all_outputs = Vec::new();
    for i in 0..50 {
        let tokens = generate(&model, &[i % 10], &config);
        assert!(!tokens.is_empty());
        for &tok in &tokens {
            assert!((tok as usize) < 100);
        }
        all_outputs.push(tokens);
    }

    assert_eq!(all_outputs.len(), 50);
}

#[test]
fn test_e2e_temperature_sampling() {
    let model = make_small_rwkv_model();
    let config = InferenceConfig {
        max_tokens: 8,
        temperature: 0.8,
        top_k: 0,
        top_p: 1.0,
        seed: Some(42),
        ..InferenceConfig::default()
    };

    let tokens = generate(&model, &[1], &config);
    assert!(!tokens.is_empty());
    for &tok in &tokens {
        assert!((tok as usize) < 100);
    }
}

#[test]
fn test_e2e_top_k_sampling() {
    let model = make_small_rwkv_model();
    let config = InferenceConfig {
        max_tokens: 8,
        temperature: 0.8,
        top_k: 10,
        seed: Some(42),
        ..InferenceConfig::default()
    };

    let tokens = generate(&model, &[1], &config);
    assert!(!tokens.is_empty());
    for &tok in &tokens {
        assert!((tok as usize) < 100);
    }
}

#[test]
fn test_e2e_top_p_sampling() {
    let model = make_small_rwkv_model();
    let config = InferenceConfig {
        max_tokens: 8,
        temperature: 0.8,
        top_k: 0,
        top_p: 0.9,
        seed: Some(42),
        ..InferenceConfig::default()
    };

    let tokens = generate(&model, &[1], &config);
    assert!(!tokens.is_empty());
    for &tok in &tokens {
        assert!((tok as usize) < 100);
    }
}

#[test]
fn test_e2e_repetition_penalty() {
    let model = make_small_rwkv_model();
    let config = InferenceConfig {
        max_tokens: 15,
        temperature: 0.0,
        repetition_penalty: 1.5,
        seed: Some(42),
        ..InferenceConfig::default()
    };

    let tokens = generate(&model, &[1, 2, 3], &config);
    assert!(!tokens.is_empty());
    for &tok in &tokens {
        assert!((tok as usize) < 100);
    }
}

#[test]
fn test_e2e_streaming_early_stop() {
    let model = make_small_rwkv_model();
    let config = InferenceConfig {
        max_tokens: 100,
        temperature: 0.0,
        seed: Some(42),
        ..InferenceConfig::default()
    };

    let mut count = 0;
    generate_streaming(&model, &[1], &config, |_| {
        count += 1;
        count < 5 // stop after 5 tokens
    });

    assert_eq!(count, 5);
}

#[test]
fn test_e2e_long_prompt() {
    let model = make_small_rwkv_model();
    let config = InferenceConfig {
        max_tokens: 5,
        temperature: 0.0,
        seed: Some(42),
        ..InferenceConfig::default()
    };

    // Long prompt (20 tokens)
    let prompt: Vec<u32> = (0..20).collect();
    let tokens = generate(&model, &prompt, &config);

    assert!(!tokens.is_empty());
    assert!(tokens.len() <= 5);
    for &tok in &tokens {
        assert!((tok as usize) < 100);
    }
}

#[test]
fn test_e2e_different_seeds_different_outputs() {
    let model = make_small_rwkv_model();

    let config1 = InferenceConfig {
        max_tokens: 10,
        temperature: 0.8,
        seed: Some(111),
        ..InferenceConfig::default()
    };

    let config2 = InferenceConfig {
        max_tokens: 10,
        temperature: 0.8,
        seed: Some(222),
        ..InferenceConfig::default()
    };

    let tokens1 = generate(&model, &[1, 2], &config1);
    let tokens2 = generate(&model, &[1, 2], &config2);

    // Different seeds should (very likely) produce different outputs
    assert_ne!(
        tokens1, tokens2,
        "different seeds should produce different outputs"
    );
}
