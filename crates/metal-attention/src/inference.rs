//! Inference pipeline: prefill, decode, and full generation loop.
//!
//! Ties together HybridModel, sampling, and the generation state machine.

use crate::config::InferenceConfig;
use crate::model::{HybridModel, ModelState};
use crate::sampling::{
    apply_repetition_penalty, sample_greedy, sample_temperature, sample_top_k, sample_top_p,
    SimpleRng,
};

/// Prefill: process all prompt tokens through the model, returning the final logits.
///
/// This processes tokens sequentially (RWKV-7 is inherently recurrent).
/// The state is updated in-place for each token.
pub fn prefill(model: &HybridModel, prompt_tokens: &[u32], state: &mut ModelState) -> Vec<f32> {
    assert!(!prompt_tokens.is_empty(), "prompt must not be empty");
    let mut logits = Vec::new();
    for &token in prompt_tokens {
        logits = model.forward_token(token, state);
    }
    logits
}

/// Decode one step: process a single token and return logits for the next token.
pub fn decode_step(model: &HybridModel, token: u32, state: &mut ModelState) -> Vec<f32> {
    model.forward_token(token, state)
}

/// Sample a token from logits using the configured sampling strategy.
fn sample_token(
    logits: &mut Vec<f32>,
    previous_tokens: &[u32],
    config: &InferenceConfig,
    rng: &mut SimpleRng,
) -> u32 {
    // Apply repetition penalty
    if config.repetition_penalty != 1.0 {
        apply_repetition_penalty(logits, previous_tokens, config.repetition_penalty);
    }

    // Sample based on temperature
    if config.temperature <= 0.0 {
        sample_greedy(logits)
    } else if config.top_k > 0 && config.top_k < logits.len() {
        sample_top_k(logits, config.top_k, config.temperature, rng)
    } else if config.top_p < 1.0 {
        sample_top_p(logits, config.top_p, config.temperature, rng)
    } else {
        sample_temperature(logits, config.temperature, rng)
    }
}

/// Full generation loop: prefill prompt, then autoregressively decode tokens.
///
/// Returns an iterator-like vector of generated token IDs (not including prompt).
pub fn generate(
    model: &HybridModel,
    prompt_tokens: &[u32],
    config: &InferenceConfig,
) -> Vec<u32> {
    let mut state = model.init_state();
    let mut rng = SimpleRng::new(config.seed.unwrap_or(42));
    let vocab_size = model.vocab_size();

    // Prefill
    let mut logits = prefill(model, prompt_tokens, &mut state);

    // Decode loop
    let mut generated = Vec::with_capacity(config.max_tokens);
    let mut all_tokens: Vec<u32> = prompt_tokens.to_vec();

    for _ in 0..config.max_tokens {
        let token = sample_token(&mut logits, &all_tokens, config, &mut rng);

        // Validate token is in vocab range
        if (token as usize) >= vocab_size {
            break;
        }

        generated.push(token);
        all_tokens.push(token);

        // Decode next step
        logits = decode_step(model, token, &mut state);
    }

    generated
}

/// Streaming generation: returns generated tokens one at a time via a callback.
///
/// The callback receives each generated token and returns `true` to continue
/// or `false` to stop generation early.
pub fn generate_streaming<F>(
    model: &HybridModel,
    prompt_tokens: &[u32],
    config: &InferenceConfig,
    mut callback: F,
) where
    F: FnMut(u32) -> bool,
{
    let mut state = model.init_state();
    let mut rng = SimpleRng::new(config.seed.unwrap_or(42));
    let vocab_size = model.vocab_size();

    // Prefill
    let mut logits = prefill(model, prompt_tokens, &mut state);

    let mut all_tokens: Vec<u32> = prompt_tokens.to_vec();

    for _ in 0..config.max_tokens {
        let token = sample_token(&mut logits, &all_tokens, config, &mut rng);

        if (token as usize) >= vocab_size {
            break;
        }

        all_tokens.push(token);

        if !callback(token) {
            break;
        }

        logits = decode_step(model, token, &mut state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_model() -> HybridModel {
        // Small model: vocab=32, hidden=16, 2 layers
        HybridModel::random(32, 16, 16, 1, 2, 42)
    }

    fn make_test_config() -> InferenceConfig {
        InferenceConfig {
            max_tokens: 10,
            temperature: 0.0, // greedy for determinism
            seed: Some(42),
            ..InferenceConfig::default()
        }
    }

    #[test]
    fn test_inference_prefill() {
        let model = make_test_model();
        let mut state = model.init_state();
        let logits = prefill(&model, &[1, 2, 3], &mut state);
        assert_eq!(logits.len(), 32); // vocab_size
        for &v in &logits {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_inference_decode_step() {
        let model = make_test_model();
        let mut state = model.init_state();
        // First prefill
        let _ = prefill(&model, &[1], &mut state);
        // Then decode
        let logits = decode_step(&model, 5, &mut state);
        assert_eq!(logits.len(), 32);
        for &v in &logits {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_inference_generate_valid_tokens() {
        let model = make_test_model();
        let config = make_test_config();
        let tokens = generate(&model, &[1, 2, 3], &config);

        // Should generate up to max_tokens
        assert!(!tokens.is_empty());
        assert!(tokens.len() <= config.max_tokens);

        // All tokens should be valid vocab IDs
        let vocab_size = model.vocab_size();
        for &tok in &tokens {
            assert!(
                (tok as usize) < vocab_size,
                "token {} >= vocab_size {}",
                tok,
                vocab_size
            );
        }
    }

    #[test]
    fn test_inference_generate_deterministic() {
        let model = make_test_model();
        let config = make_test_config();

        let tokens1 = generate(&model, &[1, 2], &config);
        let tokens2 = generate(&model, &[1, 2], &config);

        // Greedy + same seed = identical output
        assert_eq!(tokens1, tokens2, "deterministic generation should produce identical tokens");
    }

    #[test]
    fn test_inference_generate_streaming() {
        let model = make_test_model();
        let config = make_test_config();
        let vocab_size = model.vocab_size();

        let mut collected = Vec::new();
        generate_streaming(&model, &[1, 2], &config, |token| {
            assert!(
                (token as usize) < vocab_size,
                "streaming token {} >= vocab_size {}",
                token,
                vocab_size
            );
            collected.push(token);
            true
        });

        assert!(!collected.is_empty());
        assert!(collected.len() <= config.max_tokens);

        // Should match non-streaming output
        let batch = generate(&model, &[1, 2], &config);
        assert_eq!(collected, batch, "streaming and batch generation should match");
    }

    #[test]
    fn test_inference_streaming_early_stop() {
        let model = make_test_model();
        let config = InferenceConfig {
            max_tokens: 100,
            temperature: 0.0,
            seed: Some(42),
            ..InferenceConfig::default()
        };

        let mut count = 0;
        generate_streaming(&model, &[1], &config, |_| {
            count += 1;
            count < 3 // stop after 3 tokens
        });

        assert_eq!(count, 3);
    }

    #[test]
    fn test_inference_with_temperature() {
        let model = make_test_model();
        let config = InferenceConfig {
            max_tokens: 5,
            temperature: 0.8,
            top_k: 0,
            top_p: 1.0,
            seed: Some(42),
            ..InferenceConfig::default()
        };

        let tokens = generate(&model, &[1], &config);
        assert!(!tokens.is_empty());
        let vocab_size = model.vocab_size();
        for &tok in &tokens {
            assert!((tok as usize) < vocab_size);
        }
    }

    #[test]
    fn test_inference_with_top_k() {
        let model = make_test_model();
        let config = InferenceConfig {
            max_tokens: 5,
            temperature: 0.8,
            top_k: 5,
            seed: Some(42),
            ..InferenceConfig::default()
        };

        let tokens = generate(&model, &[1], &config);
        assert!(!tokens.is_empty());
        let vocab_size = model.vocab_size();
        for &tok in &tokens {
            assert!((tok as usize) < vocab_size);
        }
    }

    #[test]
    fn test_inference_with_repetition_penalty() {
        let model = make_test_model();
        let config = InferenceConfig {
            max_tokens: 10,
            temperature: 0.0,
            repetition_penalty: 1.5,
            seed: Some(42),
            ..InferenceConfig::default()
        };

        let tokens = generate(&model, &[1, 2, 3], &config);
        assert!(!tokens.is_empty());
        let vocab_size = model.vocab_size();
        for &tok in &tokens {
            assert!((tok as usize) < vocab_size);
        }
    }
}
