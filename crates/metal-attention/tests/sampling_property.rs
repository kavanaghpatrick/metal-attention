//! Property-based tests for sampling functions.
//!
//! Verifies numerical invariants of greedy, temperature, top-k, and top-p
//! sampling strategies, plus repetition penalty. CPU-only, no GPU required.

use metal_attention::sampling::{
    apply_repetition_penalty, sample_greedy, sample_temperature, sample_top_k, SimpleRng,
};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Greedy sampling properties
// ---------------------------------------------------------------------------

proptest! {
    /// Greedy sampling always returns the index of the maximum logit.
    #[test]
    fn greedy_always_returns_argmax(
        vocab_size in 2..=256usize,
        seed in 0.0f32..100.0,
    ) {
        let logits: Vec<f32> = (0..vocab_size)
            .map(|i| (i as f32 * 0.37 + seed).sin())
            .collect();

        let result = sample_greedy(&logits);

        // Find true argmax
        let mut max_idx = 0;
        let mut max_val = logits[0];
        for (i, &v) in logits.iter().enumerate().skip(1) {
            if v > max_val {
                max_val = v;
                max_idx = i;
            }
        }

        prop_assert_eq!(
            result as usize,
            max_idx,
            "greedy returned {} but argmax is {} (val={})",
            result, max_idx, max_val
        );
    }
}

// ---------------------------------------------------------------------------
// Temperature sampling properties
// ---------------------------------------------------------------------------

proptest! {
    /// Temperature <= 0.0 produces the same result as greedy.
    #[test]
    fn temperature_zero_is_greedy(
        vocab_size in 2..=128usize,
        seed in 0.0f32..100.0,
        rng_seed in 1..=10000u64,
    ) {
        let logits: Vec<f32> = (0..vocab_size)
            .map(|i| (i as f32 * 0.37 + seed).sin())
            .collect();

        let greedy_result = sample_greedy(&logits);
        let mut rng = SimpleRng::new(rng_seed);
        let temp_result = sample_temperature(&logits, 0.0, &mut rng);

        prop_assert_eq!(
            temp_result,
            greedy_result,
            "temperature=0 gave {}, greedy gave {}",
            temp_result, greedy_result
        );
    }

    /// Temperature sampling always produces a valid token index.
    #[test]
    fn temperature_output_in_range(
        vocab_size in 2..=128usize,
        seed in 0.0f32..100.0,
        temp in 0.1f32..5.0,
        rng_seed in 1..=10000u64,
    ) {
        let logits: Vec<f32> = (0..vocab_size)
            .map(|i| (i as f32 * 0.37 + seed).sin())
            .collect();

        let mut rng = SimpleRng::new(rng_seed);
        let token = sample_temperature(&logits, temp, &mut rng);

        prop_assert!(
            (token as usize) < vocab_size,
            "token {token} >= vocab_size {vocab_size}"
        );
    }
}

// ---------------------------------------------------------------------------
// Top-k sampling properties
// ---------------------------------------------------------------------------

proptest! {
    /// top_k=1 always returns the argmax (same as greedy).
    #[test]
    fn top_k_1_always_selects_max(
        vocab_size in 2..=128usize,
        seed in 0.0f32..100.0,
        rng_seed in 1..=10000u64,
    ) {
        let logits: Vec<f32> = (0..vocab_size)
            .map(|i| (i as f32 * 0.37 + seed).sin())
            .collect();

        let greedy_result = sample_greedy(&logits);
        let mut rng = SimpleRng::new(rng_seed);
        let top_k_result = sample_top_k(&logits, 1, 1.0, &mut rng);

        prop_assert_eq!(
            top_k_result,
            greedy_result,
            "top_k=1 gave {}, greedy gave {}",
            top_k_result, greedy_result
        );
    }

    /// All top-k sampled tokens are within [0, vocab_size).
    #[test]
    fn all_sampling_outputs_in_range(
        vocab_size in 2..=128usize,
        seed in 0.0f32..100.0,
        k in 1..=64usize,
        rng_seed in 1..=10000u64,
    ) {
        let logits: Vec<f32> = (0..vocab_size)
            .map(|i| (i as f32 * 0.37 + seed).sin())
            .collect();

        let mut rng = SimpleRng::new(rng_seed);
        let token = sample_top_k(&logits, k, 1.0, &mut rng);

        prop_assert!(
            (token as usize) < vocab_size,
            "token {token} >= vocab_size {vocab_size}"
        );
    }
}

// ---------------------------------------------------------------------------
// Repetition penalty properties
// ---------------------------------------------------------------------------

proptest! {
    /// Repetition penalty > 1 reduces the logit magnitude of repeated positive tokens.
    ///
    /// For positive logits: logit /= penalty => smaller.
    /// For negative logits: logit *= penalty => more negative.
    /// Either way, the probability of repeated tokens decreases.
    #[test]
    fn repetition_penalty_reduces_repeated(
        vocab_size in 4..=64usize,
        penalty in 1.01f32..3.0,
        num_repeated in 1..=4usize,
        seed in 0.0f32..100.0,
    ) {
        // Generate logits that are all positive (simplifies probability reasoning)
        let mut logits: Vec<f32> = (0..vocab_size)
            .map(|i| ((i as f32 * 0.37 + seed).sin() + 1.5).abs()) // all positive
            .collect();
        let original = logits.clone();

        // Pick some token IDs to penalize
        let previous_tokens: Vec<u32> = (0..num_repeated.min(vocab_size))
            .map(|i| i as u32)
            .collect();

        apply_repetition_penalty(&mut logits, &previous_tokens, penalty);

        // Penalized positive logits should be strictly smaller
        for &tok in &previous_tokens {
            let idx = tok as usize;
            if original[idx] > 0.0 {
                prop_assert!(
                    logits[idx] < original[idx],
                    "token {idx}: logit {:.4} should be < original {:.4} after penalty {penalty}",
                    logits[idx],
                    original[idx]
                );
            }
        }

        // Non-penalized logits should be unchanged
        for i in 0..vocab_size {
            if !previous_tokens.contains(&(i as u32)) {
                let diff = (logits[i] - original[i]).abs();
                prop_assert!(
                    diff < f32::EPSILON,
                    "token {i}: logit changed by {diff} but wasn't penalized"
                );
            }
        }
    }

    /// Repetition penalty = 1.0 is a no-op.
    #[test]
    fn repetition_penalty_one_is_noop(
        vocab_size in 2..=64usize,
        seed in 0.0f32..100.0,
    ) {
        let mut logits: Vec<f32> = (0..vocab_size)
            .map(|i| (i as f32 * 0.37 + seed).sin())
            .collect();
        let original = logits.clone();

        let all_tokens: Vec<u32> = (0..vocab_size as u32).collect();
        apply_repetition_penalty(&mut logits, &all_tokens, 1.0);

        for (i, (&a, &b)) in logits.iter().zip(original.iter()).enumerate() {
            prop_assert!(
                (a - b).abs() < f32::EPSILON,
                "token {i}: logit changed from {b} to {a} with penalty=1.0"
            );
        }
    }
}
