//! Token sampling from logit distributions.
//!
//! Provides greedy, temperature-scaled, top-p (nucleus), and top-k sampling
//! strategies, plus a repetition penalty modifier.

/// Greedy (argmax) sampling: returns the token with the highest logit.
pub fn sample_greedy(logits: &[f32]) -> u32 {
    assert!(!logits.is_empty(), "logits must not be empty");
    let mut best_idx = 0u32;
    let mut best_val = logits[0];
    for (i, &v) in logits.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}

/// Temperature-scaled sampling with softmax.
///
/// Lower temperature sharpens the distribution (more greedy),
/// higher temperature flattens it (more random).
/// Temperature of 0 falls back to greedy.
pub fn sample_temperature(logits: &[f32], temp: f32, rng: &mut SimpleRng) -> u32 {
    assert!(!logits.is_empty(), "logits must not be empty");
    if temp <= 0.0 {
        return sample_greedy(logits);
    }

    let scaled: Vec<f32> = logits.iter().map(|&x| x / temp).collect();
    let probs = softmax(&scaled);
    sample_from_probs(&probs, rng)
}

/// Top-p (nucleus) sampling: consider only the smallest set of tokens
/// whose cumulative probability exceeds `top_p`.
pub fn sample_top_p(logits: &[f32], top_p: f32, temp: f32, rng: &mut SimpleRng) -> u32 {
    assert!(!logits.is_empty(), "logits must not be empty");
    if temp <= 0.0 {
        return sample_greedy(logits);
    }

    let scaled: Vec<f32> = logits.iter().map(|&x| x / temp).collect();
    let probs = softmax(&scaled);

    // Sort by probability descending
    let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Find cutoff
    let mut cumsum = 0.0f32;
    let mut cutoff_idx = indexed.len();
    for (i, &(_, p)) in indexed.iter().enumerate() {
        cumsum += p;
        if cumsum >= top_p {
            cutoff_idx = i + 1;
            break;
        }
    }

    // Renormalize the kept tokens
    let kept = &indexed[..cutoff_idx];
    let total: f32 = kept.iter().map(|&(_, p)| p).sum();
    let renormed: Vec<(usize, f32)> = kept.iter().map(|&(idx, p)| (idx, p / total)).collect();

    // Sample from renormalized distribution
    let r = rng.next_f32();
    let mut acc = 0.0f32;
    for &(idx, p) in &renormed {
        acc += p;
        if r < acc {
            return idx as u32;
        }
    }
    // Fallback to last kept token
    renormed.last().map(|&(idx, _)| idx as u32).unwrap_or(0)
}

/// Top-k sampling: consider only the `top_k` highest-probability tokens.
pub fn sample_top_k(logits: &[f32], top_k: usize, temp: f32, rng: &mut SimpleRng) -> u32 {
    assert!(!logits.is_empty(), "logits must not be empty");
    if temp <= 0.0 {
        return sample_greedy(logits);
    }

    let k = top_k.min(logits.len()).max(1);

    // Find top-k indices
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);

    // Apply temperature and softmax over top-k only
    let max_val = indexed[0].1;
    let scaled: Vec<(usize, f32)> = indexed
        .iter()
        .map(|&(idx, v)| (idx, ((v - max_val) / temp).exp()))
        .collect();
    let total: f32 = scaled.iter().map(|&(_, v)| v).sum();
    let probs: Vec<(usize, f32)> = scaled.iter().map(|&(idx, v)| (idx, v / total)).collect();

    // Sample
    let r = rng.next_f32();
    let mut acc = 0.0f32;
    for &(idx, p) in &probs {
        acc += p;
        if r < acc {
            return idx as u32;
        }
    }
    probs.last().map(|&(idx, _)| idx as u32).unwrap_or(0)
}

/// Apply repetition penalty to logits for previously generated tokens.
///
/// For tokens in `previous_tokens`: if logit > 0, divide by penalty;
/// if logit < 0, multiply by penalty. This discourages repetition.
pub fn apply_repetition_penalty(logits: &mut [f32], previous_tokens: &[u32], penalty: f32) {
    if (penalty - 1.0).abs() < f32::EPSILON {
        return;
    }
    for &tok in previous_tokens {
        let idx = tok as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

/// Compute softmax probabilities from logits.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Sample an index from a probability distribution.
fn sample_from_probs(probs: &[f32], rng: &mut SimpleRng) -> u32 {
    let r = rng.next_f32();
    let mut acc = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if r < acc {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

/// Simple deterministic pseudo-random number generator.
///
/// Uses xorshift64 for reproducible test results. Not cryptographically secure.
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    /// Create a new RNG with the given seed.
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    /// Generate a random u64.
    pub fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Generate a random f32 in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() & 0xFFFFFF) as f32 / (0xFFFFFF as f32 + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_returns_argmax() {
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        assert_eq!(sample_greedy(&logits), 3);
    }

    #[test]
    fn test_greedy_first_element() {
        let logits = vec![10.0, 1.0, 2.0];
        assert_eq!(sample_greedy(&logits), 0);
    }

    #[test]
    fn test_greedy_last_element() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 100.0];
        assert_eq!(sample_greedy(&logits), 4);
    }

    #[test]
    fn test_temperature_zero_equals_greedy() {
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let mut rng = SimpleRng::new(42);
        assert_eq!(
            sample_temperature(&logits, 0.0, &mut rng),
            sample_greedy(&logits)
        );
    }

    #[test]
    fn test_temperature_produces_valid_token() {
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let mut rng = SimpleRng::new(42);
        let token = sample_temperature(&logits, 1.0, &mut rng);
        assert!((token as usize) < logits.len());
    }

    #[test]
    fn test_top_k_reduces_candidates() {
        // With top_k=1, should always return the argmax
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let mut rng = SimpleRng::new(42);
        for _ in 0..20 {
            let token = sample_top_k(&logits, 1, 1.0, &mut rng);
            assert_eq!(token, 3, "top_k=1 should always pick argmax");
        }
    }

    #[test]
    fn test_top_k_valid_tokens() {
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let mut rng = SimpleRng::new(123);
        for _ in 0..50 {
            let token = sample_top_k(&logits, 3, 1.0, &mut rng);
            assert!((token as usize) < logits.len());
        }
    }

    #[test]
    fn test_top_p_valid_tokens() {
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let mut rng = SimpleRng::new(456);
        for _ in 0..50 {
            let token = sample_top_p(&logits, 0.9, 1.0, &mut rng);
            assert!((token as usize) < logits.len());
        }
    }

    #[test]
    fn test_top_p_one_returns_valid() {
        // top_p=1.0 considers all tokens
        let logits = vec![1.0, 2.0, 3.0];
        let mut rng = SimpleRng::new(789);
        let token = sample_top_p(&logits, 1.0, 1.0, &mut rng);
        assert!((token as usize) < logits.len());
    }

    #[test]
    fn test_repetition_penalty_reduces_repeated() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        let original = logits.clone();
        apply_repetition_penalty(&mut logits, &[1, 3], 1.5);

        // Token 1 and 3 should be reduced (positive logits divided by penalty)
        assert!(logits[1] < original[1]);
        assert!(logits[3] < original[3]);
        // Token 0 and 2 should be unchanged
        assert!((logits[0] - original[0]).abs() < f32::EPSILON);
        assert!((logits[2] - original[2]).abs() < f32::EPSILON);
    }

    #[test]
    fn test_repetition_penalty_negative_logits() {
        let mut logits = vec![-2.0, -1.0, 0.5, 1.0];
        apply_repetition_penalty(&mut logits, &[0, 1], 2.0);

        // Negative logits should be multiplied by penalty (made more negative)
        assert!((logits[0] - (-4.0)).abs() < f32::EPSILON);
        assert!((logits[1] - (-2.0)).abs() < f32::EPSILON);
        // Positive/unchanged
        assert!((logits[2] - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_repetition_penalty_one_is_noop() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let original = logits.clone();
        apply_repetition_penalty(&mut logits, &[0, 1, 2], 1.0);
        for (a, b) in logits.iter().zip(original.iter()) {
            assert!((a - b).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax should sum to 1, got {}",
            sum
        );
    }

    #[test]
    fn test_softmax_monotonic() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);
        assert!(probs[0] < probs[1]);
        assert!(probs[1] < probs[2]);
    }

    #[test]
    fn test_rng_deterministic() {
        let mut rng1 = SimpleRng::new(42);
        let mut rng2 = SimpleRng::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }
}
