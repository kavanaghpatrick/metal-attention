//! Runtime configuration for inference.
//!
//! Bundles model path, token generation limits, and sampling parameters
//! into a single configuration struct.

use std::path::PathBuf;

/// Runtime configuration for an inference session.
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Path to the GGUF model file.
    pub model_path: PathBuf,
    /// Maximum number of tokens to generate.
    pub max_tokens: usize,
    /// Sampling temperature (0.0 = greedy, higher = more random).
    pub temperature: f32,
    /// Top-p (nucleus) sampling threshold.
    pub top_p: f32,
    /// Top-k sampling: number of highest-probability candidates to consider.
    pub top_k: usize,
    /// Repetition penalty multiplier (1.0 = no penalty).
    pub repetition_penalty: f32,
    /// Optional RNG seed for reproducible sampling.
    pub seed: Option<u64>,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            max_tokens: 128,
            temperature: 0.8,
            top_p: 0.95,
            top_k: 40,
            repetition_penalty: 1.1,
            seed: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = InferenceConfig::default();
        assert_eq!(cfg.max_tokens, 128);
        assert!((cfg.temperature - 0.8).abs() < f32::EPSILON);
        assert!((cfg.top_p - 0.95).abs() < f32::EPSILON);
        assert_eq!(cfg.top_k, 40);
        assert!((cfg.repetition_penalty - 1.1).abs() < f32::EPSILON);
        assert!(cfg.seed.is_none());
    }
}
