//! Hybrid model inference engine for Apple Silicon.
//!
//! Composable attention traits compiled to zero-overhead Metal kernels
//! via function constants. Supports prefill/decode inference with
//! schedule-driven hybrid layer dispatch.

pub mod config;
pub mod inference;
pub mod model;
pub mod sampling;

// Re-exports for convenience
pub use config::InferenceConfig;
pub use inference::{decode_step, generate, generate_streaming, prefill};
pub use model::{HybridModel, LayerState, ModelLayer, ModelState};
pub use sampling::{
    apply_repetition_penalty, sample_greedy, sample_temperature, sample_top_k, sample_top_p,
    SimpleRng,
};
