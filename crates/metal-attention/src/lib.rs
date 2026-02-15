//! Hybrid model inference engine for Apple Silicon.
//!
//! Composable attention traits compiled to zero-overhead Metal kernels
//! via function constants. Supports prefill/decode inference with
//! schedule-driven hybrid layer dispatch.
