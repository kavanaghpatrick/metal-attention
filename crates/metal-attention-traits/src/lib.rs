//! Core traits for composable attention mechanisms.
//!
//! Pure Rust crate defining `SequenceBlock`, `LinearSequenceModel`,
//! and `SoftmaxAttention` — the trait hierarchy that all model
//! implementations build on.
//!
//! This crate has **zero** Metal dependencies. GPU-specific implementations
//! live in downstream crates (e.g., `metal-attention-kernels`).

pub mod attention;
pub mod linear;
pub mod schedule;
pub mod sequence;
pub mod types;

// Re-export core types at crate root for convenience.
pub use attention::{GQAConfig, KVCacheMode, PositionEncoding, SoftmaxAttention};
pub use linear::{LinearPositionEncoding, LinearSequenceModel};
pub use schedule::{LayerSchedule, LayerType};
pub use sequence::SequenceBlock;
pub use types::{BlockConfig, DType, TensorView};
