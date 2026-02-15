//! Core traits for composable attention mechanisms.
//!
//! Pure Rust crate defining `SequenceBlock`, `LinearSequenceModel`,
//! and `SoftmaxAttention` — the trait hierarchy that all model
//! implementations build on.
//!
//! This crate has **zero** Metal dependencies. GPU-specific implementations
//! live in downstream crates (e.g., `metal-attention-kernels`).

pub mod types;
pub mod sequence;
pub mod linear;
pub mod attention;
pub mod schedule;

// Re-export core types at crate root for convenience.
pub use types::{TensorView, DType, BlockConfig};
pub use sequence::SequenceBlock;
pub use linear::{LinearSequenceModel, LinearPositionEncoding};
pub use attention::{SoftmaxAttention, PositionEncoding, KVCacheMode, GQAConfig};
pub use schedule::{LayerType, LayerSchedule};
