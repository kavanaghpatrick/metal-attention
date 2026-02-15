//! Model implementations for hybrid inference.
//!
//! Concrete implementations of `SequenceBlock` traits for RWKV-7,
//! Llama/Mistral, Jamba, Griffin, and Zamba architectures.

pub mod registry;
pub mod rwkv7;

// Re-exports
pub use registry::{create_rwkv7_block, is_supported, supported_architectures, ModelConfig};
pub use rwkv7::{Rwkv7Block, Rwkv7State};
