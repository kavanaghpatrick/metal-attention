//! Model implementations for hybrid inference.
//!
//! Concrete implementations of `SequenceBlock` traits for RWKV-7,
//! Llama/Mistral, Jamba, Griffin, and Zamba architectures.

pub mod flash_attn;
pub mod griffin;
pub mod jamba;
pub mod llama;
pub mod mamba;
pub mod registry;
pub mod rwkv7;
pub mod zamba;

// Re-exports
pub use flash_attn::{FlashAttentionLayer, FlashAttentionState};
pub use griffin::{build_griffin_layers, GriffinLayer, GriffinLayerState, RgLruBlock, RgLruState};
pub use jamba::{build_jamba_layers, JambaLayer, JambaLayerState, MoEFFN};
pub use llama::{LlamaLayer, LlamaState};
pub use mamba::{MambaBlock, MambaState};
pub use registry::{create_griffin_layers, create_jamba_layers, create_llama_layer, create_mamba_block, create_rwkv7_block, create_zamba_model, is_supported, supported_architectures, ModelConfig};
pub use rwkv7::{Rwkv7Block, Rwkv7State};
pub use zamba::{build_zamba_model, LoraProjector, ZambaLayer, ZambaLayerState, ZambaModel};
