//! Metal GPU compute kernels for attention, normalization, and inference primitives.
//!
//! Provides GPU device management, pipeline state caching, buffer pools,
//! command management, and dispatch wrappers for all Metal shader kernels.

pub mod buffer;
pub mod command;
pub mod dequant;
pub mod device;
pub mod dispatch;
pub mod embed;
pub mod ffn;
pub mod flash;
pub mod linear;
pub mod matmul;
pub mod norm;
pub mod pipeline;
pub mod rwkv;
pub mod types;

// Re-export primary types for convenience
pub use buffer::{alloc_buffer, alloc_buffer_with_data, BufferPool};
pub use command::CommandManager;
pub use dequant::{dispatch_dequantize_q4_0, dispatch_dequantize_q8_0};
pub use device::GpuDevice;
pub use dispatch::{dispatch_1d, dispatch_2d, dispatch_threadgroups, set_buffer, set_bytes};
pub use embed::dispatch_embedding_lookup;
pub use ffn::dispatch_ffn_silu;
pub use flash::dispatch_flash_attention;
pub use linear::dispatch_linear_attention;
pub use matmul::dispatch_matmul;
pub use norm::dispatch_rmsnorm;
pub use pipeline::{ConstantType, ConstantValue, PsoCache, PsoKey};
pub use rwkv::{cpu_rwkv_wkv, dispatch_rwkv_wkv};
pub use types::{AttentionParams, LayerParams, SSMParams};
