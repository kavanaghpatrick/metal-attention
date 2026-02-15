//! Metal GPU compute kernels for attention, normalization, and inference primitives.
//!
//! Provides GPU device management, pipeline state caching, buffer pools,
//! command management, and dispatch wrappers for all Metal shader kernels.

pub mod buffer;
pub mod command;
pub mod device;
pub mod dispatch;
pub mod flash;
pub mod linear;
pub mod pipeline;
pub mod types;

// Re-export primary types for convenience
pub use buffer::{alloc_buffer, alloc_buffer_with_data, BufferPool};
pub use command::CommandManager;
pub use device::GpuDevice;
pub use dispatch::{dispatch_1d, dispatch_2d, dispatch_threadgroups, set_buffer, set_bytes};
pub use flash::dispatch_flash_attention;
pub use linear::dispatch_linear_attention;
pub use pipeline::{ConstantType, ConstantValue, PsoCache, PsoKey};
pub use types::{AttentionParams, LayerParams, SSMParams};
