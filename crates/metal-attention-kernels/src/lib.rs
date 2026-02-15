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
pub mod gqa;
pub mod gpu_rope;
pub mod kv_cache;
pub mod linear;
pub mod matmul;
pub mod matvec_q4_0;
pub mod norm;
pub mod paged;
pub mod pipeline;
pub mod prefix_sum;
pub mod residual;
pub mod rope;
pub mod rwkv;
pub mod ssm;
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
pub use gqa::{cpu_gqa_remap, dispatch_gqa_remap};
pub use gpu_rope::{cpu_rope_apply, dispatch_rope_apply};
pub use kv_cache::{DenseKVCache, PagedKVCache, PAGE_SIZE};
pub use linear::dispatch_linear_attention;
pub use matmul::dispatch_matmul;
pub use matvec_q4_0::dispatch_matvec_q4_0;
pub use norm::{dispatch_rmsnorm, dispatch_rmsnorm_optimized};
pub use residual::dispatch_residual_add;
pub use paged::{cpu_paged_attention, dispatch_paged_attention, interleave_kv_pages};
pub use pipeline::{ConstantType, ConstantValue, PsoCache, PsoKey};
pub use prefix_sum::{cpu_prefix_sum, dispatch_prefix_sum};
pub use rope::{cpu_rope, dispatch_rope};
pub use rwkv::{cpu_rwkv_wkv, dispatch_rwkv_wkv};
pub use ssm::{cpu_ssm_scan, dispatch_ssm_scan};
pub use types::{AttentionParams, LayerParams, SSMParams};
