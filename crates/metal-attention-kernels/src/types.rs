//! repr(C) parameter structs matching shaders/types.h.
//!
//! These structs are passed directly to Metal compute kernels via setBytes or buffer,
//! so their layout must exactly match the MSL structs in types.h.

/// Attention kernel parameters shared between Rust host and Metal shaders.
///
/// Layout: 16 x u32/f32 fields = 64 bytes, 4-byte aligned.
/// Must be kept in sync with `shaders/types.h`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AttentionParams {
    /// Sequence length N
    pub seq_len: u32,
    /// Head dimension D
    pub head_dim: u32,
    /// Number of Q heads
    pub num_heads: u32,
    /// Number of KV heads (== num_heads for MHA, < for GQA)
    pub num_kv_heads: u32,
    /// Tile rows (queries)
    pub block_r: u32,
    /// Tile columns (keys)
    pub block_c: u32,
    /// Scaling factor: 1/sqrt(D)
    pub scale: f32,
    /// Variant selector: 0=standard, 1=RoPE, 2=ALiBi, 3=GQA
    pub variant: u32,
    /// Tokens per page (paged attention)
    pub page_size: u32,
    /// Total pages allocated (paged attention)
    pub num_pages: u32,
    /// Maximum context length (paged attention)
    pub max_context_len: u32,
    /// Partitioned reduce count (paged attention)
    pub num_partitions: u32,
    /// Explicit padding to reach 64 bytes
    pub _pad0: u32,
    /// Explicit padding
    pub _pad1: u32,
    /// Explicit padding
    pub _pad2: u32,
    /// Explicit padding
    pub _pad3: u32,
}

impl Default for AttentionParams {
    fn default() -> Self {
        Self {
            seq_len: 256,
            head_dim: 64,
            num_heads: 1,
            num_kv_heads: 1,
            block_r: 16,
            block_c: 64,
            scale: 1.0 / (64.0_f32).sqrt(),
            variant: 0,
            page_size: 16,
            num_pages: 0,
            max_context_len: 0,
            num_partitions: 1,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        }
    }
}

impl AttentionParams {
    /// Create params for standard (non-paged) flash attention.
    pub fn flash(seq_len: u32, head_dim: u32, num_heads: u32) -> Self {
        Self {
            seq_len,
            head_dim,
            num_heads,
            num_kv_heads: num_heads,
            scale: 1.0 / (head_dim as f32).sqrt(),
            ..Default::default()
        }
    }

    /// Create params for GQA (grouped-query attention).
    pub fn gqa(seq_len: u32, head_dim: u32, num_heads: u32, num_kv_heads: u32) -> Self {
        Self {
            seq_len,
            head_dim,
            num_heads,
            num_kv_heads,
            scale: 1.0 / (head_dim as f32).sqrt(),
            variant: 3,
            ..Default::default()
        }
    }

    /// Create params for paged attention.
    pub fn paged(
        seq_len: u32,
        head_dim: u32,
        num_heads: u32,
        page_size: u32,
        num_pages: u32,
        max_context_len: u32,
        num_partitions: u32,
    ) -> Self {
        Self {
            seq_len,
            head_dim,
            num_heads,
            num_kv_heads: num_heads,
            scale: 1.0 / (head_dim as f32).sqrt(),
            page_size,
            num_pages,
            max_context_len,
            num_partitions,
            ..Default::default()
        }
    }
}

/// Layer-level parameters for transformer block dispatch.
///
/// Layout: 16 fields (12 x u32 + 2 x f32 + 2 x u32 pad) = 64 bytes, 4-byte aligned.
/// Must be kept in sync with `shaders/types.h`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LayerParams {
    /// Model hidden dimension (e.g. 4096)
    pub hidden_dim: u32,
    /// FFN intermediate dimension (e.g. 11008)
    pub intermediate_dim: u32,
    /// Number of attention heads
    pub num_heads: u32,
    /// Number of KV heads (GQA)
    pub num_kv_heads: u32,
    /// Per-head dimension
    pub head_dim: u32,
    /// Vocabulary size for embedding
    pub vocab_size: u32,
    /// Current layer index
    pub layer_idx: u32,
    /// Total number of layers
    pub num_layers: u32,
    /// RMSNorm epsilon (e.g. 1e-5)
    pub rms_norm_eps: f32,
    /// RoPE base frequency (e.g. 10000.0)
    pub rope_theta: f32,
    /// Current sequence length
    pub seq_len: u32,
    /// Batch size
    pub batch_size: u32,
    /// Explicit padding
    pub _pad0: u32,
    /// Explicit padding
    pub _pad1: u32,
    /// Explicit padding
    pub _pad2: u32,
    /// Explicit padding
    pub _pad3: u32,
}

impl Default for LayerParams {
    fn default() -> Self {
        Self {
            hidden_dim: 4096,
            intermediate_dim: 11008,
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            vocab_size: 32000,
            layer_idx: 0,
            num_layers: 32,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            seq_len: 256,
            batch_size: 1,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        }
    }
}

/// SSM (State Space Model) parameters for Mamba/Jamba layers.
///
/// Layout: 16 fields (10 x u32 + 2 x f32 + 4 x u32 pad) = 64 bytes, 4-byte aligned.
/// Must be kept in sync with `shaders/types.h`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SSMParams {
    /// SSM state dimension (e.g. 16)
    pub state_dim: u32,
    /// Model hidden dimension
    pub hidden_dim: u32,
    /// SSM intermediate dimension
    pub intermediate_dim: u32,
    /// Number of SSM heads
    pub num_heads: u32,
    /// Per-head dimension
    pub head_dim: u32,
    /// Current sequence length
    pub seq_len: u32,
    /// Batch size
    pub batch_size: u32,
    /// Convolution kernel width (e.g. 4)
    pub conv_width: u32,
    /// Minimum delta time
    pub dt_min: f32,
    /// Maximum delta time
    pub dt_max: f32,
    /// Current layer index
    pub layer_idx: u32,
    /// Total number of layers
    pub num_layers: u32,
    /// Explicit padding
    pub _pad0: u32,
    /// Explicit padding
    pub _pad1: u32,
    /// Explicit padding
    pub _pad2: u32,
    /// Explicit padding
    pub _pad3: u32,
}

impl Default for SSMParams {
    fn default() -> Self {
        Self {
            state_dim: 16,
            hidden_dim: 4096,
            intermediate_dim: 8192,
            num_heads: 1,
            head_dim: 64,
            seq_len: 256,
            batch_size: 1,
            conv_width: 4,
            dt_min: 0.001,
            dt_max: 0.1,
            layer_idx: 0,
            num_layers: 32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    #[test]
    fn test_attention_params_layout() {
        assert_eq!(mem::size_of::<AttentionParams>(), 64);
        assert_eq!(mem::align_of::<AttentionParams>(), 4);

        assert_eq!(mem::offset_of!(AttentionParams, seq_len), 0);
        assert_eq!(mem::offset_of!(AttentionParams, head_dim), 4);
        assert_eq!(mem::offset_of!(AttentionParams, num_heads), 8);
        assert_eq!(mem::offset_of!(AttentionParams, num_kv_heads), 12);
        assert_eq!(mem::offset_of!(AttentionParams, block_r), 16);
        assert_eq!(mem::offset_of!(AttentionParams, block_c), 20);
        assert_eq!(mem::offset_of!(AttentionParams, scale), 24);
        assert_eq!(mem::offset_of!(AttentionParams, variant), 28);
        assert_eq!(mem::offset_of!(AttentionParams, page_size), 32);
        assert_eq!(mem::offset_of!(AttentionParams, num_pages), 36);
        assert_eq!(mem::offset_of!(AttentionParams, max_context_len), 40);
        assert_eq!(mem::offset_of!(AttentionParams, num_partitions), 44);
        assert_eq!(mem::offset_of!(AttentionParams, _pad0), 48);
        assert_eq!(mem::offset_of!(AttentionParams, _pad1), 52);
        assert_eq!(mem::offset_of!(AttentionParams, _pad2), 56);
        assert_eq!(mem::offset_of!(AttentionParams, _pad3), 60);
    }

    #[test]
    fn test_layer_params_layout() {
        assert_eq!(mem::size_of::<LayerParams>(), 64);
        assert_eq!(mem::align_of::<LayerParams>(), 4);

        assert_eq!(mem::offset_of!(LayerParams, hidden_dim), 0);
        assert_eq!(mem::offset_of!(LayerParams, intermediate_dim), 4);
        assert_eq!(mem::offset_of!(LayerParams, num_heads), 8);
        assert_eq!(mem::offset_of!(LayerParams, num_kv_heads), 12);
        assert_eq!(mem::offset_of!(LayerParams, head_dim), 16);
        assert_eq!(mem::offset_of!(LayerParams, vocab_size), 20);
        assert_eq!(mem::offset_of!(LayerParams, layer_idx), 24);
        assert_eq!(mem::offset_of!(LayerParams, num_layers), 28);
        assert_eq!(mem::offset_of!(LayerParams, rms_norm_eps), 32);
        assert_eq!(mem::offset_of!(LayerParams, rope_theta), 36);
        assert_eq!(mem::offset_of!(LayerParams, seq_len), 40);
        assert_eq!(mem::offset_of!(LayerParams, batch_size), 44);
        assert_eq!(mem::offset_of!(LayerParams, _pad0), 48);
        assert_eq!(mem::offset_of!(LayerParams, _pad1), 52);
        assert_eq!(mem::offset_of!(LayerParams, _pad2), 56);
        assert_eq!(mem::offset_of!(LayerParams, _pad3), 60);
    }

    #[test]
    fn test_ssm_params_layout() {
        assert_eq!(mem::size_of::<SSMParams>(), 64);
        assert_eq!(mem::align_of::<SSMParams>(), 4);

        assert_eq!(mem::offset_of!(SSMParams, state_dim), 0);
        assert_eq!(mem::offset_of!(SSMParams, hidden_dim), 4);
        assert_eq!(mem::offset_of!(SSMParams, intermediate_dim), 8);
        assert_eq!(mem::offset_of!(SSMParams, num_heads), 12);
        assert_eq!(mem::offset_of!(SSMParams, head_dim), 16);
        assert_eq!(mem::offset_of!(SSMParams, seq_len), 20);
        assert_eq!(mem::offset_of!(SSMParams, batch_size), 24);
        assert_eq!(mem::offset_of!(SSMParams, conv_width), 28);
        assert_eq!(mem::offset_of!(SSMParams, dt_min), 32);
        assert_eq!(mem::offset_of!(SSMParams, dt_max), 36);
        assert_eq!(mem::offset_of!(SSMParams, layer_idx), 40);
        assert_eq!(mem::offset_of!(SSMParams, num_layers), 44);
        assert_eq!(mem::offset_of!(SSMParams, _pad0), 48);
        assert_eq!(mem::offset_of!(SSMParams, _pad1), 52);
        assert_eq!(mem::offset_of!(SSMParams, _pad2), 56);
        assert_eq!(mem::offset_of!(SSMParams, _pad3), 60);
    }

    #[test]
    fn test_default_attention_params() {
        let params = AttentionParams::default();
        assert_eq!(params.seq_len, 256);
        assert_eq!(params.head_dim, 64);
        assert!((params.scale - 0.125).abs() < 1e-6);
    }

    #[test]
    fn test_flash_constructor() {
        let params = AttentionParams::flash(512, 128, 8);
        assert_eq!(params.seq_len, 512);
        assert_eq!(params.head_dim, 128);
        assert_eq!(params.num_heads, 8);
        assert_eq!(params.num_kv_heads, 8);
        assert!((params.scale - 1.0 / (128.0_f32).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_gqa_constructor() {
        let params = AttentionParams::gqa(1024, 64, 32, 8);
        assert_eq!(params.num_heads, 32);
        assert_eq!(params.num_kv_heads, 8);
        assert_eq!(params.variant, 3);
    }

    #[test]
    fn test_paged_constructor() {
        let params = AttentionParams::paged(2048, 64, 16, 32, 128, 4096, 4);
        assert_eq!(params.page_size, 32);
        assert_eq!(params.num_pages, 128);
        assert_eq!(params.max_context_len, 4096);
        assert_eq!(params.num_partitions, 4);
    }

    #[test]
    fn test_default_layer_params() {
        let params = LayerParams::default();
        assert_eq!(params.hidden_dim, 4096);
        assert_eq!(params.num_layers, 32);
        assert!((params.rms_norm_eps - 1e-5).abs() < 1e-10);
        assert!((params.rope_theta - 10000.0).abs() < 1e-3);
    }

    #[test]
    fn test_default_ssm_params() {
        let params = SSMParams::default();
        assert_eq!(params.state_dim, 16);
        assert_eq!(params.conv_width, 4);
        assert!((params.dt_min - 0.001).abs() < 1e-6);
        assert!((params.dt_max - 0.1).abs() < 1e-6);
    }
}
