//! Model registry: maps architecture enum to model construction.
//!
//! Central place to match `ModelArchitecture` from the GGUF crate
//! to concrete model block implementations.

use metal_attention_gguf::ModelArchitecture;

use crate::griffin::{build_griffin_layers, GriffinLayer};
use crate::jamba::{build_jamba_layers, JambaLayer};
use crate::llama::LlamaLayer;
use crate::mamba::MambaBlock;
use crate::rwkv7::Rwkv7Block;
use crate::zamba::{build_zamba_model, ZambaModel};

/// Model configuration extracted from GGUF metadata.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub architecture: ModelArchitecture,
    pub hidden_size: usize,
    pub head_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub num_layers: usize,
}

/// Create an RWKV-7 block from a model configuration.
///
/// Returns None if the architecture is not RWKV.
pub fn create_rwkv7_block(config: &ModelConfig, layer_index: usize) -> Option<Rwkv7Block> {
    if config.architecture != ModelArchitecture::Rwkv {
        return None;
    }

    // Create with random weights for now; real implementation will load from GGUF
    Some(Rwkv7Block::random(
        config.hidden_size,
        config.head_dim,
        config.num_heads,
        (layer_index as u64) * 31 + 7, // deterministic seed per layer
    ))
}

/// Create a Llama layer from a model configuration.
///
/// Returns None if the architecture is not Llama.
pub fn create_llama_layer(config: &ModelConfig, layer_index: usize) -> Option<LlamaLayer> {
    if config.architecture != ModelArchitecture::Llama {
        return None;
    }

    // Create with random weights for now; real implementation will load from GGUF
    Some(LlamaLayer::random(
        config.hidden_size,
        config.head_dim,
        config.num_heads,
        config.num_kv_heads,
        (layer_index as u64) * 31 + 7, // deterministic seed per layer
    ))
}

/// Create a Mamba block from a model configuration.
///
/// Returns None if the architecture is not Jamba (Mamba blocks are used inside Jamba).
pub fn create_mamba_block(config: &ModelConfig, layer_index: usize) -> Option<MambaBlock> {
    if config.architecture != ModelArchitecture::Jamba {
        return None;
    }

    Some(MambaBlock::random(
        config.hidden_size,
        16, // default d_state
        (layer_index as u64) * 31 + 7,
    ))
}

/// Create Jamba layers from a model configuration.
///
/// Returns None if the architecture is not Jamba.
/// Returns a Vec of JambaLayer with 7:1 Mamba:Attention schedule.
pub fn create_jamba_layers(config: &ModelConfig) -> Option<Vec<JambaLayer>> {
    if config.architecture != ModelArchitecture::Jamba {
        return None;
    }

    let (layers, _schedule) = build_jamba_layers(
        config.hidden_size,
        config.head_dim,
        config.num_heads,
        config.num_kv_heads,
        config.num_layers,
        16,  // d_state
        16,  // num_experts
        42,  // seed
    );

    Some(layers)
}

/// Create Griffin layers from a model configuration.
///
/// Returns None if the architecture is not Griffin.
/// Returns a Vec of GriffinLayer with 2:1 RG-LRU:Attention schedule.
pub fn create_griffin_layers(config: &ModelConfig) -> Option<Vec<GriffinLayer>> {
    if config.architecture != ModelArchitecture::Griffin {
        return None;
    }

    let (layers, _schedule) = build_griffin_layers(
        config.hidden_size,
        config.head_dim,
        config.num_heads,
        config.num_kv_heads,
        config.num_layers,
        42,  // seed
    );

    Some(layers)
}

/// Create a Zamba model from a model configuration.
///
/// Returns None if the architecture is not Zamba.
/// Returns a ZambaModel with 6:1 Mamba:SharedAttention schedule.
pub fn create_zamba_model(config: &ModelConfig) -> Option<ZambaModel> {
    if config.architecture != ModelArchitecture::Zamba {
        return None;
    }

    let model = build_zamba_model(
        config.hidden_size,
        config.head_dim,
        config.num_heads,
        config.num_kv_heads,
        config.num_layers,
        16,  // d_state
        8,   // lora_rank
        42,  // seed
    );

    Some(model)
}

/// Check whether a given architecture is supported.
pub fn is_supported(arch: ModelArchitecture) -> bool {
    matches!(
        arch,
        ModelArchitecture::Rwkv
            | ModelArchitecture::Llama
            | ModelArchitecture::Jamba
            | ModelArchitecture::Griffin
            | ModelArchitecture::Zamba
    )
}

/// List all supported architectures.
pub fn supported_architectures() -> Vec<ModelArchitecture> {
    vec![
        ModelArchitecture::Rwkv,
        ModelArchitecture::Llama,
        ModelArchitecture::Jamba,
        ModelArchitecture::Griffin,
        ModelArchitecture::Zamba,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_rwkv7_block() {
        let config = ModelConfig {
            architecture: ModelArchitecture::Rwkv,
            hidden_size: 16,
            head_dim: 16,
            num_heads: 1,
            num_kv_heads: 1,
            num_layers: 4,
        };

        let block = create_rwkv7_block(&config, 0);
        assert!(block.is_some(), "Should create RWKV-7 block for Rwkv arch");

        let block = block.unwrap();
        assert_eq!(block.hidden_size, 16);
        assert_eq!(block.head_dim, 16);
        assert_eq!(block.num_heads, 1);
    }

    #[test]
    fn test_create_rwkv7_block_wrong_arch() {
        let config = ModelConfig {
            architecture: ModelArchitecture::Llama,
            hidden_size: 16,
            head_dim: 16,
            num_heads: 1,
            num_kv_heads: 1,
            num_layers: 4,
        };

        let block = create_rwkv7_block(&config, 0);
        assert!(block.is_none(), "Should not create RWKV-7 block for Llama arch");
    }

    #[test]
    fn test_create_llama_layer() {
        let config = ModelConfig {
            architecture: ModelArchitecture::Llama,
            hidden_size: 32,
            head_dim: 16,
            num_heads: 2,
            num_kv_heads: 2,
            num_layers: 4,
        };

        let layer = create_llama_layer(&config, 0);
        assert!(layer.is_some(), "Should create Llama layer for Llama arch");

        let layer = layer.unwrap();
        assert_eq!(layer.hidden_size, 32);
        assert_eq!(layer.attention.num_heads, 2);
    }

    #[test]
    fn test_create_llama_layer_wrong_arch() {
        let config = ModelConfig {
            architecture: ModelArchitecture::Rwkv,
            hidden_size: 32,
            head_dim: 16,
            num_heads: 2,
            num_kv_heads: 2,
            num_layers: 4,
        };

        let layer = create_llama_layer(&config, 0);
        assert!(layer.is_none(), "Should not create Llama layer for Rwkv arch");
    }

    #[test]
    fn test_create_griffin_layers() {
        let config = ModelConfig {
            architecture: ModelArchitecture::Griffin,
            hidden_size: 16,
            head_dim: 8,
            num_heads: 2,
            num_kv_heads: 2,
            num_layers: 6,
        };

        let layers = create_griffin_layers(&config);
        assert!(layers.is_some(), "Should create Griffin layers for Griffin arch");

        let layers = layers.unwrap();
        assert_eq!(layers.len(), 6);

        // 2:1 schedule: 4 RG-LRU, 2 Attention
        let rglru_count = layers.iter().filter(|l| l.is_rglru()).count();
        let attn_count = layers.iter().filter(|l| l.is_attention()).count();
        assert_eq!(rglru_count, 4);
        assert_eq!(attn_count, 2);
    }

    #[test]
    fn test_create_griffin_layers_wrong_arch() {
        let config = ModelConfig {
            architecture: ModelArchitecture::Llama,
            hidden_size: 16,
            head_dim: 8,
            num_heads: 2,
            num_kv_heads: 2,
            num_layers: 6,
        };

        let layers = create_griffin_layers(&config);
        assert!(layers.is_none(), "Should not create Griffin layers for Llama arch");
    }

    #[test]
    fn test_create_zamba_model() {
        let config = ModelConfig {
            architecture: ModelArchitecture::Zamba,
            hidden_size: 16,
            head_dim: 8,
            num_heads: 2,
            num_kv_heads: 2,
            num_layers: 7,
        };

        let model = create_zamba_model(&config);
        assert!(model.is_some(), "Should create Zamba model for Zamba arch");

        let model = model.unwrap();
        assert_eq!(model.layers.len(), 7);

        // 6:1 schedule: 6 Mamba, 1 Attention
        let mamba_count = model.layers.iter().filter(|l| l.is_mamba()).count();
        let attn_count = model.layers.iter().filter(|l| l.is_attention()).count();
        assert_eq!(mamba_count, 6);
        assert_eq!(attn_count, 1);
    }

    #[test]
    fn test_create_zamba_model_wrong_arch() {
        let config = ModelConfig {
            architecture: ModelArchitecture::Jamba,
            hidden_size: 16,
            head_dim: 8,
            num_heads: 2,
            num_kv_heads: 2,
            num_layers: 7,
        };

        let model = create_zamba_model(&config);
        assert!(model.is_none(), "Should not create Zamba model for Jamba arch");
    }

    #[test]
    fn test_is_supported() {
        assert!(is_supported(ModelArchitecture::Rwkv));
        assert!(is_supported(ModelArchitecture::Llama));
        assert!(is_supported(ModelArchitecture::Jamba));
        assert!(is_supported(ModelArchitecture::Griffin));
        assert!(is_supported(ModelArchitecture::Zamba));
        assert!(!is_supported(ModelArchitecture::Unknown));
    }

    #[test]
    fn test_supported_architectures() {
        let archs = supported_architectures();
        assert_eq!(archs.len(), 5);
        assert!(archs.contains(&ModelArchitecture::Rwkv));
        assert!(archs.contains(&ModelArchitecture::Llama));
        assert!(archs.contains(&ModelArchitecture::Jamba));
        assert!(archs.contains(&ModelArchitecture::Griffin));
        assert!(archs.contains(&ModelArchitecture::Zamba));
    }
}
