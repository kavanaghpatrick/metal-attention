//! Model registry: maps architecture enum to model construction.
//!
//! Central place to match `ModelArchitecture` from the GGUF crate
//! to concrete model block implementations.

use metal_attention_gguf::ModelArchitecture;

use crate::jamba::{build_jamba_layers, JambaLayer};
use crate::llama::LlamaLayer;
use crate::mamba::MambaBlock;
use crate::rwkv7::Rwkv7Block;

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

/// Check whether a given architecture is supported.
pub fn is_supported(arch: ModelArchitecture) -> bool {
    matches!(arch, ModelArchitecture::Rwkv | ModelArchitecture::Llama | ModelArchitecture::Jamba)
}

/// List all supported architectures.
pub fn supported_architectures() -> Vec<ModelArchitecture> {
    vec![ModelArchitecture::Rwkv, ModelArchitecture::Llama, ModelArchitecture::Jamba]
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
    fn test_is_supported() {
        assert!(is_supported(ModelArchitecture::Rwkv));
        assert!(is_supported(ModelArchitecture::Llama));
        assert!(is_supported(ModelArchitecture::Jamba));
        assert!(!is_supported(ModelArchitecture::Griffin));
        assert!(!is_supported(ModelArchitecture::Zamba));
        assert!(!is_supported(ModelArchitecture::Unknown));
    }

    #[test]
    fn test_supported_architectures() {
        let archs = supported_architectures();
        assert_eq!(archs.len(), 3);
        assert!(archs.contains(&ModelArchitecture::Rwkv));
        assert!(archs.contains(&ModelArchitecture::Llama));
        assert!(archs.contains(&ModelArchitecture::Jamba));
    }
}
