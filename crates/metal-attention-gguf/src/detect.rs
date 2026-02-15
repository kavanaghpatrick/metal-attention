//! Architecture detection from GGUF metadata and tensor name patterns.

use crate::architectures::ModelArchitecture;
use crate::metadata::GgufMetadata;
use crate::tensor::GgufTensorInfo;

/// Detect the model architecture from metadata.
///
/// Strategy:
/// 1. Check `general.architecture` metadata key (primary)
/// 2. Fallback: scan tensor names for architecture-specific patterns
pub fn detect_architecture(
    metadata: &GgufMetadata,
    tensors: &[GgufTensorInfo],
) -> ModelArchitecture {
    // Primary: check explicit metadata key
    if let Some(arch_str) = metadata.get_string("general.architecture") {
        let arch = ModelArchitecture::from_str_name(arch_str);
        if arch != ModelArchitecture::Unknown {
            return arch;
        }
    }

    // Fallback: pattern match on tensor names
    detect_from_tensor_names(tensors)
}

/// Detect architecture by scanning tensor name patterns.
fn detect_from_tensor_names(tensors: &[GgufTensorInfo]) -> ModelArchitecture {
    let mut has_attn = false;
    let mut has_ssm = false;
    let mut has_time_mix = false;
    let mut has_channel_mix = false;
    let mut has_rglru = false;
    let mut has_lora = false;

    for t in tensors {
        let name = &t.name;
        if name.contains("attn_q.") || name.contains("attn_k.") || name.contains("attn_v.") {
            has_attn = true;
        }
        if name.contains(".ssm_in.") || name.contains(".ssm_out.") || name.contains(".ssm_conv1d.")
        {
            has_ssm = true;
        }
        if name.contains("time_mix_") {
            has_time_mix = true;
        }
        if name.contains("channel_mix_") {
            has_channel_mix = true;
        }
        if name.contains("rglru_") || name.contains("recurrent_gate") {
            has_rglru = true;
        }
        if name.contains("lora_") || name.contains(".shared_attn.") {
            has_lora = true;
        }
    }

    // RWKV: has time_mix and channel_mix, no standard attention
    if has_time_mix && has_channel_mix {
        return ModelArchitecture::Rwkv;
    }

    // Zamba: has SSM + attention + LoRA/shared attention markers
    if has_attn && has_ssm && has_lora {
        return ModelArchitecture::Zamba;
    }

    // Jamba: has both attention and SSM layers (but no LoRA)
    if has_attn && has_ssm {
        return ModelArchitecture::Jamba;
    }

    // Griffin: has recurrent gate (RG-LRU)
    if has_rglru {
        return ModelArchitecture::Griffin;
    }

    // Llama-like: has standard attention but no SSM/RWKV
    if has_attn && !has_ssm && !has_time_mix {
        return ModelArchitecture::Llama;
    }

    ModelArchitecture::Unknown
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantize::GgufType;
    use std::collections::HashMap;

    fn make_metadata(arch: Option<&str>) -> GgufMetadata {
        let mut map = HashMap::new();
        if let Some(a) = arch {
            map.insert(
                "general.architecture".to_string(),
                crate::metadata::GgufMetadataValue::String(a.to_string()),
            );
        }
        GgufMetadata::new(map)
    }

    fn make_tensor(name: &str) -> GgufTensorInfo {
        GgufTensorInfo {
            name: name.to_string(),
            ndim: 1,
            shape: vec![1],
            gguf_type: GgufType::F32,
            offset_in_data: 0,
        }
    }

    #[test]
    fn test_detect_from_metadata() {
        let md = make_metadata(Some("llama"));
        assert_eq!(detect_architecture(&md, &[]), ModelArchitecture::Llama);

        let md = make_metadata(Some("rwkv"));
        assert_eq!(detect_architecture(&md, &[]), ModelArchitecture::Rwkv);

        let md = make_metadata(Some("jamba"));
        assert_eq!(detect_architecture(&md, &[]), ModelArchitecture::Jamba);
    }

    #[test]
    fn test_detect_llama_from_tensors() {
        let md = make_metadata(None);
        let tensors = vec![
            make_tensor("blk.0.attn_q.weight"),
            make_tensor("blk.0.attn_k.weight"),
            make_tensor("blk.0.attn_v.weight"),
            make_tensor("blk.0.ffn_gate.weight"),
        ];
        assert_eq!(detect_architecture(&md, &tensors), ModelArchitecture::Llama);
    }

    #[test]
    fn test_detect_rwkv_from_tensors() {
        let md = make_metadata(None);
        let tensors = vec![
            make_tensor("blk.0.time_mix_key.weight"),
            make_tensor("blk.0.time_mix_value.weight"),
            make_tensor("blk.0.channel_mix_key.weight"),
        ];
        assert_eq!(detect_architecture(&md, &tensors), ModelArchitecture::Rwkv);
    }

    #[test]
    fn test_detect_jamba_from_tensors() {
        let md = make_metadata(None);
        let tensors = vec![
            make_tensor("blk.0.attn_q.weight"),
            make_tensor("blk.0.ssm_in.weight"),
            make_tensor("blk.0.ssm_out.weight"),
        ];
        assert_eq!(detect_architecture(&md, &tensors), ModelArchitecture::Jamba);
    }

    #[test]
    fn test_detect_griffin_from_tensors() {
        let md = make_metadata(None);
        let tensors = vec![
            make_tensor("blk.0.rglru_gate.weight"),
            make_tensor("blk.0.rglru_a.weight"),
        ];
        assert_eq!(
            detect_architecture(&md, &tensors),
            ModelArchitecture::Griffin
        );
    }

    #[test]
    fn test_detect_griffin_from_metadata() {
        let md = make_metadata(Some("griffin"));
        assert_eq!(detect_architecture(&md, &[]), ModelArchitecture::Griffin);

        let md = make_metadata(Some("recurrentgemma"));
        assert_eq!(detect_architecture(&md, &[]), ModelArchitecture::Griffin);
    }

    #[test]
    fn test_detect_zamba_from_metadata() {
        let md = make_metadata(Some("zamba"));
        assert_eq!(detect_architecture(&md, &[]), ModelArchitecture::Zamba);

        let md = make_metadata(Some("zamba2"));
        assert_eq!(detect_architecture(&md, &[]), ModelArchitecture::Zamba);
    }

    #[test]
    fn test_detect_zamba_from_tensors() {
        let md = make_metadata(None);
        let tensors = vec![
            make_tensor("blk.0.attn_q.weight"),
            make_tensor("blk.0.ssm_in.weight"),
            make_tensor("blk.0.ssm_out.weight"),
            make_tensor("blk.0.lora_a.weight"),
        ];
        assert_eq!(detect_architecture(&md, &tensors), ModelArchitecture::Zamba);
    }

    #[test]
    fn test_detect_unknown() {
        let md = make_metadata(None);
        let tensors = vec![make_tensor("some.random.tensor")];
        assert_eq!(
            detect_architecture(&md, &tensors),
            ModelArchitecture::Unknown
        );
    }
}
