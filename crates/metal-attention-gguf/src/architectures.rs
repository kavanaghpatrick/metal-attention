//! Model architecture detection and weight role mapping.

/// Supported model architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelArchitecture {
    Llama,
    Rwkv,
    Jamba,
    Griffin,
    Zamba,
    Unknown,
}

impl ModelArchitecture {
    /// Parse from the `general.architecture` metadata string.
    pub fn from_str_name(name: &str) -> Self {
        match name.to_lowercase().as_str() {
            "llama" => Self::Llama,
            "rwkv" | "rwkv6" | "rwkv7" => Self::Rwkv,
            "jamba" => Self::Jamba,
            "griffin" | "recurrentgemma" => Self::Griffin,
            "zamba" | "zamba2" => Self::Zamba,
            _ => Self::Unknown,
        }
    }
}

/// The role of a weight tensor in the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeightRole {
    // Common across architectures
    TokenEmbedding,
    OutputNorm,
    Output,
    // Attention
    AttnNorm,
    QProj,
    KProj,
    VProj,
    OutProj,
    // FFN
    FfnNorm,
    GateProj,
    UpProj,
    DownProj,
    // SSM (Mamba/Jamba)
    SSMIn,
    SSMOut,
    SSMConv1d,
    SSMDt,
    SSMA,
    SSMD,
    // RWKV specific
    TimeMixKey,
    TimeMixValue,
    TimeMixReceptance,
    TimeMixGate,
    TimeMixFirst,
    TimeMixDecay,
    ChannelMixKey,
    ChannelMixValue,
    ChannelMixReceptance,
    TimeMixLnX,
    // Griffin specific
    RecurrentGate,
    RecurrentLinear,
    RecurrentOut,
}

/// Map a tensor name to its (layer_index, WeightRole) for a given architecture.
///
/// Returns None if the tensor name doesn't match known patterns.
pub fn map_tensor_name(arch: ModelArchitecture, name: &str) -> Option<(Option<usize>, WeightRole)> {
    // Non-layer tensors (shared across architectures)
    if name == "token_embd.weight" {
        return Some((None, WeightRole::TokenEmbedding));
    }
    if name == "output_norm.weight" {
        return Some((None, WeightRole::OutputNorm));
    }
    if name == "output.weight" {
        return Some((None, WeightRole::Output));
    }

    // Extract layer index from "blk.N.rest"
    if let Some(rest) = name.strip_prefix("blk.") {
        if let Some(dot_pos) = rest.find('.') {
            if let Ok(layer) = rest[..dot_pos].parse::<usize>() {
                let suffix = &rest[dot_pos + 1..];
                let role = match arch {
                    ModelArchitecture::Llama
                    | ModelArchitecture::Zamba
                    | ModelArchitecture::Unknown => map_llama_suffix(suffix),
                    ModelArchitecture::Rwkv => map_rwkv_suffix(suffix),
                    ModelArchitecture::Jamba => map_jamba_suffix(suffix),
                    ModelArchitecture::Griffin => map_griffin_suffix(suffix),
                };
                return role.map(|r| (Some(layer), r));
            }
        }
    }

    None
}

fn map_llama_suffix(suffix: &str) -> Option<WeightRole> {
    match suffix {
        "attn_norm.weight" => Some(WeightRole::AttnNorm),
        "attn_q.weight" => Some(WeightRole::QProj),
        "attn_k.weight" => Some(WeightRole::KProj),
        "attn_v.weight" => Some(WeightRole::VProj),
        "attn_output.weight" => Some(WeightRole::OutProj),
        "ffn_norm.weight" => Some(WeightRole::FfnNorm),
        "ffn_gate.weight" => Some(WeightRole::GateProj),
        "ffn_up.weight" => Some(WeightRole::UpProj),
        "ffn_down.weight" => Some(WeightRole::DownProj),
        _ => None,
    }
}

fn map_rwkv_suffix(suffix: &str) -> Option<WeightRole> {
    match suffix {
        "attn_norm.weight" => Some(WeightRole::AttnNorm),
        "time_mix_key.weight" | "time_mix_k.weight" => Some(WeightRole::TimeMixKey),
        "time_mix_value.weight" | "time_mix_v.weight" => Some(WeightRole::TimeMixValue),
        "time_mix_receptance.weight" | "time_mix_r.weight" => Some(WeightRole::TimeMixReceptance),
        "time_mix_gate.weight" | "time_mix_g.weight" => Some(WeightRole::TimeMixGate),
        "time_mix_first.weight" => Some(WeightRole::TimeMixFirst),
        "time_mix_decay.weight" | "time_mix_w.weight" => Some(WeightRole::TimeMixDecay),
        "time_mix_ln.weight" => Some(WeightRole::TimeMixLnX),
        "time_mix_output.weight" | "time_mix_o.weight" => Some(WeightRole::OutProj),
        "ffn_norm.weight" | "channel_mix_norm.weight" => Some(WeightRole::FfnNorm),
        "channel_mix_key.weight" => Some(WeightRole::ChannelMixKey),
        "channel_mix_value.weight" => Some(WeightRole::ChannelMixValue),
        "channel_mix_receptance.weight" => Some(WeightRole::ChannelMixReceptance),
        _ => None,
    }
}

fn map_jamba_suffix(suffix: &str) -> Option<WeightRole> {
    match suffix {
        // Attention layers
        "attn_norm.weight" => Some(WeightRole::AttnNorm),
        "attn_q.weight" => Some(WeightRole::QProj),
        "attn_k.weight" => Some(WeightRole::KProj),
        "attn_v.weight" => Some(WeightRole::VProj),
        "attn_output.weight" => Some(WeightRole::OutProj),
        // SSM layers
        "ssm_in.weight" => Some(WeightRole::SSMIn),
        "ssm_out.weight" => Some(WeightRole::SSMOut),
        "ssm_conv1d.weight" => Some(WeightRole::SSMConv1d),
        "ssm_dt.weight" => Some(WeightRole::SSMDt),
        "ssm_a" => Some(WeightRole::SSMA),
        "ssm_d" => Some(WeightRole::SSMD),
        // FFN
        "ffn_norm.weight" => Some(WeightRole::FfnNorm),
        "ffn_gate.weight" => Some(WeightRole::GateProj),
        "ffn_up.weight" => Some(WeightRole::UpProj),
        "ffn_down.weight" => Some(WeightRole::DownProj),
        _ => None,
    }
}

fn map_griffin_suffix(suffix: &str) -> Option<WeightRole> {
    match suffix {
        "attn_norm.weight" => Some(WeightRole::AttnNorm),
        "attn_q.weight" => Some(WeightRole::QProj),
        "attn_k.weight" => Some(WeightRole::KProj),
        "attn_v.weight" => Some(WeightRole::VProj),
        "attn_output.weight" => Some(WeightRole::OutProj),
        "rglru_gate.weight" | "recurrent_gate.weight" => Some(WeightRole::RecurrentGate),
        "rglru_a.weight" | "recurrent_linear.weight" => Some(WeightRole::RecurrentLinear),
        "recurrent_out.weight" => Some(WeightRole::RecurrentOut),
        "ffn_norm.weight" => Some(WeightRole::FfnNorm),
        "ffn_gate.weight" => Some(WeightRole::GateProj),
        "ffn_up.weight" => Some(WeightRole::UpProj),
        "ffn_down.weight" => Some(WeightRole::DownProj),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_architecture_from_str() {
        assert_eq!(
            ModelArchitecture::from_str_name("llama"),
            ModelArchitecture::Llama
        );
        assert_eq!(
            ModelArchitecture::from_str_name("Llama"),
            ModelArchitecture::Llama
        );
        assert_eq!(
            ModelArchitecture::from_str_name("LLAMA"),
            ModelArchitecture::Llama
        );
        assert_eq!(
            ModelArchitecture::from_str_name("rwkv"),
            ModelArchitecture::Rwkv
        );
        assert_eq!(
            ModelArchitecture::from_str_name("rwkv6"),
            ModelArchitecture::Rwkv
        );
        assert_eq!(
            ModelArchitecture::from_str_name("rwkv7"),
            ModelArchitecture::Rwkv
        );
        assert_eq!(
            ModelArchitecture::from_str_name("jamba"),
            ModelArchitecture::Jamba
        );
        assert_eq!(
            ModelArchitecture::from_str_name("griffin"),
            ModelArchitecture::Griffin
        );
        assert_eq!(
            ModelArchitecture::from_str_name("recurrentgemma"),
            ModelArchitecture::Griffin
        );
        assert_eq!(
            ModelArchitecture::from_str_name("zamba"),
            ModelArchitecture::Zamba
        );
        assert_eq!(
            ModelArchitecture::from_str_name("zamba2"),
            ModelArchitecture::Zamba
        );
        assert_eq!(
            ModelArchitecture::from_str_name("bert"),
            ModelArchitecture::Unknown
        );
    }

    #[test]
    fn test_map_common_tensors() {
        let arch = ModelArchitecture::Llama;
        assert_eq!(
            map_tensor_name(arch, "token_embd.weight"),
            Some((None, WeightRole::TokenEmbedding))
        );
        assert_eq!(
            map_tensor_name(arch, "output_norm.weight"),
            Some((None, WeightRole::OutputNorm))
        );
        assert_eq!(
            map_tensor_name(arch, "output.weight"),
            Some((None, WeightRole::Output))
        );
    }

    #[test]
    fn test_map_llama_tensors() {
        let arch = ModelArchitecture::Llama;
        assert_eq!(
            map_tensor_name(arch, "blk.0.attn_q.weight"),
            Some((Some(0), WeightRole::QProj))
        );
        assert_eq!(
            map_tensor_name(arch, "blk.5.ffn_gate.weight"),
            Some((Some(5), WeightRole::GateProj))
        );
        assert_eq!(
            map_tensor_name(arch, "blk.31.attn_output.weight"),
            Some((Some(31), WeightRole::OutProj))
        );
    }

    #[test]
    fn test_map_rwkv_tensors() {
        let arch = ModelArchitecture::Rwkv;
        assert_eq!(
            map_tensor_name(arch, "blk.0.time_mix_key.weight"),
            Some((Some(0), WeightRole::TimeMixKey))
        );
        assert_eq!(
            map_tensor_name(arch, "blk.2.channel_mix_key.weight"),
            Some((Some(2), WeightRole::ChannelMixKey))
        );
    }

    #[test]
    fn test_map_jamba_tensors() {
        let arch = ModelArchitecture::Jamba;
        assert_eq!(
            map_tensor_name(arch, "blk.0.ssm_in.weight"),
            Some((Some(0), WeightRole::SSMIn))
        );
        assert_eq!(
            map_tensor_name(arch, "blk.1.attn_q.weight"),
            Some((Some(1), WeightRole::QProj))
        );
    }

    #[test]
    fn test_map_griffin_tensors() {
        let arch = ModelArchitecture::Griffin;
        assert_eq!(
            map_tensor_name(arch, "blk.0.rglru_gate.weight"),
            Some((Some(0), WeightRole::RecurrentGate))
        );
    }

    #[test]
    fn test_map_unknown_tensor() {
        assert_eq!(
            map_tensor_name(ModelArchitecture::Llama, "some.random.tensor"),
            None
        );
    }
}
