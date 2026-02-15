//! Model loading integration tests.
//!
//! Tests GGUF parsing, architecture detection, and weight tensor mapping
//! using synthetic GGUF data.

use metal_attention_gguf::{architectures::ModelArchitecture, GgufBuilder, GgufFile, GgufType};

#[test]
fn test_gguf_parse_minimal() {
    let data = GgufBuilder::new()
        .add_string("general.architecture", "llama")
        .add_u32("llama.block_count", 2)
        .add_tensor_zeros("token_embd.weight", &[100, 64], GgufType::F32)
        .build();

    let file = GgufFile::from_bytes(data).expect("parse failed");
    assert_eq!(file.version, 3);
    assert_eq!(
        file.metadata.get_string("general.architecture"),
        Some("llama")
    );
    assert_eq!(file.metadata.get_u32("llama.block_count"), Some(2));
    assert_eq!(file.tensors.len(), 1);
    assert_eq!(file.tensors[0].name, "token_embd.weight");
    assert_eq!(file.tensors[0].shape, vec![100, 64]);
}

#[test]
fn test_gguf_parse_metadata_types() {
    let data = GgufBuilder::new()
        .add_string("general.name", "TestModel")
        .add_u32("model.layers", 12)
        .add_f32("model.rope_freq_base", 10000.0)
        .add_bool("model.causal", true)
        .add_string_array("tokenizer.tokens", &["<s>", "</s>", "hello"])
        .add_f32_array("model.scales", &[1.0, 2.0, 3.0])
        .build();

    let file = GgufFile::from_bytes(data).expect("parse failed");
    assert_eq!(file.metadata.get_string("general.name"), Some("TestModel"));
    assert_eq!(file.metadata.get_u32("model.layers"), Some(12));
    assert_eq!(file.metadata.get_f32("model.rope_freq_base"), Some(10000.0));
    assert_eq!(file.metadata.get_bool("model.causal"), Some(true));

    let tokens = file
        .metadata
        .get_array_string("tokenizer.tokens")
        .expect("tokens array missing");
    assert_eq!(tokens, vec!["<s>", "</s>", "hello"]);

    let scales = file
        .metadata
        .get_array_f32("model.scales")
        .expect("scales array missing");
    assert_eq!(scales, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_architecture_detection_llama() {
    let data = GgufBuilder::new()
        .add_string("general.architecture", "llama")
        .add_tensor_zeros("blk.0.attn_q.weight", &[64, 64], GgufType::F32)
        .add_tensor_zeros("blk.0.ffn_gate.weight", &[128, 64], GgufType::F32)
        .build();

    let file = GgufFile::from_bytes(data).expect("parse failed");
    assert_eq!(file.architecture, ModelArchitecture::Llama);
}

#[test]
fn test_architecture_detection_rwkv() {
    let data = GgufBuilder::new()
        .add_string("general.architecture", "rwkv7")
        .add_tensor_zeros("blk.0.time_mix_key.weight", &[64, 64], GgufType::F32)
        .add_tensor_zeros("blk.0.channel_mix_key.weight", &[64, 64], GgufType::F32)
        .build();

    let file = GgufFile::from_bytes(data).expect("parse failed");
    assert_eq!(file.architecture, ModelArchitecture::Rwkv);
}

#[test]
fn test_architecture_detection_jamba() {
    let data = GgufBuilder::new()
        .add_string("general.architecture", "jamba")
        .add_tensor_zeros("blk.0.attn_q.weight", &[64, 64], GgufType::F32)
        .add_tensor_zeros("blk.1.ssm_in.weight", &[64, 64], GgufType::F32)
        .build();

    let file = GgufFile::from_bytes(data).expect("parse failed");
    assert_eq!(file.architecture, ModelArchitecture::Jamba);
}

#[test]
fn test_architecture_detection_griffin() {
    let data = GgufBuilder::new()
        .add_string("general.architecture", "griffin")
        .add_tensor_zeros("blk.0.rglru_gate.weight", &[64, 64], GgufType::F32)
        .build();

    let file = GgufFile::from_bytes(data).expect("parse failed");
    assert_eq!(file.architecture, ModelArchitecture::Griffin);
}

#[test]
fn test_architecture_detection_from_tensor_patterns() {
    // No explicit architecture metadata - detect from tensor names
    let data = GgufBuilder::new()
        .add_string("general.name", "Mystery Model")
        .add_tensor_zeros("blk.0.attn_q.weight", &[64, 64], GgufType::F32)
        .add_tensor_zeros("blk.0.attn_k.weight", &[64, 64], GgufType::F32)
        .add_tensor_zeros("blk.0.attn_v.weight", &[64, 64], GgufType::F32)
        .build();

    let file = GgufFile::from_bytes(data).expect("parse failed");
    assert_eq!(
        file.architecture,
        ModelArchitecture::Llama,
        "should detect Llama from attention tensor patterns"
    );
}

#[test]
fn test_tensor_to_buffer_mapping() {
    let tensor_data: Vec<u8> = (0u8..64).collect();
    let data = GgufBuilder::new()
        .add_string("general.architecture", "llama")
        .add_tensor("weights.test", &[16], GgufType::F32, tensor_data.clone())
        .build();

    let file = GgufFile::from_bytes(data).expect("parse failed");
    let tensor = file.find_tensor("weights.test").expect("tensor not found");

    assert_eq!(tensor.name, "weights.test");
    assert_eq!(tensor.shape, vec![16]);
    assert_eq!(tensor.gguf_type, GgufType::F32);

    let bytes = file.tensor_data(tensor);
    assert_eq!(bytes, &tensor_data[..]);
}

#[test]
fn test_multiple_tensors_correct_shapes() {
    let data = GgufBuilder::new()
        .add_string("general.architecture", "llama")
        .add_tensor_zeros("token_embd.weight", &[1000, 128], GgufType::F16)
        .add_tensor_zeros("blk.0.attn_q.weight", &[128, 128], GgufType::Q4_0)
        .add_tensor_zeros("blk.0.ffn_gate.weight", &[512, 128], GgufType::Q4_K_M)
        .add_tensor_zeros("output.weight", &[1000, 128], GgufType::F16)
        .build();

    let file = GgufFile::from_bytes(data).expect("parse failed");
    assert_eq!(file.tensors.len(), 4);

    let embd = file.find_tensor("token_embd.weight").unwrap();
    assert_eq!(embd.shape, vec![1000, 128]);
    assert_eq!(embd.gguf_type, GgufType::F16);

    let q = file.find_tensor("blk.0.attn_q.weight").unwrap();
    assert_eq!(q.shape, vec![128, 128]);
    assert_eq!(q.gguf_type, GgufType::Q4_0);

    let gate = file.find_tensor("blk.0.ffn_gate.weight").unwrap();
    assert_eq!(gate.shape, vec![512, 128]);
    assert_eq!(gate.gguf_type, GgufType::Q4_K_M);

    let output = file.find_tensor("output.weight").unwrap();
    assert_eq!(output.shape, vec![1000, 128]);
    assert_eq!(output.gguf_type, GgufType::F16);
}

#[test]
fn test_invalid_magic_returns_error() {
    let mut data = vec![0u8; 32];
    // Write wrong magic
    data[0..4].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());
    let result = GgufFile::from_bytes(data);
    assert!(result.is_err());
    let err_msg = result.err().unwrap().to_string();
    assert!(err_msg.contains("Invalid GGUF magic"));
}

#[test]
fn test_unsupported_version_returns_error() {
    let mut data = vec![0u8; 32];
    // GGUF magic
    const GGUF_MAGIC: u32 = 0x4647_5547;
    data[0..4].copy_from_slice(&GGUF_MAGIC.to_le_bytes());
    // Invalid version
    data[4..8].copy_from_slice(&99u32.to_le_bytes());
    let result = GgufFile::from_bytes(data);
    assert!(result.is_err());
    let err_msg = result.err().unwrap().to_string();
    assert!(err_msg.contains("Unsupported GGUF version"));
}

#[test]
fn test_truncated_data_returns_error() {
    // Create valid GGUF header but truncate the data section
    let data = GgufBuilder::new()
        .add_string("general.architecture", "llama")
        .add_tensor_zeros("big.weight", &[1000, 1000], GgufType::F32)
        .build();

    // Truncate to just the header
    let truncated = data[..100].to_vec();
    let result = GgufFile::from_bytes(truncated);
    assert!(result.is_err());
    let err_msg = result.err().unwrap().to_string();
    assert!(err_msg.contains("Unexpected end of data"));
}

#[test]
fn test_v2_gguf_compatibility() {
    let data = GgufBuilder::new()
        .version(2)
        .add_string("general.architecture", "llama")
        .add_tensor_zeros("token_embd.weight", &[100, 64], GgufType::F32)
        .build();

    let file = GgufFile::from_bytes(data).expect("parse failed");
    assert_eq!(file.version, 2);
    assert_eq!(file.architecture, ModelArchitecture::Llama);
}
