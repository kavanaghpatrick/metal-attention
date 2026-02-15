//! Memory-mapped GGUF model file parser with architecture detection.
//!
//! Parses GGUF binary format, extracts metadata and tensor information,
//! and detects model architectures (llama, rwkv, jamba, griffin).

pub mod architectures;
pub mod detect;
pub mod metadata;
pub mod parser;
pub mod quantize;
pub mod tensor;
pub mod tokenizer;

// Re-exports for convenience
pub use architectures::{map_tensor_name, ModelArchitecture, WeightRole};
pub use detect::detect_architecture;
pub use metadata::{GgufMetadata, GgufMetadataValue};
pub use parser::{GgufBuilder, GgufError, GgufFile};
pub use quantize::GgufType;
pub use tensor::GgufTensorInfo;
pub use tokenizer::GgufTokenizer;
