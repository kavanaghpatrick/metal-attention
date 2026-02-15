//! GGUF metadata value types and typed accessors.

use std::collections::HashMap;

/// All possible GGUF metadata value types.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufMetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufMetadataValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

/// GGUF metadata value type IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufMetadataValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufMetadataValueType {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Uint8),
            1 => Some(Self::Int8),
            2 => Some(Self::Uint16),
            3 => Some(Self::Int16),
            4 => Some(Self::Uint32),
            5 => Some(Self::Int32),
            6 => Some(Self::Float32),
            7 => Some(Self::Bool),
            8 => Some(Self::String),
            9 => Some(Self::Array),
            10 => Some(Self::Uint64),
            11 => Some(Self::Int64),
            12 => Some(Self::Float64),
            _ => None,
        }
    }
}

/// Typed accessor methods for GGUF metadata.
#[derive(Debug, Clone)]
pub struct GgufMetadata {
    pub map: HashMap<String, GgufMetadataValue>,
}

impl GgufMetadata {
    pub fn new(map: HashMap<String, GgufMetadataValue>) -> Self {
        Self { map }
    }

    /// Get a string metadata value.
    pub fn get_string(&self, key: &str) -> Option<&str> {
        match self.map.get(key) {
            Some(GgufMetadataValue::String(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Get a u32 metadata value.
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        match self.map.get(key) {
            Some(GgufMetadataValue::Uint32(v)) => Some(*v),
            _ => None,
        }
    }

    /// Get a u64 metadata value.
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        match self.map.get(key) {
            Some(GgufMetadataValue::Uint64(v)) => Some(*v),
            _ => None,
        }
    }

    /// Get an f32 metadata value.
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        match self.map.get(key) {
            Some(GgufMetadataValue::Float32(v)) => Some(*v),
            _ => None,
        }
    }

    /// Get a bool metadata value.
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        match self.map.get(key) {
            Some(GgufMetadataValue::Bool(v)) => Some(*v),
            _ => None,
        }
    }

    /// Get a string array metadata value.
    pub fn get_array_string(&self, key: &str) -> Option<Vec<&str>> {
        match self.map.get(key) {
            Some(GgufMetadataValue::Array(arr)) => {
                let mut result = Vec::with_capacity(arr.len());
                for v in arr {
                    match v {
                        GgufMetadataValue::String(s) => result.push(s.as_str()),
                        _ => return None,
                    }
                }
                Some(result)
            }
            _ => None,
        }
    }

    /// Get an f32 array metadata value.
    pub fn get_array_f32(&self, key: &str) -> Option<Vec<f32>> {
        match self.map.get(key) {
            Some(GgufMetadataValue::Array(arr)) => {
                let mut result = Vec::with_capacity(arr.len());
                for v in arr {
                    match v {
                        GgufMetadataValue::Float32(f) => result.push(*f),
                        _ => return None,
                    }
                }
                Some(result)
            }
            _ => None,
        }
    }

    /// Get a raw metadata value.
    pub fn get(&self, key: &str) -> Option<&GgufMetadataValue> {
        self.map.get(key)
    }

    /// Check if a key exists.
    pub fn contains_key(&self, key: &str) -> bool {
        self.map.contains_key(key)
    }

    /// Number of metadata entries.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether metadata is empty.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Iterate over all key-value pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &GgufMetadataValue)> {
        self.map.iter().map(|(k, v)| (k.as_str(), v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_metadata() -> GgufMetadata {
        let mut map = HashMap::new();
        map.insert(
            "general.architecture".to_string(),
            GgufMetadataValue::String("llama".to_string()),
        );
        map.insert(
            "general.name".to_string(),
            GgufMetadataValue::String("TinyLlama".to_string()),
        );
        map.insert(
            "llama.block_count".to_string(),
            GgufMetadataValue::Uint32(22),
        );
        map.insert(
            "llama.context_length".to_string(),
            GgufMetadataValue::Uint32(2048),
        );
        map.insert(
            "llama.rope.freq_base".to_string(),
            GgufMetadataValue::Float32(10000.0),
        );
        map.insert(
            "tokenizer.ggml.tokens".to_string(),
            GgufMetadataValue::Array(vec![
                GgufMetadataValue::String("<unk>".to_string()),
                GgufMetadataValue::String("<s>".to_string()),
                GgufMetadataValue::String("</s>".to_string()),
            ]),
        );
        GgufMetadata::new(map)
    }

    #[test]
    fn test_get_string() {
        let md = sample_metadata();
        assert_eq!(md.get_string("general.architecture"), Some("llama"));
        assert_eq!(md.get_string("general.name"), Some("TinyLlama"));
        assert_eq!(md.get_string("nonexistent"), None);
        // Wrong type returns None
        assert_eq!(md.get_string("llama.block_count"), None);
    }

    #[test]
    fn test_get_u32() {
        let md = sample_metadata();
        assert_eq!(md.get_u32("llama.block_count"), Some(22));
        assert_eq!(md.get_u32("llama.context_length"), Some(2048));
        assert_eq!(md.get_u32("general.architecture"), None);
    }

    #[test]
    fn test_get_f32() {
        let md = sample_metadata();
        assert_eq!(md.get_f32("llama.rope.freq_base"), Some(10000.0));
        assert_eq!(md.get_f32("nonexistent"), None);
    }

    #[test]
    fn test_get_array_string() {
        let md = sample_metadata();
        let tokens = md.get_array_string("tokenizer.ggml.tokens").unwrap();
        assert_eq!(tokens, vec!["<unk>", "<s>", "</s>"]);
        assert_eq!(md.get_array_string("nonexistent"), None);
        // Non-array returns None
        assert_eq!(md.get_array_string("general.architecture"), None);
    }

    #[test]
    fn test_len_and_contains() {
        let md = sample_metadata();
        assert_eq!(md.len(), 6);
        assert!(!md.is_empty());
        assert!(md.contains_key("general.architecture"));
        assert!(!md.contains_key("missing"));
    }
}
