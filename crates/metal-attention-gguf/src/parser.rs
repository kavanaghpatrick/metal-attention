//! GGUF binary file parser with memory-mapped I/O.

use std::collections::HashMap;
use std::path::Path;

use memmap2::Mmap;
use thiserror::Error;

use crate::architectures::ModelArchitecture;
use crate::detect::detect_architecture;
use crate::metadata::{GgufMetadata, GgufMetadataValue, GgufMetadataValueType};
use crate::quantize::GgufType;
use crate::tensor::GgufTensorInfo;

/// GGUF magic bytes: "GGUF" as little-endian u32 = 0x46475547
const GGUF_MAGIC: u32 = 0x4647_5547;
/// Default alignment for GGUF data section.
const DEFAULT_ALIGNMENT: usize = 32;

/// Errors during GGUF parsing.
#[derive(Debug, Error)]
pub enum GgufError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid GGUF magic: expected 0x46475547, got 0x{0:08X}")]
    BadMagic(u32),
    #[error("Unsupported GGUF version: {0} (expected 2 or 3)")]
    UnsupportedVersion(u32),
    #[error("Unknown metadata value type: {0}")]
    UnknownValueType(u32),
    #[error("Unknown tensor type: {0}")]
    UnknownTensorType(u32),
    #[error("Unexpected end of data at offset {0}")]
    UnexpectedEof(usize),
    #[error("Invalid UTF-8 in string at offset {0}")]
    InvalidUtf8(usize),
}

pub type Result<T> = std::result::Result<T, GgufError>;

/// A parsed GGUF file with memory-mapped backing.
pub struct GgufFile {
    /// Memory-mapped file contents.
    mmap: Mmap,
    /// GGUF format version (2 or 3).
    pub version: u32,
    /// Parsed metadata.
    pub metadata: GgufMetadata,
    /// Tensor info array.
    pub tensors: Vec<GgufTensorInfo>,
    /// Byte offset where tensor data begins.
    pub data_offset: usize,
    /// Detected model architecture.
    pub architecture: ModelArchitecture,
}

/// A cursor for reading from the mmap buffer.
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn check(&self, n: usize) -> Result<()> {
        if self.remaining() < n {
            Err(GgufError::UnexpectedEof(self.pos))
        } else {
            Ok(())
        }
    }

    fn read_u8(&mut self) -> Result<u8> {
        self.check(1)?;
        let v = self.data[self.pos];
        self.pos += 1;
        Ok(v)
    }

    fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16> {
        self.check(2)?;
        let v = u16::from_le_bytes([self.data[self.pos], self.data[self.pos + 1]]);
        self.pos += 2;
        Ok(v)
    }

    fn read_i16(&mut self) -> Result<i16> {
        self.check(2)?;
        let v = i16::from_le_bytes([self.data[self.pos], self.data[self.pos + 1]]);
        self.pos += 2;
        Ok(v)
    }

    fn read_u32(&mut self) -> Result<u32> {
        self.check(4)?;
        let v = u32::from_le_bytes([
            self.data[self.pos],
            self.data[self.pos + 1],
            self.data[self.pos + 2],
            self.data[self.pos + 3],
        ]);
        self.pos += 4;
        Ok(v)
    }

    fn read_i32(&mut self) -> Result<i32> {
        self.check(4)?;
        let v = i32::from_le_bytes([
            self.data[self.pos],
            self.data[self.pos + 1],
            self.data[self.pos + 2],
            self.data[self.pos + 3],
        ]);
        self.pos += 4;
        Ok(v)
    }

    fn read_u64(&mut self) -> Result<u64> {
        self.check(8)?;
        let v = u64::from_le_bytes([
            self.data[self.pos],
            self.data[self.pos + 1],
            self.data[self.pos + 2],
            self.data[self.pos + 3],
            self.data[self.pos + 4],
            self.data[self.pos + 5],
            self.data[self.pos + 6],
            self.data[self.pos + 7],
        ]);
        self.pos += 8;
        Ok(v)
    }

    fn read_i64(&mut self) -> Result<i64> {
        self.check(8)?;
        let v = i64::from_le_bytes([
            self.data[self.pos],
            self.data[self.pos + 1],
            self.data[self.pos + 2],
            self.data[self.pos + 3],
            self.data[self.pos + 4],
            self.data[self.pos + 5],
            self.data[self.pos + 6],
            self.data[self.pos + 7],
        ]);
        self.pos += 8;
        Ok(v)
    }

    fn read_f32(&mut self) -> Result<f32> {
        self.check(4)?;
        let v = f32::from_le_bytes([
            self.data[self.pos],
            self.data[self.pos + 1],
            self.data[self.pos + 2],
            self.data[self.pos + 3],
        ]);
        self.pos += 4;
        Ok(v)
    }

    fn read_f64(&mut self) -> Result<f64> {
        self.check(8)?;
        let v = f64::from_le_bytes([
            self.data[self.pos],
            self.data[self.pos + 1],
            self.data[self.pos + 2],
            self.data[self.pos + 3],
            self.data[self.pos + 4],
            self.data[self.pos + 5],
            self.data[self.pos + 6],
            self.data[self.pos + 7],
        ]);
        self.pos += 8;
        Ok(v)
    }

    /// Read a GGUF string: u64 length + bytes.
    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()? as usize;
        self.check(len)?;
        let s = std::str::from_utf8(&self.data[self.pos..self.pos + len])
            .map_err(|_| GgufError::InvalidUtf8(self.pos))?;
        self.pos += len;
        Ok(s.to_string())
    }

    /// Read a metadata value of the given type.
    fn read_metadata_value(&mut self, vtype: GgufMetadataValueType) -> Result<GgufMetadataValue> {
        match vtype {
            GgufMetadataValueType::Uint8 => Ok(GgufMetadataValue::Uint8(self.read_u8()?)),
            GgufMetadataValueType::Int8 => Ok(GgufMetadataValue::Int8(self.read_i8()?)),
            GgufMetadataValueType::Uint16 => Ok(GgufMetadataValue::Uint16(self.read_u16()?)),
            GgufMetadataValueType::Int16 => Ok(GgufMetadataValue::Int16(self.read_i16()?)),
            GgufMetadataValueType::Uint32 => Ok(GgufMetadataValue::Uint32(self.read_u32()?)),
            GgufMetadataValueType::Int32 => Ok(GgufMetadataValue::Int32(self.read_i32()?)),
            GgufMetadataValueType::Float32 => Ok(GgufMetadataValue::Float32(self.read_f32()?)),
            GgufMetadataValueType::Bool => {
                let v = self.read_u8()?;
                Ok(GgufMetadataValue::Bool(v != 0))
            }
            GgufMetadataValueType::String => {
                let s = self.read_string()?;
                Ok(GgufMetadataValue::String(s))
            }
            GgufMetadataValueType::Array => {
                let elem_type_raw = self.read_u32()?;
                let elem_type = GgufMetadataValueType::from_u32(elem_type_raw)
                    .ok_or(GgufError::UnknownValueType(elem_type_raw))?;
                let len = self.read_u64()? as usize;
                let mut arr = Vec::with_capacity(len);
                for _ in 0..len {
                    arr.push(self.read_metadata_value(elem_type)?);
                }
                Ok(GgufMetadataValue::Array(arr))
            }
            GgufMetadataValueType::Uint64 => Ok(GgufMetadataValue::Uint64(self.read_u64()?)),
            GgufMetadataValueType::Int64 => Ok(GgufMetadataValue::Int64(self.read_i64()?)),
            GgufMetadataValueType::Float64 => Ok(GgufMetadataValue::Float64(self.read_f64()?)),
        }
    }
}

impl GgufFile {
    /// Open and parse a GGUF file from disk using memory-mapped I/O.
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Self::parse(mmap)
    }

    /// Parse a GGUF file from raw bytes (writes to temp file for mmap).
    pub fn from_bytes(data: Vec<u8>) -> Result<Self> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir();
        let path = dir.join(format!("gguf_test_{}_{}.gguf", std::process::id(), id));
        std::fs::write(&path, &data)?;
        let result = Self::open(&path);
        let _ = std::fs::remove_file(&path);
        result
    }

    /// Internal parse from mmap.
    fn parse(mmap: Mmap) -> Result<Self> {
        let mut reader = Reader::new(&mmap);

        // Header
        let magic = reader.read_u32()?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::BadMagic(magic));
        }

        let version = reader.read_u32()?;
        if version != 2 && version != 3 {
            return Err(GgufError::UnsupportedVersion(version));
        }

        let tensor_count = reader.read_u64()? as usize;
        let metadata_kv_count = reader.read_u64()? as usize;

        // Parse metadata key-value pairs
        let mut metadata_map = HashMap::with_capacity(metadata_kv_count);
        for _ in 0..metadata_kv_count {
            let key = reader.read_string()?;
            let vtype_raw = reader.read_u32()?;
            let vtype = GgufMetadataValueType::from_u32(vtype_raw)
                .ok_or(GgufError::UnknownValueType(vtype_raw))?;
            let value = reader.read_metadata_value(vtype)?;
            metadata_map.insert(key, value);
        }
        let metadata = GgufMetadata::new(metadata_map);

        // Get alignment from metadata or use default
        let alignment = metadata
            .get_u32("general.alignment")
            .map(|v| v as usize)
            .unwrap_or(DEFAULT_ALIGNMENT);

        // Parse tensor info array
        let mut tensors = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = reader.read_string()?;
            let ndim = reader.read_u32()?;
            let mut shape = Vec::with_capacity(ndim as usize);
            for _ in 0..ndim {
                shape.push(reader.read_u64()?);
            }
            let type_raw = reader.read_u32()?;
            let gguf_type =
                GgufType::from_u32(type_raw).ok_or(GgufError::UnknownTensorType(type_raw))?;
            let offset_in_data = reader.read_u64()?;
            tensors.push(GgufTensorInfo {
                name,
                ndim,
                shape,
                gguf_type,
                offset_in_data,
            });
        }

        // Calculate data section offset (align reader position to alignment boundary)
        let data_offset = align_up(reader.pos, alignment);

        // Detect architecture
        let architecture = detect_architecture(&metadata, &tensors);

        Ok(Self {
            mmap,
            version,
            metadata,
            tensors,
            data_offset,
            architecture,
        })
    }

    /// Get the raw memory-mapped bytes.
    pub fn mmap_bytes(&self) -> &[u8] {
        &self.mmap
    }

    /// Get tensor data bytes for a specific tensor.
    pub fn tensor_data(&self, tensor: &GgufTensorInfo) -> &[u8] {
        tensor.data(&self.mmap, self.data_offset)
    }

    /// Find a tensor by name.
    pub fn find_tensor(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }
}

/// Align a value up to the given alignment boundary.
fn align_up(value: usize, alignment: usize) -> usize {
    if alignment == 0 {
        return value;
    }
    (value + alignment - 1) & !(alignment - 1)
}

// ─── Synthetic GGUF builder (for tests) ──────────────────────────────────────

/// Builder for creating synthetic GGUF files in-memory for testing.
#[derive(Default)]
pub struct GgufBuilder {
    version: u32,
    metadata: Vec<(String, GgufMetadataValueType, Vec<u8>)>,
    tensors: Vec<(String, Vec<u64>, GgufType, Vec<u8>)>,
    alignment: usize,
}

impl GgufBuilder {
    pub fn new() -> Self {
        Self {
            version: 3,
            metadata: Vec::new(),
            tensors: Vec::new(),
            alignment: DEFAULT_ALIGNMENT,
        }
    }

    pub fn version(mut self, v: u32) -> Self {
        self.version = v;
        self
    }

    fn write_string(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    pub fn add_string(mut self, key: &str, value: &str) -> Self {
        let mut val_bytes = Vec::new();
        Self::write_string(&mut val_bytes, value);
        self.metadata
            .push((key.to_string(), GgufMetadataValueType::String, val_bytes));
        self
    }

    pub fn add_u32(mut self, key: &str, value: u32) -> Self {
        self.metadata.push((
            key.to_string(),
            GgufMetadataValueType::Uint32,
            value.to_le_bytes().to_vec(),
        ));
        self
    }

    pub fn add_f32(mut self, key: &str, value: f32) -> Self {
        self.metadata.push((
            key.to_string(),
            GgufMetadataValueType::Float32,
            value.to_le_bytes().to_vec(),
        ));
        self
    }

    pub fn add_bool(mut self, key: &str, value: bool) -> Self {
        self.metadata.push((
            key.to_string(),
            GgufMetadataValueType::Bool,
            vec![value as u8],
        ));
        self
    }

    pub fn add_f32_array(mut self, key: &str, values: &[f32]) -> Self {
        let mut val_bytes = Vec::new();
        // Array header: element type (Float32=6) + count
        val_bytes.extend_from_slice(&(GgufMetadataValueType::Float32 as u32).to_le_bytes());
        val_bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for &f in values {
            val_bytes.extend_from_slice(&f.to_le_bytes());
        }
        self.metadata
            .push((key.to_string(), GgufMetadataValueType::Array, val_bytes));
        self
    }

    pub fn add_string_array(mut self, key: &str, values: &[&str]) -> Self {
        let mut val_bytes = Vec::new();
        // Array header: element type (string=8) + count
        val_bytes.extend_from_slice(&(GgufMetadataValueType::String as u32).to_le_bytes());
        val_bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for s in values {
            Self::write_string(&mut val_bytes, s);
        }
        self.metadata
            .push((key.to_string(), GgufMetadataValueType::Array, val_bytes));
        self
    }

    /// Add a tensor with explicit data.
    pub fn add_tensor(mut self, name: &str, shape: &[u64], dtype: GgufType, data: Vec<u8>) -> Self {
        self.tensors
            .push((name.to_string(), shape.to_vec(), dtype, data));
        self
    }

    /// Add a tensor with zero-filled data of the appropriate size.
    pub fn add_tensor_zeros(self, name: &str, shape: &[u64], dtype: GgufType) -> Self {
        let n_elements: u64 = shape.iter().product();
        let byte_size = dtype.tensor_byte_size(n_elements as usize);
        let data = vec![0u8; byte_size];
        self.add_tensor(name, shape, dtype, data)
    }

    /// Build the GGUF binary.
    pub fn build(self) -> Vec<u8> {
        let mut buf = Vec::new();

        // Magic
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        // Version
        buf.extend_from_slice(&self.version.to_le_bytes());
        // Tensor count
        buf.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        // Metadata KV count
        buf.extend_from_slice(&(self.metadata.len() as u64).to_le_bytes());

        // Metadata KV pairs
        for (key, vtype, val_bytes) in &self.metadata {
            Self::write_string(&mut buf, key);
            buf.extend_from_slice(&(*vtype as u32).to_le_bytes());
            buf.extend_from_slice(val_bytes);
        }

        // Calculate tensor data offsets
        // First, write tensor info headers, then pad to alignment, then data
        let mut tensor_info_bytes = Vec::new();
        let mut data_offset_within_section: u64 = 0;
        let mut tensor_offsets = Vec::new();

        for (name, shape, dtype, data) in &self.tensors {
            // Align the offset for this tensor
            let aligned = align_up(data_offset_within_section as usize, self.alignment) as u64;
            tensor_offsets.push(aligned);

            Self::write_string(&mut tensor_info_bytes, name);
            tensor_info_bytes.extend_from_slice(&(shape.len() as u32).to_le_bytes());
            for &dim in shape {
                tensor_info_bytes.extend_from_slice(&dim.to_le_bytes());
            }
            tensor_info_bytes.extend_from_slice(&(*dtype as u32).to_le_bytes());
            tensor_info_bytes.extend_from_slice(&aligned.to_le_bytes());

            data_offset_within_section = aligned + data.len() as u64;
        }

        buf.extend_from_slice(&tensor_info_bytes);

        // Align to data section
        let current = buf.len();
        let aligned_data_start = align_up(current, self.alignment);
        buf.resize(aligned_data_start, 0);

        // Write tensor data
        for (i, (_, _, _, data)) in self.tensors.iter().enumerate() {
            let target = aligned_data_start + tensor_offsets[i] as usize;
            if buf.len() < target {
                buf.resize(target, 0);
            }
            buf.extend_from_slice(data);
        }

        buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 32), 0);
        assert_eq!(align_up(1, 32), 32);
        assert_eq!(align_up(31, 32), 32);
        assert_eq!(align_up(32, 32), 32);
        assert_eq!(align_up(33, 32), 64);
    }

    #[test]
    fn test_parse_minimal_gguf() {
        // Build a minimal GGUF with 1 metadata and 1 tensor
        let data = GgufBuilder::new()
            .add_string("general.architecture", "llama")
            .add_tensor_zeros("token_embd.weight", &[32, 4096], GgufType::F32)
            .build();

        let file = GgufFile::from_bytes(data).expect("parse failed");
        assert_eq!(file.version, 3);
        assert_eq!(file.metadata.get_string("general.architecture"), Some("llama"));
        assert_eq!(file.tensors.len(), 1);
        assert_eq!(file.tensors[0].name, "token_embd.weight");
        assert_eq!(file.tensors[0].shape, vec![32, 4096]);
        assert_eq!(file.tensors[0].gguf_type, GgufType::F32);
        assert_eq!(file.architecture, ModelArchitecture::Llama);
    }

    #[test]
    fn test_parse_v2_gguf() {
        let data = GgufBuilder::new()
            .version(2)
            .add_string("general.architecture", "llama")
            .build();

        let file = GgufFile::from_bytes(data).expect("parse failed");
        assert_eq!(file.version, 2);
    }

    #[test]
    fn test_parse_multiple_metadata_types() {
        let data = GgufBuilder::new()
            .add_string("general.architecture", "llama")
            .add_string("general.name", "TestModel")
            .add_u32("llama.block_count", 22)
            .add_f32("llama.rope.freq_base", 10000.0)
            .add_bool("llama.attention.causal", true)
            .add_string_array("tokenizer.ggml.tokens", &["hello", "world"])
            .build();

        let file = GgufFile::from_bytes(data).expect("parse failed");
        assert_eq!(file.metadata.get_string("general.name"), Some("TestModel"));
        assert_eq!(file.metadata.get_u32("llama.block_count"), Some(22));
        assert_eq!(file.metadata.get_f32("llama.rope.freq_base"), Some(10000.0));
        assert_eq!(file.metadata.get_bool("llama.attention.causal"), Some(true));

        let tokens = file
            .metadata
            .get_array_string("tokenizer.ggml.tokens")
            .unwrap();
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn test_parse_multiple_tensors() {
        let data = GgufBuilder::new()
            .add_string("general.architecture", "llama")
            .add_tensor_zeros("token_embd.weight", &[32000, 4096], GgufType::F16)
            .add_tensor_zeros("blk.0.attn_q.weight", &[4096, 4096], GgufType::Q4_0)
            .add_tensor_zeros("blk.0.ffn_gate.weight", &[11008, 4096], GgufType::Q4_K_M)
            .build();

        let file = GgufFile::from_bytes(data).expect("parse failed");
        assert_eq!(file.tensors.len(), 3);

        let embd = file.find_tensor("token_embd.weight").unwrap();
        assert_eq!(embd.gguf_type, GgufType::F16);
        assert_eq!(embd.shape, vec![32000, 4096]);

        let q = file.find_tensor("blk.0.attn_q.weight").unwrap();
        assert_eq!(q.gguf_type, GgufType::Q4_0);

        let gate = file.find_tensor("blk.0.ffn_gate.weight").unwrap();
        assert_eq!(gate.gguf_type, GgufType::Q4_K_M);
    }

    #[test]
    fn test_tensor_data_access() {
        // Create a small tensor with known data
        let tensor_data: Vec<u8> = (0..16).collect();
        let data = GgufBuilder::new()
            .add_string("general.architecture", "llama")
            .add_tensor("small", &[4], GgufType::F32, tensor_data.clone())
            .build();

        let file = GgufFile::from_bytes(data).expect("parse failed");
        let t = file.find_tensor("small").unwrap();
        let bytes = file.tensor_data(t);
        assert_eq!(bytes, &tensor_data[..]);
    }

    #[test]
    fn test_bad_magic() {
        let mut data = vec![0u8; 32];
        // Write wrong magic
        data[0..4].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());
        let result = GgufFile::from_bytes(data);
        assert!(matches!(result, Err(GgufError::BadMagic(0xDEADBEEF))));
    }

    #[test]
    fn test_unsupported_version() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(&GGUF_MAGIC.to_le_bytes());
        data[4..8].copy_from_slice(&99u32.to_le_bytes()); // bad version
        let result = GgufFile::from_bytes(data);
        assert!(matches!(result, Err(GgufError::UnsupportedVersion(99))));
    }

    #[test]
    fn test_architecture_detection_from_tensors() {
        // No explicit architecture metadata -- detect from tensor names
        let data = GgufBuilder::new()
            .add_string("general.name", "TestModel")
            .add_tensor_zeros("blk.0.attn_q.weight", &[64, 64], GgufType::F32)
            .add_tensor_zeros("blk.0.attn_k.weight", &[64, 64], GgufType::F32)
            .add_tensor_zeros("blk.0.attn_v.weight", &[64, 64], GgufType::F32)
            .build();

        let file = GgufFile::from_bytes(data).expect("parse failed");
        assert_eq!(file.architecture, ModelArchitecture::Llama);
    }

    #[test]
    fn test_jamba_detection() {
        let data = GgufBuilder::new()
            .add_string("general.architecture", "jamba")
            .add_tensor_zeros("blk.0.attn_q.weight", &[64, 64], GgufType::F32)
            .add_tensor_zeros("blk.1.ssm_in.weight", &[64, 64], GgufType::F32)
            .build();

        let file = GgufFile::from_bytes(data).expect("parse failed");
        assert_eq!(file.architecture, ModelArchitecture::Jamba);
    }

    #[test]
    fn test_data_offset_alignment() {
        // With very little header data, data_offset should be aligned to 32
        let data = GgufBuilder::new()
            .add_string("general.architecture", "llama")
            .add_tensor_zeros("w", &[4], GgufType::F32)
            .build();

        let file = GgufFile::from_bytes(data).expect("parse failed");
        assert_eq!(file.data_offset % 32, 0, "data_offset must be 32-byte aligned");
    }
}
