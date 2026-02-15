//! GGUF tensor info and data access.

use crate::quantize::GgufType;

/// Information about a single tensor in a GGUF file.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    /// Tensor name (e.g. "blk.0.attn_q.weight").
    pub name: String,
    /// Number of dimensions.
    pub ndim: u32,
    /// Shape array (length = ndim).
    pub shape: Vec<u64>,
    /// Quantization / data type.
    pub gguf_type: GgufType,
    /// Byte offset of this tensor's data within the data section.
    pub offset_in_data: u64,
}

impl GgufTensorInfo {
    /// Total number of elements in this tensor.
    pub fn n_elements(&self) -> u64 {
        self.shape.iter().product::<u64>().max(1)
    }

    /// Total byte size of this tensor's data.
    pub fn byte_size(&self) -> usize {
        self.gguf_type.tensor_byte_size(self.n_elements() as usize)
    }

    /// Get a byte slice of this tensor's raw data from the mmap buffer.
    ///
    /// - `mmap`: the full memory-mapped file contents
    /// - `data_offset`: byte offset where the data section starts in the file
    pub fn data<'a>(&self, mmap: &'a [u8], data_offset: usize) -> &'a [u8] {
        let start = data_offset + self.offset_in_data as usize;
        let end = start + self.byte_size();
        &mmap[start..end]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_n_elements() {
        let info = GgufTensorInfo {
            name: "test".to_string(),
            ndim: 2,
            shape: vec![4, 8],
            gguf_type: GgufType::F32,
            offset_in_data: 0,
        };
        assert_eq!(info.n_elements(), 32);
    }

    #[test]
    fn test_byte_size_f32() {
        let info = GgufTensorInfo {
            name: "test".to_string(),
            ndim: 2,
            shape: vec![4, 8],
            gguf_type: GgufType::F32,
            offset_in_data: 0,
        };
        assert_eq!(info.byte_size(), 128); // 32 * 4
    }

    #[test]
    fn test_byte_size_q4_0() {
        let info = GgufTensorInfo {
            name: "test".to_string(),
            ndim: 1,
            shape: vec![64],
            gguf_type: GgufType::Q4_0,
            offset_in_data: 0,
        };
        // 64 elements / 32 per block = 2 blocks * 18 bytes = 36
        assert_eq!(info.byte_size(), 36);
    }

    #[test]
    fn test_data_slice() {
        let info = GgufTensorInfo {
            name: "w".to_string(),
            ndim: 1,
            shape: vec![1],
            gguf_type: GgufType::F32,
            offset_in_data: 8,
        };
        // Simulate mmap: data_offset=100, tensor at offset 8 within data section
        let mut buf = vec![0u8; 200];
        buf[108] = 0xAB;
        buf[109] = 0xCD;
        buf[110] = 0xEF;
        buf[111] = 0x12;
        let data = info.data(&buf, 100);
        assert_eq!(data.len(), 4);
        assert_eq!(data, &[0xAB, 0xCD, 0xEF, 0x12]);
    }
}
