//! GGUF quantization type definitions and block size tables.

/// GGUF tensor data types (quantization formats).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum GgufType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    // 4 and 5 are legacy types (Q4_2, Q4_3), not used
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    #[allow(non_camel_case_types)]
    Q2_K = 10,
    #[allow(non_camel_case_types)]
    Q3_K = 11,
    #[allow(non_camel_case_types)]
    Q4_K = 12,
    #[allow(non_camel_case_types)]
    Q5_K = 13,
    #[allow(non_camel_case_types)]
    Q6_K = 14,
    #[allow(non_camel_case_types)]
    Q8_K = 15,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    BF16 = 30,
}

impl GgufType {
    /// Try to create from a raw u32 type id.
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2_K),
            11 => Some(Self::Q3_K),
            12 => Some(Self::Q4_K),
            13 => Some(Self::Q5_K),
            14 => Some(Self::Q6_K),
            15 => Some(Self::Q8_K),
            24 => Some(Self::I8),
            25 => Some(Self::I16),
            26 => Some(Self::I32),
            27 => Some(Self::I64),
            28 => Some(Self::F64),
            30 => Some(Self::BF16),
            _ => None,
        }
    }

    /// Number of elements per quantization block.
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 | Self::F64 => 1,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 => 1,
            Self::Q4_0 | Self::Q4_1 => 32,
            Self::Q5_0 | Self::Q5_1 => 32,
            Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2_K => 256,
            Self::Q3_K => 256,
            Self::Q4_K => 256,
            Self::Q5_K => 256,
            Self::Q6_K => 256,
            Self::Q8_K => 256,
        }
    }

    /// Bytes per quantization block.
    pub fn bytes_per_block(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::BF16 => 2,
            Self::F64 => 8,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            // Q4_0: 2 bytes scale (f16) + 16 bytes data (32 nibbles) = 18 bytes
            Self::Q4_0 => 18,
            // Q4_1: 2 bytes scale + 2 bytes min + 16 bytes data = 20 bytes
            Self::Q4_1 => 20,
            // Q5_0: 2 bytes scale + 4 bytes high-bits + 16 bytes data = 22 bytes
            Self::Q5_0 => 22,
            // Q5_1: 2 bytes scale + 2 bytes min + 4 bytes high-bits + 16 bytes data = 24 bytes
            Self::Q5_1 => 24,
            // Q8_0: 2 bytes scale + 32 bytes data = 34 bytes
            Self::Q8_0 => 34,
            // Q8_1: 4 bytes scale + 4 bytes sum + 32 bytes data = 36 bytes (using f32 scale+sum)
            Self::Q8_1 => 36,
            // Q2_K: 256 elements, ~64+16+1+1 = 82 bytes? Actually from llama.cpp:
            // QK_K=256, 256/4=64 bytes qs + 16 bytes scales + 2 bytes dmin + 2 bytes d = 84
            Self::Q2_K => 84,
            // Q3_K: 256/4=64 qs + 256/8=32 hmask + 12 scales + 2 d = 110
            Self::Q3_K => 110,
            // Q4_K: 256/2=128 qs + 12 scales + 2 d + 2 dmin = 144
            Self::Q4_K => 144,
            // Q5_K: 256/2=128 qs + 256/8=32 qh + 12 scales + 2 d + 2 dmin = 176
            Self::Q5_K => 176,
            // Q6_K: 256/2=128 ql + 256/4=64 qh + 256/16=16 scales + 2 d = 210
            Self::Q6_K => 210,
            // Q8_K: 256 qs + 16 bsums (int16) + 4 d (float32) = 292
            Self::Q8_K => 292,
        }
    }

    /// Calculate the total byte size for `n_elements` values of this type.
    pub fn tensor_byte_size(&self, n_elements: usize) -> usize {
        let bs = self.block_size();
        let n_blocks = n_elements.div_ceil(bs);
        n_blocks * self.bytes_per_block()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_u32_roundtrip() {
        assert_eq!(GgufType::from_u32(0), Some(GgufType::F32));
        assert_eq!(GgufType::from_u32(1), Some(GgufType::F16));
        assert_eq!(GgufType::from_u32(2), Some(GgufType::Q4_0));
        assert_eq!(GgufType::from_u32(8), Some(GgufType::Q8_0));
        assert_eq!(GgufType::from_u32(13), Some(GgufType::Q5_K));
        assert_eq!(GgufType::from_u32(14), Some(GgufType::Q6_K));
        assert_eq!(GgufType::from_u32(99), None);
    }

    #[test]
    fn test_block_sizes() {
        assert_eq!(GgufType::F32.block_size(), 1);
        assert_eq!(GgufType::F16.block_size(), 1);
        assert_eq!(GgufType::Q4_0.block_size(), 32);
        assert_eq!(GgufType::Q8_0.block_size(), 32);
        assert_eq!(GgufType::Q4_K.block_size(), 256);
        assert_eq!(GgufType::Q6_K.block_size(), 256);
        assert_eq!(GgufType::Q8_K.block_size(), 256);
    }

    #[test]
    fn test_bytes_per_block() {
        assert_eq!(GgufType::F32.bytes_per_block(), 4);
        assert_eq!(GgufType::F16.bytes_per_block(), 2);
        assert_eq!(GgufType::Q4_0.bytes_per_block(), 18);
        assert_eq!(GgufType::Q8_0.bytes_per_block(), 34);
        assert_eq!(GgufType::Q4_K.bytes_per_block(), 144);
        assert_eq!(GgufType::Q5_K.bytes_per_block(), 176);
        assert_eq!(GgufType::Q6_K.bytes_per_block(), 210);
        assert_eq!(GgufType::Q8_K.bytes_per_block(), 292);
    }

    #[test]
    fn test_tensor_byte_size() {
        // 1024 F32 elements = 4096 bytes
        assert_eq!(GgufType::F32.tensor_byte_size(1024), 4096);
        // 1024 F16 elements = 2048 bytes
        assert_eq!(GgufType::F16.tensor_byte_size(1024), 2048);
        // 32 Q4_0 elements = 1 block * 18 bytes = 18
        assert_eq!(GgufType::Q4_0.tensor_byte_size(32), 18);
        // 64 Q4_0 elements = 2 blocks * 18 = 36
        assert_eq!(GgufType::Q4_0.tensor_byte_size(64), 36);
        // 256 Q4_K elements = 1 block * 144 bytes = 144
        assert_eq!(GgufType::Q4_K.tensor_byte_size(256), 144);
        assert_eq!(GgufType::Q5_K.tensor_byte_size(256), 176);
        assert_eq!(GgufType::Q6_K.tensor_byte_size(256), 210);
    }
}
