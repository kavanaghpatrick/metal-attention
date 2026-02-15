//! Core types shared across the trait hierarchy.
//!
//! These types are pure Rust with zero Metal dependencies.

/// Tensor descriptor for inference. Does not own data -- references Metal buffers.
#[derive(Debug, Clone)]
pub struct TensorView {
    /// Byte offset into the backing Metal buffer.
    pub offset: usize,
    /// Shape in elements (e.g., [seq_len, head_dim]).
    pub shape: Vec<usize>,
    /// Element stride per dimension.
    pub strides: Vec<usize>,
    /// Element data type.
    pub dtype: DType,
}

impl TensorView {
    /// Create a new TensorView with the given shape and dtype.
    /// Strides are computed assuming contiguous row-major layout.
    pub fn new(shape: Vec<usize>, dtype: DType) -> Self {
        let strides = Self::compute_strides(&shape);
        Self {
            offset: 0,
            shape,
            strides,
            dtype,
        }
    }

    /// Create a new TensorView with explicit offset, shape, strides, and dtype.
    pub fn with_offset(offset: usize, shape: Vec<usize>, strides: Vec<usize>, dtype: DType) -> Self {
        Self {
            offset,
            shape,
            strides,
            dtype,
        }
    }

    /// Compute contiguous row-major strides for a given shape.
    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1usize; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Total number of elements in this tensor.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Total size in bytes based on dtype and element count.
    pub fn size_bytes(&self) -> usize {
        self.num_elements() * self.dtype.size_bytes()
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
}

/// Data types supported by the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum DType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_K_M,
    Q8_0,
}

impl DType {
    /// Size of a single element in bytes.
    ///
    /// For quantized types, returns the average bytes per element
    /// based on the block structure.
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::BF16 => 2,
            // Q4_0: 32 elements per block, 18 bytes per block (16 bytes data + 2 bytes scale)
            // Average: 18/32 ≈ 0.5625, but we use 1 as minimum per-element accounting
            DType::Q4_0 => 1,
            // Q4_K_M: similar to Q4_0 but with more metadata
            DType::Q4_K_M => 1,
            // Q8_0: 32 elements per block, 34 bytes per block (32 bytes data + 2 bytes scale)
            DType::Q8_0 => 1,
        }
    }
}

/// Configuration for a single sequence block within a layer.
#[derive(Debug, Clone)]
pub struct BlockConfig {
    pub hidden_size: usize,
    pub head_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub layer_index: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_view_construction() {
        let tv = TensorView::new(vec![4, 8], DType::F32);
        assert_eq!(tv.shape, vec![4, 8]);
        assert_eq!(tv.strides, vec![8, 1]);
        assert_eq!(tv.offset, 0);
        assert_eq!(tv.dtype, DType::F32);
        assert_eq!(tv.num_elements(), 32);
        assert_eq!(tv.size_bytes(), 128); // 32 * 4 bytes
        assert_eq!(tv.ndim(), 2);
    }

    #[test]
    fn test_tensor_view_with_offset() {
        let tv = TensorView::with_offset(256, vec![2, 3, 4], vec![12, 4, 1], DType::F16);
        assert_eq!(tv.offset, 256);
        assert_eq!(tv.shape, vec![2, 3, 4]);
        assert_eq!(tv.strides, vec![12, 4, 1]);
        assert_eq!(tv.dtype, DType::F16);
        assert_eq!(tv.num_elements(), 24);
        assert_eq!(tv.size_bytes(), 48); // 24 * 2 bytes
    }

    #[test]
    fn test_tensor_view_scalar() {
        let tv = TensorView::new(vec![1], DType::F32);
        assert_eq!(tv.strides, vec![1]);
        assert_eq!(tv.num_elements(), 1);
        assert_eq!(tv.size_bytes(), 4);
    }

    #[test]
    fn test_tensor_view_3d_strides() {
        let tv = TensorView::new(vec![2, 3, 4], DType::F32);
        assert_eq!(tv.strides, vec![12, 4, 1]);
    }

    #[test]
    fn test_dtype_sizes() {
        assert_eq!(DType::F32.size_bytes(), 4);
        assert_eq!(DType::F16.size_bytes(), 2);
        assert_eq!(DType::BF16.size_bytes(), 2);
        assert_eq!(DType::Q4_0.size_bytes(), 1);
        assert_eq!(DType::Q4_K_M.size_bytes(), 1);
        assert_eq!(DType::Q8_0.size_bytes(), 1);
    }

    #[test]
    fn test_block_config() {
        let config = BlockConfig {
            hidden_size: 4096,
            head_dim: 128,
            num_heads: 32,
            num_kv_heads: 8,
            layer_index: 0,
        };
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.layer_index, 0);
    }

    #[test]
    fn test_type_sizes() {
        // TensorView should be reasonably small (heap-allocated vecs)
        assert!(std::mem::size_of::<DType>() <= 1);
        assert!(std::mem::size_of::<BlockConfig>() <= 48);
    }
}
