//! Layer scheduling for hybrid models.
//!
//! Determines which layers use linear attention vs softmax attention
//! in architectures like Jamba, Zamba, and RWKV-7.

/// Layer type in a hybrid model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    Linear,
    Attention,
}

/// Schedule determining which layers use Linear vs Attention.
#[derive(Debug, Clone)]
pub struct LayerSchedule {
    /// Layer types in order. Length = total_layers.
    pub types: Vec<LayerType>,
}

impl LayerSchedule {
    /// Create a periodic schedule: every (ratio+1)-th layer is Attention,
    /// the rest are Linear. E.g., ratio=7 means layers 0-6 are Linear,
    /// layer 7 is Attention, layers 8-14 are Linear, layer 15 is Attention, etc.
    ///
    /// Special case: ratio=0 produces pure transformer (all attention).
    pub fn periodic(total_layers: usize, ratio: usize) -> Self {
        let mut types = Vec::with_capacity(total_layers);
        for i in 0..total_layers {
            if ratio == 0 {
                // Pure transformer: all attention
                types.push(LayerType::Attention);
            } else if (i + 1) % (ratio + 1) == 0 {
                types.push(LayerType::Attention);
            } else {
                types.push(LayerType::Linear);
            }
        }
        Self { types }
    }

    /// Create from explicit list (for architectures with irregular patterns).
    pub fn explicit(types: Vec<LayerType>) -> Self {
        Self { types }
    }

    /// Pure transformer (all attention layers).
    pub fn pure_transformer(total_layers: usize) -> Self {
        Self::periodic(total_layers, 0)
    }

    /// Pure linear (all linear layers).
    pub fn pure_linear(total_layers: usize) -> Self {
        Self {
            types: vec![LayerType::Linear; total_layers],
        }
    }

    /// Total number of layers.
    pub fn total_layers(&self) -> usize {
        self.types.len()
    }

    /// Count of linear layers.
    pub fn linear_count(&self) -> usize {
        self.types
            .iter()
            .filter(|t| **t == LayerType::Linear)
            .count()
    }

    /// Count of attention layers.
    pub fn attention_count(&self) -> usize {
        self.types
            .iter()
            .filter(|t| **t == LayerType::Attention)
            .count()
    }

    /// Get the layer type at a given index.
    pub fn layer_type(&self, index: usize) -> Option<LayerType> {
        self.types.get(index).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_periodic_7_1_pattern() {
        // periodic(32, 7) should produce 7 Linear : 1 Attention pattern
        let schedule = LayerSchedule::periodic(32, 7);
        assert_eq!(schedule.total_layers(), 32);

        // Attention layers should be at positions 7, 15, 23, 31
        for i in 0..32 {
            let expected = if (i + 1) % 8 == 0 {
                LayerType::Attention
            } else {
                LayerType::Linear
            };
            assert_eq!(
                schedule.types[i], expected,
                "Layer {i}: expected {expected:?}, got {:?}",
                schedule.types[i]
            );
        }

        // 4 attention layers, 28 linear layers
        assert_eq!(schedule.attention_count(), 4);
        assert_eq!(schedule.linear_count(), 28);
    }

    #[test]
    fn test_periodic_ratio_1() {
        // ratio=1 means alternating: L, A, L, A, ...
        let schedule = LayerSchedule::periodic(8, 1);
        assert_eq!(schedule.types[0], LayerType::Linear);
        assert_eq!(schedule.types[1], LayerType::Attention);
        assert_eq!(schedule.types[2], LayerType::Linear);
        assert_eq!(schedule.types[3], LayerType::Attention);
        assert_eq!(schedule.attention_count(), 4);
        assert_eq!(schedule.linear_count(), 4);
    }

    #[test]
    fn test_pure_transformer() {
        let schedule = LayerSchedule::pure_transformer(16);
        assert_eq!(schedule.total_layers(), 16);
        assert_eq!(schedule.attention_count(), 16);
        assert_eq!(schedule.linear_count(), 0);
        assert!(schedule.types.iter().all(|t| *t == LayerType::Attention));
    }

    #[test]
    fn test_pure_linear() {
        let schedule = LayerSchedule::pure_linear(16);
        assert_eq!(schedule.total_layers(), 16);
        assert_eq!(schedule.attention_count(), 0);
        assert_eq!(schedule.linear_count(), 16);
        assert!(schedule.types.iter().all(|t| *t == LayerType::Linear));
    }

    #[test]
    fn test_explicit() {
        let types = vec![
            LayerType::Linear,
            LayerType::Linear,
            LayerType::Attention,
            LayerType::Linear,
            LayerType::Attention,
        ];
        let schedule = LayerSchedule::explicit(types.clone());
        assert_eq!(schedule.types, types);
        assert_eq!(schedule.attention_count(), 2);
        assert_eq!(schedule.linear_count(), 3);
    }

    #[test]
    fn test_layer_type_accessor() {
        let schedule = LayerSchedule::periodic(4, 1);
        assert_eq!(schedule.layer_type(0), Some(LayerType::Linear));
        assert_eq!(schedule.layer_type(1), Some(LayerType::Attention));
        assert_eq!(schedule.layer_type(4), None);
    }

    #[test]
    fn test_empty_schedule() {
        let schedule = LayerSchedule::periodic(0, 7);
        assert_eq!(schedule.total_layers(), 0);
        assert_eq!(schedule.attention_count(), 0);
        assert_eq!(schedule.linear_count(), 0);
    }
}
