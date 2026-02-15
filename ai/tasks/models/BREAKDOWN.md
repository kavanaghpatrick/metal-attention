---
id: models.BREAKDOWN
module: models
priority: 4
status: failing
version: 1
origin: spec-workflow
dependsOn: [traits.BREAKDOWN, kernels.BREAKDOWN, gguf.BREAKDOWN]
tags: [breakdown]
testRequirements:
  unit:
    required: false
    pattern: "tests/models/**/*.test.*"
---
# Models -- Breakdown

## Context

The models crate implements concrete model architectures by composing the trait interfaces defined in `metal-attention-traits` with the GPU kernels from `metal-attention-kernels` and weight loading from `metal-attention-gguf`. Each model type maps GGUF weights to the correct layer structure, configures the `LayerSchedule`, and provides architecture-specific behavior (e.g., RWKV token shifting, Jamba MoE routing, Zamba shared attention). A registry maps architecture names to constructors for auto-detection.

## Scope

- **Llama/Mistral** (`llama.rs`): Pure transformer baseline. All attention layers. GQA with RoPE. SwiGLU FFN. Validates `SoftmaxAttention` trait in isolation. Serves as performance comparison baseline.
- **RWKV-7** (`rwkv7.rs`): Pure linear model. All linear layers with token shift + ReLU^2 FFN + Dynamic State Evolution. No KV cache. Validates `LinearSequenceModel` trait in isolation. P0 target.
- **Jamba** (`jamba.rs`): 7:1 SSM:Attention hybrid with Mamba-2 blocks + grouped-query attention + 16-expert MoE routing. 72 total layers (63 linear + 9 attention). P1 target.
- **Griffin** (`griffin.rs`): 2:1 RG-LRU:Attention hybrid. Gated linear recurrence + local sliding-window attention. RecurrentGemma open weights. P1 target.
- **Zamba** (`zamba.rs`): 6:1 Mamba:SharedAttention. Shared attention block applied at regular intervals with LoRA projectors for depth specialization. P1 target.
- **Registry** (`registry.rs`): Maps architecture name string to model constructor. Used by `Engine::load_model()` for auto-detection.

## Key Decisions

- **From TECH.md**: Each model struct composes `HybridModel<L, A>` with concrete type parameters. RWKV-7 uses `PureLinearModel<RWKV7Block>` (special case, no attention). Llama uses `HybridModel<NoLinear, FlashAttention>` with ratio=0 (all attention).
- **From TECH.md**: Weight loading goes through GGUF `map_tensor_name()` which returns `(layer_index, WeightRole)`. Each model defines its own tensor name patterns.
- **From TECH.md**: RWKV-7 key characteristics: Dynamic State Evolution, vector-valued gating, in-context learning rates, token-shift mechanism, bonus terms, ReLU^2 FFN. Fixed-size state matrix per layer.
- **From TECH.md**: Jamba key characteristics: Mamba-2 blocks interleaved with GQA at 7:1 ratio, 16-expert MoE with top-2 routing, KV cache only for 1/8 of layers.
- **From PM.md**: P0 requires RWKV-7 end-to-end inference (US-1). P1 requires Jamba (US-2), Griffin, Zamba, Llama baseline.
- **From PM.md**: P0-1 requires the trait hierarchy to be implemented through concrete models.

## Acceptance Criteria

1. `registry::get_model("llama")` returns a valid constructor
2. `registry::get_model("rwkv")` returns a valid constructor
3. Llama model loads weights from GGUF and constructs correct all-attention `LayerSchedule`
4. RWKV-7 model loads weights from GGUF and constructs all-linear `LayerSchedule`
5. Jamba model constructs 7:1 periodic `LayerSchedule` with correct 72-layer pattern
6. Griffin model constructs 2:1 periodic `LayerSchedule`
7. Each model correctly maps GGUF tensor names to `WeightRole` assignments
8. RWKV-7 can perform a single forward pass through one layer (prefill + decode) without error
9. Llama can perform a single forward pass through one layer (prefill + decode) without error
10. Model `info()` method returns correct `ModelInfo` with architecture name, layer count, quantization, and SSM:attention ratio

## Technical Notes

- From TECH.md: `HybridModel<L, A>` holds `schedule: LayerSchedule`, `linear_impl: L`, `attention_impl: A`, `layers: Vec<LayerWeights>`, `states: Vec<LayerState<L::State, A::State>>`, `embedding`, `output_proj`, `final_norm`, `config: ModelConfig`.
- From TECH.md: Jamba MoE routing uses top-2 out of 16 experts. Active parameters are ~3.8B out of 12B total. MoE routing logic is part of the FFN path, not the attention path.
- From TECH.md: Zamba uses shared attention blocks -- one attention block is reused at multiple layer positions with LoRA projectors for per-position specialization. This is a weight-sharing pattern not present in other architectures.
- From QA.md: Model architecture detection tests verify correct `LayerSchedule` from GGUF metadata. Hybrid-specific quality tests validate each layer type independently AND in composition.
- From QA.md: Token-level accuracy tests compare predictions between metal-attention and reference implementation (>99% greedy match, >0.999 cosine similarity).
