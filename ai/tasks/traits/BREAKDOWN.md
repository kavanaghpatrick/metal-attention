---
id: traits.BREAKDOWN
module: traits
priority: 1
status: failing
version: 1
origin: spec-workflow
dependsOn: [devops.BREAKDOWN]
tags: [breakdown]
testRequirements:
  unit:
    required: false
    pattern: "tests/traits/**/*.test.*"
---
# Traits -- Breakdown

## Context

The composable trait system is the core differentiator of metal-attention. Unlike llama.cpp or MLX which hard-code each model architecture, metal-attention defines abstract interfaces for sequence processing blocks. Each combination of `LinearSequenceModel` + `SoftmaxAttention` + ratio compiles to fully specialized Metal kernels via function constants. The traits crate has zero Metal dependency -- it defines pure Rust interfaces that the kernels crate implements.

## Scope

- `trait SequenceBlock` -- Root trait for any sequence-processing block
- `trait LinearSequenceModel: SequenceBlock` -- O(N) or O(1) per-token models (FLA, Mamba, RG-LRU, RWKV-7)
- `trait SoftmaxAttention: SequenceBlock` -- O(N^2) attention with KV cache (Flash, Paged)
- Shared types: `TensorView`, `DType`, `BlockConfig`, `LayerType`, `LayerSchedule`
- Position encoding types: `PositionEncoding` (RoPE, ALiBi, None), `LinearPositionEncoding` (None, TokenShift)
- KV cache types: `KVCacheMode` (Auto, Dense, Paged), `GQAConfig`
- Model configuration: `ModelConfig` struct with all architecture parameters
- `HybridModel<L, A>` struct and `LayerState<LS, AS>` enum

## Key Decisions

- **From TECH.md**: `TensorView` does not own data -- it references Metal buffers via byte offset + shape + strides + dtype. This is the bridge between pure Rust trait interfaces and Metal GPU buffers.
- **From TECH.md**: `SequenceBlock::State` associated type enables polymorphic per-layer state (KV cache for attention, hidden state matrix for SSM). `init_state()`, `forward_prefill()`, `forward_decode()` are the three core methods.
- **From TECH.md**: `LinearSequenceModel` adds `prefill_chunked()` with chunk_size parameter (respecting 32KB threadgroup limit) and `decode_step()` for O(D^2) single-token update.
- **From TECH.md**: `SoftmaxAttention` adds KV cache management (`cached_length`, `max_length`), position encoding configuration, GQA config, and separate `prefill_attention`/`decode_attention` methods with explicit Q/K/V parameters.
- **From TECH.md**: `LayerSchedule::periodic(total_layers, ratio)` creates the interleaving pattern. Ratio=0 means pure transformer (all attention). Pure linear is a separate constructor.
- **From PM.md**: P0-1 requires trait hierarchy. This is the foundation without which the project has no differentiator.

## Acceptance Criteria

1. `trait SequenceBlock` compiles with associated `State` type and `init_state`/`forward_prefill`/`forward_decode` methods
2. `trait LinearSequenceModel: SequenceBlock` compiles with `prefill_chunked` and `decode_step` methods
3. `trait SoftmaxAttention: SequenceBlock` compiles with KV cache methods and `prefill_attention`/`decode_attention`
4. `TensorView`, `DType`, `BlockConfig` types compile and derive Debug/Clone
5. `LayerSchedule::periodic(32, 7)` produces correct 7:1 pattern (28 linear + 4 attention)
6. `LayerSchedule::pure_transformer(32)` produces all-attention schedule
7. `LayerSchedule::pure_linear(32)` produces all-linear schedule
8. `ModelConfig` holds all architecture parameters from GGUF metadata
9. `HybridModel<L, A>` compiles with generic bounds `L: LinearSequenceModel, A: SoftmaxAttention`
10. Crate has zero dependencies on Metal or any GPU-specific code

## Technical Notes

- From TECH.md: `DType` enum covers F32, F16, BF16, Q4_0, Q4_K_M, Q8_0. Must match quantization types from GGUF parser.
- From TECH.md: `BlockConfig` contains `hidden_size`, `head_dim`, `num_heads`, `num_kv_heads`, `layer_index`.
- From TECH.md: `ModelConfig` includes optional SSM fields (`ssm_state_size`, `ssm_conv_size`) and MoE fields (`num_experts`, `num_active_experts`).
- From QA.md: Unit tests should verify type sizes, default values, and schedule generation patterns. These are CPU-only tests.
- From UX.md: The `KvCacheMode` enum (Auto, Dense, Paged) surfaces to both the library API and CLI flags.
