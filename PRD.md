# PRD: Hybrid Model Inference Engine for Apple Silicon

> **Project**: metal-attention
> **Location**: `gpu_kernel/attention-proto/` (prototype) -> `metal-attention/` (production)
> **Author**: Patrick Kavanagh
> **Date**: 2026-02-14
> **Status**: Draft

---

## 1. Problem Statement

Hybrid AI models (Jamba, Griffin, Nemotron-H, Zamba, RWKV) mix linear/SSM layers with sparse softmax attention layers. They're faster and more memory-efficient than pure transformers, especially at long sequences. The industry is moving toward them.

**No existing inference engine optimizes hybrid models for Apple Silicon.**

- **llama.cpp**: Transformer-focused. Mamba support is bolted on, not optimized for Apple GPU. Uses a generic path for SSM layers.
- **MLX**: Supports some architectures but has no composable kernel layer. Each new architecture requires new C++ kernel code.
- **CoreML**: Black box. Can't customize attention mechanism or compose layer types.
- **Burn**: Rust ML framework with `Backend` trait, but no attention-level parameterization. Can't access `simdgroup_matrix` via CubeCL/wgpu.

Meanwhile, our prototype investigation (8 prototypes, 48 benchmarks) proved that linear attention is **50-70x faster** than softmax on Apple Silicon at N=1024 (GPU kernel time). Hybrid models that are 7:1 SSM:attention spend the majority of compute in the fast path. An engine optimized for this pattern would significantly outperform generic inference runtimes on Apple hardware.

## 2. Target Users

### Primary: Developers running hybrid models locally on Mac

- Running Jamba, Griffin, Zamba, RWKV models for local AI applications
- Currently using llama.cpp with suboptimal hybrid model support
- Want fast inference without cloud dependency
- Care about: tok/s, memory usage, model compatibility

### Secondary: Researchers experimenting with hybrid architectures

- Designing new SSM:attention ratios
- Prototyping custom attention variants
- Need to benchmark architecture choices on Apple Silicon
- Care about: flexibility, composability, benchmark tooling

### Tertiary: Rust ML ecosystem developers

- Building on Burn framework, need attention on Apple Silicon
- Can't access Metal-specific features through CubeCL/wgpu
- Care about: API ergonomics, trait compatibility, zero unsafe

## 3. Product Vision

A Rust inference engine where hybrid model architectures are **type parameters, not code forks**.

```rust
// Jamba: 7:1 SSM-to-attention ratio, MoE with 16 experts
let model = HybridModel::<
    LinearAttention,       // SSM/linear layers (7 of every 8)
    FlashAttention,        // Sparse attention layers (1 of every 8)
    RoPE,                  // Position encoding
    PagedKVCache,          // KV cache strategy
    7,                     // RATIO: 7 linear layers per 1 attention layer
>::load("jamba-1.5-mini.gguf")?;

let output = model.generate("Explain quantum computing", &params)?;
```

Changing the architecture is a type parameter change, not a rewrite. Each combination compiles to zero-overhead specialized Metal kernels via function constants.

## 4. Validated Technical Foundation

All core technical decisions are backed by prototype benchmarks on Apple M4. This is not speculative design.

### 4.1 Prototype Results Summary

| Proto | What it proved | Key metric |
|-------|---------------|------------|
| 1: Flash Attention | Softmax baseline on simdgroup_matrix | 0.16 TFLOPS at N=2048, D=64 |
| 2: Function Stitching | No runtime dispatch in GPU inner loops | `noinline` costs +39% overhead |
| 3: PagedAttention V2 | Paged KV cache viable on Apple Silicon | ~9% overhead at page_size=16 |
| 4: Function Constants | Zero-cost compile-time specialization | 0% runtime overhead, 178ns cache hit |
| 5: CubeCL | wgpu/CubeCL NOT viable for attention | 58-70% of hand-written MSL |
| 6: Linear Attention | Linear beats softmax at ALL tested N | **0.12x wall-clock at N=1024** |
| 7: RoPE/ALiBi/GQA | All variants have negligible overhead | <0.1% of base attention compute |
| 8: Burn Extension | Framework integration without forking | 2-17us bridge overhead, ~150 lines |

### 4.2 Design-Breaking Constraints (Empirically Validated)

1. **32KB threadgroup memory limit**: Dictates all tile/page/chunk sizes on Apple Silicon (M1-M4)
2. **Linear attention dominance**: Faster than softmax at ALL tested N >= 256 on Apple GPU
3. **No runtime dispatch in GPU inner loops**: 39% overhead from function calls
4. **CubeCL cannot access simdgroup_matrix**: Hand-written MSL is mandatory
5. **Function constants are the dispatch strategy**: 0% GPU overhead, 34-63us cold compile

### 4.3 Performance Baselines (M4, D=64)

| Kernel | N=256 | N=512 | N=1024 | Scaling |
|--------|-------|-------|--------|---------|
| Flash Attention | 389us | 762us | 2.42ms | O(N^2) |
| Linear Attention (GPU only) | ~35us | ~35us | ~35us | ~constant |
| Linear Attention (wall-clock) | 205us | 231us | 280us | O(N) |
| PagedAttention V2 | 438us | 1.31ms | 1.72ms | O(N^2) + 9% |

### 4.4 Existing Assets

- 8 Metal shader files (flash, paged, linear, rope, GQA, stitched variants)
- PsoCache with 178ns hit latency
- Device initialization, timing infrastructure, benchmark harness
- 34 passing tests, 48 criterion benchmarks
- 58 KB findings in gpu-forge knowledge base
- SYNTHESIS.md with complete architecture recommendations

## 5. Requirements

### 5.1 Core: Hybrid Model Inference (P0)

**R1**: Support composable attention mechanisms via Rust traits
- `trait SequenceBlock` as root trait
- `trait LinearSequenceModel: SequenceBlock` for linear/SSM layers
- `trait SoftmaxAttention: SequenceBlock` for transformer layers
- Each trait impl maps to specialized Metal kernels via function constants

**R2**: Support hybrid architectures as const generic parameters
- `HybridModel<R: LinearSequenceModel, A: SoftmaxAttention, const RATIO: usize>`
- Known ratios: Griffin 2:1, Jamba 7:1, Nemotron-H ~12:1, Zamba 6:1
- Pure transformer (RATIO=0) and pure linear (RATIO=inf) as degenerate cases

**R3**: Zero-overhead dispatch between layer types
- Function constant specialization for each (mechanism, variant, tile size) combination
- PsoCache for compiled pipeline states (178ns lookup)
- No runtime branching in GPU kernel inner loops

**R4**: GGUF model loading
- Parse GGUF format for model weights, tokenizer, architecture metadata
- Detect hybrid architecture type from model metadata
- Map layer types to appropriate trait implementations

**R5**: Text generation API
- Prompt processing (prefill) and token generation (decode)
- Configurable sampling (temperature, top-p, top-k, repetition penalty)
- Streaming token output

### 5.2 Attention Mechanisms (P0)

**R6**: Linear attention backend
- FLA chunk-based kernels (chunk_h, chunk_o) from Proto 6
- GPU prefix sum for H_cumulative (eliminate ~300us CPU overhead)
- chunk_size selection via function constants respecting 32KB limit

**R7**: Flash attention backend
- simdgroup_matrix-based tiled attention from Proto 1
- Multi-simdgroup support (target >1 TFLOPS vs current 0.16)
- Tile sizes: Br=16, Bc=64 for D=64; Br=16, Bc=16 for D=128

**R8**: Position encoding variants
- RoPE: standalone kernel (~10us/head, negligible)
- ALiBi: fused via function constant (0% overhead, dead-code elimination)
- No position encoding (for SSM layers)

**R9**: KV cache strategies
- Dense contiguous (default for short sequences)
- PagedAttention V2 (page_size=16 for D=64, ~9% overhead)
- Page size selection via function constants

**R10**: Grouped-Query Attention
- GQA remap kernel with runtime group_size parameter
- Support GQA group sizes: 1 (MHA), 2, 4, 8 (MQA)

### 5.3 Model Support (P1)

**R11**: Target model architectures (in priority order)
1. RWKV-7 (pure linear attention — validates LinearSequenceModel)
2. Jamba 1.5 (7:1 SSM:attention + MoE — validates full hybrid path)
3. Griffin/Hawk (2:1 RG-LRU:attention — validates different ratio)
4. Zamba (6:1 shared attention — validates weight sharing)
5. Standard transformer (Llama 3, Mistral — baseline comparison)

**R12**: Quantization support
- Q4_0, Q4_K_M, Q8_0 weight quantization (GGUF standard formats)
- FP16 and FP32 activations
- Dequantization in attention kernel where applicable

### 5.4 Performance Targets (P0)

**R13**: Performance requirements (M4 Pro 20-core GPU, 7B parameter model)
- Prompt processing (prefill): target competitive with llama.cpp on same model
- Token generation (decode): target competitive with llama.cpp on same model
- Hybrid model advantage: >30% faster than llama.cpp on hybrid architectures where llama.cpp uses generic SSM path
- Memory: fit within unified memory (no disk swapping for model sizes <= available RAM)

**R14**: Latency requirements
- First token latency: <500ms for prompts under 512 tokens
- PSO cold compile: <100ms total for all kernel variants at startup
- PSO cache hit: <1us per dispatch (measured: 178ns)

### 5.5 CLI Interface (P1)

**R15**: Command-line inference tool
```
metal-attention run --model jamba-1.5-mini.gguf --prompt "..." [options]
metal-attention bench --model <path> --seq-lengths 256,512,1024,2048
metal-attention info --model <path>  # show architecture, layer types, sizes
```

**R16**: Output format
- Streaming token output to stdout
- Performance stats on stderr (tok/s, memory, kernel times)
- JSON output mode for programmatic use

### 5.6 Framework Integration (P2)

**R17**: Burn framework integration
- `trait AttentionBackend: Backend` supertrait (Proto 8 pattern)
- Bridge function with 2-17us overhead
- Backend delegation via proc-macro
- Compatible with Burn 0.20+

### 5.7 Apple Silicon Compatibility (P1)

**R18**: Hardware support
- M1, M2, M3, M4 families (all support simdgroup_matrix)
- Metal feature set detection for optimal tile/chunk sizes
- Graceful degradation on older hardware (smaller tiles, fewer simdgroups)

## 6. Non-Requirements (Explicit Exclusions)

- **Training**: This is inference only. No backward pass, no gradient computation.
- **Cross-platform**: Apple Silicon Metal only. No CUDA, no Vulkan, no WebGPU. CubeCL proved this would sacrifice 30-42% throughput.
- **Serving / multi-tenant**: Single-user local inference. No HTTP server, no batched requests, no multi-sequence scheduling. PagedAttention is included for KV cache efficiency, not multi-tenant memory sharing.
- **Model conversion**: Users provide GGUF files. No PyTorch/SafeTensors import.
- **Metal 4**: Initial release targets Metal 3 (MSL 3.1). Metal 4 cooperative tensor migration is Phase E optimization work (requires macOS 26+, M5 for Neural Accelerator benefit).
- **GUI**: CLI only. No TUI, no desktop app.

## 7. Architecture Overview

```
┌─────────────────────────────────────────────────┐
│                   CLI / API                      │
├─────────────────────────────────────────────────┤
│              Model Runtime Loop                  │
│  ┌──────────────────────────────────────────┐   │
│  │ HybridModel<R, A, RATIO>                 │   │
│  │   for layer in layers:                   │   │
│  │     match layer.type:                    │   │
│  │       Linear => R::forward(...)          │   │
│  │       Attention => A::forward(...)       │   │
│  └──────────────────────────────────────────┘   │
├─────────────────────────────────────────────────┤
│              Trait Dispatch Layer                 │
│  ┌───────────────┐    ┌──────────────────────┐  │
│  │ LinearSeqModel│    │  SoftmaxAttention     │  │
│  │  - FLA chunks │    │  - Flash (simdgroup)  │  │
│  │  - Mamba SSM  │    │  - Paged KV cache     │  │
│  │  - RG-LRU     │    │  - RoPE/ALiBi/GQA     │  │
│  └───────┬───────┘    └──────────┬───────────┘  │
│          │                       │               │
├──────────┴───────────────────────┴──────────────┤
│              Metal Kernel Layer                   │
│  PsoCache (178ns/hit)                            │
│  Function constants → specialized kernels        │
│  simdgroup_matrix, threadgroup memory            │
│  Hand-written MSL (mandatory per Proto 5)        │
├─────────────────────────────────────────────────┤
│              Apple Silicon Hardware               │
│  Unified memory (zero-copy), 32KB threadgroup    │
│  M1/M2/M3/M4 GPU, MSL 3.1                       │
└─────────────────────────────────────────────────┘
```

## 8. Success Metrics

| Metric | Target | How to measure |
|--------|--------|----------------|
| Hybrid model speedup vs llama.cpp | >30% on Jamba/Griffin | criterion benchmarks, same model, same hardware |
| Linear attention throughput | >1 TFLOPS at D=64 | GPU kernel timing via MTLCommandBuffer timestamps |
| Flash attention throughput | >1 TFLOPS at D=64 | Same (current: 0.16, needs multi-simdgroup) |
| Model load time | <5s for 7B Q4 model | wall-clock, mmap-based loading |
| First token latency | <500ms (512-token prompt) | wall-clock from prompt submit to first token |
| Abstraction overhead | 0% GPU, <20us host | function constant dispatch, Burn bridge |
| Supported architectures | >= 5 hybrid + 2 transformer | model compatibility tests |
| Test coverage | >80% on kernel dispatch paths | cargo test + shader validation |

## 9. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Linear attention quality loss vs softmax | Medium | High | Document quality trade-offs per model. Softmax fallback is always available. User chooses. |
| GGUF format doesn't encode hybrid architecture metadata | Medium | Medium | Custom metadata keys or architecture detection heuristics. Contribute upstream if needed. |
| Flash attention stuck at 0.16 TFLOPS | Low | Medium | Multi-simdgroup is the known fix. Metal Flash Attention (MFA) achieves >1 TFLOPS as reference. |
| Mamba/SSM kernel complexity | Medium | Medium | Start with linear attention (simpler). Add Mamba SSM as second LinearSequenceModel impl. |
| Small initial user base | High | Low | Acceptable for open-source project. Grows as hybrid models proliferate. |
| Metal 4 makes Metal 3 kernels obsolete | Low | Low | Metal 3/4 coexist. Metal 4 migration is Phase E. Kernels remain valid on M1-M4 hardware. |

## 10. Implementation Phases

### Phase A: Core Trait + Linear Attention (4-6 weeks)

Foundation. Define traits, implement linear attention with GPU prefix sum, validate on RWKV-7.

1. Define `trait SequenceBlock`, `trait LinearSequenceModel`, `trait SoftmaxAttention`
2. Implement `LinearAttention` using Proto 6 chunk_h/chunk_o kernels
3. Add GPU prefix sum kernel (eliminate ~300us CPU overhead)
4. GGUF model loader (weights + tokenizer + architecture metadata)
5. Token generation loop (greedy decode)
6. CLI: `metal-attention run --model rwkv-7.gguf --prompt "..."`
7. Benchmark: linear attention tok/s vs llama.cpp on same RWKV model

**Exit criteria**: RWKV-7 generates coherent text, tok/s measured and published.

### Phase B: Softmax Flash Attention (3-4 weeks)

Add the softmax path. Validate on standard transformer models.

1. Implement `FlashAttention` using Proto 1 kernel
2. Multi-simdgroup support (target >1 TFLOPS)
3. Tile size selection via function constants
4. Validate on Llama 3 / Mistral (pure transformer baseline)
5. Benchmark: flash attention tok/s vs llama.cpp on same model

**Exit criteria**: Llama 3 generates coherent text, >1 TFLOPS flash attention.

### Phase C: Hybrid Runtime + KV Cache (3-4 weeks)

The differentiating feature. Compose linear + softmax layers in hybrid architectures.

1. `HybridModel<R, A, RATIO>` runtime loop
2. PagedAttention V2 for KV cache (Proto 3)
3. RoPE/ALiBi/GQA fusion (Proto 7)
4. Jamba model loading and inference
5. Griffin model loading and inference
6. Benchmark: hybrid model tok/s vs llama.cpp

**Exit criteria**: Jamba generates coherent text, >30% faster than llama.cpp on hybrid layers.

### Phase D: Burn Integration (2-3 weeks)

Framework integration for the Rust ML ecosystem.

1. `AttentionBackend: Backend` supertrait (Proto 8)
2. Backend delegation via proc-macro
3. Bridge function connecting Burn tensors to Metal kernels
4. Example: Burn model using metal-attention kernels

**Exit criteria**: Burn model runs with metal-attention backend, benchmark published.

### Phase E: Optimization + Hardware Support (ongoing)

1. Multi-simdgroup flash attention optimization
2. Async copy / memory prefetching
3. Multi-generation support (M1-M4 feature detection)
4. Metal 4 cooperative tensor migration (when macOS 26 ships)
5. Additional quantization formats
6. Additional model architectures as they release

## 11. Open Questions

1. **GGUF hybrid architecture metadata**: Do Jamba/Griffin GGUF files encode the SSM:attention ratio, or do we need to detect it from layer names?
2. **Mamba SSM kernel complexity**: The selective scan requires custom Metal kernels beyond the linear attention prototype. Scope this before committing to Mamba support in Phase A.
3. **Multi-simdgroup flash attention**: The path from 0.16 to >1 TFLOPS is known (Metal Flash Attention demonstrates it) but the implementation effort for our codebase needs scoping.
4. **Tokenizer strategy**: Use `tokenizers` crate (HuggingFace) or implement minimal GGUF tokenizer? Trade-off: dependency size vs compatibility.
5. **Benchmark methodology for hybrid advantage claim**: How to fairly compare hybrid model tok/s when llama.cpp and metal-attention use different kernel implementations for both the linear and attention layers?

## 12. References

- [SYNTHESIS.md](SYNTHESIS.md) — Complete prototype results and architecture recommendations
- [gpu-forge Knowledge Base](https://github.com/kavanaghpatrick/gpu-forge) — 1,635+ verified GPU computing findings
- [Issue #20 Comment](https://github.com/kavanaghpatrick/gpu-forge/issues/20#issuecomment-3893597613) — Investigation summary
- Dao & Gu, "Transformers are SSMs" (ICML 2024) — SSD duality framework
- Katharopoulos et al., "Transformers are RNNs" (ICML 2020) — Linear attention
- LithOS (SOSP'25) — GPU OS reference architecture, kernel atomizer
- Jamba (AI21, 2024) — 7:1 SSM:Transformer hybrid architecture
