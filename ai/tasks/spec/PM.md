# Product Manager Analysis: metal-attention

> **Project**: metal-attention -- Hybrid Model Inference Engine for Apple Silicon
> **Date**: 2026-02-14
> **Status**: Spec Phase
> **Basis**: 8 validated prototypes, 48 benchmarks, 38 findings, PRD.md, SYNTHESIS.md

---

## Research Findings

### 1. Hybrid Model Architectures Are the Industry Direction

The transformer monopoly is ending. Multiple production-grade hybrid architectures now combine state space models (SSMs) or linear recurrences with sparse attention layers, achieving transformer-quality results with fundamentally better inference scaling:

- **Jamba** (AI21): 7:1 SSM-to-attention ratio with MoE. Jamba 1.5 supports 256K-token context. Jamba Reasoning 3B delivers 2-4x faster inference on consumer hardware and runs on laptops, iPhones, and Pixel phones. Released under Apache 2.0. ([AI21 Jamba Blog](https://www.ai21.com/blog/announcing-jamba/), [Jamba Reasoning 3B](https://www.ai21.com/blog/introducing-jamba-reasoning-3b/))
- **RWKV-7 "Goose"**: Pure RNN with constant memory and constant per-token inference time. Only matrix-vector multiplications at inference -- no matrix-matrix multiplications. Can run on phones. RWKV-7 2.9B matches 3B SoTA on English benchmarks despite training on dramatically fewer tokens. RWKV-V8 "DeepEmbed" enables sparse large models on edge devices without consuming VRAM. ([RWKV-7 Paper](https://arxiv.org/abs/2503.14456), [RWKV GitHub](https://github.com/BlinkDL/RWKV-LM))
- **Griffin** (Google DeepMind): 2:1 gated linear recurrence to local attention. Matches Llama-2 quality on 6x fewer training tokens. Fixed state size vs growing KV cache gives lower latency and higher throughput for long sequences. RecurrentGemma ships as open weights. ([Griffin Paper](https://arxiv.org/abs/2402.19427), [RecurrentGemma](https://github.com/google-deepmind/recurrentgemma))
- **Zamba** (Zyphra): 6:1 Mamba backbone with shared attention layers (one shared block applied 13 times). Zamba2-7B achieves 25% faster TTFT, 20% better tok/s, and significant memory reduction vs Llama3-8B. Highest-performing dense SSM at 7B scale. ([Zamba Paper](https://arxiv.org/abs/2405.16712), [Zamba2 Blog](https://www.zyphra.com/post/zamba2-7b))
- **Mamba-3**: Inference-first SSM with MIMO updates for efficient hardware parallelism during decoding. 5x throughput over transformers with linear sequence scaling. ([Mamba-3 OpenReview](https://openreview.net/forum?id=HwCvaJOiCj))

**Key insight**: These architectures share a pattern -- the majority of compute happens in linear/SSM layers (fast path), with sparse attention layers (slow path) for retrieval and in-context learning. An engine optimized for this ratio pattern would outperform generic engines that treat all layers identically.

### 2. Apple Silicon Is a Growing Local Inference Platform

Apple has built a vertically integrated stack for local AI that gives it a practical advantage for on-device LLMs ([Apple Sleeper Advantage](https://www.xda-developers.com/apple-sleeper-advantage-local-llms/)):

- **M5 GPU Neural Accelerators** yield up to 4x speedup on TTFT vs M4 baseline ([Apple M5 MLX Research](https://machinelearning.apple.com/research/exploring-llms-mlx-m5))
- **M4 Pro** delivers 60-120 tok/s on 7-8B models via llama.cpp Metal ([llama.cpp M-Series Discussion](https://github.com/ggml-org/llama.cpp/discussions/4167))
- **Memory bandwidth** is the primary bottleneck: M3 MacBook Air limited to 100 GB/s while M4 Pro doubles it ([llama.cpp Discussion #12985](https://github.com/ggml-org/llama.cpp/discussions/12985))
- A production-grade comparative study evaluated five local LLM runtimes on Apple Silicon (MLX, MLC-LLM, llama.cpp, Ollama, PyTorch MPS), confirming the platform's readiness for serious local inference ([Apple Silicon LLM Study](https://arxiv.org/abs/2511.05502))
- Apple Silicon frameworks are "rapidly maturing into viable, production-grade solutions for private, on-device LLM inference" ([Same study](https://arxiv.org/abs/2511.05502))

### 3. No Existing Engine Optimizes Hybrid Models for Metal

Every existing inference engine on Apple Silicon treats hybrid architectures as a second-class citizen:

- **llama.cpp**: Transformer-focused. Has RWKV-6/7 support bolted on ([llama.cpp RWKV Wiki](https://wiki.rwkv.com/inference/llamacpp.html)), but uses generic paths for SSM layers. No Metal-specific SSM kernel optimization.
- **MLX**: Apple-optimized with highest sustained generation throughput (230 tok/s vs llama.cpp's 150 tok/s in benchmarks), but no composable kernel architecture for hybrid models. Each new architecture requires new C++ kernel code. ([MLX vs llama.cpp Benchmark](https://medium.com/@andreask_75652/benchmarking-apples-mlx-vs-llama-cpp-bbbebdc18416))
- **web-rwkv**: Pure Rust/WebGPU RWKV inference with Metal support via wgpu, but RWKV-only. No hybrid architecture support. Int8/Float4 quantization. ([web-rwkv GitHub](https://github.com/cryscan/web-rwkv))

### 4. Rust ML Ecosystem Is Maturing

- **Burn**: Pure Rust deep learning framework with Metal backend in development for 2025. API stabilization underway. Burn+CUDA achieves 97% of PyTorch+CUDA on Phi3 3.8B. Developing a native Metal backend beyond wgpu capabilities. ([Burn 2025 Plans](https://burn.dev/blog/going-big-and-small-for-2025/), [Burn GitHub](https://github.com/tracel-ai/burn))
- **Candle**: HuggingFace's minimalist Rust ML framework. metal-candle achieves 25.9x faster than MLX for embeddings with near-constant batch scaling (1-100 batch only +13%). ([metal-candle Benchmarks](https://github.com/GarthDB/metal-candle/blob/main/BENCHMARKS.md), [Candle GitHub](https://github.com/huggingface/candle))
- **Cloudflare Infire**: Production Rust LLM inference engine, 7% faster than vLLM with lower CPU overhead. Validates Rust for production inference at scale. ([Rust Enterprise 2025](https://rust-trends.com/newsletter/rust-enterprise-breakthrough-2025/))
- **Lele**: Compiles ONNX models into pure Rust for bare-metal inference. ([Lele Forum Post](https://users.rust-lang.org/t/lele-bare-metal-ml-inference-engine-in-pure-rust-compile-onnx-into-rust/138195))

### 5. Metal Function Constants Enable Zero-Overhead Architecture Dispatch

Apple's function constant specialization eliminates dynamic branching entirely. The compiler folds constants, removes dead code, and eliminates buffer reads for material parameters. This produces the most optimal code path per variant. ([Apple Developer: Function Specialization](https://developer.apple.com/documentation/metal/using-function-specialization-to-build-pipeline-variants), [Metal Shader Best Practices](https://developer.apple.com/videos/play/tech-talks/111373/))

Our prototype validation confirms: 0% runtime overhead, 178ns cache hit, 34-63us cold compile per variant. This is the mechanism that makes "architecture as type parameter" feasible -- each `HybridModel<R, A, RATIO>` compiles to fully specialized Metal kernels with zero dispatch overhead.

---

## Product Vision

**metal-attention** is a Rust inference engine where hybrid AI model architectures are type parameters, not code forks.

The core insight is that hybrid models (Jamba, Griffin, RWKV, Zamba) spend 70-90% of their compute in linear/SSM layers -- the fast path. Our prototype data proves linear attention is 50-70x faster than softmax on Apple GPU at N=1024. An engine that treats this fast path as the primary compute path, with sparse softmax attention as the occasional slow path, will fundamentally outperform engines designed around the assumption that every layer is expensive.

The architecture exploits two validated technical capabilities unique to our approach:

1. **Composable traits compiled to specialized kernels**: `trait SequenceBlock` with `LinearSequenceModel` and `SoftmaxAttention` subtypes, where each combination maps to Metal function constants that compile to zero-overhead specialized GPU code.

2. **Hand-written MSL with simdgroup_matrix**: Mandatory for competitive throughput (CubeCL/wgpu achieves only 58-70%). Direct Metal access via objc2-metal with ~20 crate dependencies vs CubeCL's ~350.

The result: changing a model from Jamba (7:1 SSM:attention) to Griffin (2:1) is a type parameter change, not a rewrite. Each combination compiles to optimal GPU code automatically.

```rust
// Jamba: 7 linear layers per 1 attention layer, MoE routing
let jamba = HybridModel::<LinearAttention, FlashAttention, RoPE, PagedKVCache, 7>
    ::load("jamba-1.5-mini.gguf")?;

// Griffin: 2 linear recurrence layers per 1 local attention layer
let griffin = HybridModel::<GatedLinearRecurrence, LocalAttention, RoPE, DenseKVCache, 2>
    ::load("recurrentgemma-2b.gguf")?;

// RWKV-7: Pure linear -- no attention layers at all
let rwkv = PureLinearModel::<RWKVAttention>::load("rwkv-7-3b.gguf")?;
```

---

## Target Users & Market

### Primary: Developers Running Hybrid Models Locally on Mac

**Profile**: Software engineers and indie developers building local AI applications on macOS. Currently using Ollama/llama.cpp for inference, frustrated by suboptimal hybrid model support. They want to run Jamba, RWKV, or Griffin locally without cloud dependency.

**Size estimate**: The local LLM market on Apple Silicon is growing rapidly. Five production-grade runtimes now compete on the platform ([Apple Silicon LLM Study](https://arxiv.org/abs/2511.05502)). Apple's M-series installed base exceeds 100M devices. The subset running local LLMs is small but growing fast as models shrink (Jamba Reasoning 3B, Zamba2 3B) and local inference becomes practical.

**Pain points**:
- llama.cpp's Mamba/SSM support is not Metal-optimized
- No engine lets them easily switch between hybrid architectures
- Performance on hybrid models is disappointing compared to pure transformer performance

**What they care about**: tok/s, memory usage, model compatibility, ease of use (CLI)

### Secondary: Researchers Experimenting with Hybrid Architectures

**Profile**: ML researchers and PhD students designing new SSM:attention ratios, evaluating hybrid architectures for specific tasks, or benchmarking architecture choices on consumer hardware.

**Size estimate**: Small but high-influence. Hybrid architecture research is one of the hottest areas in ML (Mamba-3, RWKV-7, Jamba 1.5, Griffin all published 2024-2025). Researchers publishing benchmarks on consumer hardware create outsized visibility.

**Pain points**:
- Prototyping a new ratio requires forking an entire inference engine
- No standardized way to benchmark SSM vs attention trade-offs on Apple Silicon
- CubeCL/wgpu abstractions prevent access to Metal-specific features

**What they care about**: Flexibility, composability, benchmark tooling, reproducibility

### Tertiary: Rust ML Ecosystem Developers

**Profile**: Developers building on Burn framework who need high-performance attention on Apple Silicon. Currently blocked by CubeCL's inability to access simdgroup_matrix.

**Size estimate**: Niche but strategic. Burn has 9.2K GitHub stars and growing. The Rust ML ecosystem is reaching an inflection point with Cloudflare Infire validating Rust for production inference.

**Pain points**:
- Burn's wgpu backend achieves only 58-70% of native Metal throughput for attention
- No way to access simdgroup_matrix from CubeCL user kernels
- Metal-specific features require leaving the Burn ecosystem entirely

**What they care about**: API ergonomics, trait compatibility, zero unsafe blocks

### Market Timing

The market timing is favorable for three converging reasons:

1. **Hybrid models are proliferating**: Jamba, Griffin, Zamba, RWKV-7, Mamba-3, Nemotron-H -- the pace of hybrid architecture releases is accelerating. Each one is slightly different (different ratios, different SSM flavors, different attention patterns). A composable engine becomes more valuable with every new architecture.

2. **Models are shrinking to consumer hardware**: Jamba Reasoning 3B, Zamba2 3B, RWKV-7 3B all fit on entry-level Macs. The 7B class (Zamba2-7B, Jamba Mini) fits comfortably on M4 Pro with 24GB.

3. **Apple Silicon is purpose-built for this**: Unified memory eliminates PCIe bottleneck. Metal function constants enable zero-overhead kernel specialization. The hardware advantage is real but no existing engine fully exploits it for hybrid architectures.

---

## User Stories

### Core Inference (P0)

**US-1**: As a developer, I want to run RWKV-7 locally on my Mac so that I can use a high-quality linear model without cloud dependency.
- Acceptance: `metal-attention run --model rwkv-7-3b.gguf --prompt "..."` generates coherent text with measured tok/s reported.

**US-2**: As a developer, I want to run Jamba locally on my Mac so that I can use a hybrid SSM:attention model with long-context support.
- Acceptance: Jamba 1.5 Mini generates coherent text. Hybrid layer dispatch is measurably faster than llama.cpp on the same model and hardware.

**US-3**: As a developer, I want to see real-time performance statistics during inference so that I can understand throughput and identify bottlenecks.
- Acceptance: Streaming tok/s, memory usage, and kernel timing breakdown printed to stderr during generation.

### Architecture Flexibility (P1)

**US-4**: As a researcher, I want to benchmark different attention mechanisms on the same model so that I can evaluate quality/performance trade-offs.
- Acceptance: `metal-attention bench --model <path> --seq-lengths 256,512,1024,2048` produces criterion-quality benchmarks with GPU kernel timing.

**US-5**: As a researcher, I want to experiment with different SSM:attention ratios so that I can prototype new hybrid architectures without modifying kernel code.
- Acceptance: Changing the ratio parameter and recompiling produces a working engine with different layer dispatch.

**US-6**: As a developer, I want model architecture to be auto-detected from GGUF metadata so that I don't have to manually specify layer types.
- Acceptance: `metal-attention run --model jamba-1.5-mini.gguf` automatically detects Jamba architecture and configures 7:1 SSM:attention dispatch.

### Ecosystem Integration (P2)

**US-7**: As a Burn framework developer, I want to use metal-attention's kernels as a Burn backend so that I can get native Metal attention performance without leaving the Burn ecosystem.
- Acceptance: `AttentionBackend` supertrait works with Burn 0.20+. Bridge overhead < 20us. Zero unsafe blocks.

**US-8**: As a developer, I want JSON output mode so that I can integrate metal-attention into automated pipelines.
- Acceptance: `--output json` flag produces structured JSON with tokens, timing, and metadata.

---

## Success Metrics

### Performance (Quantitative)

| Metric | Target | Measurement Method | Baseline |
|--------|--------|-------------------|----------|
| Hybrid model tok/s vs llama.cpp | >30% faster on Jamba/Griffin hybrid layers | criterion benchmark, same model, same M4 hardware | llama.cpp current hybrid model tok/s |
| RWKV-7 inference tok/s | Competitive with or faster than llama.cpp RWKV path | Wall-clock tok/s measurement | llama.cpp RWKV-7 tok/s on M4 |
| Linear attention throughput | >1 TFLOPS at D=64 (with GPU prefix sum) | MTLCommandBuffer GPU timestamps | Current: 0.04 TFLOPS (CPU prefix sum bottleneck) |
| Flash attention throughput | >1 TFLOPS at D=64 (multi-simdgroup) | MTLCommandBuffer GPU timestamps | Current: 0.16 TFLOPS (single simdgroup) |
| Dispatch overhead per layer | 0% GPU overhead, <1us host cache hit | PsoCache lookup timing | Current: 178ns cache hit (validated) |
| Model load time | <5s for 7B Q4 GGUF | Wall-clock measurement | N/A (new capability) |
| First token latency | <500ms for 512-token prompt | Wall-clock prompt-to-first-token | N/A (new capability) |
| Memory efficiency | No disk swapping for model sizes <= available RAM | Activity Monitor / vm_stat | N/A (new capability) |

### Adoption (Qualitative -- 6-month horizon)

| Metric | Target | Measurement |
|--------|--------|-------------|
| GitHub stars | >500 at 6 months | GitHub API |
| Supported architectures | >= 5 hybrid + 2 transformer | Model compatibility test suite |
| Community benchmark citations | >= 3 external benchmarks citing metal-attention | Search / mentions |
| Burn ecosystem adoption | >= 1 third-party project using AttentionBackend | GitHub dependency graph |
| Test coverage on kernel dispatch | >80% | cargo test + shader validation |

---

## Competitive Analysis

### llama.cpp

| Dimension | llama.cpp | metal-attention |
|-----------|-----------|-----------------|
| **Language** | C/C++ | Rust |
| **Metal support** | Yes (GPU offloading) | Yes (native Metal, hand-written MSL) |
| **Hybrid model support** | Bolted-on Mamba/RWKV | First-class, composable via type parameters |
| **SSM kernel optimization** | Generic path, not Metal-optimized | Specialized Metal kernels via function constants |
| **Architecture flexibility** | Hard-coded per model | Trait-based, ratio as const generic |
| **Performance (M4, 7-8B)** | 60-120 tok/s | Target: >30% faster on hybrid layers |
| **Model format** | GGUF (native) | GGUF (compatible) |
| **Community** | Massive (60K+ stars) | New entrant |
| **Maturity** | Production | Pre-alpha |

**Our advantage**: Hybrid architecture dispatch. llama.cpp treats Mamba layers as a compatibility feature; we treat them as the primary fast path. The 50-70x linear attention advantage on Metal is the wedge.

**Their advantage**: Maturity, community, model coverage, cross-platform. We cannot and should not try to replace llama.cpp for pure transformer inference. We win only on hybrid architectures where the fast-path optimization matters.

### MLX (Apple)

| Dimension | MLX | metal-attention |
|-----------|-----|-----------------|
| **Language** | C++/Python | Rust |
| **Metal optimization** | Apple-internal, best-in-class | Hand-written MSL, prototype-validated |
| **Throughput** | ~230 tok/s (highest on Apple Silicon) | Target: competitive on hybrid, faster on hybrid layers |
| **Hybrid support** | Per-model implementation | Composable trait architecture |
| **Extensibility** | Requires C++ kernel code | Rust traits + function constants |
| **Ecosystem** | Apple-first, growing | Rust ML ecosystem |

**Our advantage**: Composable architecture for hybrid models. MLX requires new C++ kernels for each architecture variant. metal-attention composes architectures from traits and compiles to specialized kernels automatically.

**Their advantage**: Apple-internal expertise, Neural Accelerator access (M5+), highest baseline throughput. MLX will likely always be faster for pure transformer inference due to Apple's internal Metal knowledge. vllm-mlx already achieves 21-87% higher throughput than llama.cpp ([MLX vs llama.cpp study](https://arxiv.org/pdf/2511.05502)).

### CoreML

| Dimension | CoreML | metal-attention |
|-----------|--------|-----------------|
| **Optimization** | Black-box Apple compiler | Transparent, hand-written kernels |
| **Custom attention** | Not possible | Full control via traits |
| **Hybrid support** | None | First-class |
| **Neural Engine** | Yes | No (Metal GPU only) |
| **Model format** | CoreML model package | GGUF |

**Our advantage**: Full control over attention mechanism. CoreML is a black box -- you cannot customize the attention pattern, compose layer types, or optimize for hybrid ratios.

**Their advantage**: Neural Engine access, Apple's optimization pipeline, integration with iOS/macOS app development. CoreML is the right choice for deploying standard models in production apps; we target the power-user and researcher segment.

### Burn (Rust)

| Dimension | Burn | metal-attention |
|-----------|------|-----------------|
| **Scope** | General DL framework | Inference engine (attention-focused) |
| **Metal backend** | In development (via wgpu/CubeCL) | Hand-written MSL (validated) |
| **simdgroup_matrix** | Inaccessible via CubeCL | Direct access (Proto 5 validated) |
| **Throughput** | 58-70% of native MSL | 100% (hand-written MSL baseline) |
| **Integration** | N/A | AttentionBackend supertrait (Proto 8) |

**Relationship**: Complementary, not competitive. metal-attention's AttentionBackend integrates as a Burn backend extension. Burn provides the framework; we provide the Metal attention kernels Burn cannot generate through CubeCL.

### Candle (HuggingFace)

| Dimension | Candle | metal-attention |
|-----------|--------|-----------------|
| **Language** | Rust | Rust |
| **Metal support** | Yes (metal-candle crate) | Yes (hand-written MSL) |
| **Focus** | General ML inference | Hybrid attention dispatch |
| **Hybrid models** | Per-model implementation | Composable trait system |
| **Maturity** | Active development | Pre-alpha |

**Our advantage**: Composable hybrid architecture dispatch. Candle implements models individually; we make architecture a type parameter.

**Their advantage**: Broader model support, HuggingFace ecosystem integration, embeddings performance (25.9x faster than MLX per metal-candle benchmarks).

### web-rwkv

| Dimension | web-rwkv | metal-attention |
|-----------|----------|-----------------|
| **Language** | Rust | Rust |
| **GPU API** | WebGPU (wgpu) | Native Metal (MSL) |
| **Models** | RWKV only (v4-v7) | Hybrid models (RWKV, Jamba, Griffin, Zamba, transformers) |
| **Platform** | Cross-platform (Vulkan/DX12/Metal/WebGPU) | Apple Silicon only |
| **Metal throughput** | Limited by wgpu abstraction | Full native Metal performance |

**Our advantage**: Native Metal performance (wgpu loses 30-42% throughput), broader architecture support beyond RWKV-only.

**Their advantage**: Cross-platform, browser support via WASM, mature RWKV implementation.

---

## Scope & Prioritization

### P0 -- Must Have (Phase A+B: 7-10 weeks)

These are the foundation without which the project has no reason to exist.

| ID | Requirement | Rationale |
|----|-------------|-----------|
| P0-1 | `trait SequenceBlock` + `LinearSequenceModel` + `SoftmaxAttention` trait hierarchy | Core architecture. Without composable traits, we are just another hard-coded inference engine. |
| P0-2 | Linear attention backend (FLA chunk_h/chunk_o + GPU prefix sum) | The 50-70x advantage over softmax is our wedge. GPU prefix sum eliminates the ~300us CPU bottleneck. |
| P0-3 | Flash attention backend (simdgroup_matrix, multi-simdgroup for >1 TFLOPS) | Needed for softmax attention layers in hybrid models and for pure transformer baseline. |
| P0-4 | Function constant specialization + PsoCache dispatch | Zero-overhead kernel selection is what makes trait composition feasible without runtime cost. |
| P0-5 | GGUF model loading (weights, tokenizer, architecture metadata) | GGUF is the de facto standard for local model deployment. Without it, no models can run. |
| P0-6 | Token generation loop (prefill + decode + sampling) | The minimum viable inference capability. |
| P0-7 | RWKV-7 end-to-end inference | First validation target. Pure linear model proves LinearSequenceModel trait works. |
| P0-8 | CLI: `metal-attention run` with streaming output | Users need a way to actually use the engine. |
| P0-9 | Correctness validation against CPU FP64 reference | Every kernel must produce correct results. Non-negotiable. |

### P1 -- Should Have (Phase C+D: 5-7 weeks)

These deliver the hybrid advantage and ecosystem integration.

| ID | Requirement | Rationale |
|----|-------------|-----------|
| P1-1 | `HybridModel<R, A, RATIO>` runtime dispatch | The differentiating feature. Hybrid layer composition with ratio as const generic. |
| P1-2 | Jamba model inference | Primary hybrid model target. 7:1 SSM:attention + MoE validates full hybrid path. |
| P1-3 | Griffin model inference | Different ratio (2:1) validates architecture flexibility. |
| P1-4 | Zamba model inference (shared attention) | 6:1 with shared attention block validates weight sharing pattern. |
| P1-5 | PagedAttention V2 KV cache | Memory-efficient KV cache for long-context hybrid inference. ~9% overhead validated. |
| P1-6 | RoPE/ALiBi/GQA position encoding + attention variants | Required for production model support. All validated at <0.1% overhead. |
| P1-7 | Quantization (Q4_0, Q4_K_M, Q8_0) | Required for 7B models to fit in consumer memory. |
| P1-8 | `metal-attention bench` CLI command | Benchmark tooling for researchers and performance validation. |
| P1-9 | `metal-attention info` CLI command | Model inspection and architecture detection. |
| P1-10 | M1/M2/M3/M4 hardware compatibility | Metal feature set detection for optimal tile/chunk sizes per GPU generation. |
| P1-11 | Standard transformer baseline (Llama 3/Mistral) | Needed for fair benchmarking and as a degenerate case of the hybrid architecture. |

### P2 -- Nice to Have (Phase D+E: ongoing)

These expand the ecosystem and optimize further.

| ID | Requirement | Rationale |
|----|-------------|-----------|
| P2-1 | Burn `AttentionBackend` supertrait integration | Ecosystem play. Proto 8 validated the pattern (2-17us bridge, ~150 lines). |
| P2-2 | JSON output mode | Programmatic integration for pipelines and benchmarking scripts. |
| P2-3 | Multi-simdgroup flash attention optimization (target >1 TFLOPS) | Current 0.16 TFLOPS is 4-10% of MFA reference. Optimization headroom is large. |
| P2-4 | Async copy / memory prefetching | GPU memory optimization for improved bandwidth utilization. |
| P2-5 | Metal 4 cooperative tensor migration | Future-proofing for M5+ and macOS 26+. Not needed for M1-M4. |
| P2-6 | Additional model architectures (Nemotron-H, Mamba-3, future hybrids) | Expanding model coverage as new architectures release. |
| P2-7 | Speculative decoding | Performance optimization for multi-token generation. |
| P2-8 | Additional quantization formats (Q5_K_M, Q6_K, IQ variants) | Broader model compatibility. |

### Explicit Non-Scope

| Item | Reason |
|------|--------|
| Training / backward pass | Inference-only engine. Training is a different product. |
| Cross-platform (CUDA/Vulkan) | CubeCL proved 30-42% throughput loss. Metal-only is a deliberate choice. |
| HTTP serving / multi-tenant | Single-user local inference. No server mode. |
| Model conversion (PyTorch/SafeTensors) | Users provide GGUF. Conversion is someone else's problem. |
| GUI / TUI | CLI only. Desktop apps are someone else's problem. |
| iOS deployment | macOS Metal only. iOS has different constraints (tile deferred rendering, thermal). |

---

## Risk Assessment

### Technical Risks

| Risk | Likelihood | Impact | Mitigation | Residual Risk |
|------|-----------|--------|------------|---------------|
| **Flash attention stuck at 0.16 TFLOPS** | Low | Medium | Multi-simdgroup is the known fix. Metal Flash Attention (MFA) achieves >1 TFLOPS as reference implementation. The optimization path is documented, not speculative. | If stuck, hybrid models still win because 70-90% of their layers use linear attention (50-70x faster), making flash TFLOPS less critical. |
| **GGUF format lacks hybrid architecture metadata** | Medium | Medium | Custom metadata keys or architecture detection heuristics (layer name patterns). Contribute upstream to llama.cpp's GGUF spec if needed. Jamba and RWKV models already exist in GGUF format. | May need manual architecture specification for some models initially. |
| **Mamba SSM kernel complexity** | Medium | Medium | Start with linear attention (FLA), which is simpler and validated. Add Mamba selective scan as a second LinearSequenceModel implementation. web-rwkv and llama.cpp both have reference implementations. | Mamba-specific kernel may take longer than estimated. Does not block RWKV or FLA-based models. |
| **Linear attention quality loss vs softmax** | Medium | High | This is a model property, not an engine property. Models trained with linear attention (RWKV, Jamba) are designed for it. Softmax fallback is always available. Document quality trade-offs per model. | Some users may be confused about quality differences between attention mechanisms. Clear documentation required. |
| **GPU prefix sum kernel correctness at scale** | Low | Medium | Validated algorithm (parallel scan). Start with small prefix sums, expand to full N. Test against CPU reference at every size. | Edge cases in large prefix sums (> 1024 chunks) may need debugging. |
| **32KB threadgroup memory limit constrains D=128 models** | High (certain) | Medium | Smaller tile/chunk sizes with function constants. D=128 uses Br=16, Bc=16 (24KB). Accepted trade-off: slightly lower throughput for D=128. | D=256 models (rare but emerging) may not fit at all. |

### Market / Product Risks

| Risk | Likelihood | Impact | Mitigation | Residual Risk |
|------|-----------|--------|------------|---------------|
| **Small initial user base** | High | Low | Acceptable for open-source project. Focus on quality benchmarks that get cited. Hybrid models are proliferating; user base grows with model ecosystem. | May take 6-12 months to reach critical mass. |
| **llama.cpp adds optimized hybrid support** | Medium | High | Our structural advantage (trait composition, function constants) is architectural, not just feature parity. Even if llama.cpp optimizes Mamba kernels, they cannot match trait-based composability without a rewrite. | If llama.cpp optimizes well enough, the performance delta may not justify switching for most users. |
| **MLX adds hybrid architecture support** | Medium | High | Similar to llama.cpp risk. MLX would add per-model support; our advantage is composability. MLX would need to expose a trait-like composition mechanism to match our flexibility. | Apple's internal knowledge of Metal gives them a throughput advantage we may never match for individual kernels. |
| **Hybrid model hype fades, transformers remain dominant** | Low | Critical | Hedged by supporting pure transformer inference (RATIO=0). The engine works for all architectures; hybrid optimization is the differentiator, not the only capability. Evidence against this risk: RWKV-7, Jamba 1.5, Mamba-3, Griffin all shipping with strong quality metrics. | If hybrids fade, we become a niche Rust inference engine with good Metal performance but no unique positioning. |
| **Metal 4 + M5 changes kernel landscape** | Low | Low | Metal 3/4 coexist. Metal 4 cooperative tensors are additive (Phase E optimization), not replacement. All current kernels remain valid on M1-M4 hardware, which will be 90%+ of the installed base for years. | M5 Neural Accelerator benefits may be MLX-exclusive initially. |
| **Burn framework pivots away from supertrait pattern** | Low | Low | AttentionBackend is additive (supertrait, not fork). If Burn changes Backend trait, we adapt the bridge. Proto 8 showed the pattern works with ~150 lines. | Minor maintenance burden if Burn API changes. |

### Execution Risks

| Risk | Likelihood | Impact | Mitigation | Residual Risk |
|------|-----------|--------|------------|---------------|
| **Scope creep into model coverage** | High | Medium | Strict P0/P1/P2 prioritization. Ship RWKV-7 first (pure linear, simplest), then Jamba (full hybrid). Do not chase model coverage before the trait architecture is proven. | Pressure to add models before the foundation is solid. Requires discipline. |
| **GGUF parser complexity** | Medium | Medium | Use existing Rust GGUF parsing crate if available, or port from llama.cpp's well-tested parser. This is mechanical work, not research. | Unexpected GGUF format variations across model providers. |
| **Tokenizer integration** | Medium | Low | Use HuggingFace `tokenizers` crate (Rust native, well-tested). Adds dependencies but eliminates a class of bugs entirely. | Dependency size (~40 crates) may concern minimalists. |
| **Single maintainer bottleneck** | High | Medium | Open source early. Write clear contributing guidelines. Prioritize code readability (always_inline helpers validated at 0% overhead -- use them freely for code organization). | Bus factor = 1 until community forms. |

### Risk Summary Matrix

```
                    Low Impact    Medium Impact    High Impact     Critical
                  +-------------+----------------+---------------+-----------+
  High Likelihood | user base   | scope creep    |               |           |
                  |             | 32KB D=128     |               |           |
                  +-------------+----------------+---------------+-----------+
  Med Likelihood  |             | GGUF metadata  | llama.cpp     |           |
                  |             | Mamba kernel   | improves      |           |
                  |             | tokenizer      | MLX improves  |           |
                  +-------------+----------------+---------------+-----------+
  Low Likelihood  | Metal 4     | GPU prefix sum | flash TFLOPS  | hybrid    |
                  | Burn pivot  |                | stuck         | hype fades|
                  +-------------+----------------+---------------+-----------+
```

The highest-consequence risk is "hybrid model hype fades" (Low likelihood, Critical impact). This is mitigated by supporting pure transformer inference as a degenerate case, but if hybrids do not become mainstream, the project loses its primary differentiator. All available evidence (RWKV-7 shipping, Jamba 1.5 at 256K context, Mamba-3 inference-first design, Griffin/RecurrentGemma from Google, Zamba2 beating Llama3 efficiency) suggests this risk is low.

The highest-likelihood risk is "small initial user base" (High likelihood, Low impact). This is acceptable for an open-source project targeting a specific niche. The strategy is to produce benchmark results that get cited by researchers, which drives organic discovery.

---

## Appendix: Research Sources

1. [Jamba: Hybrid Transformer-Mamba Language Model](https://arxiv.org/abs/2403.19887) -- AI21, 2024
2. [RWKV-7 "Goose" with Expressive Dynamic State Evolution](https://arxiv.org/abs/2503.14456) -- BlinkDL, 2025
3. [Griffin: Mixing Gated Linear Recurrences with Local Attention](https://arxiv.org/abs/2402.19427) -- Google DeepMind, 2024
4. [Zamba: A Compact 7B SSM Hybrid Model](https://arxiv.org/abs/2405.16712) -- Zyphra, 2024
5. [Production-Grade Local LLM Inference on Apple Silicon](https://arxiv.org/abs/2511.05502) -- Comparative Study, 2025
6. [Attention was never enough: Tracing the rise of hybrid LLMs](https://www.ai21.com/blog/rise-of-hybrid-llms/) -- AI21, 2025
7. [Performance of llama.cpp on Apple Silicon M-series](https://github.com/ggml-org/llama.cpp/discussions/4167) -- GitHub Discussion
8. [Benchmarking Apple's MLX vs. llama.cpp](https://medium.com/@andreask_75652/benchmarking-apples-mlx-vs-llama-cpp-bbbebdc18416) -- Andreas Kunar, 2025
9. [Burn: Going Big and Small for 2025](https://burn.dev/blog/going-big-and-small-for-2025/) -- Tracel AI
10. [metal-candle Benchmarks](https://github.com/GarthDB/metal-candle/blob/main/BENCHMARKS.md) -- GarthDB
11. [web-rwkv: Pure WebGPU/Rust RWKV Implementation](https://github.com/cryscan/web-rwkv) -- cryscan
12. [Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU](https://machinelearning.apple.com/research/exploring-llms-mlx-m5) -- Apple ML Research
13. [Metal Function Specialization](https://developer.apple.com/documentation/metal/using-function-specialization-to-build-pipeline-variants) -- Apple Developer
14. [RecurrentGemma](https://github.com/google-deepmind/recurrentgemma) -- Google DeepMind
15. [Jamba Reasoning 3B](https://www.ai21.com/blog/introducing-jamba-reasoning-3b/) -- AI21, 2025
16. [Mamba-3: Improved Sequence Modeling](https://openreview.net/forum?id=HwCvaJOiCj) -- OpenReview
17. [Zamba2-7B](https://www.zyphra.com/post/zamba2-7b) -- Zyphra
18. [Native LLM and MLLM Inference at Scale on Apple Silicon](https://arxiv.org/html/2601.19139v1) -- 2026
19. [Forge: High-performance LLM inference engine built on Candle](https://github.com/Ataraxy-Labs/forge) -- Ataraxy Labs
20. [RWKV Language Model Wiki](https://wiki.rwkv.com/) -- RWKV Foundation
