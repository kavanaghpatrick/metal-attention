# metal-attention -- Spec Overview

## Executive Summary

metal-attention is a Rust inference engine for Apple Silicon that treats hybrid AI model architectures (Jamba, RWKV-7, Griffin, Zamba) as composable type parameters rather than hard-coded forks. The core insight is that hybrid models spend 70-90% of their compute in linear/SSM layers (the fast path), where prototype data proves 50-70x faster throughput than softmax attention on Apple GPU. An engine optimized for this ratio pattern will fundamentally outperform generic engines that treat all layers identically.

The architecture exploits two validated technical capabilities: (1) composable traits (`SequenceBlock` -> `LinearSequenceModel` / `SoftmaxAttention`) where each combination maps to Metal function constants compiled to zero-overhead specialized GPU code (178ns cache hit, 34-63us cold compile), and (2) hand-written MSL with `simdgroup_matrix` for competitive throughput (CubeCL/wgpu achieves only 58-70% of native performance). The engine reads GGUF model files, auto-detects architecture from metadata, and dispatches through the appropriate kernel paths.

The project targets developers running hybrid models locally on Mac (primary), researchers experimenting with SSM:attention ratios (secondary), and Rust ML ecosystem developers needing high-performance Metal attention via Burn integration (tertiary). It competes with llama.cpp and MLX specifically on hybrid architecture dispatch, not on general transformer coverage. The workspace is organized as 6 Rust crates: traits (no Metal dep), kernels (Metal GPU layer), gguf (parser), models (concrete architectures), main lib (inference pipeline), and optional burn integration.

## PM Summary

- **Target users**: Developers running hybrid models locally on Mac; ML researchers; Burn framework developers
- **P0 scope**: Core traits, linear + flash attention backends, function constant dispatch, GGUF loading, RWKV-7 E2E inference, CLI `run` command, correctness validation
- **P1 scope**: `HybridModel<R, A, RATIO>` dispatch, Jamba/Griffin/Zamba models, PagedAttention V2, RoPE/ALiBi/GQA, quantization (Q4_0, Q4_K_M, Q8_0), `bench`/`info` CLI commands
- **P2 scope**: Burn `AttentionBackend` supertrait, JSON output, multi-simdgroup flash optimization, speculative decoding
- **Key metrics**: >30% faster than llama.cpp on hybrid layers, >1 TFLOPS flash attention (D=64), <5s model load (7B Q4), <500ms first token latency
- **Non-scope**: Training, cross-platform (CUDA/Vulkan), HTTP serving, model conversion, GUI, iOS
- **Highest risk**: "Hybrid hype fades" (Low likelihood, Critical impact) -- mitigated by supporting pure transformer as degenerate case

## UX Summary

- **CLI structure**: Three subcommands -- `run` (inference), `bench` (benchmarks), `info` (model inspection)
- **Convention alignment**: Short flags match llama.cpp (`-m`, `-p`, `-n`, `-s`); kebab-case long flags
- **Output design**: Tokens to stdout, everything else to stderr; streaming by default; `--json` for JSONL machine output
- **Error philosophy**: Conversational errors with actionable hints; `thiserror` for library, `anyhow` for CLI
- **Configuration**: TOML config file at `~/Library/Application Support/metal-attention/config.toml`; CLI > env > config > defaults precedence
- **Library API**: `Engine::new()` -> `load_model()` -> `model.generate()` returning `TokenStream` iterator; builder pattern for `GenerationParams`
- **Key crates**: clap 4.x, indicatif, console, serde_json, tracing, directories

## Tech Summary

- **Workspace**: 6 crates -- `metal-attention-traits` (pure Rust interfaces), `metal-attention-kernels` (Metal GPU layer + MSL shaders), `metal-attention-gguf` (mmap parser), `metal-attention-models` (Llama/RWKV/Jamba/Griffin/Zamba), `metal-attention` (inference pipeline), `metal-attention-burn` (optional Burn bridge)
- **Trait hierarchy**: `SequenceBlock` (root) -> `LinearSequenceModel` (FLA, Mamba, RG-LRU, RWKV7) + `SoftmaxAttention` (Flash, Paged); `HybridModel<L, A>` composes them via `LayerSchedule`
- **Kernel architecture**: 13 MSL shader files compiled via `build.rs` (`xcrun metal -std=metal3.1`); PsoCache with function constants for zero-overhead specialization; `types.h` shared structs (AttentionParams, LayerParams, SSMParams)
- **Memory strategy**: mmap GGUF with `newBufferWithBytesNoCopy` (zero-copy when 32-byte aligned); BufferPool ring allocator for activations; triple-buffered CommandManager with `dispatch_semaphore(3)`
- **GGUF loading**: Binary parser with mmap, architecture auto-detection from metadata + tensor name patterns, weight mapping via `(layer_index, WeightRole)` system
- **Inference pipeline**: Prefill (parallel prompt processing) + Decode (single-token autoregressive) through `[RMSNorm -> SequenceBlock -> RMSNorm -> FFN]` per layer
- **Proto migration**: 8 validated prototypes provide shaders, device/pipeline code, CPU FP64 references; proto preserved as reference, not linked to workspace

## QA Summary

- **Test tiers**: Unit (CPU, every commit), GPU correctness (MTL_SHADER_VALIDATION=1, every PR), benchmarks (criterion, every PR), stress/leak (nightly), model quality (weekly)
- **Dual-reference architecture**: FP64 CPU reference implementations (authoritative ground truth) + golden output files (regression detection)
- **Tolerance table**: Flash 5e-3/1e-2, Linear 1e-3/1e-2, Paged 1e-3/1e-2, RoPE 1e-4/1e-3, GQA 1e-6/1e-6 (exact copy)
- **GPU validation**: All 4 MTL_SHADER_VALIDATION flags enabled; memory leak detection via `currentAllocatedSize()` over 100+ iterations (<1% growth); threadgroup memory budget static assertions
- **CI pipeline**: Tier 1 CPU-only (GitHub-hosted, <5min), Tier 2 GPU (self-hosted M4, <15min), Tier 3 full bench (nightly, <60min), Tier 4 model quality (weekly, <120min)
- **Quality gates**: Pre-merge (clippy, fmt, tests, <10% perf regression), Pre-release (full benchmarks, stress tests, perplexity validation, cross-device), Per-kernel (FP64 reference, tolerance documented, golden file, leak test)
- **Model quality**: Token accuracy >99% greedy match vs reference, logit cosine similarity >0.999, perplexity within +/- 0.5 of published values

## Module Roadmap

| Priority | Module | Description | Dependencies | Phase |
|----------|--------|-------------|--------------|-------|
| 0 | **devops** | Workspace setup, Cargo.toml configs, CI pipelines, build.rs shader compilation, linting | None | Foundation |
| 1 | **traits** | Core trait definitions: `SequenceBlock`, `LinearSequenceModel`, `SoftmaxAttention`, shared types (`TensorView`, `DType`, `BlockConfig`) | devops | Foundation |
| 2 | **kernels** | Metal shader compilation, `GpuDevice`, `PsoCache`, `BufferPool`, `CommandManager`, kernel dispatch (flash, linear, paged, RoPE, GQA, RMSNorm, FFN, matmul, dequant, SSM, prefix_sum) | devops, traits | Foundation |
| 3 | **gguf** | GGUF binary parser with mmap, metadata accessor, tensor info, quantization types, embedded BPE/SPM tokenizer, architecture detection, weight mapping | devops | Foundation |
| 4 | **models** | Concrete model implementations: RWKV-7 (pure linear), Jamba (7:1 hybrid + MoE), Griffin (2:1), Zamba (6:1 shared), Llama/Mistral (pure transformer baseline) | traits, kernels, gguf | Core |
| 5 | **inference** | Inference pipeline: prefill/decode loop, sampling engine (temperature, top-p/k, repetition penalty), token generation, `Engine` + `Model` public API | traits, kernels, gguf, models | Core |
| 6 | **cli** | CLI binary: `run`/`bench`/`info` subcommands, streaming output, JSON mode, TOML config, progress indicators, error formatting, shell completions | inference | Interface |
| 7 | **burn** | Optional Burn framework integration: `AttentionBackend` supertrait, Burn tensor <-> Metal buffer bridge, backend trait delegation | traits, kernels | Ecosystem |
| 999999 | **integration** | End-to-end tests, model quality validation (perplexity, token accuracy, golden outputs), stress/leak tests, cross-device compatibility, benchmark regression suite | All modules | Validation |
