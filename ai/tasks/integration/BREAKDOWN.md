---
id: integration.BREAKDOWN
module: integration
priority: 999999
status: failing
version: 1
origin: spec-workflow
dependsOn: [devops.BREAKDOWN, traits.BREAKDOWN, kernels.BREAKDOWN, gguf.BREAKDOWN, models.BREAKDOWN, inference.BREAKDOWN, cli.BREAKDOWN, burn.BREAKDOWN]
tags: [breakdown]
testRequirements:
  unit:
    required: false
    pattern: "tests/integration/**/*.test.*"
---
# Integration -- Breakdown

## Context

The integration module contains end-to-end tests, model quality validation, stress/leak tests, cross-device compatibility checks, and the benchmark regression suite. This is the final validation layer that ensures all modules compose correctly and the system meets its performance and quality targets. It depends on all other modules being functional.

## Scope

- **End-to-end inference tests**: Load real GGUF models, run prefill + decode, verify coherent text output
- **Kernel correctness suite**: GPU output vs CPU FP64 reference for all kernels across sweep configurations (N, D, tile sizes, heads)
- **Cross-kernel consistency**: Flash vs paged attention (contiguous pages), ALiBi-disabled vs vanilla flash, bridge vs direct GPU
- **Model quality validation**:
  - Tier 2: Token-level accuracy (>99% greedy match, >0.999 cosine similarity vs reference engine)
  - Tier 3: Perplexity evaluation on held-out corpus (within +/- 0.5 of published values)
  - Tier 4: Coherence spot-checks (valid UTF-8, no token loops, expected keywords)
- **Stress tests**: Memory leak detection (all kernels, 1000 iterations, <1% growth), large-N correctness (N=4096, 8192), long generation (10K+ tokens), near-OOM handling
- **Performance benchmarks**: Criterion suite for flash/linear/paged attention TFLOPS, PSO compile latency, E2E inference tok/s, prefill/decode throughput
- **Scaling validation**: O(N^2) for flash attention, O(N) for linear attention, constant PagedAttention overhead percentage
- **Edge case tests**: Empty prompt, single token, prompt exceeding max length, all-zero activations, NaN injection
- **CLI integration tests**: `metal-attention run`, `bench`, `info` produce expected output format
- **Cross-device compatibility**: Feature detection, capability validation across M1/M2/M3/M4
- **Golden output management**: Fixture files for regression detection, regeneration behind feature flags
- **Benchmark regression tracking**: Bencher integration, baseline management per hardware class, 5%/10% thresholds

## Key Decisions

- **From QA.md**: Dual-reference architecture: FP64 CPU reference (authoritative ground truth) + golden output files (regression detection). Both must be maintained.
- **From QA.md**: Test data uses deterministic LCG-based pseudo-random generators with standard seeds (Q=42, K=137, V=999) for reproducibility across platforms.
- **From QA.md**: Tolerance table is centralized in `src/tolerances.rs` with rationale strings. A dedicated test verifies code matches documented tolerances.
- **From QA.md**: Reference models (RWKV-7 1.5B Q4_K_M, TinyLlama 1.1B Q4_0, Jamba Mini Q4_K_M) fetched via download script, NOT checked into repo. CI caches between runs.
- **From QA.md**: Performance regression blocks merge at >10%, warns at >5%. Bencher tracks with Delta Interquartile Range model.
- **From PM.md**: Performance targets: >30% faster than llama.cpp on hybrid layers, >1 TFLOPS flash attention (D=64), <5s model load, <500ms first token latency.

## Acceptance Criteria

1. E2E test: RWKV-7 model loads and generates 10+ coherent tokens without error
2. E2E test: Llama baseline model loads and generates 10+ coherent tokens without error
3. Flash attention GPU output matches CPU FP64 reference within atol=5e-3 for N in {64, 128, 256, 512, 1024}
4. Linear attention GPU output matches CPU FP64 reference within atol=1e-3 for N in {64, 128, 256, 512}
5. Paged attention matches dense attention within atol=1e-3 (contiguous pages)
6. Memory leak test passes for all kernels: <1% growth over 100 iterations
7. All outputs are finite (no NaN, no Inf) across all test configurations
8. Greedy decode produces >99% token match rate vs reference engine (when reference available)
9. Benchmark suite runs and reports TFLOPS, tok/s, and latency for all kernels
10. CLI `run` command produces streaming text output from a real model
11. CLI `info` command displays correct architecture info from a real GGUF file
12. All tests pass with `MTL_SHADER_VALIDATION=1` enabled
13. Stress test: 1000-token generation shows no memory growth beyond initial allocation

## Technical Notes

- From QA.md: GPU warmup: 8 throwaway dispatches before measurement. Timing via hardware GPU timestamps. Coefficient of variation must be <5% to accept measurement.
- From QA.md: Tests must use `--test-threads=1` for Metal device contention avoidance.
- From QA.md: Golden output regeneration behind `--features regen-golden` flag. FP64 reference regeneration behind `--features regen-reference`.
- From QA.md: Numerical edge cases to test: large magnitude Q/K (>10.0), near-zero (<1e-6), uniform values, one-hot K, maximum N (8192+), D=128 with reduced tiles.
- From QA.md: Hybrid-specific tests: SSM-only (RWKV-7), attention-only (Llama), hybrid 7:1 (Jamba), varying ratios (2:1, 6:1, 7:1). Quality must be stable across ratios.
- From QA.md: Pre-release gates include: full benchmark suite, all stress tests, perplexity within tolerance, cross-device test (M1 + M4 minimum), all supported models load and generate.
