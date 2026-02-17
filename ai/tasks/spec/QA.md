# QA Strategy: metal-attention Hybrid Model Inference Engine

> **Project**: metal-attention
> **Scope**: Rust inference engine with composable attention traits, Metal GPU kernels, hybrid models (Jamba, RWKV, Griffin)
> **Baseline**: 34 passing tests, 48 criterion benchmarks, 8 Metal shader files, FP64 CPU reference implementations
> **Date**: 2026-02-14

---

## Research Findings

### GPU Kernel Testing Strategies

Metal provides two validation layers for GPU-side correctness: API validation (CPU-side usage correctness) and shader validation (GPU-side memory access, null texture detection, out-of-bounds threadgroup access). The `MTL_SHADER_VALIDATION` environment variable instruments all global memory accesses, with `MTL_SHADER_VALIDATION_FAIL_MODE` defaulting to "zerofill" where invalid reads return 0 and invalid writes are silently dropped. Apple recommends enabling shader validation during development and QA but not in production due to high performance and memory overhead.

Sources:
- [Validating Metal Shader Usage - Apple Developer](https://developer.apple.com/documentation/xcode/validating-your-apps-metal-shader-usage/)
- [MetalValidation(1) Man Page](https://keith.github.io/xcode-man-pages/MetalValidation.1.html)
- [Debug GPU-side Errors in Metal - WWDC20](https://developer.apple.com/videos/play/wwdc2020/10616/)

### Flash Attention Numerical Accuracy Testing

Flash Attention is tested to produce the same output and gradient as a reference implementation up to numerical tolerance, with maximum numerical error checked to be at most twice the numerical error of a baseline implementation in PyTorch. Dual-Delta Testing is an emerging methodology that evaluates two error distributions against a high-precision oracle, enabling rigorous comparison between custom and baseline implementations. FlashAttention-3 achieves close to 1.2 PFLOPS on NVIDIA with FP8, demonstrating 2.6x smaller error than baseline FP8 attention.

Sources:
- [Flash Attention - Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
- [FlashAttention-3 - PyTorch Blog](https://pytorch.org/blog/flashattention-3/)
- [Dual-Delta Testing for Numerical Accuracy - arXiv](https://arxiv.org/html/2602.10605)

### Rust Criterion Benchmark Best Practices

Criterion.rs is the de facto standard for Rust benchmarking, providing statistics-driven micro-benchmarking with automatic detection of performance improvements and regressions. Bencher extends this for continuous benchmarking in CI, using state-of-the-art customizable analytics (including Delta Interquartile Range model used by Rustls) to detect performance regressions before production. Iai provides an alternative using instruction counts instead of wall clock time, ideal for CI environments where shared runners lack consistent timing.

Sources:
- [Criterion.rs Documentation](https://bheisler.github.io/criterion.rs/book/)
- [How to Track Criterion Benchmarks in CI - Bencher](https://bencher.dev/learn/track-in-ci/rust/criterion/)
- [Rustls: Continuous Benchmarking Case Study - Bencher](https://bencher.dev/learn/case-study/rustls/)
- [Benchmarking and Analyzing Rust Performance - LambdaClass](https://blog.lambdaclass.com/benchmarking-and-analyzing-rust-performance-with-criterion-and-iai/)

### Property-Based Testing for Numerical Code

Proptest, inspired by Python's Hypothesis, allows testing that properties hold for arbitrary inputs with automatic minimal test case shrinking. For GPU kernel testing, CPU-side property tests enable validation without GPU hardware in CI. The development cycle is: write the algorithm, test thoroughly on CPU with standard tools, then run on GPU knowing the logic is correct. This eliminates debugging cycles by validating numerical correctness before GPU execution.

Sources:
- [Proptest on GitHub](https://github.com/proptest-rs/proptest)
- [Property-Based Testing in Rust - LogRocket](https://blog.logrocket.com/property-based-testing-in-rust-with-proptest/)
- [Rust Running on Every GPU - Rust GPU Blog](https://rust-gpu.github.io/blog/2025/07/25/rust-on-every-gpu/)

### Hybrid Model Architecture Testing

Hybrid models like Jamba (7:1 SSM:Transformer) interleave Mamba and attention layers with grouped-query attention and MoE routing. RWKV-X achieves near-perfect accuracy on 64K passkey retrieval benchmarks. Testing hybrid quality requires evaluating each layer type independently AND in composition, as hybrid quality cannot be inferred from standalone performance of individual components.

Sources:
- [Jamba: A Hybrid Transformer-Mamba Language Model - arXiv](https://arxiv.org/html/2403.19887v1)
- [RWKV-X: A Linear Complexity Hybrid Language Model - arXiv](https://arxiv.org/html/2504.21463v2)
- [A Systematic Analysis of Hybrid Linear Attention - arXiv](https://arxiv.org/pdf/2507.06457)

### CI/CD with Metal GPU on macOS

GitHub-hosted Apple Silicon runners (M1) now have GPU hardware acceleration enabled by default. However, Metal GPU passthrough for full shader execution is still not universally guaranteed on GitHub-hosted runners. Self-hosted runners on Mac mini/Studio provide reliable GPU access. Cilicon enables ephemeral CI VMs with near-native performance using Apple's Virtualization Framework.

Sources:
- [GitHub Actions Apple Silicon M1 Runners](https://github.com/orgs/community/discussions/69211)
- [GPU Passthrough for GitHub-hosted macOS Runners - Issue #7085](https://github.com/actions/runner-images/issues/7085)
- [Cilicon - Ephemeral macOS CI](https://github.com/traderepublic/Cilicon)

### Perplexity and Model Quality Evaluation

Perplexity measures model uncertainty when predicting the next token, calculated as the exponential of average negative log-likelihood. Lower perplexity indicates better performance. Token-level cross-entropy analysis identifies specific vocabulary items where models show high uncertainty. Perplexity is commonly used to track quality after quantization but cannot directly measure semantic understanding or factual accuracy. It is not comparable across models with different tokenizers.

Sources:
- [Perplexity for LLM Evaluation - Comet](https://www.comet.com/site/blog/perplexity-for-llm-evaluation/)
- [LLM Performance Metrics - Galileo](https://galileo.ai/blog/llm-performance-metrics)
- [Evaluating Perplexity on Language Models - MachineLearningMastery](https://machinelearningmastery.com/evaluating-perplexity-on-language-models/)

---

## Test Architecture

### Test Categories

| Category | Tool | Requires GPU | Runs in CI | Frequency |
|----------|------|-------------|-----------|-----------|
| Unit tests | `cargo test` | No (CPU reference) | Yes | Every commit |
| GPU correctness tests | `cargo test` + MTL_SHADER_VALIDATION=1 | Yes | Self-hosted only | Every commit |
| Property-based tests | `cargo test` + proptest | No (CPU logic) | Yes | Every commit |
| Integration tests | `cargo test --test` | Yes | Self-hosted only | Every commit |
| Criterion benchmarks | `cargo bench` | Yes | Self-hosted only | Every PR |
| Stress/leak tests | `cargo test --release -- --ignored` | Yes | Self-hosted only | Nightly |
| Model quality tests | Custom harness | Yes | Self-hosted only | Weekly / pre-release |
| End-to-end inference | `metal-attention run` | Yes | Self-hosted only | Pre-release |

### Directory Structure

```
metal-attention/
  src/
    lib.rs
    attention/          # Trait definitions + impls
      mod.rs
      flash.rs          # SoftmaxAttention impl
      linear.rs         # LinearSequenceModel impl
      paged.rs          # PagedAttention KV cache
      variants.rs       # RoPE, ALiBi, GQA
    kernel/             # Metal shader management
      mod.rs
      pso_cache.rs      # PSO compilation + caching
      encode.rs         # Command buffer encoding
      device.rs         # Metal device init
    model/              # Model loading + runtime
      mod.rs
      gguf.rs           # GGUF parser
      hybrid.rs         # HybridModel<R, A, RATIO>
      generate.rs       # Token generation loop
    types.rs            # Shared type definitions
  shaders/              # Hand-written MSL
    flash_attention.metal
    linear_attention.metal
    paged_attention.metal
    rope.metal
    gqa_remap.metal
    prefix_sum.metal
  tests/
    correctness/        # GPU vs CPU FP64 reference
      flash.rs
      linear.rs
      paged.rs
      rope.rs
      alibi.rs
      gqa.rs
      prefix_sum.rs
    integration/        # Multi-kernel pipeline tests
      hybrid_dispatch.rs
      kv_cache_lifecycle.rs
      model_loading.rs
      token_generation.rs
    property/           # Proptest-based
      numerical_invariants.rs
      attention_properties.rs
      memory_layout.rs
    stress/             # Long-running, ignored by default
      memory_leak.rs
      large_sequence.rs
      repeated_compile.rs
      oom_recovery.rs
    model_quality/      # Output quality validation
      perplexity.rs
      token_accuracy.rs
      golden_outputs.rs
  benches/
    flash_attention.rs
    linear_attention.rs
    paged_attention.rs
    hybrid_dispatch.rs
    pso_compile.rs
    e2e_inference.rs
    variant_overhead.rs
  fixtures/
    golden/             # Golden output files (checked in)
    models/             # Small test model weights (git-lfs or download script)
    reference/          # CPU FP64 reference outputs for regression
```

### Test Naming Convention

All tests follow the pattern: `test_{kernel}_{what}_{variant}` for correctness tests, `bench_{kernel}_{metric}_{config}` for benchmarks, and `stress_{concern}_{scenario}` for stress tests. Examples:

- `test_flash_correctness_n256_d64` -- Correctness against FP64 reference
- `test_flash_finiteness_all_outputs` -- No NaN/Inf in outputs
- `bench_flash_tflops_n2048_d64_br16_bc64` -- Throughput benchmark
- `stress_flash_leak_1000_iterations` -- Memory leak detection

---

## Correctness Testing

### Dual-Reference Architecture

Every GPU kernel is validated against two references:

1. **FP64 CPU reference implementation** -- The authoritative "ground truth." Implemented in pure Rust using f64 arithmetic with naive (non-tiled, non-optimized) algorithms. These reference implementations are already proven in the prototype (34 tests passing).

2. **Known-good GPU output (golden files)** -- Serialized outputs from validated GPU runs, used for regression detection when reference implementations are too slow for large-N tests.

### Tolerance Table (Established from Prototypes)

| Kernel | atol | rtol | Rationale |
|--------|------|------|-----------|
| Flash Attention | 5e-3 | 1e-2 | FP32 online softmax vs FP64 naive; tiled accumulation reordering |
| Linear Attention (FLA) | 1e-3 | 1e-2 | FP32 chunk accumulation across D=64 dimensions |
| PagedAttention V2 | 1e-3 | 1e-2 | Scalar dot products + paging indirection |
| RoPE | 1e-4 | 1e-3 | Element-wise trig; FP32 sin/cos well-approximated |
| ALiBi | 5e-3 | 1e-2 | Dominated by flash attention softmax accumulation error |
| GQA Remap | 1e-6 | 1e-6 | Pure memory copy; should be exact |
| GPU Prefix Sum | 1e-6 | 1e-6 | Integer-like FP32 accumulation; exact for small values |
| Hybrid Layer Output | 5e-3 | 1e-2 | Composed error from linear + softmax layers |

### Correctness Test Categories

**1. Kernel-Level Correctness (per kernel, per config)**

```rust
// Pattern: GPU result vs FP64 CPU reference
#[test]
fn test_flash_correctness_n256_d64() {
    let device = GpuDevice::shared();
    let (q, k, v) = generate_deterministic_data(256, 64, seed: 42);
    let gpu_result = run_flash_attention(device, &q, &k, &v, 256, 64);
    let cpu_result = cpu_attention_f64(&q, &k, &v, 256, 64);
    assert_all_finite(&gpu_result);
    assert_allclose(&gpu_result, &cpu_result, 5e-3, 1e-2, "flash N=256 D=64");
}
```

Sweep configurations:
- Sequence lengths: N in {64, 128, 256, 512, 1024, 2048}
- Head dimensions: D in {64, 128}
- Chunk/page sizes: all valid sizes within 32KB threadgroup budget
- Number of heads: 1, 4, 8, 32

**2. Finiteness Checks**

Every test verifies that ALL GPU output elements are finite (not NaN, not Inf). This catches:
- Uninitialized threadgroup memory reads
- Division by zero in softmax normalization
- Overflow in FP32 accumulation

**3. Dimensional Correctness**

Every test asserts output shape matches expected dimensions before comparing values:
```rust
assert_eq!(gpu_result.len(), seq_len * head_dim);
```

**4. Symmetry and Invariant Tests**

- Flash attention with all-zero V should produce all-zero output
- GQA remap with group_size=1 (MHA) should be identity
- RoPE at position 0 should not modify input
- Linear attention with identity K (K^T = I) should produce V directly

**5. Cross-Kernel Consistency**

- Flash attention and paged attention (with contiguous pages) should produce matching output within paged tolerance
- ALiBi-disabled flash attention should match vanilla flash attention exactly
- Bridge output (Proto 8 path) should match direct GPU output exactly (atol=1e-6)

### Numerical Stability Edge Cases

| Test Case | What It Catches |
|-----------|----------------|
| Large magnitude Q/K values (>10.0) | Softmax overflow in exp() |
| Near-zero Q/K values (<1e-6) | Underflow in attention scores |
| Uniform Q/K (all same value) | Degenerate softmax (uniform distribution) |
| One-hot K (single nonzero position) | Attention should select the corresponding V |
| Maximum sequence length (N=8192+) | Accumulation error growth over long sequences |
| D=128 with reduced tile sizes | Correctness under constrained threadgroup memory |

---

## Performance Testing

### Benchmark Framework

- **Primary**: Criterion.rs 0.5 with `iter_custom` for GPU timing via `MTLCommandBuffer::GPUStartTime/GPUEndTime`
- **Regression tracking**: Bencher for continuous benchmarking with Delta Interquartile Range statistical model
- **Supplementary**: Iai for instruction-count-based benchmarks on CPU-side logic (PSO cache lookup, GGUF parsing)

### Benchmark Categories

**1. Kernel Throughput (TFLOPS)**

| Benchmark | Metric | Target | Current |
|-----------|--------|--------|---------|
| Flash Attention N=2048 D=64 | TFLOPS | >1.0 | 0.16 |
| Flash Attention N=1024 D=64 | TFLOPS | >0.5 | 0.11 |
| Linear Attention N=1024 D=64 | Wall-clock | <100us | 280us (with CPU prefix sum) |
| Linear Attention N=1024 D=64 (GPU-only) | Kernel time | <50us | ~35us |
| PagedAttention N=1024 D=64 ps=16 | Overhead vs dense | <15% | ~9% |

**2. Infrastructure Latency**

| Benchmark | Metric | Target | Current |
|-----------|--------|--------|---------|
| PSO cold compile (per variant) | Latency | <100us | 34-63us |
| PSO cache hit | Latency | <1us | 178ns |
| RoPE per head | Latency | <20us | ~10us |
| GQA remap (gs=4, H=8) | Latency | <200us | ~78us |
| ALiBi fused | Overhead | 0% | ~0% |
| Burn bridge | Overhead | <20us | 2-17us |

**3. End-to-End Inference**

| Benchmark | Metric | Target |
|-----------|--------|--------|
| First token latency (512-token prompt) | Wall-clock | <500ms |
| Token generation (7B Q4, M4 Pro) | tok/s | Competitive with llama.cpp |
| Hybrid model speedup (Jamba vs llama.cpp) | Relative | >30% on SSM layers |
| Model load time (7B Q4 GGUF) | Wall-clock | <5s |
| Total PSO compile (all variants at startup) | Wall-clock | <100ms |

**4. Scaling Tests**

Benchmark each kernel across a sequence length sweep to validate asymptotic complexity:
- Flash attention: verify O(N^2) scaling (2x N should yield ~4x time)
- Linear attention: verify O(N) scaling (2x N should yield ~2x time)
- PagedAttention: verify overhead remains constant percentage as N grows

### Benchmark Methodology

1. **GPU warmup**: 8 throwaway dispatches before measurement to stabilize GPU clocks and populate caches
2. **Timing**: Hardware GPU timestamps via `MTLCommandBuffer::GPUStartTime/GPUEndTime`, not wall-clock
3. **Statistical rigor**: Criterion default 50 samples, 10s measurement time, 5s warmup
4. **Coefficient of variation**: Report CV%; reject measurements with CV > 5%
5. **Deterministic data**: LCG-generated pseudo-random inputs with fixed seeds for reproducibility
6. **Buffer pre-allocation**: Allocate all Metal buffers before benchmark loop to isolate kernel time

### Regression Detection

- **Threshold**: 5% regression triggers warning; 10% regression blocks merge
- **Tracking**: Bencher stores historical benchmark results per commit
- **Baseline**: `main` branch benchmarks updated on merge; PR benchmarks compared against baseline
- **Noise floor**: Criterion's statistical analysis filters measurement noise; only statistically significant changes flagged

---

## GPU-Specific Testing

### Shader Validation

All test runs execute with:
```
MTL_SHADER_VALIDATION=1
MTL_SHADER_VALIDATION_DEVICE_AND_CONSTANT_MEMORY=1
MTL_SHADER_VALIDATION_THREADGROUP_MEMORY=1
MTL_SHADER_VALIDATION_TEXTURE_USAGE=1
```

This catches:
- Out-of-bounds device/constant memory access
- Out-of-bounds threadgroup memory access
- Null buffer/texture access
- Incorrect buffer binding indices

Shader validation is set in the test harness environment, not in production builds, due to significant performance overhead.

### Memory Leak Detection

**Established pattern from prototype** (already passing): Track `MTLDevice::currentAllocatedSize()` over repeated iterations and assert growth < 1%.

| Test | Iterations | Threshold | Status |
|------|-----------|-----------|--------|
| Flash Attention leak test | 100 | <1% growth | Passing |
| PagedAttention leak test | 100 | <1% growth | Passing |
| Linear Attention leak test | 100 | <1% growth | Passing |
| RoPE leak test | 1000 | <1% growth | Passing |
| GQA Remap leak test | 1000 | <1% growth | Passing |

**Production extensions**:
- PSO cache growth: Verify PsoCache does not leak compiled pipeline states when entries are evicted
- KV cache lifecycle: Allocate, fill, evict, reallocate KV cache pages; verify no net memory growth
- Model load/unload: Load model, run inference, drop model, verify `currentAllocatedSize` returns to baseline
- Long-running inference: 10,000+ token generation; verify steady-state memory after initial allocation

### Threadgroup Memory Budget Validation

Static compile-time assertions for all (kernel, tile_size, D) combinations:

```rust
// Enforced per-kernel: total threadgroup memory <= 32KB
const_assert!(Q_TILE + K_CHUNK + S_TILE <= 32 * 1024);
```

Runtime validation: Each PsoKey encodes tile/chunk/page sizes. The PSO compilation step verifies that the Metal compiler accepts the threadgroup memory budget. A test matrix covers all valid configurations:

| Kernel | Parameter | Valid Values (D=64) | Valid Values (D=128) |
|--------|-----------|--------------------|--------------------|
| Flash Attention | (Br, Bc) | (16,64), (32,64) | (16,16), (16,32) |
| PagedAttention | page_size | 8, 16, 32 | 8, 16 |
| Linear Attention | chunk_size | 32 | 16 |

### Device Compatibility Testing

| Feature | M1 | M2 | M3 | M4 | Test Strategy |
|---------|----|----|----|----|---------------|
| simdgroup_matrix | Yes | Yes | Yes | Yes | Core requirement; fail fast if absent |
| 32KB threadgroup memory | Yes | Yes | Yes | Yes | Budget validated at compile time |
| MSL 3.1 | Yes | Yes | Yes | Yes | Checked at device initialization |
| GPU core count | 7-8 | 8-10 | 10 | 10 | Threadgroup count scales; benchmark reports per-device |
| Memory bandwidth | 100GB/s | 100GB/s | 100GB/s | 120GB/s | Performance targets scaled per-device |

**Feature detection at startup**:
```rust
fn validate_device(device: &MTLDevice) -> Result<DeviceCapabilities> {
    assert!(device.supportsFamily(MTLGPUFamily::Apple7)); // M1+
    let caps = DeviceCapabilities {
        max_threadgroup_memory: 32 * 1024, // All M-series
        supports_simdgroup_matrix: true,    // All M-series
        gpu_core_count: detect_core_count(device),
    };
    Ok(caps)
}
```

### Command Buffer Error Handling

Every command buffer commit is followed by status checking:
```rust
cmd_buf.waitUntilCompleted();
match cmd_buf.status() {
    MTLCommandBufferStatus::Completed => { /* success */ },
    MTLCommandBufferStatus::Error => {
        let error = cmd_buf.error().unwrap();
        panic!("GPU command buffer error: {}", error.localizedDescription());
    },
    _ => panic!("Unexpected command buffer status"),
}
```

Tests specifically verify error handling for:
- Buffer size mismatches (buffer too small for dispatch grid)
- Invalid function constant combinations
- Exceeding threadgroup memory limits at PSO compile time

---

## Model Quality Testing

### Tier 1: Numerical Equivalence (Per-Kernel)

Already covered in Correctness Testing above. Verifies GPU kernels produce mathematically equivalent results to CPU references within documented tolerances.

### Tier 2: Token-Level Accuracy

Compare token predictions (argmax of logits) between metal-attention and a reference implementation (llama.cpp or HuggingFace Transformers) for the same model weights.

| Test | Metric | Target |
|------|--------|--------|
| Greedy decode match (100 tokens) | Token match rate | >99% identical tokens |
| Top-5 agreement (100 tokens) | Top-5 overlap | >95% |
| Logit cosine similarity | Per-token | >0.999 |

**Procedure**:
1. Load identical GGUF model in both metal-attention and reference engine
2. Feed identical prompt tokens
3. Extract raw logits before sampling from both engines
4. Compare logit vectors token-by-token via cosine similarity and top-k agreement
5. Compare greedy-decoded token sequences

### Tier 3: Perplexity Evaluation

Compute perplexity on a held-out text corpus (WikiText-2 or a project-local evaluation set):

| Model | Quantization | Perplexity Target | Tolerance |
|-------|-------------|-------------------|-----------|
| RWKV-7 (7B) | FP16 | Match published perplexity | +/- 0.5 |
| RWKV-7 (7B) | Q4_K_M | Match llama.cpp Q4_K_M perplexity | +/- 0.5 |
| Jamba 1.5 Mini | FP16 | Match published perplexity | +/- 0.5 |
| Llama 3 (8B) | Q4_0 | Match llama.cpp perplexity | +/- 0.3 |

Perplexity is tracked per-release and per-quantization format. Regressions > 1.0 perplexity point block release.

### Tier 4: Coherence Spot-Check

Automated generation tests with heuristic quality checks:
- Generated text is valid UTF-8
- No repeated token loops (same 3+ tokens repeating indefinitely)
- Output length matches requested length (within sampling variance)
- Known factual prompts produce expected keywords in output

These are smoke tests, not precision metrics. They catch catastrophic failures (garbage output, infinite loops, empty output).

### Hybrid-Specific Quality Tests

| Test | What It Validates |
|------|-------------------|
| SSM-only inference (RWKV-7) | LinearSequenceModel produces coherent text in isolation |
| Attention-only inference (Llama 3) | SoftmaxAttention produces coherent text in isolation |
| Hybrid 7:1 inference (Jamba) | Layer composition does not degrade quality vs individual layers |
| Varying RATIO (2:1, 6:1, 7:1) | Quality stable across different SSM:attention ratios |
| Linear vs softmax on same layer | Quantify quality delta when substituting layer types |

---

## Stress & Edge Cases

### Large Model / Long Sequence Tests

| Scenario | Configuration | What It Tests |
|----------|--------------|---------------|
| Maximum sequence length | N=8192, D=64, 32 heads | Accumulation error growth, memory limits |
| Maximum model size (fits in RAM) | 13B Q4 on 32GB M4 Pro | Memory management, mmap loading |
| Minimum model size | 0.5B FP16 | Degenerate case; all overhead, minimal compute |
| Many heads | N=256, D=64, 128 heads | Threadgroup grid scaling, GQA with large group sizes |
| D=128 head dimension | N=1024, D=128 | Reduced tile sizes, correctness under memory pressure |
| Single token decode | N=1, D=64 | Degenerate attention (1x1 score matrix) |

### Memory Pressure Tests

| Scenario | Method | Expected Behavior |
|----------|--------|-------------------|
| Near-OOM model loading | Load model consuming >90% available RAM | Graceful error, no crash, memory reclaimed |
| KV cache growth to limit | Generate tokens until KV cache exhausts available memory | Graceful stop or cache eviction |
| Concurrent Metal allocations | Multiple inference streams sharing device | No silent corruption; serialized or explicit error |
| Buffer allocation failure | Mock allocation returning nil | Rust Result propagation, no panic |

### Error Recovery Tests

| Scenario | Trigger | Expected Behavior |
|----------|---------|-------------------|
| Invalid GGUF file | Truncated/corrupted file | Error::InvalidModel returned, no panic |
| Unsupported architecture | GGUF with unknown layer type | Error::UnsupportedArchitecture with model name |
| Missing Metal device | (Hard to test on Mac) | Descriptive error at startup, not deep in pipeline |
| PSO compilation failure | Invalid function constant combination | Error::PsoCompileFailed with shader name + constants |
| Command buffer error | Dispatch with wrong buffer sizes | GPU error caught and propagated |
| Model file not found | Non-existent path | Standard io::Error propagation |

### Edge Case Input Tests

| Input | Expected |
|-------|----------|
| Empty prompt (0 tokens) | Error or empty output, no crash |
| Single token prompt | Valid first-token generation |
| Prompt exceeding max sequence length | Truncation with warning, or error |
| All-zero input activations | Valid output (likely uniform distribution) |
| NaN in input (injected) | Propagates as NaN in output; test documents this behavior |
| Extremely long generation (10K+ tokens) | Stable output, no memory growth, no quality degradation |

---

## CI/CD Strategy

### Pipeline Architecture

```
                        ┌─────────────────────┐
                        │  GitHub PR Created   │
                        └──────────┬──────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
              ┌─────┴─────┐                ┌──────┴──────┐
              │  CI: CPU   │                │ CI: GPU     │
              │ (GitHub    │                │ (Self-hosted│
              │  hosted)   │                │  M4 runner) │
              └─────┬─────┘                └──────┬──────┘
                    │                             │
            ┌───────┴───────┐          ┌──────────┴──────────┐
            │ cargo check   │          │ cargo test           │
            │ cargo clippy  │          │  (MTL_SHADER_VALID.) │
            │ cargo fmt     │          │ cargo bench (subset) │
            │ cargo test    │          │  (regression check)  │
            │  --no-default │          └──────────┬──────────┘
            │  -features    │                     │
            │ proptest      │                     │
            └───────┬───────┘                     │
                    │                             │
                    └──────────────┬──────────────┘
                                   │
                        ┌──────────┴──────────┐
                        │   Merge to main     │
                        └──────────┬──────────┘
                                   │
                        ┌──────────┴──────────┐
                        │  Nightly (GPU):     │
                        │  - Full bench suite │
                        │  - Stress tests     │
                        │  - Leak tests       │
                        │  - Model quality    │
                        └─────────────────────┘
```

### CI Tier 1: CPU-Only (GitHub-Hosted Runner, Every Commit)

Runs on standard GitHub Actions macOS ARM64 runner (no GPU required):

- `cargo check --all-targets` -- Compilation verification
- `cargo clippy --all-targets -- -D warnings` -- Lint enforcement
- `cargo fmt --check` -- Format enforcement
- `cargo test --lib` -- Unit tests (CPU reference implementations, type tests, parsing)
- Property-based tests via proptest (CPU-only numerical invariants)
- `cargo test --doc` -- Documentation example tests
- GGUF parsing tests (no GPU needed)
- Tolerance table consistency check (documented tolerances match code)

**Time budget**: <5 minutes

### CI Tier 2: GPU Correctness (Self-Hosted M4 Runner, Every PR)

Runs on dedicated Mac mini/Studio with M4 GPU:

- `MTL_SHADER_VALIDATION=1 cargo test` -- All correctness tests with shader validation
- `cargo bench -- --quick` -- Abbreviated benchmark run (2 samples) for regression detection
- Device capability detection tests
- PSO compilation tests for all valid function constant combinations

**Time budget**: <15 minutes

### CI Tier 3: Full Benchmark (Self-Hosted, Nightly)

- Complete criterion benchmark suite (50 samples, 10s measurement)
- Benchmark results uploaded to Bencher for tracking
- Regression report generated comparing against `main` baseline
- Memory leak stress tests (`--ignored` tests)
- Large-N correctness tests (N=4096, N=8192)

**Time budget**: <60 minutes

### CI Tier 4: Model Quality (Self-Hosted, Weekly / Pre-Release)

- Perplexity evaluation on evaluation corpus
- Token accuracy comparison vs reference implementations
- End-to-end inference smoke tests (generate 100 tokens from standard prompts)
- Cross-model consistency (same engine, different architectures)

**Time budget**: <120 minutes

### Headless Metal Testing

Metal commands execute without a display on macOS when:
1. Running as a daemon or background process (Metal compute does not require a window)
2. The `MTLCreateSystemDefaultDevice()` call succeeds (available on all Mac hardware)
3. No render pass is needed (all tests use compute shaders only)

Self-hosted runner configuration:
```bash
# Runner environment variables
export MTL_SHADER_VALIDATION=1
export MTL_SHADER_VALIDATION_FAIL_MODE=zerofill
export RUST_TEST_THREADS=1  # Avoid Metal device contention
```

Tests must use `--test-threads=1` to prevent multiple tests from competing for the shared Metal device and command queue.

---

## Regression Prevention

### Benchmark Baselines

Baselines are stored per-hardware-class (not per-specific-machine) to allow runner replacement:

| Hardware Class | Identifier | Storage |
|---------------|-----------|---------|
| M4 (10-core GPU, 16GB) | `m4-10c-16gb` | Bencher cloud + `baselines/m4-10c-16gb.json` in repo |
| M4 Pro (20-core GPU, 48GB) | `m4pro-20c-48gb` | Bencher cloud + `baselines/m4pro-20c-48gb.json` |
| M1 (8-core GPU, 16GB) | `m1-8c-16gb` | Bencher cloud + `baselines/m1-8c-16gb.json` |

Baseline format (per benchmark):
```json
{
  "benchmark": "flash_attention/N=2048_D=64_Br=16_Bc=64",
  "hardware": "m4-10c-16gb",
  "median_ns": 6500000,
  "mad_ns": 150000,
  "tflops": 0.163,
  "commit": "abc123",
  "date": "2026-02-14"
}
```

### Tolerance Tracking

Correctness tolerances are encoded as constants in a shared module, not scattered across tests:

```rust
// src/tolerances.rs
pub struct KernelTolerance {
    pub atol: f64,
    pub rtol: f64,
    pub kernel: &'static str,
    pub rationale: &'static str,
}

pub const FLASH_TOLERANCE: KernelTolerance = KernelTolerance {
    atol: 5e-3,
    rtol: 1e-2,
    kernel: "flash_attention",
    rationale: "FP32 online softmax vs FP64 reference; tiled accumulation reordering",
};

pub const ROPE_TOLERANCE: KernelTolerance = KernelTolerance {
    atol: 1e-4,
    rtol: 1e-3,
    kernel: "rope",
    rationale: "Element-wise trig; FP32 sin/cos well-approximated",
};
// ... all kernels
```

A dedicated test verifies that the tolerance constants in code match this document:
```rust
#[test]
fn test_tolerance_table_consistency() {
    assert_eq!(FLASH_TOLERANCE.atol, 5e-3);
    assert_eq!(ROPE_TOLERANCE.atol, 1e-4);
    assert_eq!(GQA_TOLERANCE.atol, 1e-6);
    // ... ensures doc and code stay synchronized
}
```

### Performance Regression Workflow

1. PR opened: CI Tier 2 runs abbreviated benchmarks
2. Bencher compares against main baseline using Delta Interquartile Range
3. If regression > 5%: Warning comment on PR with affected benchmarks
4. If regression > 10%: PR check fails; requires manual override or fix
5. On merge to main: Full benchmark suite runs; baseline updated

### Correctness Regression Workflow

1. New tolerance required (e.g., new kernel): Must add entry to tolerance table with rationale
2. Tolerance loosened: Requires approval + comment explaining why (e.g., algorithm change)
3. Tolerance tightened: Encouraged; indicates improved numerical stability
4. Golden output mismatch: Investigate whether output changed for better or worse; update golden file only with justification

---

## Test Data & Fixtures

### Synthetic Data Generation

All test data uses deterministic pseudo-random generators for reproducibility:

```rust
/// LCG-based deterministic f32 generator.
/// Same seed always produces same sequence across platforms.
fn random_f32(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len).map(|_| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let bits = (state >> 33) as u32;
        (bits as f32 / (u32::MAX >> 1) as f32) * 2.0 - 1.0
    }).collect()
}
```

**Standard seeds** (used across all tests for consistency):
- Q data: seed = 42
- K data: seed = 137
- V data: seed = 999

**Specialized data generators**:
- `gen_large_magnitude(seed)` -- Values in [-100, 100] for overflow testing
- `gen_near_zero(seed)` -- Values in [-1e-6, 1e-6] for underflow testing
- `gen_uniform(value, len)` -- All-same-value for degenerate case testing
- `gen_one_hot(position, len, dim)` -- Single nonzero position for selectivity testing
- `gen_identity_matrix(dim)` -- Identity for invariant testing

### Reference Models (For Model Quality Tests)

| Model | Size | Quantization | Purpose | Storage |
|-------|------|-------------|---------|---------|
| RWKV-7 (1.5B) | ~1.5GB | Q4_K_M | Linear attention validation | Git LFS / download script |
| TinyLlama (1.1B) | ~0.6GB | Q4_0 | Flash attention validation | Git LFS / download script |
| Jamba Mini test | ~2GB | Q4_K_M | Hybrid dispatch validation | Download script |

Models are NOT checked into the repository. A `scripts/download_test_models.sh` script fetches them to `fixtures/models/` with checksum verification. CI caches these between runs.

### Golden Output Files

Checked into `fixtures/golden/` for regression detection:

```
fixtures/golden/
  flash_n256_d64_seed42.bin       # 256*64 f32 values
  linear_n128_d64_chunk32.bin     # 128*64 f32 values
  paged_n64_d64_ps16.bin          # 64*64 f32 values
  rope_n64_d64.bin                # 64*64 f32 values (Q output)
  gqa_n32_d64_h8_kv2.bin          # 8*32*64 f32 values
```

Golden files are regenerated when:
1. A kernel algorithm changes (e.g., tiling strategy, accumulation order)
2. Tolerance is tightened (new golden file validates tighter accuracy)
3. Bug fix that changes numerical output (must be documented)

Golden file updates require an explicit `cargo test --features regen-golden` flag to prevent accidental overwrites.

### CPU FP64 Reference Outputs

Pre-computed reference outputs for large-N configurations where FP64 CPU computation is slow:

```
fixtures/reference/
  flash_n2048_d64_f64.bin    # Pre-computed FP64 reference for N=2048
  flash_n4096_d64_f64.bin    # Pre-computed FP64 reference for N=4096
  linear_n1024_d64_f64.bin   # Pre-computed FP64 reference for N=1024
```

These are regenerated by `cargo test --features regen-reference` and checked in. They enable large-N correctness tests without re-running expensive FP64 CPU computation on every test run.

---

## Quality Gates

### Gate 1: Pre-Merge (Every PR)

All of these must pass before a PR can be merged:

| Check | Tool | Blocking |
|-------|------|----------|
| Compilation (all targets, all features) | `cargo check` | Yes |
| Clippy (zero warnings) | `cargo clippy -- -D warnings` | Yes |
| Formatting | `cargo fmt --check` | Yes |
| Unit tests (CPU) | `cargo test --lib` | Yes |
| GPU correctness tests | `MTL_SHADER_VALIDATION=1 cargo test` | Yes |
| Property-based tests | `cargo test` (proptest) | Yes |
| No new `unsafe` without justification | Manual review | Yes |
| Tolerance table consistency | Automated test | Yes |
| Benchmark regression < 10% | Bencher comparison | Yes |
| No memory leak (if touching GPU code) | Stress test subset | Yes |

### Gate 2: Pre-Release (Version Tag)

Additional checks required before tagging a release:

| Check | Tool | Blocking |
|-------|------|----------|
| Full benchmark suite (50 samples) | `cargo bench` | Yes |
| All stress tests pass | `cargo test --release -- --ignored` | Yes |
| Memory leak tests (all kernels, 1000 iterations) | Stress harness | Yes |
| Model quality: perplexity within tolerance | Quality harness | Yes |
| Token accuracy vs reference engine | Quality harness | Yes |
| End-to-end inference produces coherent output | Smoke test | Yes |
| All supported models load and generate | Integration test | Yes |
| Performance targets met (documented in SYNTHESIS.md) | Benchmark analysis | Yes |
| Cross-device test (at least M1 + M4) | Manual or multi-runner | Yes |

### Gate 3: Architecture-Specific (New Kernel / New Model)

When adding a new kernel or model architecture:

| Check | Requirement |
|-------|-------------|
| FP64 CPU reference implementation | Must exist for the new kernel |
| Tolerance documented with rationale | Entry in tolerance table |
| Benchmark added to criterion suite | At least (N=256, N=1024, N=2048) |
| Threadgroup memory budget verified | Static assertion + runtime test |
| Memory leak test | 100+ iterations with growth < 1% |
| Golden output file | Generated and checked in |
| Shader validation clean | Zero errors with MTL_SHADER_VALIDATION=1 |

### Gate 4: Performance Target Achievement

Tracked across the project lifecycle:

| Milestone | Gate |
|-----------|------|
| Phase A complete | Linear attention tok/s measured, RWKV-7 generates coherent text |
| Phase B complete | Flash attention >1 TFLOPS, Llama 3 generates coherent text |
| Phase C complete | Hybrid >30% faster than llama.cpp on Jamba, Jamba generates coherent text |
| Phase D complete | Burn backend benchmark published, bridge overhead <20us |
| v1.0 release | All performance targets in PRD Section 8 met |

---

## Appendix: Test Execution Quick Reference

```bash
# Run all CPU-only tests (no GPU needed)
cargo test --lib

# Run all tests with shader validation (requires GPU)
MTL_SHADER_VALIDATION=1 cargo test -- --test-threads=1

# Run specific kernel correctness test
MTL_SHADER_VALIDATION=1 cargo test test_flash_correctness -- --test-threads=1

# Run property-based tests
cargo test property -- --test-threads=1

# Run stress/leak tests (slow, ignored by default)
cargo test --release -- --ignored --test-threads=1

# Run all benchmarks
cargo bench

# Run specific benchmark group
cargo bench -- flash_attention

# Quick regression check (2 samples)
cargo bench -- --quick

# Regenerate golden output files
cargo test --features regen-golden -- --test-threads=1

# Regenerate FP64 reference outputs
cargo test --features regen-reference -- --test-threads=1

# Full pre-release validation
MTL_SHADER_VALIDATION=1 cargo test -- --test-threads=1 && \
cargo test --release -- --ignored --test-threads=1 && \
cargo bench
```
