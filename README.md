# metal-attention

Composable attention kernels for Apple Silicon, in Rust.
Swap mechanisms. Keep performance.

```
trait Attention<Q, K, V> -> specialized Metal kernel -> 0% abstraction overhead
```

## What is this?

An inference engine for hybrid AI models (Jamba, Griffin, RWKV, Zamba) on Apple Silicon. Model architectures are type parameters, not code forks. Each combination compiles to zero-overhead specialized Metal kernels.

```rust
let model = HybridModel::<
    LinearAttention,    // 7 of every 8 layers (fast path)
    FlashAttention,     // 1 of every 8 layers (precise path)
    RoPE,               // position encoding
    PagedKVCache,       // KV cache strategy
    7,                  // Jamba's 7:1 SSM:attention ratio
>::load("jamba-1.5-mini.gguf")?;
```

## Why?

Hybrid models mix fast linear/SSM layers with sparse softmax attention layers. They're the future of efficient AI. But no inference engine optimizes the hybrid pattern for Apple Silicon.

| | llama.cpp | MLX | CoreML | **metal-attention** |
|---|---|---|---|---|
| Language | C++ | Python/C++ | Swift (black box) | **Rust** |
| Attention types | Hardcoded | Hardcoded | Hardcoded | **Composable** |
| Add new variant | Fork & rewrite | Fork & rewrite | Can't | **Add trait impl** |
| Hybrid arch support | Bolted on | Partial | Limited | **First-class** |
| simdgroup_matrix | Yes | Yes | Unknown | **Yes** |
| Abstraction overhead | N/A | N/A | N/A | **0% (measured)** |

## Validated, Not Theoretical

Every design decision is backed by prototype benchmarks on Apple M4. Not guessed. Not hallucinated.

**8 prototypes. 48 criterion benchmarks. 34 tests. 58 knowledge base findings.**

### Key results

| Finding | Data |
|---------|------|
| Linear attention vs softmax at N=1024 | **0.12x wall-clock** (linear wins decisively) |
| Function constant dispatch overhead | **0%** runtime, 178ns cache hit |
| CubeCL/wgpu vs hand-written MSL | 58-70% throughput (MSL mandatory) |
| PagedAttention V2 overhead | ~9% vs contiguous (viable) |
| RoPE/ALiBi/GQA overhead | <0.1% of base attention |
| Burn framework integration | Works without forking (~150 lines) |

### Performance baselines (M4, D=64)

| Kernel | N=256 | N=512 | N=1024 | Scaling |
|--------|-------|-------|--------|---------|
| Flash Attention | 389us | 762us | 2.42ms | O(N^2) |
| Linear Attention (GPU) | ~35us | ~35us | ~35us | ~constant |
| PagedAttention V2 | 438us | 1.31ms | 1.72ms | O(N^2) + 9% |

## Architecture

```
trait SequenceBlock (root)
|-- trait LinearSequenceModel    // linear attention, SSMs, RWKV
|   |-- impl LinearAttention    // FLA chunk-based (Proto 6)
|   |-- impl MambaSSM           // selective state space
|   +-- impl RGLRU              // Griffin gated linear recurrence
|
+-- trait SoftmaxAttention       // standard transformers
    |-- impl FlashAttention      // tiled simdgroup_matrix (Proto 1)
    +-- impl PagedAttention      // V2 with block table (Proto 3)
```

Hybrid architectures compose these as const generics:

```rust
HybridModel<R: LinearSequenceModel, A: SoftmaxAttention, const RATIO: usize>
// Griffin 2:1, Jamba 7:1, Nemotron-H ~12:1, Zamba 6:1
```

## Target models

| Model | Architecture | Ratio | Priority |
|-------|-------------|-------|----------|
| RWKV-7 | Pure linear attention | N/A | P0 (validates linear path) |
| Jamba 1.5 | Mamba + Transformer + MoE | 7:1 | P0 (validates hybrid) |
| Griffin/Hawk | RG-LRU + Attention | 2:1 | P1 |
| Zamba | Mamba + shared attention | 6:1 | P1 |
| Llama 3 / Mistral | Pure transformer | N/A | P1 (baseline comparison) |

## Status

**Phase**: Pre-implementation (PRD complete, prototypes validated)

See [PRD.md](PRD.md) for full product requirements and implementation phases.
See [SYNTHESIS.md](SYNTHESIS.md) for detailed prototype results and architecture recommendations.

### Roadmap

| Phase | Focus | Duration |
|-------|-------|----------|
| A | Core traits + linear attention + RWKV-7 | 4-6 weeks |
| B | Flash attention (>1 TFLOPS) + Llama 3 | 3-4 weeks |
| C | Hybrid runtime + Jamba + Griffin | 3-4 weeks |
| D | Burn framework integration | 2-3 weeks |
| E | Optimization + Metal 4 migration | Ongoing |

## Requirements

- Apple Silicon Mac (M1 or later)
- macOS 14+ (Sonoma)
- Rust stable toolchain
- Xcode Command Line Tools (for Metal compiler)

## Related

- [gpu-forge](https://github.com/kavanaghpatrick/gpu-forge) — GPU computing knowledge base (1,635+ verified findings) that informed every design decision
- [Issue #20](https://github.com/kavanaghpatrick/gpu-forge/issues/20) — Original investigation: trait Attention<Q,K,V> on Apple Silicon

## License

MIT OR Apache-2.0
