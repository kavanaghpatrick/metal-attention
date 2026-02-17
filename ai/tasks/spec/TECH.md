# Technical Architecture: metal-attention

> Hybrid Model Inference Engine for Apple Silicon
> Author: Technical Architect Agent
> Date: 2026-02-14
> Status: Specification

---

## Research Findings

### GGUF Format

The [GGUF (GGML Universal File)](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) format is a binary container storing model weights, tokenizer data, and architecture metadata in a single file. Key structural details:

- **Header**: 4-byte magic (`GGUF`), format version (uint32), tensor count (uint64), metadata KV count (uint64). Little-endian by default.
- **Metadata**: Key-value store with typed values (uint8/16/32/64, int8/16/32/64, float32/64, bool, string, arrays). Architecture metadata lives under `general.architecture`, `general.name`, and architecture-specific prefixes (e.g., `llama.attention.head_count`, `mamba.ssm.state_size`).
- **Tensor Info Array**: Each tensor has a name (GGUF string), ndim (uint32), shape (uint64[ndim]), quantization type (enum), and byte offset into the data section.
- **Tensor Data**: Aligned to `general.alignment` (default 32 bytes). Quantized blocks packed contiguously. Data section is designed for direct mmap.
- **Quantization types**: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M, Q6_K, F16, F32, BF16. Block sizes vary (typically 32 elements per block for Q4/Q8).

Hybrid model support in GGUF is emerging. Jamba GGUF files encode Mamba parameters under `jamba.ssm.*` keys and attention parameters under `jamba.attention.*`. RWKV-7 GGUF files use `rwkv.*` prefixes. The layer type (attention vs SSM vs linear) can be inferred from tensor name patterns (`blk.N.attn.*` vs `blk.N.ssm.*` vs `blk.N.channel_mixing.*`).

Sources:
- [GGUF Specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [GGUF File Format Explained](https://apxml.com/courses/practical-llm-quantization/chapter-5-quantization-formats-tooling/gguf-format)
- [GGUF DeepWiki](https://deepwiki.com/ggml-org/llama.cpp/6.1-gguf-file-format)
- [HuggingFace GGUF Documentation](https://huggingface.co/docs/hub/en/gguf)

### Metal Compute Architecture

Apple Silicon GPUs (M1-M4) share key constraints validated by our 8 prototypes:

- **32KB threadgroup memory**: Hard limit across all M-series. Dictates tile/page/chunk sizes for all kernels.
- **simdgroup_matrix**: `simdgroup_float8x8` operations (load, multiply_accumulate, store) provide cooperative 8x8 matrix ops across 32 threads. Available since Apple GPU family 7 (M1+), MSL 3.1.
- **Dynamic Caching (M3+)**: Hardware-managed local memory allocation. On Apple Family 9 GPUs (M3/M4), direct device buffer reads may outperform explicit threadgroup staging.
- **Unified memory**: CPU and GPU share physical memory. `MTLResourceOptions::StorageModeShared` enables zero-copy buffer access. M4 baseline: 120 GB/s bandwidth; M4 Pro: 273 GB/s; M4 Max: 546 GB/s.
- **Dispatch overhead**: ~100-200us per command buffer commit. Multi-head batching into a single dispatch is critical for small workloads.
- **Function constants**: Compile-time specialization with 0% runtime overhead, 34-63us cold compile, 178ns cache hit. Decisively superior to runtime dispatch (39% overhead for `noinline`).

Sources:
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [Metal Performance Best Practices](https://developer.apple.com/videos/play/tech-talks/111373/)
- [Apple GPU Microarchitecture Benchmarks](https://github.com/philipturner/metal-benchmarks)

### Rust ML Framework Patterns

**Burn** (tracel-ai): Backend trait abstraction enabling composable, swappable backends (CUDA, Metal via wgpu, CPU). `Backend` trait is the core abstraction; `Autodiff` decorates any backend with automatic differentiation. CubeCL provides cross-platform kernel compilation using `comptime` specialization. Metal compiler added to wgpu runtime in 2025. However, CubeCL cannot access `simdgroup_matrix` or function constants -- our Proto 5 showed 58-70% throughput vs hand-written MSL.

**Candle** (huggingface): Minimalist ML framework. `Tensor` struct with backend dispatch. `VarBuilder` for loading safetensors/PyTorch/quantized formats. Compiles to single binary, millisecond startup. First-class WASM support. Good reference for `VarBuilder` pattern and quantized tensor loading.

Key lesson: Neither framework provides attention-level parameterization for Metal. Our `AttentionBackend: Backend` supertrait pattern (Proto 8) bridges this gap with 2-17us overhead.

Sources:
- [Burn Framework](https://github.com/tracel-ai/burn)
- [Candle Framework](https://github.com/huggingface/candle)
- [Burn Metal Backend](https://burn.dev/)
- [Candle DeepWiki](https://deepwiki.com/huggingface/candle)

### Target Model Architectures

**RWKV-7 "Goose"**: Pure linear-time, constant-space RNN. Uses Dynamic State Evolution with vector-valued gating and in-context learning rates. No KV cache needed. Recurrent state is a fixed-size matrix (state_size x state_size) per layer, updated per token. Token-shift mechanism with bonus terms and ReLU^2 FFN. Surpasses TC0 expressive power of attention/linear attention.

**Jamba 1.5** (AI21): Hybrid Transformer-Mamba architecture. 72 layers interleaving Mamba-2 blocks with grouped-query attention layers at a 7:1 ratio. 16-expert MoE routing. Mini model: 12B active parameters. Effective 256K context. KV cache only needed for the attention layers (1/8 of total layers).

**Griffin** (Google DeepMind): Hybrid gated linear recurrence + local sliding-window attention. Residual block + MLP block + temporal-mixing block (local MQA or RG-LRU). Matches Llama-2 quality on 6x fewer training tokens. RecurrentGemma is the open-weights implementation. 2:1 recurrence:attention ratio.

**Zamba** (Zyphra): Mamba backbone with shared attention layers. Zamba1: one shared attention every 6 Mamba blocks (6:1 ratio). Zamba2: two shared attention blocks with LoRA projectors for depth-specialization. Mamba2 blocks have ~4x throughput of equivalent transformer blocks. Minimal KV cache (only shared attention layers).

**Mamba SSM**: Selective State Space Model. Key mechanism is the selective scan -- input-dependent state transitions. Hardware-aware implementation uses parallel associative scan, kernel fusion, and gradient recomputation. State update: h_t = A_t * h_{t-1} + B_t * x_t; y_t = C_t * h_t. State size is (d_model x d_state) per layer.

Sources:
- [RWKV-7 Repository](https://github.com/BlinkDL/RWKV-LM)
- [Jamba-1.5 Paper](https://arxiv.org/abs/2408.12570)
- [Griffin Paper](https://arxiv.org/abs/2402.19427)
- [Zamba Architecture](https://www.zyphra.com/post/zamba2-7b)
- [Mamba Paper](https://arxiv.org/abs/2312.00752)
- [RecurrentGemma](https://github.com/google-deepmind/recurrentgemma)

### Tokenizer Strategy

**HuggingFace `tokenizers` crate**: Rust-native, supports BPE, WordPiece, Unigram (SentencePiece). 20s for 1GB text tokenization. Well-maintained, widely used. Loads `tokenizer.json` format. Available on crates.io.

**GGUF-embedded tokenizer**: GGUF files store tokenizer vocabulary and merges in metadata (`tokenizer.ggml.model`, `tokenizer.ggml.tokens`, `tokenizer.ggml.merges`). llama.cpp implements its own BPE/SentencePiece decoder directly from these fields.

**Decision**: Implement a minimal GGUF tokenizer reader for the core path (avoids ~350 crate dependency tree of `tokenizers`). Support `tokenizer.json` sidecar loading via `tokenizers` crate behind an optional feature flag for models that need it.

Sources:
- [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers)
- [rust-tokenizers](https://github.com/guillaume-be/rust-tokenizers)
- [kitoken](https://github.com/Systemcluster/kitoken)

---

## System Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI Binary                               │
│  metal-attention run|bench|info                                  │
│  Argument parsing, streaming output, JSON mode                   │
├─────────────────────────────────────────────────────────────────┤
│                       Inference Engine                            │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │  Text        │  │  Model Runtime   │  │  Sampling        │  │
│  │  Pipeline    │  │  Loop            │  │  Engine          │  │
│  │              │  │                  │  │                  │  │
│  │  tokenize →  │  │  prefill() →     │  │  temperature,    │  │
│  │  prompt ids  │  │  decode() →      │  │  top-p, top-k,   │  │
│  │  → detokenize│  │  next token      │  │  repetition pen  │  │
│  └──────┬───────┘  └────────┬─────────┘  └────────┬─────────┘  │
│         │                   │                      │             │
├─────────┴───────────────────┴──────────────────────┴────────────┤
│                     Model Abstraction Layer                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  HybridModel<L: LinearSequenceModel,                    │    │
│  │              A: SoftmaxAttention,                        │    │
│  │              const RATIO: usize>                         │    │
│  │                                                          │    │
│  │  Layers: Vec<LayerBlock>                                 │    │
│  │  Each LayerBlock: RMSNorm → SequenceBlock → RMSNorm → FFN│    │
│  │  SequenceBlock: Linear (L) or Attention (A) by schedule  │    │
│  └──────────┬──────────────────┬────────────────────────────┘    │
│             │                  │                                  │
├─────────────┴──────────────────┴────────────────────────────────┤
│                   Trait Dispatch Layer                            │
│  ┌────────────────────┐      ┌──────────────────────────┐       │
│  │  LinearSequenceModel│      │  SoftmaxAttention         │       │
│  │                    │      │                          │       │
│  │  - FLALinear       │      │  - FlashAttention        │       │
│  │  - MambaSSM        │      │    (simdgroup_matrix)    │       │
│  │  - RGLRU           │      │  - PagedFlashAttention   │       │
│  │  - RWKV7Block      │      │                          │       │
│  │                    │      │  Variants: RoPE, ALiBi,  │       │
│  │  State: fixed-size │      │  GQA (function constants)│       │
│  │  per layer         │      │  KV Cache: dense / paged │       │
│  └────────┬───────────┘      └──────────┬───────────────┘       │
│           │                             │                        │
├───────────┴─────────────────────────────┴───────────────────────┤
│                   Metal Kernel Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  PsoCache    │  │  BufferPool  │  │  CommandManager      │  │
│  │  (178ns/hit) │  │  (ring alloc)│  │  (triple buffer)     │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
│         │                 │                      │               │
│  ┌──────┴─────────────────┴──────────────────────┴───────────┐  │
│  │  GpuDevice: MTLDevice + MTLCommandQueue + MTLLibrary      │  │
│  │  Hand-written MSL shaders compiled via build.rs           │  │
│  │  Function constants for zero-overhead specialization      │  │
│  └───────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                   Model Loading Layer                            │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │  GGUF Parser     │  │  Weight Mapper   │  │  Tokenizer   │  │
│  │  mmap + header   │  │  tensor name →   │  │  BPE/SPM     │  │
│  │  metadata + info │  │  layer + role    │  │  from GGUF   │  │
│  └──────────────────┘  └──────────────────┘  └──────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                   Apple Silicon Hardware                          │
│  Unified Memory (zero-copy), 32KB threadgroup, MSL 3.1          │
│  M1/M2/M3/M4: simdgroup_matrix, function constants              │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
GGUF File
  │
  ├─ mmap ─────────────────────────────> WeightTensors (zero-copy device buffers)
  ├─ parse metadata ──────────────────> ArchitectureConfig (layer types, dims, ratios)
  └─ parse tokenizer ─────────────────> Tokenizer (BPE vocab + merges)
  │
  v
Model Construction
  │
  ├─ ArchitectureConfig ──> LayerSchedule (which layers are Linear vs Attention)
  ├─ WeightTensors ────────> LayerWeights (bound to Metal buffers per layer)
  └─ PsoCache ─────────────> Compiled kernel variants (lazy, 34-63us each)
  │
  v
Inference Loop
  │
  ├─ Prefill (prompt tokens):
  │    tokens → embed → [for each layer: norm → seq_block → norm → ffn] → logits
  │    Parallel processing of all prompt tokens.
  │    Linear layers: process in chunks, update hidden state.
  │    Attention layers: full Q*K*V over all prompt positions, populate KV cache.
  │
  └─ Decode (generation):
       loop:
         last_token → embed → [for each layer: norm → seq_block → norm → ffn] → logits
         logits → sample → next_token
         Linear layers: single-token state update (O(D^2) per layer).
         Attention layers: single-query against full KV cache (O(N*D) per layer).
         yield next_token
```

---

## Module Breakdown

### Workspace Crate Structure

```
metal-attention/
├── Cargo.toml                    # Workspace root
├── crates/
│   ├── metal-attention/          # Main library crate
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs            # Public API re-exports
│   │       ├── model.rs          # HybridModel<L, A, RATIO>
│   │       ├── inference.rs      # Prefill + decode loop
│   │       ├── sampling.rs       # Temperature, top-p/k, repetition penalty
│   │       └── config.rs         # Runtime configuration
│   │
│   ├── metal-attention-traits/   # Core trait definitions (no Metal dep)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── sequence.rs       # trait SequenceBlock
│   │       ├── linear.rs         # trait LinearSequenceModel
│   │       ├── attention.rs      # trait SoftmaxAttention
│   │       └── types.rs          # Shared types, tensor shapes
│   │
│   ├── metal-attention-kernels/  # Metal GPU layer
│   │   ├── Cargo.toml
│   │   ├── build.rs              # xcrun metal compiler pipeline
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── device.rs         # GpuDevice singleton
│   │   │   ├── pipeline.rs       # PsoCache + PsoKey
│   │   │   ├── buffer.rs         # BufferPool, alloc, readback
│   │   │   ├── command.rs        # CommandManager, triple buffering
│   │   │   ├── dispatch.rs       # Kernel dispatch helpers
│   │   │   ├── flash.rs          # Flash attention dispatch
│   │   │   ├── linear.rs         # FLA chunk_h/chunk_o + prefix sum dispatch
│   │   │   ├── paged.rs          # PagedAttention dispatch
│   │   │   ├── rope.rs           # RoPE kernel dispatch
│   │   │   ├── gqa.rs            # GQA remap dispatch
│   │   │   ├── norm.rs           # RMSNorm kernel dispatch
│   │   │   ├── ffn.rs            # FFN (SwiGLU / ReLU^2) kernel dispatch
│   │   │   ├── embed.rs          # Embedding lookup kernel
│   │   │   ├── matmul.rs         # General matrix multiply kernel
│   │   │   ├── dequant.rs        # Dequantization kernels (Q4_0, Q4_K_M, Q8_0)
│   │   │   └── ssm.rs            # Selective scan / state update kernels
│   │   └── shaders/
│   │       ├── types.h           # Shared MSL types (#repr(C) match)
│   │       ├── flash_attention.metal
│   │       ├── linear_attention.metal
│   │       ├── paged_attention.metal
│   │       ├── paged_reduce.metal
│   │       ├── rope.metal
│   │       ├── gqa_remap.metal
│   │       ├── rmsnorm.metal
│   │       ├── ffn.metal
│   │       ├── embedding.metal
│   │       ├── matmul.metal
│   │       ├── dequantize.metal
│   │       ├── ssm_scan.metal
│   │       └── prefix_sum.metal
│   │
│   ├── metal-attention-gguf/     # GGUF parser + weight mapper
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── parser.rs         # Binary GGUF parser (mmap-based)
│   │       ├── metadata.rs       # KV metadata accessor
│   │       ├── tensor.rs         # Tensor info + data accessor
│   │       ├── quantize.rs       # Quantization type handling
│   │       ├── tokenizer.rs      # GGUF-embedded BPE/SPM tokenizer
│   │       ├── architectures.rs  # Per-model weight mapping tables
│   │       └── detect.rs         # Architecture detection from metadata/tensors
│   │
│   ├── metal-attention-models/   # Concrete model implementations
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── llama.rs          # Llama 3 / Mistral (pure transformer)
│   │       ├── rwkv7.rs          # RWKV-7 (pure linear)
│   │       ├── jamba.rs          # Jamba 1.5 (7:1 Mamba:Attention + MoE)
│   │       ├── griffin.rs        # Griffin / RecurrentGemma (2:1 RG-LRU:Attention)
│   │       ├── zamba.rs          # Zamba (6:1 Mamba:SharedAttention)
│   │       └── registry.rs      # Model name → constructor lookup
│   │
│   └── metal-attention-burn/     # Optional Burn integration (feature-gated)
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs
│           ├── backend.rs        # MetalAttentionBackend<B> newtype
│           ├── bridge.rs         # Burn tensor ↔ Metal buffer bridge
│           └── ops.rs            # Backend trait delegation
│
├── src/
│   └── main.rs                   # CLI binary entry point
│
├── proto/                        # Existing prototype (preserved for reference)
│   ├── Cargo.toml
│   ├── build.rs
│   ├── src/
│   ├── shaders/
│   └── benches/
│
├── benches/                      # Production benchmarks
│   ├── attention.rs
│   ├── inference.rs
│   └── model_load.rs
│
└── tests/
    ├── correctness.rs            # Kernel output vs CPU reference
    ├── model_load.rs             # GGUF loading round-trip
    └── e2e.rs                    # End-to-end text generation
```

### Module Dependency Graph

```
metal-attention-cli (binary)
  └─> metal-attention (lib)
        ├─> metal-attention-traits
        ├─> metal-attention-kernels
        │     └─> metal-attention-traits
        ├─> metal-attention-gguf
        └─> metal-attention-models
              ├─> metal-attention-traits
              ├─> metal-attention-kernels
              └─> metal-attention-gguf

metal-attention-burn (optional)
  ├─> metal-attention-traits
  ├─> metal-attention-kernels
  └─> burn
```

The key boundary: `metal-attention-traits` has zero Metal dependencies. It defines the abstract interface. `metal-attention-kernels` is the only crate that links Metal frameworks. Model implementations in `metal-attention-models` use the trait abstractions and call into `metal-attention-kernels` for GPU dispatch.

---

## Trait Hierarchy

### Core Traits

```rust
// metal-attention-traits/src/sequence.rs

/// Tensor descriptor for inference. Does not own data -- references Metal buffers.
#[derive(Debug, Clone)]
pub struct TensorView {
    /// Byte offset into the backing Metal buffer
    pub offset: usize,
    /// Shape in elements (e.g., [seq_len, head_dim])
    pub shape: Vec<usize>,
    /// Element stride per dimension
    pub strides: Vec<usize>,
    /// Element data type
    pub dtype: DType,
}

/// Data types supported by the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_K_M,
    Q8_0,
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

/// Root trait for any sequence-processing block.
///
/// A SequenceBlock processes a sequence of token embeddings, producing
/// a transformed sequence of the same shape. It may maintain internal
/// state (e.g., recurrent hidden state, KV cache).
pub trait SequenceBlock {
    /// Per-layer persistent state type (KV cache for attention, hidden state for SSM).
    type State;

    /// Initialize empty state for a new sequence.
    fn init_state(&self, config: &BlockConfig) -> Self::State;

    /// Process a full sequence (prefill). Updates state in place.
    /// Input shape: [seq_len, hidden_size]
    /// Output shape: [seq_len, hidden_size]
    fn forward_prefill(
        &self,
        input: &TensorView,
        state: &mut Self::State,
        config: &BlockConfig,
    ) -> TensorView;

    /// Process a single token (decode). Updates state in place.
    /// Input shape: [1, hidden_size]
    /// Output shape: [1, hidden_size]
    fn forward_decode(
        &self,
        input: &TensorView,
        state: &mut Self::State,
        config: &BlockConfig,
    ) -> TensorView;

    /// Memory footprint of this block's state in bytes.
    fn state_size_bytes(&self, config: &BlockConfig) -> usize;
}
```

```rust
// metal-attention-traits/src/linear.rs

/// Position encoding configuration for linear models (most use none).
#[derive(Debug, Clone, Copy)]
pub enum LinearPositionEncoding {
    None,
    TokenShift,  // RWKV-style token shift
}

/// A sequence block with O(N) or O(1) per-token processing.
///
/// LinearSequenceModels maintain a fixed-size hidden state that is updated
/// per token (or per chunk during prefill). No KV cache is needed --
/// the entire context is compressed into the state matrix.
///
/// Examples: FLA linear attention, Mamba SSM, RG-LRU, RWKV-7 blocks.
pub trait LinearSequenceModel: SequenceBlock {
    /// Fixed-size recurrent state (e.g., D x D matrix for FLA, d_model x d_state for Mamba).
    /// This is Self::State from SequenceBlock.

    /// Process input in chunks during prefill (parallel over chunks).
    /// chunk_size is selected based on head_dim and 32KB threadgroup memory budget.
    fn prefill_chunked(
        &self,
        input: &TensorView,
        state: &mut Self::State,
        config: &BlockConfig,
        chunk_size: usize,
    ) -> TensorView;

    /// Single-token recurrent update during decode. O(D^2) or O(d_model * d_state).
    fn decode_step(
        &self,
        input: &TensorView,
        state: &mut Self::State,
        config: &BlockConfig,
    ) -> TensorView;

    /// Maximum supported head dimension for this implementation.
    fn max_head_dim(&self) -> usize;

    /// Optimal chunk size for the given head dimension (respecting 32KB threadgroup limit).
    fn optimal_chunk_size(&self, head_dim: usize) -> usize;
}
```

```rust
// metal-attention-traits/src/attention.rs

/// Position encoding variant for softmax attention.
#[derive(Debug, Clone, Copy)]
pub enum PositionEncoding {
    None,
    RoPE { theta_base: f32 },
    ALiBi,
}

/// KV cache strategy.
#[derive(Debug, Clone, Copy)]
pub enum KVCacheMode {
    /// Contiguous buffer, simple indexing.
    Dense,
    /// PagedAttention V2 with block table indirection.
    Paged { page_size: u32 },
}

/// GQA configuration.
#[derive(Debug, Clone, Copy)]
pub struct GQAConfig {
    /// Number of Q heads per KV head group. 1 = MHA, num_heads = MQA.
    pub group_size: usize,
}

/// A softmax attention block with O(N^2) compute and KV cache.
///
/// Processes sequences using scaled dot-product attention with
/// Flash Attention kernel implementation (tiled, online softmax,
/// simdgroup_matrix).
///
/// Examples: standard multi-head attention, GQA, MQA.
pub trait SoftmaxAttention: SequenceBlock {
    /// KV cache state. This is Self::State from SequenceBlock.

    /// Current sequence length in the KV cache.
    fn cached_length(&self, state: &Self::State) -> usize;

    /// Maximum sequence length this cache can hold.
    fn max_length(&self, state: &Self::State) -> usize;

    /// Position encoding used by this attention implementation.
    fn position_encoding(&self) -> PositionEncoding;

    /// KV cache mode (dense or paged).
    fn cache_mode(&self) -> KVCacheMode;

    /// GQA configuration.
    fn gqa_config(&self) -> GQAConfig;

    /// Prefill: compute attention over full prompt and populate KV cache.
    /// Uses Flash Attention kernel for O(N^2) attention with tiling.
    fn prefill_attention(
        &self,
        q: &TensorView,
        k: &TensorView,
        v: &TensorView,
        state: &mut Self::State,
        config: &BlockConfig,
    ) -> TensorView;

    /// Decode: compute single-query attention against full KV cache.
    /// Appends new K/V to cache, computes attention over all cached positions.
    fn decode_attention(
        &self,
        q: &TensorView,
        k: &TensorView,
        v: &TensorView,
        state: &mut Self::State,
        config: &BlockConfig,
    ) -> TensorView;
}
```

### Hybrid Model Composition

```rust
// metal-attention/src/model.rs

/// Layer type in a hybrid model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    Linear,
    Attention,
}

/// Schedule determining which layers use Linear vs Attention.
pub struct LayerSchedule {
    /// Layer types in order. Length = total_layers.
    pub types: Vec<LayerType>,
}

impl LayerSchedule {
    /// Create a periodic schedule: every (RATIO+1)-th layer is Attention,
    /// the rest are Linear. E.g., RATIO=7 means layers 0-6 are Linear,
    /// layer 7 is Attention, layers 8-14 are Linear, layer 15 is Attention, etc.
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
}

/// A hybrid model composing linear and attention sequence blocks.
///
/// The generic parameters determine the concrete block implementations
/// and the interleaving ratio. Each combination compiles to specialized
/// Metal kernels via function constants -- zero runtime dispatch overhead.
pub struct HybridModel<L: LinearSequenceModel, A: SoftmaxAttention> {
    /// Layer schedule (which layers are Linear vs Attention).
    pub schedule: LayerSchedule,

    /// Linear sequence model implementation (shared across all linear layers).
    pub linear_impl: L,

    /// Softmax attention implementation (shared across all attention layers).
    pub attention_impl: A,

    /// Per-layer weights (embedding projections, norms, FFN weights).
    pub layers: Vec<LayerWeights>,

    /// Per-layer state (recurrent states for linear, KV caches for attention).
    pub states: Vec<LayerState<L::State, A::State>>,

    /// Token embedding table.
    pub embedding: TensorView,

    /// Output projection (typically tied to embedding).
    pub output_proj: TensorView,

    /// RMSNorm weight for final normalization.
    pub final_norm: TensorView,

    /// Architecture configuration.
    pub config: ModelConfig,
}

/// Per-layer state, polymorphic over layer type.
pub enum LayerState<LS, AS> {
    Linear(LS),
    Attention(AS),
}

/// Full model configuration parsed from GGUF metadata.
pub struct ModelConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub intermediate_size: usize,  // FFN hidden size
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_seq_len: usize,
    pub architecture: String,      // e.g., "jamba", "rwkv", "griffin"

    // SSM-specific (optional)
    pub ssm_state_size: Option<usize>,
    pub ssm_conv_size: Option<usize>,

    // MoE-specific (optional)
    pub num_experts: Option<usize>,
    pub num_active_experts: Option<usize>,
}
```

---

## Metal Kernel Architecture

### Shader Organization

All Metal shaders live in `crates/metal-attention-kernels/shaders/`. The `build.rs` script compiles them via `xcrun metal` (-std=metal3.1) into `.air` files and links them into a single `shaders.metallib`.

**Kernel categories**:

| Category | Shaders | Function Constants |
|----------|---------|-------------------|
| Attention | `flash_attention.metal` | HEAD_DIM, BLOCK_R, BLOCK_C, ALIBI_ENABLED |
| Linear Attention | `linear_attention.metal` | HEAD_DIM, CHUNK_SIZE |
| Paged Attention | `paged_attention.metal`, `paged_reduce.metal` | HEAD_DIM, PAGE_SIZE |
| Position Encoding | `rope.metal` | (none -- always applied standalone) |
| GQA | `gqa_remap.metal` | (runtime params -- pure memory copy) |
| Normalization | `rmsnorm.metal` | HIDDEN_SIZE |
| FFN | `ffn.metal` | HIDDEN_SIZE, INTERMEDIATE_SIZE, FFN_TYPE (SwiGLU/ReLU2/GeGLU) |
| Embedding | `embedding.metal` | HIDDEN_SIZE, VOCAB_SIZE |
| Matrix Multiply | `matmul.metal` | M, N, K tile sizes |
| Dequantization | `dequantize.metal` | QUANT_TYPE (Q4_0/Q4_K_M/Q8_0), BLOCK_SIZE |
| SSM | `ssm_scan.metal` | STATE_SIZE, D_MODEL |
| Utility | `prefix_sum.metal` | ELEMENT_SIZE (for D*D matrix prefix sum) |

### Shared Header (`types.h`)

The `types.h` header defines `#repr(C)` structs shared between Rust host code and MSL. The existing `AttentionParams` (64 bytes, 4-byte aligned) will be extended with additional parameter structs:

```c
// types.h (extended)

struct AttentionParams {
    uint seq_len;
    uint head_dim;
    uint num_heads;
    uint num_kv_heads;
    uint block_r;
    uint block_c;
    float scale;
    uint variant;
    uint page_size;
    uint num_pages;
    uint max_context_len;
    uint num_partitions;
    uint _pad0;
    uint _pad1;
    uint _pad2;
    uint _pad3;
};
// 64 bytes, 4-byte aligned

struct LayerParams {
    uint hidden_size;
    uint intermediate_size;
    uint seq_len;
    uint batch_size;           // always 1 for this engine
    float rms_norm_eps;
    uint ffn_type;             // 0=SwiGLU, 1=ReLU2, 2=GeGLU
    uint quant_type;           // 0=F32, 1=F16, 2=Q4_0, 3=Q4_K_M, 4=Q8_0
    uint quant_block_size;     // typically 32
    uint _pad0;
    uint _pad1;
    uint _pad2;
    uint _pad3;
    uint _pad4;
    uint _pad5;
    uint _pad6;
    uint _pad7;
};
// 64 bytes, 4-byte aligned

struct SSMParams {
    uint d_model;
    uint d_state;
    uint d_conv;
    uint seq_len;
    uint expand;               // expansion factor (typically 2)
    uint _pad0;
    uint _pad1;
    uint _pad2;
};
// 32 bytes, 4-byte aligned
```

### PsoCache Architecture

The PSO cache is the central dispatch mechanism. Every kernel dispatch goes through the cache:

```rust
// metal-attention-kernels/src/pipeline.rs (evolved from proto)

pub struct PsoCache {
    cache: HashMap<PsoKey, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
}

impl PsoCache {
    /// Get or compile a PSO. Cache hit: 178ns. Cache miss: 34-63us.
    pub fn get_or_compile(&mut self, key: &PsoKey)
        -> &ProtocolObject<dyn MTLComputePipelineState>;

    /// Pre-warm the cache with all kernel variants needed by a model config.
    /// Called once at model load time. Total time: ~100ms for typical models.
    pub fn prewarm(&mut self, config: &ModelConfig) {
        // Flash attention variants: per (head_dim, block_r, block_c, alibi)
        // Linear attention variants: per (head_dim, chunk_size)
        // Paged attention variants: per (head_dim, page_size)
        // FFN variants: per (hidden_size, intermediate_size, ffn_type)
        // Norm variants: per (hidden_size)
        // Dequant variants: per (quant_type, block_size)
    }
}
```

**PsoKey composition** for each kernel:

| Kernel | Function Constants in PsoKey |
|--------|------------------------------|
| `flash_attention` | `HEAD_DIM:u32`, `BLOCK_R:u32`, `BLOCK_C:u32`, `ALIBI_ENABLED:bool` |
| `chunk_h` | `HEAD_DIM:u32`, `CHUNK_SIZE:u32` |
| `chunk_o` | `HEAD_DIM:u32`, `CHUNK_SIZE:u32` |
| `paged_attention_partition` | `HEAD_DIM:u32`, `PAGE_SIZE:u32` |
| `paged_attention_reduce` | `HEAD_DIM:u32`, `PAGE_SIZE:u32` |
| `rmsnorm` | `HIDDEN_SIZE:u32` |
| `silu_mul` (SwiGLU) | `SIZE:u32` |
| `dequantize_q4_0` | `BLOCK_SIZE:u32` |

### Kernel Fusion Strategy

Function constants enable compile-time kernel fusion without runtime cost:

1. **ALiBi fusion**: `ALIBI_ENABLED` bool in flash_attention. When false, the compiler dead-code-eliminates the bias computation (validated: 0% overhead).
2. **Causal mask fusion**: `CAUSAL_MASK` bool in flash_attention. Skips upper-triangular attention scores.
3. **In-kernel dequantization**: For quantized KV caches, the attention kernel dequantizes K/V inline during tile loading, avoiding a separate dequant pass.
4. **Fused RMSNorm + projection**: A single kernel can compute norm(x) * W, avoiding one global memory round-trip.

Kernels that are NOT fused (keeping them separate for flexibility):
- RoPE: Always standalone (10us/head, negligible).
- GQA remap: Pure memory copy, no compute to fuse with.
- Embedding lookup: One-time per token, trivial.

---

## Model Loading

### GGUF Parser Design

```rust
// metal-attention-gguf/src/parser.rs

use std::fs::File;
use std::io;
use memmap2::Mmap;

/// A parsed GGUF file providing zero-copy access to metadata and tensor data.
pub struct GgufFile {
    /// Memory-mapped file contents.
    mmap: Mmap,
    /// Parsed header.
    header: GgufHeader,
    /// Metadata key-value pairs (references into mmap).
    metadata: Vec<GgufMetadataKV>,
    /// Tensor descriptors (references into mmap).
    tensor_infos: Vec<GgufTensorInfo>,
    /// Byte offset where tensor data begins.
    data_offset: usize,
}

pub struct GgufHeader {
    pub magic: [u8; 4],         // "GGUF"
    pub version: u32,           // 3
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

pub struct GgufTensorInfo {
    pub name: String,
    pub n_dims: u32,
    pub dimensions: Vec<u64>,
    pub type_id: GgufType,      // Quantization type enum
    pub offset: u64,            // Relative to data section start
}

impl GgufFile {
    /// Open and parse a GGUF file using mmap for zero-copy access.
    pub fn open(path: &std::path::Path) -> io::Result<Self>;

    /// Get a metadata value by key. Returns None if not found.
    pub fn get_metadata(&self, key: &str) -> Option<&GgufMetadataValue>;

    /// Get a string metadata value.
    pub fn get_string(&self, key: &str) -> Option<&str>;

    /// Get a u32 metadata value.
    pub fn get_u32(&self, key: &str) -> Option<u32>;

    /// Get a f32 metadata value.
    pub fn get_f32(&self, key: &str) -> Option<f32>;

    /// Look up a tensor by name. Returns its info and a byte slice into the data.
    pub fn get_tensor(&self, name: &str) -> Option<(&GgufTensorInfo, &[u8])>;

    /// Iterate over all tensor descriptors.
    pub fn tensors(&self) -> impl Iterator<Item = &GgufTensorInfo>;

    /// Total file size in bytes.
    pub fn file_size(&self) -> usize;

    /// Get the raw byte slice for a tensor's data (for direct Metal buffer creation).
    pub fn tensor_data(&self, info: &GgufTensorInfo) -> &[u8];
}
```

### Architecture Detection

```rust
// metal-attention-gguf/src/detect.rs

/// Detected model architecture with layer type information.
pub struct DetectedArchitecture {
    pub name: String,                    // "llama", "jamba", "rwkv", "griffin", "zamba"
    pub total_layers: usize,
    pub layer_schedule: LayerSchedule,   // Which layers are Linear vs Attention
    pub config: ModelConfig,
}

/// Detect architecture from GGUF metadata and tensor names.
pub fn detect_architecture(gguf: &GgufFile) -> Result<DetectedArchitecture, String> {
    // 1. Check general.architecture metadata key
    let arch = gguf.get_string("general.architecture")
        .unwrap_or("unknown");

    match arch {
        "llama" | "mistral" => detect_transformer(gguf),
        "jamba" => detect_jamba(gguf),
        "rwkv" => detect_rwkv(gguf),
        "griffin" | "recurrentgemma" => detect_griffin(gguf),
        "zamba" => detect_zamba(gguf),
        "mamba" => detect_mamba(gguf),
        _ => {
            // Fallback: infer from tensor name patterns
            detect_from_tensors(gguf)
        }
    }
}
```

### Weight Mapping

Each model architecture defines a mapping from GGUF tensor names to layer roles:

```rust
// metal-attention-gguf/src/architectures.rs

/// Role of a weight tensor within a layer.
pub enum WeightRole {
    // Attention weights
    QueryProj,      // blk.N.attn_q.weight
    KeyProj,        // blk.N.attn_k.weight
    ValueProj,      // blk.N.attn_v.weight
    OutputProj,     // blk.N.attn_output.weight

    // FFN weights
    GateProj,       // blk.N.ffn_gate.weight
    UpProj,         // blk.N.ffn_up.weight
    DownProj,       // blk.N.ffn_down.weight

    // Normalization
    AttnNorm,       // blk.N.attn_norm.weight
    FFNNorm,        // blk.N.ffn_norm.weight

    // SSM/Linear weights
    SSMIn,          // blk.N.ssm_in.weight
    SSMOut,         // blk.N.ssm_out.weight
    SSMConv1d,      // blk.N.ssm_conv1d.weight
    SSMA,           // blk.N.ssm_a.weight (state matrix)
    SSMB,           // blk.N.ssm_b.weight
    SSMC,           // blk.N.ssm_c.weight
    SSMD,           // blk.N.ssm_d.weight (skip connection)
    SSMDt,          // blk.N.ssm_dt.weight (time step)

    // RWKV-specific
    TimeMix,        // blk.N.time_mix_*
    ChannelMix,     // blk.N.channel_mix_*

    // Global
    TokenEmbedding, // token_embd.weight
    OutputWeight,   // output.weight (may be tied to embedding)
    OutputNorm,     // output_norm.weight
}

/// Maps a GGUF tensor name to (layer_index, weight_role).
pub fn map_tensor_name(name: &str, arch: &str) -> Option<(usize, WeightRole)>;
```

### Memory Layout for Weight Tensors

Weights are loaded as Metal buffers using the mmap'd GGUF data directly when possible:

```rust
// metal-attention-kernels/src/buffer.rs

/// Create a Metal buffer backed by the mmap'd GGUF tensor data.
/// For StorageModeShared, this is effectively zero-copy on Apple Silicon
/// (unified memory means the GPU reads from the same physical pages).
pub fn create_weight_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    tensor_data: &[u8],
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    // Use newBufferWithBytesNoCopy when alignment permits (32-byte aligned)
    // Fall back to newBufferWithBytes for unaligned data
    if tensor_data.as_ptr() as usize % 32 == 0 {
        // Zero-copy: Metal references the mmap'd pages directly
        unsafe {
            device.newBufferWithBytesNoCopy_length_options_deallocator(
                NonNull::new(tensor_data.as_ptr() as *mut _).unwrap(),
                tensor_data.len(),
                MTLResourceOptions::StorageModeShared,
                None,  // no deallocator -- mmap outlives the buffer
            ).expect("Failed to create no-copy buffer")
        }
    } else {
        // Copy: Metal allocates and copies (rare for GGUF-aligned data)
        alloc_buffer_with_data(device, tensor_data)
    }
}
```

---

## Inference Pipeline

### Prefill Phase

```rust
// metal-attention/src/inference.rs

impl<L: LinearSequenceModel, A: SoftmaxAttention> HybridModel<L, A> {
    /// Process all prompt tokens in parallel.
    /// Returns logits for the last prompt position (used for first generated token).
    pub fn prefill(&mut self, token_ids: &[u32]) -> Vec<f32> {
        let seq_len = token_ids.len();

        // 1. Embedding lookup: token_ids → [seq_len, hidden_size]
        let hidden = self.embed(token_ids);

        // 2. Process each layer
        let mut x = hidden;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Pre-attention/SSM normalization
            let normed = self.rmsnorm(&x, &layer.attn_norm);

            // Sequence block (linear or attention)
            let seq_out = match self.schedule.types[layer_idx] {
                LayerType::Linear => {
                    let state = self.states[layer_idx].as_linear_mut();
                    let projected = self.linear_proj(&normed, layer);
                    self.linear_impl.prefill_chunked(
                        &projected, state, &layer.block_config,
                        self.linear_impl.optimal_chunk_size(layer.block_config.head_dim),
                    )
                }
                LayerType::Attention => {
                    let state = self.states[layer_idx].as_attention_mut();
                    let (q, k, v) = self.qkv_proj(&normed, layer);
                    // Apply RoPE if configured
                    let (q, k) = self.apply_position_encoding(q, k, 0, seq_len);
                    // Apply GQA remap if needed
                    let k = self.maybe_gqa_remap(&k, layer);
                    let v = self.maybe_gqa_remap(&v, layer);
                    self.attention_impl.prefill_attention(
                        &q, &k, &v, state, &layer.block_config,
                    )
                }
            };

            // Output projection + residual
            let projected = self.out_proj(&seq_out, layer);
            x = self.residual_add(&x, &projected);

            // Post-FFN normalization
            let normed = self.rmsnorm(&x, &layer.ffn_norm);

            // FFN (SwiGLU / ReLU^2 / GeGLU depending on architecture)
            let ffn_out = self.ffn(&normed, layer);
            x = self.residual_add(&x, &ffn_out);
        }

        // 3. Final norm + output projection → logits
        let normed = self.rmsnorm(&x, &self.final_norm);
        self.compute_logits(&normed, seq_len - 1)  // logits for last position only
    }
}
```

### Decode Phase

```rust
impl<L: LinearSequenceModel, A: SoftmaxAttention> HybridModel<L, A> {
    /// Generate next token given the previous token.
    /// Processes a single token through all layers.
    /// Returns logits for sampling.
    pub fn decode(&mut self, token_id: u32, position: usize) -> Vec<f32> {
        // 1. Embed single token: [1, hidden_size]
        let hidden = self.embed(&[token_id]);

        // 2. Process each layer (single-token path)
        let mut x = hidden;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let normed = self.rmsnorm(&x, &layer.attn_norm);

            let seq_out = match self.schedule.types[layer_idx] {
                LayerType::Linear => {
                    // O(D^2) per layer -- fast path
                    let state = self.states[layer_idx].as_linear_mut();
                    let projected = self.linear_proj(&normed, layer);
                    self.linear_impl.decode_step(
                        &projected, state, &layer.block_config,
                    )
                }
                LayerType::Attention => {
                    // O(N*D) per layer -- scales with context length
                    let state = self.states[layer_idx].as_attention_mut();
                    let (q, k, v) = self.qkv_proj(&normed, layer);
                    let (q, k) = self.apply_position_encoding(q, k, position, 1);
                    let k = self.maybe_gqa_remap(&k, layer);
                    let v = self.maybe_gqa_remap(&v, layer);
                    self.attention_impl.decode_attention(
                        &q, &k, &v, state, &layer.block_config,
                    )
                }
            };

            let projected = self.out_proj(&seq_out, layer);
            x = self.residual_add(&x, &projected);

            let normed = self.rmsnorm(&x, &layer.ffn_norm);
            let ffn_out = self.ffn(&normed, layer);
            x = self.residual_add(&x, &ffn_out);
        }

        // 3. Final norm + logits
        let normed = self.rmsnorm(&x, &self.final_norm);
        self.compute_logits(&normed, 0)
    }
}
```

### Sampling Engine

```rust
// metal-attention/src/sampling.rs

/// Sampling parameters for text generation.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,           // 0.0 = greedy, 1.0 = default
    pub top_p: f32,                 // nucleus sampling threshold
    pub top_k: usize,              // 0 = disabled
    pub repetition_penalty: f32,   // 1.0 = no penalty
    pub max_tokens: usize,         // maximum generation length
    pub stop_tokens: Vec<u32>,     // EOS token IDs
    pub seed: Option<u64>,         // reproducible sampling
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repetition_penalty: 1.1,
            max_tokens: 2048,
            stop_tokens: vec![],
            seed: None,
        }
    }
}

/// Sample a token from logits using the configured strategy.
pub fn sample(logits: &[f32], params: &SamplingParams, past_tokens: &[u32]) -> u32 {
    let mut logits = logits.to_vec();

    // 1. Apply repetition penalty
    if params.repetition_penalty != 1.0 {
        for &token in past_tokens {
            if (token as usize) < logits.len() {
                let score = logits[token as usize];
                logits[token as usize] = if score > 0.0 {
                    score / params.repetition_penalty
                } else {
                    score * params.repetition_penalty
                };
            }
        }
    }

    // 2. Temperature scaling
    if params.temperature == 0.0 {
        // Greedy: return argmax
        return logits.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap().0 as u32;
    }

    for logit in &mut logits {
        *logit /= params.temperature;
    }

    // 3. Top-K filtering
    if params.top_k > 0 {
        top_k_filter(&mut logits, params.top_k);
    }

    // 4. Top-P (nucleus) filtering
    if params.top_p < 1.0 {
        top_p_filter(&mut logits, params.top_p);
    }

    // 5. Softmax + categorical sample
    softmax_sample(&logits, params.seed)
}
```

### KV Cache Management

```rust
// metal-attention-kernels/src/kv_cache.rs

/// Dense KV cache: contiguous buffer, simple append.
pub struct DenseKVCache {
    /// K buffer: [max_seq_len, num_kv_heads, head_dim]
    k_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// V buffer: [max_seq_len, num_kv_heads, head_dim]
    v_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Current number of cached tokens.
    len: usize,
    /// Maximum capacity.
    max_len: usize,
}

/// Paged KV cache: block table indirection, fragmentation-free.
pub struct PagedKVCache {
    /// KV page pool: [num_pages, 2, page_size, head_dim]
    page_pool: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Block table: [max_pages_per_seq] mapping logical page → physical page.
    block_table: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Free page list (host-side management).
    free_pages: Vec<u32>,
    /// Number of pages currently allocated for this sequence.
    allocated_pages: usize,
    /// Current token count.
    len: usize,
    /// Page size (tokens per page).
    page_size: usize,
}
```

---

## Memory Management

### Unified Memory Strategy

Apple Silicon's unified memory eliminates the CPU-GPU transfer bottleneck that dominates discrete GPU inference. The memory strategy:

1. **Model weights**: mmap the GGUF file. Create Metal buffers with `newBufferWithBytesNoCopy` when alignment permits (zero additional memory for weights). The GPU reads directly from the mmap'd pages. This is why model load time can be <5s even for 7B models.

2. **Activation buffers**: Allocate with `StorageModeShared`. Reuse across layers via a ring allocator (each layer's output becomes the next layer's input). Peak activation memory: `2 * seq_len * hidden_size * sizeof(f32)` (double-buffered for residual connection).

3. **KV caches**: Largest runtime allocation. Dense mode: `2 * max_seq_len * num_attn_layers * num_kv_heads * head_dim * sizeof(f16)`. For Jamba 1.5 Mini (12B, 9 attention layers out of 72, GQA with 8 KV heads, D=128, max 4096 tokens): `2 * 4096 * 9 * 8 * 128 * 2 = ~144 MB`. Compare to pure transformer (72 attention layers): `2 * 4096 * 72 * 8 * 128 * 2 = ~1.15 GB`. The 8:1 hybrid ratio yields 8x KV cache savings.

4. **Recurrent states**: Fixed-size per linear layer. For FLA (D x D matrix): `64 * 64 * 4 = 16 KB` per layer. For Mamba (d_model x d_state): `4096 * 16 * 4 = 256 KB` per layer. Total for 63 linear layers in Jamba: 63 * 256 KB = ~16 MB. Negligible compared to KV cache.

### Buffer Pool

```rust
// metal-attention-kernels/src/buffer.rs

/// A pool of reusable Metal buffers to avoid repeated allocation.
///
/// Buffers are keyed by size class (rounded up to power-of-2).
/// When a buffer is returned to the pool, it becomes available for
/// the next request of the same or smaller size.
pub struct BufferPool {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    /// Available buffers keyed by size class.
    free_buffers: HashMap<usize, Vec<Retained<ProtocolObject<dyn MTLBuffer>>>>,
    /// Total bytes currently allocated (for tracking).
    total_allocated: usize,
}

impl BufferPool {
    /// Acquire a buffer of at least `size` bytes.
    /// Returns a pooled buffer if available, otherwise allocates a new one.
    pub fn acquire(&mut self, size: usize) -> Retained<ProtocolObject<dyn MTLBuffer>>;

    /// Return a buffer to the pool for reuse.
    pub fn release(&mut self, buffer: Retained<ProtocolObject<dyn MTLBuffer>>);

    /// Current total memory allocated by this pool.
    pub fn total_allocated_bytes(&self) -> usize;

    /// Release all pooled buffers back to the system.
    pub fn drain(&mut self);
}
```

### Triple Buffering

Triple buffering allows CPU work (weight loading, token processing) to overlap with GPU work (kernel execution):

```rust
// metal-attention-kernels/src/command.rs

use std::sync::Arc;

/// Manages command buffer submission with triple buffering.
///
/// Uses dispatch_semaphore(3) to allow up to 3 command buffers in flight.
/// While the GPU executes frame N, the CPU prepares frame N+1 and N+2.
pub struct CommandManager {
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    /// Semaphore limiting in-flight command buffers to 3.
    in_flight_semaphore: dispatch::Semaphore,
    /// Current frame index (0, 1, 2, cycling).
    frame_index: usize,
    /// Per-frame activation buffers (triple-buffered).
    frame_buffers: [FrameResources; 3],
}

struct FrameResources {
    /// Scratch buffer for this frame's activations.
    activation_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Command buffer for this frame (set when encoding, cleared on completion).
    command_buffer: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
}

impl CommandManager {
    /// Begin encoding a new frame. Blocks if all 3 frames are in flight.
    pub fn begin_frame(&mut self) -> &ProtocolObject<dyn MTLComputeCommandEncoder>;

    /// Commit the current frame's command buffer and advance frame index.
    pub fn end_frame(&mut self);

    /// Wait for all in-flight frames to complete.
    pub fn drain(&mut self);
}
```

---

## Performance Architecture

### Dispatch Strategy

The dispatch strategy is designed around the empirically validated constraints from the 8 prototypes:

**Principle**: All kernel variant selection happens at PSO compile time via function constants. Zero runtime branching in GPU kernel inner loops.

```
Host-side dispatch flow (per layer, per token):

1. Determine layer type from LayerSchedule → LayerType::Linear or LayerType::Attention
2. Construct PsoKey from (kernel_name, head_dim, tile_sizes, variant_flags)
3. PsoCache::get_or_compile(key) → 178ns cache hit (after first call)
4. Bind buffers + params → set on compute encoder
5. dispatchThreadgroups → GPU executes specialized kernel

No branches on GPU. Function constants are baked into the compiled kernel code.
The Metal compiler dead-code-eliminates unused paths (validated: ALiBi OFF = 0% overhead).
```

### Kernel Dispatch Patterns

| Phase | Kernel Sequence | Dispatch Shape |
|-------|----------------|----------------|
| **Prefill (attention layer)** | RoPE → FlashAttention → OutProj | (seq_len/Br, num_heads, 1) |
| **Prefill (linear layer)** | chunk_h → prefix_sum → chunk_o → OutProj | (num_chunks, 1, 1) per pass |
| **Decode (attention layer)** | RoPE → DecodeAttention → OutProj | (1, num_heads, 1) |
| **Decode (linear layer)** | state_update → OutProj | (1, 1, 1) |
| **Every layer** | RMSNorm → [above] → RMSNorm → FFN | (seq_len, 1, 1) |

### Kernel Fusion Opportunities

Priority-ordered fusion plan:

1. **GPU prefix sum** (Phase A): Eliminate ~300us CPU bottleneck in linear attention. Replace CPU readback + prefix sum + re-upload with a single Metal kernel over D*D matrices. Expected savings: 250-350us per linear attention dispatch.

2. **Multi-simdgroup flash attention** (Phase B): Scale from 1 simdgroup (0.16 TFLOPS) to 4+ simdgroups per threadgroup. Use simdgroup_matrix across multiple 8x8 tiles concurrently. Target: >1 TFLOPS, matching Metal Flash Attention (MFA) reference.

3. **Fused QKV projection** (Phase C): Single matmul kernel computing Q, K, V projections from the normed hidden state. Saves 2 global memory round-trips per attention layer.

4. **Fused RMSNorm + linear projection** (Phase E): Compute `norm(x) * W` in a single kernel. Saves 1 global memory round-trip per layer.

5. **In-kernel dequantization** (Phase E): Dequantize Q4/Q8 weights inside matmul kernels, avoiding a separate dequant pass. Key for memory-bandwidth-bound decode.

### Async Compute

For decode, where single-token processing is bottlenecked by memory bandwidth rather than compute, async operations provide latency hiding:

```
Frame N:     GPU executing layer 0-3 kernels
Frame N+1:   CPU encoding layer 4-7 commands
Frame N+2:   (available for further pipelining)

The triple-buffered CommandManager ensures the GPU is never idle
waiting for CPU command encoding.
```

For prefill, all layers are submitted as a single command buffer (sequential within-buffer execution is fine since layers are dependent). Metal's implicit barrier between dispatches within a command buffer handles synchronization.

---

## Dependencies & Build System

### Cargo.toml (Workspace Root)

```toml
[workspace]
members = [
    "crates/metal-attention",
    "crates/metal-attention-traits",
    "crates/metal-attention-kernels",
    "crates/metal-attention-gguf",
    "crates/metal-attention-models",
    "crates/metal-attention-burn",
]
resolver = "2"

[workspace.package]
version = "0.1.0"
edition = "2021"
license = "MIT OR Apache-2.0"
repository = "https://github.com/kavanaghpatrick/metal-attention"

[workspace.dependencies]
# Metal bindings
objc2 = "0.6"
objc2-metal = "0.3"
objc2-foundation = "0.3"
block2 = "0.6"

# Memory mapping
memmap2 = "0.9"

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# Error handling
thiserror = "2"
anyhow = "1"

# CLI
clap = { version = "4", features = ["derive"] }

# Logging
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

# Random sampling
rand = "0.8"

# Testing / benchmarks
criterion = { version = "0.5", features = ["html_reports"] }

# Optional: HuggingFace tokenizers (feature-gated)
tokenizers = { version = "0.21", optional = true }

# Optional: Burn integration (feature-gated)
burn = { version = "0.20", default-features = false, optional = true }

[profile.release]
opt-level = 3
lto = "thin"
codegen-units = 1
```

### metal-attention-kernels/Cargo.toml

```toml
[package]
name = "metal-attention-kernels"
version.workspace = true
edition.workspace = true

[dependencies]
objc2.workspace = true
objc2-metal.workspace = true
objc2-foundation.workspace = true
block2.workspace = true
thiserror.workspace = true
tracing.workspace = true

[build-dependencies]
# No build deps -- build.rs uses std::process::Command for xcrun
```

### metal-attention-gguf/Cargo.toml

```toml
[package]
name = "metal-attention-gguf"
version.workspace = true
edition.workspace = true

[dependencies]
memmap2.workspace = true
thiserror.workspace = true
tracing.workspace = true

[features]
default = []
hf-tokenizers = ["dep:tokenizers"]

[dependencies.tokenizers]
workspace = true
optional = true
```

### build.rs (Metal Shader Compilation)

The build.rs from the prototype is carried forward with enhancements:

```rust
// crates/metal-attention-kernels/build.rs

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let shader_dir = Path::new("shaders");
    let profile = env::var("PROFILE").unwrap_or_default();

    // Collect all .metal files
    let metal_files = collect_metal_files(shader_dir);

    // Compile each .metal → .air with Metal 3.1, include path for types.h
    let air_files: Vec<PathBuf> = metal_files.iter()
        .map(|f| compile_metal_to_air(f, shader_dir, &out_dir, &profile))
        .collect();

    // Link all .air → shaders.metallib
    link_metallib(&air_files, &out_dir);

    // Rerun triggers
    println!("cargo:rerun-if-changed=shaders/");

    // Export metallib path for runtime discovery
    println!("cargo:rustc-env=METALLIB_PATH={}", out_dir.join("shaders.metallib").display());
}

fn compile_metal_to_air(metal_file: &Path, include_dir: &Path, out_dir: &Path, profile: &str) -> PathBuf {
    let stem = metal_file.file_stem().unwrap().to_str().unwrap();
    let air_file = out_dir.join(format!("{stem}.air"));

    let mut cmd = Command::new("xcrun");
    cmd.args(["-sdk", "macosx", "metal", "-std=metal3.1", "-c"]);
    cmd.args(["-I", include_dir.to_str().unwrap()]);

    if profile == "release" {
        cmd.arg("-O2");
    }

    // Enable Metal validation in debug builds
    if profile != "release" {
        cmd.arg("-gline-tables-only");
    }

    cmd.arg(metal_file.to_str().unwrap());
    cmd.args(["-o", air_file.to_str().unwrap()]);

    let status = cmd.status().expect("Failed to run xcrun metal");
    assert!(status.success(), "Metal compilation failed for {}", metal_file.display());

    air_file
}
```

---

## Testing Strategy

### Unit Tests

Located in each crate's `src/` modules with `#[cfg(test)]` blocks.

| Crate | Test Focus | Examples |
|-------|-----------|----------|
| `metal-attention-traits` | Type sizes, default values, schedule generation | `LayerSchedule::periodic(32, 7)` produces correct pattern |
| `metal-attention-kernels` | Kernel output vs CPU reference | Flash attention output matches `cpu_attention_f64` (atol=5e-3) |
| `metal-attention-gguf` | GGUF parsing, metadata access | Parse real GGUF header, read tensor shapes, tokenizer round-trip |
| `metal-attention-models` | Architecture detection | Detect Jamba from GGUF metadata, correct layer schedule |
| `metal-attention` | Prefill/decode pipeline | Single-layer model produces non-zero logits |

### Integration Tests

Located in workspace-level `tests/` directory.

```rust
// tests/correctness.rs

/// Verify that flash attention GPU output matches CPU reference.
/// Uses the proto's cpu_attention_f64 as ground truth.
#[test]
fn flash_attention_matches_cpu_reference() {
    // Random Q, K, V at N=256, D=64
    // GPU flash attention output
    // CPU FP64 reference output
    // assert_allclose(gpu, cpu, atol=5e-3, rtol=1e-2)
}

/// Verify that linear attention GPU output matches CPU reference.
#[test]
fn linear_attention_matches_cpu_reference() {
    // assert_allclose(gpu, cpu, atol=1e-3, rtol=1e-2)
}

/// Verify PagedAttention produces same output as dense attention.
#[test]
fn paged_attention_matches_dense() {
    // Same Q, K, V → paged vs dense → assert_allclose(atol=1e-3)
}

// tests/model_load.rs

/// Verify GGUF file can be opened and metadata read.
#[test]
fn load_gguf_metadata() {
    // Open a small test GGUF file
    // Verify architecture, layer count, head dimensions
}

/// Verify tensors can be mapped to Metal buffers.
#[test]
fn gguf_tensor_to_metal_buffer() {
    // Load tensor data → create Metal buffer → verify contents
}

// tests/e2e.rs

/// Verify end-to-end text generation produces coherent output.
/// Uses a small quantized model (e.g., TinyLlama Q4_0).
#[test]
fn generate_text_e2e() {
    // Load model → prefill("Hello") → decode 10 tokens
    // Verify: all tokens are valid vocab IDs, no NaN logits
}
```

### Benchmark Suite

Located in workspace-level `benches/` directory, using criterion.

```rust
// benches/attention.rs

fn bench_flash_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("flash_attention");
    for n in [256, 512, 1024, 2048] {
        group.bench_with_input(
            BenchmarkId::new("N", n),
            &n,
            |b, &n| b.iter(|| run_flash_attention(...)),
        );
    }
    group.finish();
}

fn bench_linear_attention(c: &mut Criterion) {
    // Same pattern, includes GPU-only timing via command buffer timestamps
}

fn bench_decode_throughput(c: &mut Criterion) {
    // Measure tokens/second during autoregressive decode
}

fn bench_prefill_throughput(c: &mut Criterion) {
    // Measure tokens/second during prompt processing
}
```

### Correctness Tolerances (from Prototype Validation)

| Kernel | atol | rtol | Notes |
|--------|------|------|-------|
| Flash Attention | 5e-3 | 1e-2 | FP32 online softmax vs FP64 reference |
| Linear Attention | 1e-3 | 1e-2 | FP32 chunk accumulation |
| PagedAttention | 1e-3 | 1e-2 | Scalar dot products |
| RoPE | 1e-4 | 1e-3 | Element-wise trig |
| ALiBi | 5e-3 | 1e-2 | Dominated by softmax accumulation |
| GQA remap | 1e-6 | 1e-6 | Exact (pure memory copy) |
| RMSNorm | 1e-5 | 1e-4 | Square root + division |
| FFN (SwiGLU) | 1e-4 | 1e-3 | Sigmoid + multiplication |

### CI / Validation Requirements

- All tests run with `MTL_SHADER_VALIDATION=1` environment variable (catches memory access violations, out-of-bounds reads, uninitialized threadgroup memory).
- Memory leak detection: `currentAllocatedSize()` sampled before/after 100-iteration loops. Zero growth required.
- Benchmark regression detection: criterion comparison against baseline stored in `bench.json`.

---

## Migration Path from Proto

### What Moves Forward (Direct Reuse)

| Proto Asset | Target Location | Changes Needed |
|-------------|----------------|----------------|
| `shaders/flash_attention.metal` | `kernels/shaders/flash_attention.metal` | Add multi-simdgroup support, causal mask function constant |
| `shaders/linear_attention.metal` | `kernels/shaders/linear_attention.metal` | Add multi-head support (grid dispatch includes head dim) |
| `shaders/paged_attention.metal` | `kernels/shaders/paged_attention.metal` | Add multi-head support, connect to PagedKVCache |
| `shaders/paged_reduce.metal` | `kernels/shaders/paged_reduce.metal` | Minimal changes |
| `shaders/rope.metal` | `kernels/shaders/rope.metal` | Add multi-head support |
| `shaders/gqa_remap.metal` | `kernels/shaders/gqa_remap.metal` | No changes needed |
| `shaders/types.h` | `kernels/shaders/types.h` | Extend with LayerParams, SSMParams |
| `src/device.rs` (GpuDevice) | `kernels/src/device.rs` | Multi-metallib search for workspace layout |
| `src/pipeline.rs` (PsoCache) | `kernels/src/pipeline.rs` | Add `prewarm()` method |
| `src/encode.rs` (buffer helpers) | `kernels/src/buffer.rs` | Add BufferPool, zero-copy mmap support |
| `src/types.rs` (AttentionParams) | `kernels/src/types.rs` + `traits/src/types.rs` | Split into host-side types and GPU-side params |
| `src/proto1_flash.rs` (cpu_attention_f64, assert_allclose) | `tests/correctness.rs` | Move to test utilities |
| `src/proto6_fla.rs` (cpu_linear_attention_f64) | `tests/correctness.rs` | Move to test utilities |
| `build.rs` | `kernels/build.rs` | Add METALLIB_PATH env export, debug symbols |
| `benches/*.rs` | `benches/*.rs` | Adapt to new API surface |

### What Gets Rewritten

| Proto Component | Why Rewrite | New Design |
|----------------|-------------|------------|
| Per-proto host dispatch (run_flash_attention, run_linear_attention) | One-off test harnesses, not composable | Generic dispatch through SequenceBlock trait + PsoCache |
| Timing infrastructure | Prototype-specific, not integrated with inference loop | tracing spans + command buffer GPU timestamps |
| KB findings system | Development tool, not runtime | Remove entirely from production crate |
| Proto modules (proto2_stitch, proto4_constants, proto5_cubecl, proto7_variants, proto8_burn) | Prototypes that validated decisions -- decisions are now baked in | Architecture document captures findings, no code needed |

### What Gets Added (Not in Proto)

| New Component | Why Not in Proto | Design Source |
|---------------|-----------------|---------------|
| GGUF parser | Proto used hardcoded test data | GGUF spec + llama.cpp reference |
| Weight mapper | Proto had no model loading | Architecture detection from GGUF metadata |
| Tokenizer | Proto had no text processing | GGUF-embedded BPE + optional HF tokenizers |
| FFN kernels (SwiGLU, ReLU^2, GeGLU) | Proto focused on attention only | Standard feed-forward network layers |
| RMSNorm kernel | Proto used pre-normalized inputs | Llama-style RMS normalization |
| Embedding/output projection kernels | Proto used pre-projected Q/K/V | Standard embedding + matmul |
| Dequantization kernels | Proto used FP32 only | Q4_0, Q4_K_M, Q8_0 block dequant |
| SSM scan kernel | Proto implemented FLA only | Selective scan for Mamba/Mamba-2 |
| Prefix sum kernel | Proto used CPU prefix sum | GPU parallel prefix sum over D x D matrices |
| Inference loop (prefill + decode) | Proto benchmarked kernels in isolation | Full forward pass through all layers |
| Sampling engine | Not needed for kernel benchmarks | Temperature, top-p, top-k, repetition penalty |
| CLI | Not needed for benchmarks | clap-based run/bench/info commands |
| Triple buffering / CommandManager | Proto committed + waited synchronously | Overlap CPU encoding with GPU execution |
| BufferPool | Proto allocated fresh buffers each call | Reuse buffers across decode steps |

### Migration Sequence

1. **Create workspace structure** with all crate directories and Cargo.toml files.
2. **Copy shaders** from `proto/shaders/` to `kernels/shaders/`, extending `types.h`.
3. **Port device.rs, pipeline.rs, encode.rs** from proto to kernels crate, adapting imports.
4. **Define traits** in traits crate (no Metal dependency, pure Rust interfaces).
5. **Implement GGUF parser** in gguf crate (mmap + binary parsing).
6. **Implement FLA linear attention** as first `LinearSequenceModel` using ported shaders.
7. **Implement flash attention** as first `SoftmaxAttention` using ported shaders.
8. **Build inference loop** connecting traits, kernels, and GGUF loading.
9. **Add new kernels** (RMSNorm, FFN, embedding, matmul, dequant) as needed for end-to-end inference.
10. **CLI binary** for run/bench/info commands.

The proto directory is preserved as a reference but not linked into the workspace build. All prototype findings are captured in SYNTHESIS.md and this document.
