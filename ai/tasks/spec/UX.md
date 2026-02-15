# UX Analysis: metal-attention CLI & API

> **Agent**: UX Designer
> **Date**: 2026-02-14
> **Status**: Complete
> **Scope**: CLI interface design, API ergonomics, output formatting, error handling, configuration

---

## Research Findings

### CLI UX Conventions for ML Inference Tools

Research into established ML inference CLIs reveals consistent patterns that metal-attention should adopt:

**llama.cpp CLI patterns** (the de facto standard for local inference):
- Model specification via `-m` / `--model` with path or HuggingFace shorthand (`-hf user/model[:quant]`)
- Threading control via `-t` / `--threads`
- GPU layer offloading via `-ngl` (not applicable to our unified-memory architecture, but the concept of hardware hints is relevant)
- Verbosity via `-lv` / `--verbosity` with numeric levels
- Sampling parameters: `-s` (seed), `--temp`, `--top-p`, `--top-k`, `--repeat-penalty`
- Benchmarking via separate `llama-bench` binary with `--n-prompt`, `--n-gen` parameters
- Output: performance stats printed to stderr, tokens to stdout
- JSON output from benchmarks includes `avg_ts` (tokens/second), `stddev_ts`, `samples_ns`

Sources:
- [llama-cli manpage](https://manpages.debian.org/testing/llama.cpp-tools/llama-cli.1.en.html)
- [llama.cpp CLI guide discussion](https://github.com/ggml-org/llama.cpp/discussions/15709)
- [llama-bench discussion](https://github.com/ggml-org/llama.cpp/discussions/9124)

**MLX-LM CLI patterns** (Apple Silicon native):
- `mlx_lm.generate --model <name> --prompt "..."` as the primary command
- `--max-tokens` for generation length
- `--max-kv-size` for KV cache control
- Direct HuggingFace model download integration
- Python API mirrors CLI arguments closely

Sources:
- [MLX-LM GitHub](https://github.com/ml-explore/mlx-lm)
- [MLX framework](https://ml-explore.github.io/mlx/)

**Command Line Interface Guidelines (clig.dev)**:
- Human-readable output by default; machine-readable via explicit flag
- Detect TTY to adjust output formatting (colors, progress bars)
- Error messages should be conversational: state what went wrong and suggest a fix
- Send diagnostics to stderr, data to stdout
- Follow existing conventions (GNU long options, kebab-case flags)
- Make functionality discoverable via help text and examples

Sources:
- [Command Line Interface Guidelines](https://clig.dev/)
- [CLI Guidelines GitHub](https://github.com/cli-guidelines/cli-guidelines)

**Rust CLI ecosystem**:
- `clap` 4.x with `#[derive(Parser)]` is the standard for argument parsing
- `anyhow` for application errors with context chains; `thiserror` for library error enums
- `indicatif` for progress bars (Sync + Send, multi-threaded safe, 20 FPS refresh, stderr by default)
- `serde` + `serde_json` for structured output
- `directories` crate for cross-platform config paths (XDG on Linux, `~/Library` on macOS)
- `figment` or `config-rs` for layered configuration (defaults < file < env < CLI args)
- Kebab-case for multi-word flags: `--seq-lengths`, not `--seq_lengths`

Sources:
- [clap documentation](https://docs.rs/clap)
- [Rust CLI error handling](https://rust-cli.github.io/book/tutorial/errors.html)
- [indicatif crate](https://docs.rs/indicatif)
- [Rust CLI machine communication](https://rust-cli.github.io/book/in-depth/machine-communication.html)
- [Clap + Figment config](https://www.hecatron.com/posts/2025/rust-cli-cfg-opts/)

### Hybrid Model Landscape (User Mental Model)

Users of metal-attention will think in terms of model architectures with SSM-to-attention ratios:
- Jamba: 7:1 SSM:Attention + MoE (AI21)
- Griffin/Hawk: 2:1 RG-LRU:Attention (Google)
- Zamba: 6:1 shared attention (Zyphra)
- RWKV-7: pure linear attention
- Nemotron-H: ~12:1 (NVIDIA)

The CLI should surface architecture details clearly so users understand what kernel paths are active and why performance differs from pure-transformer baselines.

Sources:
- [Jamba ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/a9ed43fa31dc8b4a7d7a673d713dcb5f-Paper-Conference.pdf)
- [Hybrid Architecture Analysis](https://arxiv.org/html/2510.04800v1)
- [Rise of Hybrid LLMs (AI21)](https://www.ai21.com/blog/rise-of-hybrid-llms/)

---

## CLI Interface Design

### Command Structure

metal-attention uses a subcommand pattern following clap's derive API. Three primary subcommands cover the complete user workflow:

```
metal-attention <COMMAND> [OPTIONS]

Commands:
  run     Run inference on a model with a prompt
  bench   Benchmark model performance across sequence lengths
  info    Display model architecture, layer composition, and memory estimates

Options:
  -v, --verbose          Increase verbosity (repeat for more: -vv, -vvv)
  -q, --quiet            Suppress all non-essential output
  --json                 Output in JSON format (all commands)
  --no-color             Disable colored output (also respects NO_COLOR env var)
  --config <PATH>        Path to config file (default: ~/.config/metal-attention/config.toml)
  -h, --help             Print help information
  -V, --version          Print version information
```

### `run` Command

```
metal-attention run [OPTIONS] --model <PATH> --prompt <TEXT>

Required:
  -m, --model <PATH>        Path to GGUF model file

Input (one required):
  -p, --prompt <TEXT>        Text prompt for generation
  --prompt-file <PATH>       Read prompt from file (- for stdin)
  --interactive              Interactive chat mode (reads from stdin line-by-line)

Generation:
  -n, --max-tokens <N>       Maximum tokens to generate (default: 256)
  --temp <FLOAT>             Sampling temperature (default: 0.7)
  --top-p <FLOAT>            Top-p (nucleus) sampling (default: 0.9)
  --top-k <INT>              Top-k sampling (default: 40)
  --repeat-penalty <FLOAT>   Repetition penalty (default: 1.1)
  -s, --seed <INT>           RNG seed for reproducibility (default: random)

Performance:
  --threads <N>              CPU threads for host-side work (default: auto)
  --context-size <N>         Maximum context window size (default: from model)
  --kv-cache <MODE>          KV cache strategy: dense | paged (default: auto)
  --page-size <N>            Page size for paged KV cache (default: 16)

Output:
  --no-stream                Wait for full completion before printing
  --stats                    Print performance statistics after generation
  --stats-interval <N>       Print running stats every N tokens (default: off)
  --json                     Output JSON (streaming JSONL or single object)

Architecture override (advanced):
  --attention <TYPE>         Override attention backend: flash | linear | auto (default: auto)
  --position-encoding <TYPE> Override position encoding: rope | alibi | none (default: from model)
```

### `bench` Command

```
metal-attention bench [OPTIONS] --model <PATH>

Required:
  -m, --model <PATH>          Path to GGUF model file

Benchmark parameters:
  --seq-lengths <LIST>        Comma-separated sequence lengths (default: 128,256,512,1024,2048)
  --batch-sizes <LIST>        Comma-separated batch sizes (default: 1)
  -r, --runs <N>              Repetitions per configuration (default: 5)
  --warmup <N>                Warmup iterations before measurement (default: 3)

Scope:
  --prefill-only              Only benchmark prompt processing (prefill)
  --decode-only               Only benchmark token generation (decode)
  --kernel-only               Benchmark raw kernel time (exclude host overhead)
  --all-backends              Benchmark all applicable backends (flash, linear, paged)

Output:
  --json                      Output results as JSON
  --csv                       Output results as CSV
  -o, --output <PATH>         Write results to file instead of stdout
  --compare <PATH>            Compare against previous benchmark results
```

### `info` Command

```
metal-attention info [OPTIONS] --model <PATH>

Required:
  -m, --model <PATH>          Path to GGUF model file

Display:
  --layers                    Show per-layer type breakdown
  --memory                    Show memory estimates for different quantizations
  --kernels                   Show which Metal kernels will be used
  --json                      Output as JSON
```

### Short Flag Conventions

Frequently-used flags get short aliases following established conventions:

| Short | Long | Rationale |
|-------|------|-----------|
| `-m` | `--model` | Matches llama.cpp convention |
| `-p` | `--prompt` | Matches llama.cpp convention |
| `-n` | `--max-tokens` | Matches llama.cpp `-n` for token count |
| `-s` | `--seed` | Matches llama.cpp convention |
| `-r` | `--runs` | Matches llama-bench convention |
| `-o` | `--output` | Standard Unix convention |
| `-v` | `--verbose` | Standard Unix convention |
| `-q` | `--quiet` | Standard Unix convention |

---

## User Workflows

### Workflow 1: First Run (New User)

The first-run experience is critical. A user who has just installed metal-attention should go from zero to generating text in under 60 seconds.

```bash
# Step 1: Install
cargo install metal-attention

# Step 2: Run with a model they already have
metal-attention run -m ~/models/rwkv-7-1.6b-q4_0.gguf -p "Hello, world"

# Expected output:
# Loading model: rwkv-7-1.6b-q4_0.gguf
# Architecture: RWKV-7 (pure linear attention)
# Layers: 24 linear
# Memory: 1.2 GB (Q4_0)
# Device: Apple M4 Pro (20-core GPU, 48 GB unified memory)
#
# Hello, world! I'm a language model running entirely on your Mac's GPU...
#
# --- Stats ---
# Tokens: 47 | Time: 1.23s | Speed: 38.2 tok/s | Memory: 1.4 GB
```

Key UX decisions for first run:
- Model loading message appears on stderr so stdout contains only generated tokens
- Architecture detection is automatic -- no flags needed for standard GGUF models
- Device capabilities are shown once to confirm Metal is active
- Performance stats are printed at the end by default (suppress with `-q`)
- If the model file is not found, the error message suggests common model directories

### Workflow 2: Interactive Exploration

```bash
# Interactive mode for conversational use
metal-attention run -m jamba-1.5-mini.gguf --interactive

# Expected behavior:
# Loading model: jamba-1.5-mini.gguf
# Architecture: Jamba 1.5 (7:1 SSM:Attention, MoE 16 experts)
# Layers: 32 total (28 linear + 4 flash attention)
# Ready. Type your prompt (Ctrl+D to exit, Ctrl+C to cancel generation).
#
# > What is quantum computing?
# Quantum computing is a type of computation that harnesses...
# [38.5 tok/s]
#
# > Can you explain that more simply?
# Sure! Think of it like this...
# [41.2 tok/s]
```

### Workflow 3: Benchmarking for Hardware Evaluation

```bash
# Full benchmark suite
metal-attention bench -m jamba-1.5-mini.gguf --all-backends --json -o bench.json

# Human-readable table output (default):
#
# metal-attention bench v0.1.0 | Apple M4 Pro (20-core GPU)
# Model: jamba-1.5-mini (Jamba 1.5, 7:1, Q4_K_M)
#
# Prefill (prompt processing):
# ┌───────────┬──────────┬──────────┬──────────┬──────────┐
# │ Seq Length │ Flash    │ Linear   │ Paged    │ Hybrid   │
# ├───────────┼──────────┼──────────┼──────────┼──────────┤
# │       128 │  1842 t/s│  3920 t/s│  1690 t/s│  3654 t/s│
# │       256 │  1523 t/s│  3890 t/s│  1402 t/s│  3601 t/s│
# │       512 │  1190 t/s│  3844 t/s│  1095 t/s│  3498 t/s│
# │      1024 │   820 t/s│  3801 t/s│   754 t/s│  3350 t/s│
# │      2048 │   445 t/s│  3756 t/s│   409 t/s│  3197 t/s│
# └───────────┴──────────┴──────────┴──────────┴──────────┘
#
# Decode (token generation):
# ┌───────────┬──────────┬────────────┐
# │ Seq Length │  tok/s   │ std dev    │
# ├───────────┼──────────┼────────────┤
# │       128 │  42.3    │ +/- 0.8    │
# │       256 │  41.9    │ +/- 0.6    │
# │       512 │  40.2    │ +/- 1.1    │
# │      1024 │  38.1    │ +/- 0.9    │
# │      2048 │  34.5    │ +/- 1.3    │
# └───────────┴──────────┴────────────┘
#
# Kernel times (GPU only, median of 5 runs):
#   flash_attention: 2.42ms @ N=1024, D=64
#   linear_attention: 35us @ N=1024, D=64
#   rope: 10us/head
#   gqa_remap: 78us (group_size=4)
#   pso_cache_hit: 178ns
```

### Workflow 4: Programmatic Integration (JSON Mode)

```bash
# JSON streaming output for piping to other tools
metal-attention run -m model.gguf -p "Count to 5" --json --stats

# JSONL output (one object per line, streamable):
{"type":"status","message":"loading","model":"model.gguf","architecture":"jamba"}
{"type":"status","message":"ready","device":"Apple M4 Pro","memory_used_mb":1200}
{"type":"token","text":"One"}
{"type":"token","text":","}
{"type":"token","text":" two"}
{"type":"token","text":","}
{"type":"token","text":" three"}
{"type":"token","text":","}
{"type":"token","text":" four"}
{"type":"token","text":","}
{"type":"token","text":" five"}
{"type":"token","text":"."}
{"type":"done","tokens_generated":11,"elapsed_ms":287,"tokens_per_second":38.3,"peak_memory_mb":1450}
```

### Workflow 5: Comparing Architectures (Researcher)

```bash
# Compare flash vs linear attention on the same model
metal-attention bench -m model.gguf --all-backends --seq-lengths 256,512,1024,2048,4096

# Compare results against a previous run
metal-attention bench -m model.gguf --compare baseline.json

# Output:
# Comparison vs baseline.json:
#   prefill N=1024: 3350 t/s -> 3520 t/s (+5.1%)
#   decode  N=1024: 38.1 t/s -> 39.4 t/s (+3.4%)
#   flash kernel:   2.42ms -> 2.31ms (-4.5%)
```

### Workflow 6: Model Information

```bash
metal-attention info -m jamba-1.5-mini.gguf --layers --memory --kernels

# Output:
# Model: Jamba 1.5 Mini
# Architecture: Jamba (Hybrid Transformer-Mamba)
# Parameters: 12B (active: 3.8B with MoE)
# Quantization: Q4_K_M
# Context window: 256K tokens
#
# Layer composition:
#   Total layers: 32
#   Linear (SSM): 28 (87.5%) -> LinearAttention kernel
#   Attention:     4 (12.5%) -> FlashAttention kernel
#   Ratio: 7:1 SSM:Attention
#   MoE: 16 experts, top-2 routing
#
# Position encoding: RoPE (theta=10000)
# GQA: group_size=8 (8 KV heads, 64 attention heads)
#
# Memory estimates:
#   Model weights: 2.1 GB (Q4_K_M)
#   KV cache (1K ctx): 128 MB
#   KV cache (4K ctx): 512 MB
#   KV cache (256K ctx): 32 GB
#   Total (1K ctx): 2.2 GB
#
# Metal kernels:
#   flash_attention (Br=16, Bc=64, D=64) -> PSO variant 1
#   linear_attention (chunk_size=32, D=64) -> PSO variant 2
#   rope (D=64) -> PSO variant 3
#   gqa_remap (group_size=8) -> PSO variant 4
#   prefix_sum (D=64) -> PSO variant 5
#   PSO cold compile estimate: ~315us (5 variants @ ~63us each)
```

---

## API Design

### Public Rust Library Interface

metal-attention is published as both a CLI binary and a library crate. The library API follows Rust conventions: builder pattern for configuration, `Result<T, Error>` for fallible operations, and trait-based extensibility.

### Core Types

```rust
// Top-level entry point
pub struct Engine {
    device: MetalDevice,
    pso_cache: PsoCache,
}

impl Engine {
    /// Create an engine with default configuration for the current device.
    pub fn new() -> Result<Self, Error>;

    /// Create an engine with custom configuration.
    pub fn with_config(config: EngineConfig) -> Result<Self, Error>;

    /// Load a model from a GGUF file.
    /// Architecture is auto-detected from model metadata.
    pub fn load_model(&self, path: impl AsRef<Path>) -> Result<Model, Error>;
}
```

### Model and Generation

```rust
pub struct Model {
    // opaque internals
}

impl Model {
    /// Get model metadata (architecture, layer count, quantization, etc.)
    pub fn info(&self) -> &ModelInfo;

    /// Generate tokens from a prompt.
    /// Returns a stream of generated tokens for progressive consumption.
    pub fn generate(
        &self,
        prompt: &str,
        params: &GenerationParams,
    ) -> Result<TokenStream, Error>;

    /// Generate and collect all tokens into a single string.
    /// Convenience wrapper around `generate()`.
    pub fn generate_all(
        &self,
        prompt: &str,
        params: &GenerationParams,
    ) -> Result<GenerationResult, Error>;

    /// Run benchmarks on the loaded model.
    pub fn benchmark(&self, config: &BenchConfig) -> Result<BenchResult, Error>;
}
```

### Builder-Pattern Configuration

```rust
/// Generation parameters with builder pattern.
/// All fields have sensible defaults.
pub struct GenerationParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repeat_penalty: f32,
    pub seed: Option<u64>,
    pub stop_sequences: Vec<String>,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repeat_penalty: 1.1,
            seed: None,
            stop_sequences: Vec::new(),
        }
    }
}

impl GenerationParams {
    pub fn new() -> Self { Self::default() }
    pub fn max_tokens(mut self, n: usize) -> Self { self.max_tokens = n; self }
    pub fn temperature(mut self, t: f32) -> Self { self.temperature = t; self }
    pub fn top_p(mut self, p: f32) -> Self { self.top_p = p; self }
    pub fn top_k(mut self, k: usize) -> Self { self.top_k = k; self }
    pub fn seed(mut self, s: u64) -> Self { self.seed = Some(s); self }
    pub fn deterministic(self) -> Self { self.temperature(0.0).seed(42) }
}
```

### Streaming Token Output

```rust
/// A stream of generated tokens. Implements Iterator.
pub struct TokenStream {
    // internal state
}

impl Iterator for TokenStream {
    type Item = Result<Token, Error>;
}

pub struct Token {
    /// The decoded text of this token.
    pub text: String,
    /// Token ID in the model's vocabulary.
    pub id: u32,
    /// Cumulative generation statistics at this point.
    pub stats: TokenStats,
}

pub struct TokenStats {
    /// Tokens generated so far.
    pub tokens_generated: usize,
    /// Elapsed time since generation started.
    pub elapsed: Duration,
    /// Current tokens per second.
    pub tokens_per_second: f64,
}

/// Complete generation result (from generate_all).
pub struct GenerationResult {
    pub text: String,
    pub tokens: Vec<Token>,
    pub stats: GenerationStats,
}

pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_time: Duration,
    pub decode_time: Duration,
    pub total_time: Duration,
    pub prefill_tokens_per_second: f64,
    pub decode_tokens_per_second: f64,
    pub peak_memory_bytes: usize,
}
```

### Engine Configuration

```rust
pub struct EngineConfig {
    /// CPU threads for host-side work (tokenization, sampling).
    pub threads: Option<usize>,
    /// KV cache strategy.
    pub kv_cache: KvCacheMode,
    /// Maximum context size override.
    pub max_context_size: Option<usize>,
}

pub enum KvCacheMode {
    /// Automatically choose based on sequence length and memory.
    Auto,
    /// Dense contiguous cache.
    Dense,
    /// Paged cache with specified page size.
    Paged { page_size: u32 },
}

pub struct BenchConfig {
    pub seq_lengths: Vec<usize>,
    pub batch_sizes: Vec<usize>,
    pub runs: usize,
    pub warmup: usize,
    pub scope: BenchScope,
}

pub enum BenchScope {
    Full,
    PrefillOnly,
    DecodeOnly,
    KernelOnly,
}
```

### Usage Examples

```rust
use metal_attention::{Engine, GenerationParams};

// Minimal usage
fn main() -> anyhow::Result<()> {
    let engine = Engine::new()?;
    let model = engine.load_model("jamba-1.5-mini.gguf")?;

    // Streaming output
    for token in model.generate("Explain quantum computing", &GenerationParams::default())? {
        let token = token?;
        print!("{}", token.text);
    }

    Ok(())
}

// Full control
fn advanced() -> anyhow::Result<()> {
    let engine = Engine::with_config(EngineConfig {
        threads: Some(8),
        kv_cache: KvCacheMode::Paged { page_size: 16 },
        max_context_size: Some(4096),
    })?;

    let model = engine.load_model("jamba-1.5-mini.gguf")?;

    // Check architecture before running
    let info = model.info();
    println!("Architecture: {} ({}:1 ratio)", info.architecture, info.ssm_attention_ratio);

    let params = GenerationParams::new()
        .max_tokens(512)
        .temperature(0.3)
        .top_p(0.95)
        .seed(42);

    let result = model.generate_all("Summarize this paper...", &params)?;
    println!("{}", result.text);
    println!("Speed: {:.1} tok/s", result.stats.decode_tokens_per_second);

    Ok(())
}
```

---

## Error Handling & Messages

### Error Design Philosophy

Following clig.dev guidelines, errors should be conversational: state what went wrong, explain why, and suggest how to fix it. The library uses `thiserror` for structured error enums; the CLI wraps these with `anyhow` for context chains.

### Error Enum (Library)

```rust
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Model file not found: {path}")]
    ModelNotFound { path: PathBuf },

    #[error("Invalid GGUF format: {reason}")]
    InvalidModel { reason: String },

    #[error("Unsupported architecture: {arch} (supported: jamba, griffin, rwkv, zamba, llama, mistral)")]
    UnsupportedArchitecture { arch: String },

    #[error("Unsupported quantization: {quant} (supported: Q4_0, Q4_K_M, Q8_0, F16, F32)")]
    UnsupportedQuantization { quant: String },

    #[error("Metal device not available")]
    NoMetalDevice,

    #[error("Insufficient memory: model requires {required_mb} MB, device has {available_mb} MB")]
    InsufficientMemory { required_mb: usize, available_mb: usize },

    #[error("Kernel compilation failed: {shader} ({reason})")]
    KernelCompilation { shader: String, reason: String },

    #[error("Context size exceeded: {requested} tokens exceeds maximum {maximum}")]
    ContextOverflow { requested: usize, maximum: usize },

    #[error("Generation cancelled")]
    Cancelled,

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Metal(#[from] MetalError),
}
```

### CLI Error Formatting

The CLI formats errors for human consumption with actionable guidance. Errors go to stderr and use color when connected to a TTY.

```
Examples of well-formatted CLI errors:

Error: Model file not found: ~/models/jamba.gguf
  The file does not exist at the specified path.
  Hint: Check the path. Common model locations:
    ~/models/
    ~/.cache/huggingface/hub/

Error: Unsupported architecture: mamba-pure
  metal-attention does not yet support this architecture.
  Supported architectures: jamba, griffin, rwkv, zamba, llama, mistral
  Hint: Check https://github.com/.../metal-attention for architecture support status.

Error: Insufficient memory for model
  Model requires 6.2 GB but only 4.1 GB is available.
  The model weights (3.8 GB) plus KV cache at context size 4096 (2.4 GB) exceed available memory.
  Hint: Try a smaller quantization (--model jamba-q4_0.gguf) or reduce context size (--context-size 1024).

Error: Metal device not available
  No Metal-compatible GPU was found on this system.
  metal-attention requires Apple Silicon (M1 or later) with Metal support.
  Hint: Run 'system_profiler SPDisplaysDataType' to check your GPU.
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error (model loading, generation failure) |
| 2 | Invalid arguments (clap handles this automatically) |
| 3 | Model file not found or invalid |
| 4 | Hardware not supported (no Metal device) |
| 5 | Out of memory |
| 130 | Interrupted (Ctrl+C) |

### Error Context Chain

For internal errors, the CLI uses `anyhow::Context` to build a chain from low-level to high-level:

```
Error: Failed to load model 'jamba-1.5-mini.gguf'

Caused by:
  0: Failed to parse GGUF header
  1: Invalid tensor metadata at offset 0x1A3F
  2: Expected FP16 tensor, got unknown type 99

This may indicate a corrupted or incompatible GGUF file.
Hint: Re-download the model or try a different quantization variant.
```

Verbose mode (`-v`) adds file/line information. Debug mode (`-vvv`) adds full backtraces.

---

## Output Formatting

### Streaming Token Output (Default)

When connected to a TTY, tokens stream to stdout character-by-character as they are generated. The experience mirrors typing -- each token appears immediately with no buffering delay.

```
$ metal-attention run -m model.gguf -p "What is Rust?"
Rust is a systems programming language focused on safety, concurrency,
and performance. It achieves memory safety without garbage collection
through its ownership system...

---
47 tokens | 1.23s | 38.2 tok/s
```

Design details:
- Tokens are flushed immediately to stdout (no line buffering)
- The stats line appears on stderr after generation completes
- Stats line is suppressed with `-q` and always shown with `--stats`
- When piped (not a TTY), no stats line appears unless `--stats` is explicit
- Ctrl+C cleanly cancels generation and prints partial stats

### Progress Indicators

Model loading uses an `indicatif` spinner on stderr since load times can be 1-5 seconds:

```
Loading model: jamba-1.5-mini.gguf
  [==>                    ] 45% Loading tensors (layer 14/32)
```

For benchmarking, a progress bar shows completion across all configurations:

```
Benchmarking: jamba-1.5-mini.gguf
  [=========>             ] 40% (4/10 configs, run 3/5 @ N=512)
```

Progress bars are automatically hidden when stdout is not a TTY or when `--json` is specified.

### Performance Stats Format

After generation (when `--stats` is active or by default in TTY mode):

```
--- Performance ---
Prefill:  512 tokens in 142ms (3,606 tok/s)
Decode:   128 tokens in 3.21s (39.9 tok/s)
Total:    3.35s
Memory:   2.1 GB model + 128 MB KV cache = 2.2 GB peak
Kernels:  28x linear_attention (35us avg) + 4x flash_attention (2.4ms avg)
PSO:      5 variants compiled (315us cold), cache hit rate 99.97%
Device:   Apple M4 Pro, 20-core GPU, Metal 3.1
```

### JSON Output Format

When `--json` is specified, all output is machine-readable JSON. For streaming, each line is a self-contained JSON object (JSONL/NDJSON format):

```jsonl
{"type":"meta","version":"0.1.0","model":"jamba-1.5-mini.gguf","architecture":"jamba","ratio":"7:1"}
{"type":"status","phase":"loading","progress":0.0}
{"type":"status","phase":"loading","progress":1.0,"load_time_ms":2340}
{"type":"status","phase":"compiling","variants":5,"estimated_ms":315}
{"type":"status","phase":"ready"}
{"type":"token","id":1234,"text":"Rust","tokens_generated":1,"elapsed_ms":142}
{"type":"token","id":5678,"text":" is","tokens_generated":2,"elapsed_ms":167}
...
{"type":"done","prompt_tokens":12,"generated_tokens":47,"prefill_ms":142,"decode_ms":1230,"total_ms":1372,"tokens_per_second":38.2,"peak_memory_mb":2200}
```

For `bench --json`, the output is a single JSON object:

```json
{
  "version": "0.1.0",
  "model": "jamba-1.5-mini.gguf",
  "architecture": "jamba",
  "device": "Apple M4 Pro",
  "gpu_cores": 20,
  "timestamp": "2026-02-14T10:30:00Z",
  "results": [
    {
      "test": "prefill",
      "seq_length": 1024,
      "batch_size": 1,
      "backend": "hybrid",
      "runs": 5,
      "avg_tokens_per_second": 3350.0,
      "stddev_tokens_per_second": 45.2,
      "avg_time_ms": 305.7,
      "samples_ms": [308.1, 302.4, 306.9, 303.1, 308.0]
    }
  ]
}
```

### Table Formatting (Bench)

Benchmark tables use Unicode box-drawing characters when connected to a TTY, plain ASCII when piped:

TTY output:
```
┌───────────┬──────────┬──────────┬──────────┐
│ Seq Length │  Prefill │   Decode │   Memory │
├───────────┼──────────┼──────────┼──────────┤
│       256 │ 3601 t/s │ 41.9 t/s │   2.1 GB │
│      1024 │ 3350 t/s │ 38.1 t/s │   2.3 GB │
│      4096 │ 3102 t/s │ 32.4 t/s │   3.1 GB │
└───────────┴──────────┴──────────┴──────────┘
```

Piped output:
```
Seq Length | Prefill  | Decode   | Memory
---------- ---------- ---------- --------
       256 | 3601 t/s | 41.9 t/s | 2.1 GB
      1024 | 3350 t/s | 38.1 t/s | 2.3 GB
      4096 | 3102 t/s | 32.4 t/s | 3.1 GB
```

---

## Accessibility & Documentation

### Color and Terminal Behavior

- Respect the `NO_COLOR` environment variable (https://no-color.org/)
- Provide `--no-color` flag as explicit override
- Auto-detect TTY: use colors and progress bars only when stdout/stderr is a terminal
- Use semantic colors, not decorative: red for errors, yellow for warnings, green for success, dim for metadata
- All information conveyed by color is also conveyed by text (no color-only signaling)

### Help Text Quality

Every command includes examples in its help text. Help should be scannable and useful, not a wall of flags:

```
$ metal-attention run --help

Run inference on a hybrid AI model

Usage: metal-attention run [OPTIONS] --model <PATH> --prompt <TEXT>

Examples:
  metal-attention run -m model.gguf -p "Hello, world"
  metal-attention run -m model.gguf -p "Explain AI" --temp 0.3 -n 512
  metal-attention run -m model.gguf --prompt-file input.txt --json
  echo "prompt" | metal-attention run -m model.gguf --prompt-file -

Options:
  ...
```

### Shell Completions

Generate shell completion scripts via a hidden subcommand:

```bash
# Generate completions for the user's shell
metal-attention completions bash > /usr/local/share/bash-completion/completions/metal-attention
metal-attention completions zsh > ~/.zfunc/_metal-attention
metal-attention completions fish > ~/.config/fish/completions/metal-attention.fish
```

Implemented via `clap_complete` crate with zero manual maintenance.

### Man Pages

Generate man pages from clap definitions via `clap_mangen`:

```bash
metal-attention manpage > metal-attention.1
```

### Discoverability

- `metal-attention` with no arguments prints a brief usage summary with the three commands and a "Getting Started" hint
- Each error message includes a relevant `Hint:` line
- `metal-attention info -m model.gguf --kernels` shows exactly which GPU kernels will be used, demystifying the hardware layer
- Version output includes build info: `metal-attention 0.1.0 (rustc 1.82, Metal 3.1, Apple M4 Pro)`

---

## Configuration

### Configuration File

metal-attention supports an optional TOML configuration file for persistent defaults. The file is located at:

- macOS: `~/Library/Application Support/metal-attention/config.toml`
- Override: `--config <PATH>` or `METAL_ATTENTION_CONFIG` env var

```toml
# ~/.config/metal-attention/config.toml
# All fields are optional. CLI flags override config file values.

[generation]
max_tokens = 512
temperature = 0.7
top_p = 0.9
top_k = 40
repeat_penalty = 1.1

[performance]
threads = 8              # CPU threads for host work (default: auto)
kv_cache = "auto"        # "auto" | "dense" | "paged"
page_size = 16           # Page size for paged KV cache
max_context_size = 4096  # Maximum context window

[output]
color = "auto"           # "auto" | "always" | "never"
stats = true             # Show performance stats after generation
format = "text"          # "text" | "json"

[paths]
model_dir = "~/models"   # Default directory to search for models

[bench]
runs = 5
warmup = 3
seq_lengths = [128, 256, 512, 1024, 2048]
```

### Configuration Precedence

Configuration is resolved in priority order (highest wins):

1. CLI flags (`--temp 0.3`)
2. Environment variables (`METAL_ATTENTION_TEMP=0.3`)
3. Config file (`temperature = 0.3`)
4. Built-in defaults

### Environment Variables

Every CLI flag has a corresponding environment variable with `METAL_ATTENTION_` prefix:

| Environment Variable | CLI Flag | Default |
|---------------------|----------|---------|
| `METAL_ATTENTION_MODEL` | `--model` | (none) |
| `METAL_ATTENTION_THREADS` | `--threads` | auto |
| `METAL_ATTENTION_TEMP` | `--temp` | 0.7 |
| `METAL_ATTENTION_MAX_TOKENS` | `--max-tokens` | 256 |
| `METAL_ATTENTION_KV_CACHE` | `--kv-cache` | auto |
| `METAL_ATTENTION_CONFIG` | `--config` | (platform default) |
| `NO_COLOR` | `--no-color` | (unset) |
| `METAL_ATTENTION_LOG` | `--verbose` | warn |

The `METAL_ATTENTION_LOG` variable follows `env_logger` / `tracing-subscriber` conventions: `error`, `warn`, `info`, `debug`, `trace`. It can also filter by module: `METAL_ATTENTION_LOG=metal_attention::kernel=debug`.

### Model Path Resolution

When `--model` is specified without an absolute path, metal-attention searches in this order:

1. Current working directory
2. `$METAL_ATTENTION_MODEL_DIR` (env var)
3. `model_dir` from config file
4. `~/models/`
5. `~/.cache/metal-attention/models/`

This allows common usage without full paths:

```bash
# These all work if the model is in a known location:
metal-attention run -m jamba-1.5-mini.gguf -p "Hello"
metal-attention run -m ~/models/jamba-1.5-mini.gguf -p "Hello"
```

---

## Recommended Crate Dependencies

| Crate | Purpose | Version |
|-------|---------|---------|
| `clap` | Argument parsing (derive + completions) | 4.x |
| `clap_complete` | Shell completion generation | 4.x |
| `clap_mangen` | Man page generation | latest |
| `serde` + `serde_json` | JSON serialization | 1.x |
| `anyhow` | Application error handling with context | 1.x |
| `thiserror` | Library error enum derivation | 2.x |
| `indicatif` | Progress bars and spinners | 0.17.x |
| `console` | Terminal detection, colors, TTY queries | 0.15.x |
| `toml` | Config file parsing | 0.8.x |
| `directories` | Platform-specific config/cache paths | 5.x |
| `tracing` + `tracing-subscriber` | Structured logging | 0.1.x |

---

## Summary of Key UX Decisions

1. **Tokens to stdout, everything else to stderr** -- enables clean piping and composition with other tools.

2. **Auto-detect architecture from GGUF metadata** -- users should never need to specify what kind of model they have. The tool tells them.

3. **Stats shown by default in TTY, hidden when piped** -- follows the principle of human-first output that degrades gracefully for machines.

4. **`--json` for machine-readable output** -- single flag transforms all output to JSONL (streaming) or JSON (batch), matching the pattern established by `rustc --error-format json` and `cargo --message-format json`.

5. **Builder pattern for API, flat flags for CLI** -- the library API uses idiomatic Rust builders; the CLI keeps a flat flag structure for discoverability. Both share the same defaults.

6. **Actionable error messages** -- every error includes what went wrong and what the user can try. No raw Metal error codes or panic backtraces in normal output.

7. **Architecture transparency** -- the `info` command and model-loading output always show the SSM:attention ratio and which kernel paths are active. Users should understand why metal-attention is fast on their model.

8. **Benchmark comparison** -- `bench --compare` enables before/after analysis, critical for researchers tuning architectures and for validating optimization work.

9. **Convention over configuration** -- works with zero config. The config file exists for power users who want persistent defaults, not as a requirement.

10. **Follow llama.cpp conventions where applicable** -- `-m` for model, `-p` for prompt, `-n` for token count, `-s` for seed. Users migrating from llama.cpp should feel immediately at home.
