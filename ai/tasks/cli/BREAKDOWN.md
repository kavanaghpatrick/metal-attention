---
id: cli.BREAKDOWN
module: cli
priority: 6
status: failing
version: 1
origin: spec-workflow
dependsOn: [inference.BREAKDOWN]
tags: [breakdown]
testRequirements:
  unit:
    required: false
    pattern: "tests/cli/**/*.test.*"
---
# CLI -- Breakdown

## Context

The CLI is the primary user interface for metal-attention. It provides three subcommands (`run`, `bench`, `info`) following established conventions from llama.cpp and the Rust CLI ecosystem. The CLI wraps the library API, adding streaming output, progress indicators, JSON mode, TOML configuration, error formatting with actionable hints, and shell completions. Design follows clig.dev guidelines: tokens to stdout, everything else to stderr, human-readable by default, machine-readable via `--json`.

## Scope

- **Binary entry point**: `src/main.rs` with clap derive-based argument parsing
- **`run` subcommand**: Load model, tokenize prompt, generate tokens with streaming output, report stats
- **`bench` subcommand**: Run criterion-style benchmarks across sequence lengths, batch sizes, backends; output tables or JSON/CSV
- **`info` subcommand**: Display model architecture, layer composition, memory estimates, Metal kernel assignments
- **Streaming output**: Flush tokens immediately to stdout; stats to stderr; detect TTY for formatting
- **JSON mode**: `--json` flag produces JSONL for streaming, single JSON for batch operations
- **Configuration**: TOML config file at `~/Library/Application Support/metal-attention/config.toml`; precedence: CLI > env > config > defaults
- **Environment variables**: `METAL_ATTENTION_*` prefix for all settings; `NO_COLOR` support; `METAL_ATTENTION_LOG` for tracing levels
- **Error formatting**: Conversational errors with `Hint:` lines; exit codes (0=success, 1=general, 2=args, 3=model, 4=hardware, 5=OOM, 130=interrupted)
- **Progress indicators**: `indicatif` spinner for model loading, progress bar for benchmarks (stderr only)
- **Shell completions**: `clap_complete` for bash/zsh/fish via hidden `completions` subcommand
- **Table formatting**: Unicode box-drawing for TTY, plain ASCII when piped
- **Model path resolution**: Search CWD, env var, config, `~/models/`, `~/.cache/metal-attention/models/`

## Key Decisions

- **From UX.md**: Short flags match llama.cpp: `-m` (model), `-p` (prompt), `-n` (max-tokens), `-s` (seed), `-r` (runs), `-o` (output), `-v` (verbose), `-q` (quiet).
- **From UX.md**: Stats shown by default when TTY, hidden when piped. `--stats` forces display. `-q` suppresses.
- **From UX.md**: Interactive mode via `--interactive` reads stdin line-by-line for chat-style interaction.
- **From UX.md**: Bench output includes both table format (human) and JSON/CSV (machine). `--compare <PATH>` enables before/after regression analysis.
- **From UX.md**: Recommended crates: clap 4.x, clap_complete, serde+serde_json, anyhow, thiserror, indicatif 0.17.x, console 0.15.x, toml 0.8.x, directories 5.x, tracing+tracing-subscriber.
- **From PM.md**: P0-8 requires `metal-attention run` with streaming output. P1-8 requires `bench` command. P1-9 requires `info` command.
- **From UX.md**: Version output includes build info: `metal-attention 0.1.0 (rustc X.Y, Metal X.Y, Apple MX)`.

## Acceptance Criteria

1. `metal-attention --help` prints usage with three subcommands and examples
2. `metal-attention run -m model.gguf -p "test"` loads model and generates tokens to stdout
3. `metal-attention run -m model.gguf -p "test" --json` produces valid JSONL output
4. `metal-attention bench -m model.gguf --seq-lengths 128,256` produces benchmark table
5. `metal-attention info -m model.gguf` displays architecture, layer count, memory estimates
6. `metal-attention info -m model.gguf --json` produces valid JSON
7. Stats line appears on stderr when stdout is TTY; hidden when piped
8. `--no-color` and `NO_COLOR` env var disable colored output
9. Missing model file produces actionable error with path suggestions
10. Invalid arguments produce clap-formatted error with usage hint
11. Config file at platform-specific path is loaded when present; CLI flags override config values
12. `METAL_ATTENTION_MODEL` environment variable is respected
13. `Ctrl+C` during generation prints partial stats and exits with code 130

## Technical Notes

- From UX.md: JSONL streaming format: `{"type":"meta",...}`, `{"type":"status",...}`, `{"type":"token",...}`, `{"type":"done",...}`.
- From UX.md: Bench JSON format: single object with `version`, `model`, `architecture`, `device`, `timestamp`, and `results` array.
- From UX.md: Model info output shows: architecture name, SSM:attention ratio, per-layer type breakdown, memory estimates at different context sizes, which Metal kernels will be used with their PSO variants.
- From QA.md: CLI error formatting uses anyhow context chains. Verbose mode (`-v`) adds file/line info. Debug mode (`-vvv`) adds full backtraces.
- From UX.md: Configuration file uses TOML with sections: `[generation]`, `[performance]`, `[output]`, `[paths]`, `[bench]`.
