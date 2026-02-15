---
id: devops.BREAKDOWN
module: devops
priority: 0
status: failing
version: 1
origin: spec-workflow
dependsOn: []
tags: [breakdown]
testRequirements:
  unit:
    required: false
    pattern: "tests/devops/**/*.test.*"
---
# DevOps -- Breakdown

## Context

metal-attention is structured as a Rust workspace with 6 crates (`metal-attention`, `metal-attention-traits`, `metal-attention-kernels`, `metal-attention-gguf`, `metal-attention-models`, `metal-attention-burn`) plus a CLI binary. The project inherits 8 validated prototypes in `proto/` and requires Metal shader compilation via `xcrun metal -std=metal3.1` in the build system. CI must support both CPU-only testing (GitHub-hosted runners) and GPU testing (self-hosted Apple Silicon runners with `MTL_SHADER_VALIDATION=1`).

## Scope

- Workspace `Cargo.toml` with all workspace members, shared dependencies, and release profile
- Per-crate `Cargo.toml` files with correct inter-crate dependencies
- `build.rs` for `metal-attention-kernels` that compiles `.metal` -> `.air` -> `shaders.metallib`
- GitHub Actions CI pipeline: Tier 1 (CPU-only, GitHub-hosted) and Tier 2 (GPU, self-hosted)
- Clippy, rustfmt, and cargo check configuration
- Criterion benchmark scaffolding in workspace `benches/`
- `.gitignore` for build artifacts, metallib files, model fixtures
- Workspace-level test scaffolding in `tests/`

## Key Decisions

- **From TECH.md**: Workspace uses `resolver = "2"`, edition 2021, dual MIT/Apache-2.0 license. Core dependencies: `objc2 0.6`, `objc2-metal 0.3`, `memmap2 0.9`, `clap 4.x`, `thiserror 2`, `anyhow 1`, `serde 1`, `tracing 0.1`, `criterion 0.5`, `rand 0.8`. Release profile: `opt-level = 3`, `lto = "thin"`, `codegen-units = 1`.
- **From TECH.md**: `build.rs` uses `xcrun -sdk macosx metal -std=metal3.1 -c` for compilation, `-O2` in release, `-gline-tables-only` in debug. All `.air` files linked into single `shaders.metallib`. `METALLIB_PATH` exported as rustc env var.
- **From QA.md**: CI Tier 1 (CPU): `cargo check`, `cargo clippy -- -D warnings`, `cargo fmt --check`, `cargo test --lib`, proptest, <5 minutes. CI Tier 2 (GPU): `MTL_SHADER_VALIDATION=1 cargo test -- --test-threads=1`, `cargo bench -- --quick`, <15 minutes.
- **From QA.md**: Tests must use `--test-threads=1` for Metal device contention. Self-hosted runner needs `RUST_TEST_THREADS=1`.

## Acceptance Criteria

1. `cargo check --workspace` compiles without errors
2. `cargo clippy --workspace --all-targets -- -D warnings` passes with zero warnings
3. `cargo fmt --all --check` passes
4. `build.rs` successfully compiles at least one `.metal` shader to `.metallib` (can be a stub shader initially)
5. CI workflow file exists and runs Tier 1 checks on PR
6. Workspace member crates resolve dependencies correctly (no circular deps)
7. `cargo test --workspace --lib` runs (even if tests are empty placeholders)
8. Criterion benchmark target compiles (`cargo bench --no-run`)

## Technical Notes

- From TECH.md: The module dependency graph is:
  ```
  cli -> metal-attention -> traits, kernels, gguf, models
  kernels -> traits
  models -> traits, kernels, gguf
  burn -> traits, kernels, burn-framework
  ```
- From TECH.md: `metal-attention-traits` has zero Metal dependencies (pure Rust interfaces). Only `metal-attention-kernels` links Metal frameworks.
- From QA.md: All test runs must be compatible with `MTL_SHADER_VALIDATION=1` environment. Shader validation catches out-of-bounds access, null buffer access, incorrect binding indices.
- From TECH.md: Proto directory is preserved as reference but NOT linked into workspace build. Proto shaders are copied to `kernels/shaders/` during module setup.
- The `metal-attention-burn` crate is feature-gated and optional.
