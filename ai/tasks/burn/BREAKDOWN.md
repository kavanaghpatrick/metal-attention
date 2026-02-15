---
id: burn.BREAKDOWN
module: burn
priority: 7
status: failing
version: 1
origin: spec-workflow
dependsOn: [traits.BREAKDOWN, kernels.BREAKDOWN]
tags: [breakdown]
testRequirements:
  unit:
    required: false
    pattern: "tests/burn/**/*.test.*"
---
# Burn Integration -- Breakdown

## Context

The Burn integration provides a bridge between the Burn deep learning framework and metal-attention's native Metal kernels. Burn's existing Metal support via CubeCL/wgpu achieves only 58-70% of native MSL throughput because CubeCL cannot access `simdgroup_matrix` or Metal function constants. The `AttentionBackend` supertrait pattern, validated in Proto 8, bridges this gap with 2-17us overhead and ~150 lines of code. This crate is optional and feature-gated.

## Scope

- **Backend newtype**: `MetalAttentionBackend<B: Backend>` that wraps any Burn backend and adds attention operations
- **`AttentionBackend` supertrait**: `trait AttentionBackend: Backend` defining attention-specific operations that use metal-attention kernels
- **Tensor bridge**: Convert Burn tensors to Metal buffers and back. Zero-copy when both use the same underlying memory (unified memory on Apple Silicon).
- **Backend trait delegation**: Forward all standard `Backend` trait methods to the inner backend; intercept attention operations to dispatch through metal-attention kernels
- **Feature gating**: Entire crate behind `burn` feature flag in workspace; depends on `burn 0.20+`

## Key Decisions

- **From TECH.md**: `AttentionBackend: Backend` supertrait adds attention operations without forking Burn. The bridge converts Burn tensor handles to Metal buffer pointers and back.
- **From TECH.md**: Proto 8 validated: 2-17us bridge overhead, ~150 lines of code, zero `unsafe` blocks needed in the public API.
- **From PM.md**: P2-1 -- Nice to Have. Ecosystem play. Bridge overhead must be <20us.
- **From PM.md**: Risk "Burn framework pivots away from supertrait pattern" (Low likelihood, Low impact) -- minor maintenance burden if Burn API changes.
- **From PM.md**: Target user: Burn framework developers who need native Metal attention performance without leaving the Burn ecosystem. Burn has 9.2K GitHub stars.

## Acceptance Criteria

1. `MetalAttentionBackend<B>` compiles as a valid Burn backend
2. `AttentionBackend` supertrait extends `Backend` with at least one attention operation
3. Burn tensor -> Metal buffer conversion works for F32 tensors
4. Metal buffer -> Burn tensor conversion produces correct values
5. Bridge round-trip overhead is <20us (measured with criterion)
6. Public API exposes zero `unsafe` blocks
7. Feature flag `burn` enables/disables the entire crate cleanly
8. Example code demonstrating `MetalAttentionBackend` usage compiles and runs

## Technical Notes

- From TECH.md: Module dependency: `metal-attention-burn -> metal-attention-traits + metal-attention-kernels + burn`. Does NOT depend on `metal-attention` (main lib) or `metal-attention-models`.
- From TECH.md: The bridge needs to handle Burn's tensor ownership model. Burn tensors own their data; Metal buffers are reference-counted via `Retained`. The bridge must manage lifetimes correctly.
- From QA.md: Bridge output should match direct GPU output exactly (atol=1e-6). Benchmark: bridge overhead included in comparison.
- From PM.md: Burn 0.20+ is the target version. API stabilization is underway in Burn's 2025 roadmap.
- The Burn integration is the lowest-priority functional module (priority 7). It can be deferred if core inference is not yet stable.
