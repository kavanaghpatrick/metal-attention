---
id: kernels.BREAKDOWN
module: kernels
priority: 2
status: failing
version: 1
origin: spec-workflow
dependsOn: [devops.BREAKDOWN, traits.BREAKDOWN]
tags: [breakdown]
testRequirements:
  unit:
    required: false
    pattern: "tests/kernels/**/*.test.*"
---
# Kernels -- Breakdown

## Context

The kernels crate is the GPU compute engine of metal-attention. It contains all hand-written MSL shaders, the Metal device management, PSO (Pipeline State Object) cache with function constant specialization, buffer pool for memory management, and command buffer triple-buffering. This crate evolved from 8 validated prototypes that proved key metrics: 178ns PSO cache hit, 34-63us cold compile, 0% function constant overhead, simdgroup_matrix-based flash attention, FLA linear attention with chunk_h/chunk_o, PagedAttention V2, RoPE, ALiBi, and GQA. The crate is the only one that links Metal frameworks.

## Scope

- **Metal device management**: `GpuDevice` singleton with `MTLDevice`, `MTLCommandQueue`, `MTLLibrary`
- **Pipeline management**: `PsoCache` with `PsoKey` composition, `get_or_compile()`, and `prewarm()` for model-specific variant pre-compilation
- **Buffer management**: `BufferPool` with size-class allocation and reuse; `create_weight_buffer()` for zero-copy mmap'd GGUF tensors
- **Command management**: `CommandManager` with triple-buffering via `dispatch_semaphore(3)`, `begin_frame()`/`end_frame()`/`drain()`
- **MSL shaders** (13 files):
  - `flash_attention.metal` -- Tiled flash attention with simdgroup_matrix, online softmax
  - `linear_attention.metal` -- FLA chunk_h/chunk_o kernels
  - `paged_attention.metal` + `paged_reduce.metal` -- PagedAttention V2
  - `rope.metal` -- Rotary position encoding
  - `gqa_remap.metal` -- Grouped-query attention head remapping
  - `rmsnorm.metal` -- RMS normalization
  - `ffn.metal` -- Feed-forward network (SwiGLU, ReLU^2, GeGLU)
  - `embedding.metal` -- Token embedding lookup
  - `matmul.metal` -- General matrix multiply with tiling
  - `dequantize.metal` -- Q4_0, Q4_K_M, Q8_0 block dequantization
  - `ssm_scan.metal` -- Selective state space scan (Mamba)
  - `prefix_sum.metal` -- GPU parallel prefix sum over D*D matrices
- **Shared header**: `types.h` with `AttentionParams`, `LayerParams`, `SSMParams` `#repr(C)` structs
- **Dispatch helpers**: Per-kernel Rust dispatch functions (flash, linear, paged, rope, gqa, norm, ffn, embed, matmul, dequant, ssm)

## Key Decisions

- **From TECH.md**: Function constants provide zero-overhead kernel specialization. PsoKey composition per kernel encodes HEAD_DIM, BLOCK_R, BLOCK_C, CHUNK_SIZE, PAGE_SIZE, ALIBI_ENABLED, HIDDEN_SIZE, FFN_TYPE, QUANT_TYPE etc.
- **From TECH.md**: `build.rs` compiles shaders via `xcrun -sdk macosx metal -std=metal3.1 -c` to `.air`, then links to single `shaders.metallib`. Release uses `-O2`, debug uses `-gline-tables-only`.
- **From TECH.md**: 32KB threadgroup memory is a hard limit across all M-series. Valid tile sizes documented: Flash D=64 uses (Br=16,Bc=64) or (Br=32,Bc=64); D=128 uses (Br=16,Bc=16) or (Br=16,Bc=32).
- **From TECH.md**: Buffer creation uses `newBufferWithBytesNoCopy` when 32-byte aligned (zero-copy from mmap), fallback to `newBufferWithBytes` for unaligned data.
- **From TECH.md**: Triple buffering with 3 in-flight command buffers. Per-frame activation buffers cycle through frame indices 0-2.
- **From TECH.md**: Proto shaders are copied and extended (add multi-simdgroup, causal mask, multi-head dispatch). Proto device.rs, pipeline.rs, encode.rs are ported with new imports.

## Acceptance Criteria

1. `GpuDevice` initializes Metal device, command queue, and loads compiled metallib
2. `PsoCache::get_or_compile()` returns valid PSO for flash attention with function constants; cache hit < 1us
3. `PsoCache::prewarm()` pre-compiles all kernel variants for a given `ModelConfig`
4. `BufferPool::acquire()`/`release()` allocates and reuses Metal buffers by size class
5. `CommandManager` supports triple-buffered frame submission with semaphore gating
6. Flash attention shader compiles and produces output matching CPU FP64 reference (atol=5e-3)
7. Linear attention shader (chunk_h + chunk_o) compiles and produces correct output (atol=1e-3)
8. RoPE shader produces correct output (atol=1e-4)
9. RMSNorm shader produces correct output (atol=1e-5)
10. At least one dequantization kernel (Q4_0) compiles and round-trips correctly
11. `types.h` shared structs match Rust-side `#[repr(C)]` layouts exactly (verified by size assertions)
12. All shaders compile without errors via `build.rs` on macOS with Xcode installed

## Technical Notes

- From TECH.md: `AttentionParams` is 64 bytes, 4-byte aligned. `LayerParams` is 64 bytes. `SSMParams` is 32 bytes. Size/alignment must be verified with compile-time assertions.
- From QA.md: All GPU tests must run with `MTL_SHADER_VALIDATION=1`. Correctness tolerances: Flash 5e-3, Linear 1e-3, Paged 1e-3, RoPE 1e-4, GQA 1e-6 (exact). Memory leak tests: <1% growth over 100 iterations.
- From QA.md: GPU warmup (8 throwaway dispatches) before benchmarks. Timing via `MTLCommandBuffer::GPUStartTime/GPUEndTime`, not wall-clock.
- From TECH.md: Kernel fusion priorities -- GPU prefix sum first (eliminates ~300us CPU bottleneck), then multi-simdgroup flash attention (target >1 TFLOPS), then fused QKV projection.
- From TECH.md: Kernels NOT fused (kept separate): RoPE (10us/head, negligible), GQA remap (pure memory copy), embedding lookup (one-time per token).
