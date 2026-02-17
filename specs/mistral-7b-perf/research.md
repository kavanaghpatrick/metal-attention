---
spec: mistral-7b-perf
phase: research
created: 2026-02-16
generated: auto
---

# Research: mistral-7b-perf

## Executive Summary

Close 42 -> 100+ tok/s gap on Mistral-7B Q4_0 decode (M4 Pro, 273 GB/s) via four targeted optimizations: Q6_K native matvec kernel, page-aligned zero-copy weight buffers, GQA batch prefill, and speculative decoding scaffolding. Feasibility is HIGH -- all four build on proven patterns in the existing codebase.

## Codebase Analysis

### Existing Patterns

- **Matvec kernels**: `matvec_q4_0_v5_coalesced.metal`, `matvec_q8_0.metal`, `matvec_f32_v2.metal` all use 256 threads (8 simdgroups, ROWS_PER_TG=8), `simd_sum` reduction, buffer binding convention (weight=0, input=1, output=2, out_dim=3, in_dim=4). Q6_K kernel follows same pattern with different dequant math.
- **Multi-token kernels**: `multi_token_matvec_q4_0.metal` adds batch_size param at buffer(5), loops over tokens within same TG. Q6_K batched variant follows same structure.
- **Weight loading**: `gpu_weight_store.rs` handles Q4_0/Q8_0/F32/Q5_K/Q6_K. Q6_K lm_head currently dequantized to F32 at load (line 520), creating 512 MB bandwidth overhead per token.
- **Page alignment**: `make_weight_buffer()` (line 97-128) checks `ptr.is_multiple_of(page_size)`, falls back to copy for non-aligned. Most GGUF tensors are 32-byte aligned, not 4096-byte aligned.
- **PSO prewarm**: All kernels prewarmed in `from_gguf()` (lines 272-299). New kernels must be added here.
- **KV cache**: `GpuKVCache` in `gpu_kv_cache.rs` has `len` counter, `reset()` method. No `truncate()` yet -- needed for speculative decode rollback.
- **Batch prefill**: `forward_prompt()` (line 545) already works for SmolLM. Uses `multi_token_matvec_q4_0`, per-token RoPE and attention. GQA-parameterized via `num_heads`/`num_kv_heads`.
- **lm_head dispatch chain**: Priority: Q8_0 > F32 > Q4_0 (lines 495-520 in forward_token, lines 857-880 in forward_prompt). Q6_K slot missing.

### Dependencies

- `objc2-metal` for Metal buffer/encoder APIs
- `metal_attention_gguf::GgufFile` for mmap tensor access
- `metal_attention_kernels::buffer::create_weight_buffer` for `newBufferWithBytesNoCopy`
- `metal_attention_kernels::pipeline::PsoCache` for PSO management
- `half` crate for fp16 handling
- Existing Rust `dequantize_q6_k_to_f32` in `gpu_weight_store.rs` (lines 205-262) as reference implementation

### Constraints

- Apple Silicon only (Apple Family 7+ for `simd_sum`)
- GGUF `general.alignment` typically 32 bytes, not 4096 -- page alignment requires offset-based approach
- Metal `newBufferWithBytesNoCopy` requires page-aligned pointer + page-multiple length
- Speculative decode requires matching tokenizer between draft and target models (SmolLM=49K vocab != Mistral=32K vocab -- blocker for actual deployment, scaffolding only)
- `GpuKVCache` needs `truncate()` for rollback (O(1): just adjust `len`)

## Bandwidth Budget

| Component | Bytes/Token | % Total |
|-----------|-------------|---------|
| 32x attention Q/K/V/O (Q4_0) | 1,830 MB | 47.2% |
| 32x FFN gate/up/down (Q4_0) | 1,530 MB | 39.5% |
| lm_head F32 (current) | 512 MB | 13.2% |
| lm_head Q6_K (after opt 1) | 108 MB | 3.1% |
| Norms, KV, embeddings | 3 MB | 0.1% |
| **Total (current)** | **3,875 MB** | |
| **Total (after Q6_K)** | **3,471 MB** | **-10.4%** |

Theoretical max at 273 GB/s: 70 tok/s (current), 79 tok/s (after Q6_K).

## Feasibility Assessment

| Aspect | Assessment | Notes |
|--------|------------|-------|
| Q6_K kernel | HIGH | Direct copy of Q8_0 pattern; dequant math already in Rust |
| Page-aligned buffers | HIGH | `WeightBuffer{buf,offset}` wrapper; offset propagation through all encode methods |
| Batch prefill GQA | HIGH | `forward_prompt()` already works for SmolLM; Mistral just has larger dims + GQA (already parameterized) |
| Speculative scaffolding | MEDIUM | New `speculative.rs` module; KV rollback trivial; full draft model selection unresolved |
| Overall effort | L (12-18 days) | 4 independent work streams |
| Risk level | MEDIUM | Spec decode highest risk (tokenizer mismatch); others LOW |

## Recommendations

1. Ship Q6_K kernel first -- lowest risk, immediate bandwidth win, blocks batch prefill Q6_K lm_head path
2. Page-aligned buffers can be done in parallel -- independent of kernel work
3. Batch prefill likely works already for Mistral; test first, fix only what's broken
4. Speculative decode is scaffolding only -- implement struct + loop, defer draft model resolution
