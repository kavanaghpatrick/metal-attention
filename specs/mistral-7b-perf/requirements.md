---
spec: mistral-7b-perf
phase: requirements
created: 2026-02-16
generated: auto
---

# Requirements: mistral-7b-perf

## Summary

Close 42 -> 100+ tok/s gap on Mistral-7B Q4_0 decode via four optimizations: Q6_K native kernel (-404 MB bandwidth), page-aligned zero-copy buffers (-3.8 GB memcpy at load), GQA batch prefill (34 -> 500+ tok/s prefill), and speculative decoding scaffolding (2-3x effective decode).

## User Stories

### US-1: Native Q6_K Metal Matvec Kernel
As a developer running Mistral-7B Q4_0, I want the lm_head to use a native Q6_K matvec kernel instead of dequantized F32, so that per-token bandwidth drops by 404 MB and decode throughput increases by 5-8 tok/s.

**Acceptance Criteria**:
- AC-1.1: `matvec_q6_k` output matches CPU `dequantize_q6_k_to_f32` + F32 matvec within atol=5e-2 on 4096x32000 matrix
- AC-1.2: lm_head stored as raw Q6_K (108 MB) instead of F32 (512 MB) -- buffer size assertion
- AC-1.3: Mistral-7B decode tok/s increases by >= 5 (42 -> 47+) with Q6_K as only change
- AC-1.4: No regression on SmolLM-135M or any model without Q6_K tensors
- AC-1.5: `multi_token_matvec_q6_k` batched variant for prefill and speculative verify

### US-2: Page-Aligned Zero-Copy Weight Buffers
As a developer loading any GGUF model, I want weight buffers to use zero-copy Metal buffers via page-offset alignment, so that model loading eliminates 3.8 GB memcpy and memory usage drops.

**Acceptance Criteria**:
- AC-2.1: `WeightBuffer{buffer, offset}` struct replaces raw MTLBuffer fields in weight store
- AC-2.2: All encode methods propagate offset through `set_buffer(encoder, buf, offset, idx)`
- AC-2.3: Zero-copy tensors produce bit-identical inference output vs copy-based path
- AC-2.4: Graceful fallback to copy when mmap region doesn't extend far enough for page padding
- AC-2.5: No correctness regression on SmolLM or Mistral-7B

### US-3: Batch Prefill for Mistral-7B GQA
As a developer running Mistral-7B with multi-token prompts, I want `forward_prompt()` to process all prompt tokens in batched GPU dispatches, so that prefill throughput reaches 500+ tok/s.

**Acceptance Criteria**:
- AC-3.1: `forward_prompt()` on Mistral-7B processes 128-token prompt at >= 100 tok/s
- AC-3.2: GQA (32Q/8KV heads) handled correctly -- K/V use kv_dim, Q uses q_dim
- AC-3.3: Output token from batched prefill matches single-token loop (greedy argmax identity)
- AC-3.4: KV cache correctly populated after batch prefill (coherent decode follows)
- AC-3.5: Q6_K lm_head path used in `forward_prompt()` for Mistral-7B

### US-4: Speculative Decoding Scaffolding
As a developer seeking higher effective decode throughput, I want a `SpeculativeDecoder` struct that drafts N=8 tokens with a small model and verifies with the target model in one batched pass, so that effective decode can reach 80-120 tok/s.

**Acceptance Criteria**:
- AC-4.1: `SpeculativeDecoder::new(draft_path, target_path, n_draft)` loads two models sharing one Metal device
- AC-4.2: Greedy speculative decode (temp=0) produces output identical to target-only greedy decode
- AC-4.3: KV cache rollback via O(1) `truncate(new_len)` on both draft and target models
- AC-4.4: `forward_prompt_logits()` returns logits for ALL positions (not just last token)
- AC-4.5: CLI flags `--draft <PATH>` and `--draft-tokens N` added to inference entry point
- AC-4.6: Graceful fallback to standard decode when all draft tokens rejected

## Functional Requirements

| ID | Requirement | Priority | Source |
|----|-------------|----------|--------|
| FR-1 | `matvec_q6_k.metal` shader: fused dequant + dot product for Q6_K superblocks. 256 threads, 8 simdgroups, 8 rows/TG. | P0 | US-1 |
| FR-2 | `multi_token_matvec_q6_k.metal`: batched Q6_K matvec for prefill/verify | P0 | US-1 |
| FR-3 | `GpuWeightStore` stores Q6_K lm_head as raw bytes; `lm_head_q6k` field + accessor | P0 | US-1 |
| FR-4 | `WeightBuffer{buffer, offset}` struct; `make_weight_buffer_aligned()` using page-offset trick | P0 | US-2 |
| FR-5 | All `encode_*` methods propagate `WeightBuffer.offset` through `set_buffer` calls | P0 | US-2 |
| FR-6 | `forward_prompt()` works for Mistral-7B dimensions (hidden=4096, 32Q/8KV, head_dim=128, ffn=14336) | P0 | US-3 |
| FR-7 | Q6_K lm_head dispatch in both `forward_token()` and `forward_prompt()` | P0 | US-3 |
| FR-8 | `GpuKVCache::truncate(new_len)` for O(1) rollback | P1 | US-4 |
| FR-9 | `GpuForwardPass::rollback_to(position)` resets position + all KV caches | P1 | US-4 |
| FR-10 | `forward_prompt_logits()` returning logits for all input positions | P1 | US-4 |
| FR-11 | `SpeculativeDecoder` struct with `generate()`, `reset()`, greedy accept/reject loop | P1 | US-4 |
| FR-12 | CLI `--draft` and `--draft-tokens` flags in inference entry point | P1 | US-4 |

## Non-Functional Requirements

| ID | Requirement | Category |
|----|-------------|----------|
| NFR-1 | Mistral-7B decode >= 47 tok/s after Q6_K (target-only) | Performance |
| NFR-2 | Mistral-7B prefill >= 100 tok/s at 128-token prompt | Performance |
| NFR-3 | Speculative effective decode >= 80 tok/s (when draft model compatible) | Performance |
| NFR-4 | Q6_K matvec kernel < 1e-3 atol vs CPU dequant; < 5e-2 atol vs F32 matvec | Correctness |
| NFR-5 | Greedy speculative output bit-identical to target-only output | Correctness |
| NFR-6 | Model load time < 1s for Mistral-7B with page-aligned buffers | Performance |
| NFR-7 | No regression on existing 324 tests | Compatibility |

## Out of Scope

- Other quant kernels (Q4_K_M, Q5_K, IQ formats)
- Eagle/Medusa tree-based speculative decoding
- Temperature > 0 speculative sampling (greedy-only for Phase 1)
- Multi-GPU / distributed inference
- Actual draft model selection (scaffolding only; tokenizer mismatch unresolved)
- HTTP/API serving

## Dependencies

- Mistral-7B Q4_0 GGUF file for testing (3.8 GB)
- SmolLM-135M Q4_0 GGUF file for draft model testing (87 MB)
- Existing `forward_prompt()` infrastructure
- Existing matvec kernel patterns (`matvec_q8_0`, `multi_token_matvec_q4_0`)
