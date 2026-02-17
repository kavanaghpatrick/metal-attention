---
spec: rwkv7-inference
phase: requirements
created: 2026-02-16
generated: auto
---

# Requirements: rwkv7-inference

## Summary
Implement GPU inference for RWKV-7 7.2B model on Apple Silicon M4 Pro using Metal compute kernels. Load Q4_0 quantized GGUF model, perform recurrent decode with WKV-7 operator, generate text via tokenizer. Measure and benchmark token/second throughput.

## User Stories

### US-1: Load RWKV-7 GGUF model weights
As a developer, I want to load RWKV-7 7.2B Q4_0 GGUF model into Metal buffers so that I can run GPU inference.

**Acceptance Criteria**:
- AC-1.1: GpuRwkv7ForwardPass::from_gguf() loads all 32 layer weights (~30 tensors per layer)
- AC-1.2: Embedding table (vocab_size=65536, hidden=4096) loaded as GPU buffer
- AC-1.3: LM head projection loaded (65536 x 4096)
- AC-1.4: All LayerNorm weights loaded (66 norms: 1 pre-embed + 2*32 per-layer + 1 final)
- AC-1.5: LoRA projection tensors loaded (w0/w1/w2 for decay, alpha, v-blend, gate per layer)
- AC-1.6: Zero-copy mmap buffers used where page-aligned (similar to existing GpuWeightStore)

### US-2: Single token forward pass with simplified WKV
As a developer, I want to run a single token through all 32 RWKV-7 layers on GPU so that I can validate the inference pipeline.

**Acceptance Criteria**:
- AC-2.1: forward_token(token_id) returns logits [vocab_size]
- AC-2.2: Embedding lookup kernel runs on GPU
- AC-2.3: Token shift mixing (mix_r, mix_k, mix_v, mix_w, mix_a, mix_g) computed per layer
- AC-2.4: 6 full-rank matvec operations per layer execute on GPU (R, K, V, W, O, plus FFN K/V)
- AC-2.5: Simplified WKV kernel (existing rwkv_wkv.metal) computes state update
- AC-2.6: Channel mix FFN (squared ReLU) executes on GPU
- AC-2.7: LayerNorm operations (4 per layer: pre-layer, attn_norm, time_mix GroupNorm, ffn_norm) on GPU
- AC-2.8: Final norm + LM head matvec produce logits
- AC-2.9: Output shape [65536] with finite values (no NaN/Inf)

### US-3: Full WKV-7 recurrence with delta rule
As a developer, I want the WKV kernel to implement full RWKV-7 recurrence so that the model produces correct outputs.

**Acceptance Criteria**:
- AC-3.1: Data-dependent decay: w = exp(-0.606531 * sigmoid(w0 + tanh(xw @ w1) @ w2))
- AC-3.2: In-context learning rate: a = sigmoid(a0 + (xa @ a1) @ a2)
- AC-3.3: Delta rule self-modification: ab = (-kk)^T @ (kk * a)
- AC-3.4: Full state update: state = diag(w)*state + state@ab + v^T*k
- AC-3.5: GroupNorm applied to WKV output
- AC-3.6: Bonus r*k attention term computed
- AC-3.7: Gated output projection: output = sigmoid(g) * (output @ O)
- AC-3.8: CPU reference implementation matches GPU within 1e-3 tolerance

### US-4: LoRA projection kernels
As a developer, I want LoRA projections (low-rank 4096 -> 64 -> 4096) to run on GPU so that WKV-7 can compute data-dependent parameters.

**Acceptance Criteria**:
- AC-4.1: Decay LoRA: w = w0 + (xw @ w1) @ w2 (rank 64)
- AC-4.2: Alpha LoRA: a = a0 + (xa @ a1) @ a2 (rank 64)
- AC-4.3: V-blend LoRA: v_mix = v0 + (xv @ v1) @ v2 (rank 32)
- AC-4.4: Gate LoRA: g = (xg @ g1) @ g2 (rank 128)
- AC-4.5: Small matvec kernel optimized for ranks 32-128 (reuse or adapt existing Q4_0 matvec)
- AC-4.6: All LoRA results have correct shape and finite values

### US-5: End-to-end text generation
As a user, I want to generate text from a prompt using RWKV-7 7.2B on GPU so that I can test the complete inference pipeline.

**Acceptance Criteria**:
- AC-5.1: Tokenizer loaded from GGUF metadata (RWKV uses rwkv_vocab_v20230424.txt)
- AC-5.2: Prompt tokenized to token IDs
- AC-5.3: Prefill: all prompt tokens processed sequentially (recurrent state updated)
- AC-5.4: Decode: autore gressive generation up to max_tokens or EOS
- AC-5.5: Greedy sampling (argmax) produces next token
- AC-5.6: Generated tokens decoded to text
- AC-5.7: Text output is coherent (passes sanity check: no repetition loops, grammatical)

### US-6: Performance benchmarking
As a developer, I want to measure RWKV-7 GPU inference throughput so that I can compare vs CPU and other implementations.

**Acceptance Criteria**:
- AC-6.1: Benchmark command: `metal-attention bench -m model.gguf --gpu`
- AC-6.2: Measure decode tok/s (exclude prefill from measurement)
- AC-6.3: Report: tokens generated, decode time, tok/s
- AC-6.4: Compare GPU vs CPU HybridModel path (existing RWKV-7 CPU implementation)
- AC-6.5: Optional: compare vs llama.cpp if RWKV-7 support exists

## Functional Requirements

| ID | Requirement | Priority | Source |
|----|-------------|----------|--------|
| FR-1 | Load RWKV-7 7.2B Q4_0 GGUF model to Metal buffers | Must | US-1 |
| FR-2 | Detect RWKV architecture from GGUF metadata/tensors | Must | US-1 |
| FR-3 | Map 30+ RWKV-7 tensor suffixes to WeightRole enum | Must | US-1 |
| FR-4 | Single token forward pass returns logits [vocab_size] | Must | US-2 |
| FR-5 | GPU embedding lookup kernel | Must | US-2 |
| FR-6 | GPU Q4_0 matvec for 6 full-rank projections per layer | Must | US-2 |
| FR-7 | GPU simplified WKV kernel (state = w*state + k^T*v) | Must | US-2 |
| FR-8 | GPU channel mix FFN (squared ReLU + gate/value matvecs) | Must | US-2 |
| FR-9 | GPU LayerNorm kernel (or reuse RMSNorm with bias) | Must | US-2 |
| FR-10 | Full WKV-7 recurrence (delta rule + data-dependent decay) | Must | US-3 |
| FR-11 | GPU GroupNorm kernel for WKV output | Must | US-3 |
| FR-12 | GPU LoRA matvec kernel (4096 -> 32-128 -> 4096) | Must | US-4 |
| FR-13 | RWKV tokenizer integration from GGUF | Must | US-5 |
| FR-14 | Autoregressive decode loop with recurrent state | Must | US-5 |
| FR-15 | Greedy sampling (GPU-side argmax or CPU fallback) | Must | US-5 |
| FR-16 | Benchmark command with --gpu flag | Must | US-6 |
| FR-17 | Tok/s measurement and reporting | Must | US-6 |
| FR-18 | CPU reference validation for each GPU kernel | Should | US-3, US-4 |
| FR-19 | Error handling for invalid GGUF files | Should | US-1 |
| FR-20 | Temperature/top-p/top-k sampling modes | Could | US-5 |

## Non-Functional Requirements

| ID | Requirement | Category |
|----|-------------|----------|
| NFR-1 | Decode throughput > CPU HybridModel path | Performance |
| NFR-2 | GPU kernel accuracy within 1e-3 of CPU reference | Correctness |
| NFR-3 | Model loading < 5 seconds (zero-copy mmap) | Performance |
| NFR-4 | Memory usage ≤ 5 GB (4.44 GB model + 32 MB state + overhead) | Resource |
| NFR-5 | Compilation time < 60 seconds (PSO cache) | Developer Experience |
| NFR-6 | Compatible with Apple Family 7+ (M1, M2, M3, M4) | Compatibility |
| NFR-7 | No CUDA/non-Metal dependencies | Portability |
| NFR-8 | CLI matches existing Run/Bench/Info pattern | Usability |

## Out of Scope
- Batch inference (batch_size > 1)
- Prompt caching / KV cache (RWKV is recurrent, no KV cache)
- Multi-GPU or distributed inference
- Training or fine-tuning
- BF16/FP16 native precision (Q4_0 only for POC)
- RWKV-5/6 compatibility (RWKV-7 only)
- Alternative attention mechanisms (RWKV-7 WKV only)

## Dependencies
- RWKV-7 7.2B Q4_0 GGUF model file (user-provided)
- Existing metal-attention infrastructure (GpuDevice, PsoCache, buffer helpers)
- Existing kernels: Q4_0 matvec, RMSNorm, embedding, argmax
- New kernels: GroupNorm, LoRA small matvec (if RMSNorm/Q4_0 matvec not adaptable)
