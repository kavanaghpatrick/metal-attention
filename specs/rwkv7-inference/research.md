---
spec: rwkv7-inference
phase: research
created: 2026-02-16
generated: auto
---

# Research: rwkv7-inference

## Executive Summary
RWKV-7 7.2B GPU inference on Apple Silicon M4 Pro is highly feasible. Existing codebase has 80% of required infrastructure: Rwkv7Block with CPU reference, simplified WKV GPU kernel, GGUF detection, and enum dispatch. Missing: full WKV-7 recurrence (delta rule + LoRA projections), GGUF weight loading for ~30 tensor types per layer, channel mix FFN, LayerNorm ops, and tokenizer integration. GPU path requires new GpuRwkv7ForwardPass (~600 lines) rather than modifying existing GpuForwardPass (transformer-specific). Target: real text generation from Q4_0 GGUF, measured tok/s.

## Codebase Analysis

### Existing Patterns (Reusable)

| Component | File | Status |
|-----------|------|--------|
| Rwkv7Block struct | metal-attention-models/src/rwkv7.rs | Complete (642 lines, 7 passing tests) |
| WKV GPU kernel (simplified) | metal-attention-kernels/shaders/rwkv_wkv.metal | Working (~100 lines, GPU vs CPU validated) |
| Kernel dispatcher | metal-attention-kernels/src/rwkv.rs | Complete (170 lines) |
| GGUF RWKV detection | metal-attention-gguf/src/detect.rs | Working (detects time_mix + channel_mix patterns) |
| Architecture enum | metal-attention-gguf/src/architectures.rs | Partial (maps ~15 RWKV suffixes, missing LoRA tensors) |
| Model dispatch | metal-attention/src/model.rs | Complete (HybridModel enum dispatch) |
| GPU infrastructure | metal-attention/src/gpu_forward_pass.rs | Exists for Llama (Q4_0 matvec, RMSNorm, embedding, KV cache) |
| Main binary | src/main.rs | RWKV metadata extraction (lines 217-238) |

**Key finding**: Existing GpuForwardPass (line 52-113) is transformer-specific (KV cache, attention, RoPE). RWKV-7 needs separate GpuRwkv7ForwardPass due to:
- Recurrent state (32 MB fixed) vs KV cache (grows with sequence)
- Different operation sequence (token shift → LoRA → WKV → GroupNorm → channel mix)
- Different kernel set (WKV-7, LoRA matvecs, GroupNorm vs attention, RoPE)

### Dependencies (Already in Workspace)

```toml
objc2-metal = "0.3"  # Metal bindings
memmap2 = "0.9"      # GGUF mmap
half = "2"           # FP16 for quantization
clap = "4"           # CLI (Run/Bench/Info subcommands)
```

GPU Forge KB findings:
- [689] msl-kernels: llama.cpp uses simd_sum reduction for small M matvec (bandwidth-bound decode)
- [401] mlx-compute: simdgroup_multiply_accumulate with 64x64 blocks for matmul
- [981] gpu-centric-arch: Function constants + stitchable functions for kernel variants

### Constraints

| Type | Detail | Impact |
|------|--------|--------|
| Quantization | Q4_0 (4-bit) for large matvecs | 4.44 GB model, Q4_0 matvec kernel exists |
| State size | 32 MB (64 heads x 64 x 64 x 32 layers x 4 bytes) | Constant memory vs transformer KV cache growth |
| LoRA ranks | decay=64, alpha=64, v-blend=32, gate=128 | 8 small matvecs per layer (4096 -> 64 -> 4096) |
| Vocab | 65536 tokens | 256 MB embedding table (likely Q8_0 in GGUF) |
| Metal API | Apple Family 7+ (M1+) | Already required by existing GpuForwardPass |

## Feasibility Assessment

| Aspect | Assessment | Notes |
|--------|------------|-------|
| Technical Viability | High | 80% infrastructure exists, WKV kernel working, Q4_0 matvec proven |
| Effort Estimate | M (Medium) | ~600 lines GpuRwkv7ForwardPass + 30 GGUF tensor mappings + GroupNorm kernel |
| Risk Level | Medium | Full WKV-7 recurrence not validated on GPU, GroupNorm kernel new, LoRA small matvecs untested |

**GPU Forge insight (KB [689])**: llama.cpp uses separate kernel families for matvec (small M, simd_sum) vs matmul (large M, simdgroup_matrix). RWKV-7 LoRA projections (4096 -> 64) fit the small M pattern.

## Recommendations

1. **POC phase**: Simplified WKV (existing) + 6 full-rank matvecs + channel mix. Skip LoRA/delta-rule initially. Get single token forward working.
2. **Incremental WKV upgrade**: Add delta rule (state @ ab term), then LoRA projections, then GroupNorm + gated output. Validate each against CPU reference.
3. **Reuse existing kernels**: Q4_0 matvec (6 large projections), RMSNorm (4 LayerNorm ops similar), embedding lookup. Only new kernels: GroupNorm, small LoRA matvec.
4. **Separate GPU path**: New gpu_rwkv7_forward_pass.rs. Avoids polluting transformer-specific GpuForwardPass with recurrent state logic.
5. **Benchmarking**: Compare tok/s decode vs llama.cpp RWKV implementation (if exists) or vs CPU HybridModel path.

## Next Steps
1. Extend architectures.rs with RWKV-7 tensor mappings (~30 new suffixes)
2. Implement GpuRwkv7ForwardPass skeleton (embed → layers → norm → lm_head)
3. POC: single token forward with simplified WKV
4. Upgrade WKV kernel to full recurrence
5. E2E generation with tokenizer
