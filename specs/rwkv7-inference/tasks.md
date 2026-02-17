---
spec: rwkv7-inference
phase: tasks
total_tasks: 32
created: 2026-02-16
generated: auto
---

# Tasks: rwkv7-inference

## Phase 1: Make It Work (POC)

Focus: Single token forward pass with simplified WKV. Skip LoRA/delta-rule. Validate end-to-end pipeline.

- [ ] 1.1 Extend GGUF tensor mappings for RWKV-7
  - **Do**: Add 30+ WeightRole enum variants to architectures.rs (TimeMixLerpFused, TimeMixW0/W1/W2, TimeMixA0/A1/A2, TimeMixV0/V1/V2, TimeMixG1/G2, TimeMixKK/KA/RK, TimeMixLnWeight/LnBias, ChannelMixLerpK, TokenEmbdNorm, OutputNorm). Extend map_rwkv_suffix() with all cases. Add unit tests for each mapping.
  - **Files**: crates/metal-attention-gguf/src/architectures.rs
  - **Done when**: All 30+ tensor name patterns map correctly, cargo test passes
  - **Verify**: cargo test --package metal-attention-gguf architectures
  - **Commit**: `feat(gguf): add RWKV-7 tensor mapping for 30+ weight types`
  - _Requirements: FR-1, FR-2, FR-3_
  - _Design: Component C (GGUF Tensor Mapping Extension)_

- [ ] 1.2 Create Rwkv7WeightStore struct skeleton
  - **Do**: Add Rwkv7WeightStore to gpu_weight_store.rs with fields for per-layer buffers (time_mix_lerp_fused, w0/w1/w2, a0/a1/a2, v0/v1/v2, g1/g2, k_k/k_a/r_k, time_mix_ln_weight/bias, channel_mix_lerp_k/key/value, attn_norm/ffn_norm) and shared buffers (token_embd, token_embd_norm, output_norm, output). Implement from_gguf() method to load tensors using existing create_weight_buffer(). Add _gguf: Arc<GgufFile> to keep mmap alive.
  - **Files**: crates/metal-attention/src/gpu_weight_store.rs
  - **Done when**: Struct compiles, from_gguf() loads all tensors without panics
  - **Verify**: cargo check --package metal-attention
  - **Commit**: `feat(gpu): add Rwkv7WeightStore for GGUF weight loading`
  - _Requirements: FR-1_
  - _Design: Component B (GpuWeightStoreRwkv7)_

- [ ] 1.3 Create GpuRwkv7ForwardPass skeleton with buffer allocation
  - **Do**: Create crates/metal-attention/src/gpu_rwkv7_forward_pass.rs with GpuRwkv7ForwardPass struct. Add fields: device, pso_cache, weight_store, hidden_a/hidden_b (ping-pong), scratch buffers (q/k/v/attn_out/gate/up/silu/ffn), logits_buf, argmax buffers, recurrent_state (32 layers x num_heads x head_dim x head_dim). Implement from_gguf() to parse metadata, load weights, allocate all buffers.
  - **Files**: crates/metal-attention/src/gpu_rwkv7_forward_pass.rs, crates/metal-attention/src/lib.rs
  - **Done when**: GpuRwkv7ForwardPass::from_gguf() allocates all buffers, compiles
  - **Verify**: cargo check --package metal-attention
  - **Commit**: `feat(gpu): add GpuRwkv7ForwardPass skeleton with buffer allocation`
  - _Requirements: FR-1, FR-4_
  - _Design: Component A (GpuRwkv7ForwardPass)_

- [ ] 1.4 Implement embedding lookup for RWKV-7
  - **Do**: In GpuRwkv7ForwardPass, add embed_token() method to dispatch embedding kernel. Reuse existing embed.rs dispatch_embedding() with Q8_0 dequant. Handle vocab_size=65536. Write to hidden_a buffer.
  - **Files**: crates/metal-attention/src/gpu_rwkv7_forward_pass.rs
  - **Done when**: Single token embedding lookup produces [4096] F32 vector on GPU
  - **Verify**: Unit test: embed known token, readback hidden_a, verify non-zero
  - **Commit**: `feat(gpu): implement RWKV-7 embedding lookup with Q8_0 dequant`
  - _Requirements: FR-5_
  - _Design: Component A (GpuRwkv7ForwardPass)_

- [ ] 1.5 Implement token shift mixing
  - **Do**: Add token_shift kernel to mix current input with prev_token using lerp factors (mix_r, mix_k, mix_v, mix_w, mix_a, mix_g). Create shaders/token_shift.metal with kernel signature. Dispatch in forward_layer() before matvecs. Use time_mix_lerp_fused buffer (6 lerp values).
  - **Files**: crates/metal-attention-kernels/shaders/token_shift.metal, crates/metal-attention-kernels/src/token_shift.rs, crates/metal-attention/src/gpu_rwkv7_forward_pass.rs
  - **Done when**: Token shift produces 6 mixed vectors (r_shift, k_shift, v_shift, w_shift, a_shift, g_shift)
  - **Verify**: Unit test: token_shift with known inputs/prev, verify output = lerp formula
  - **Commit**: `feat(gpu): add token shift kernel for RWKV-7 time mixing`
  - _Requirements: FR-4_
  - _Design: Component A, Data Flow step 2_

- [ ] 1.6 Wire 6 full-rank Q4_0 matvecs per layer
  - **Do**: In forward_layer(), dispatch 6 Q4_0 matvecs using existing matvec_q4_0 kernel: R (attn_q), K (attn_k), V (attn_v), W (time_mix_w), O (attn_output), plus FFN channel_mix_key and channel_mix_value. Use Rwkv7WeightStore buffers. Write outputs to scratch_q/k/v buffers. Apply sigmoid to W output (decay must be in (0,1)).
  - **Files**: crates/metal-attention/src/gpu_rwkv7_forward_pass.rs
  - **Done when**: All 6 matvecs execute, outputs finite
  - **Verify**: Unit test: forward_layer with random input, verify 6 scratch buffers non-zero
  - **Commit**: `feat(gpu): wire 6 Q4_0 matvecs per RWKV-7 layer`
  - _Requirements: FR-6_
  - _Design: Component A, Data Flow step 2_

- [ ] 1.7 Dispatch simplified WKV kernel (reuse existing)
  - **Do**: In forward_layer(), call dispatch_rwkv_wkv() from existing rwkv.rs with r/k/v/w buffers and per-layer recurrent state slice. Update state in-place. Write output to attn_out buffer. Simplified WKV: state = w*state + k^T*v, output = r*state.
  - **Files**: crates/metal-attention/src/gpu_rwkv7_forward_pass.rs
  - **Done when**: WKV kernel executes, state updated, output finite
  - **Verify**: Unit test: dispatch WKV with known r/k/v/w, compare output vs CPU cpu_rwkv_wkv
  - **Commit**: `feat(gpu): integrate simplified WKV kernel in RWKV-7 forward pass`
  - _Requirements: FR-7_
  - _Design: Component A, Data Flow step 2_

- [ ] 1.8 Implement channel mix FFN with squared ReLU
  - **Do**: Add channel_mix kernel to ffn.metal. Compute: k = relu(k_input @ K_weight)^2, output = k @ V_weight. K_weight: 4096 → 16384 (Q4_0), V_weight: 16384 → 4096 (Q4_0). Dispatch from forward_layer() after WKV. Use channel_mix_lerp_k for token shift on FFN input.
  - **Files**: crates/metal-attention-kernels/shaders/ffn.metal, crates/metal-attention-kernels/src/ffn.rs, crates/metal-attention/src/gpu_rwkv7_forward_pass.rs
  - **Done when**: Channel mix produces [4096] output with squared ReLU activation
  - **Verify**: Unit test: channel_mix with known input, verify output > 0 (ReLU²)
  - **Commit**: `feat(gpu): add channel mix FFN with squared ReLU for RWKV-7`
  - _Requirements: FR-8_
  - _Design: Component G (Channel Mix FFN Kernel)_

- [ ] 1.9 Wire LayerNorm operations (reuse RMSNorm)
  - **Do**: In forward_layer(), dispatch RMSNorm kernel 4 times: (1) token_embd_norm at start, (2) attn_norm before time_mix, (3) ffn_norm before channel_mix, (4) output_norm at end. Adapt dispatch_rmsnorm() to handle bias (LayerNorm = RMSNorm + bias). Use Rwkv7WeightStore norm buffers.
  - **Files**: crates/metal-attention/src/gpu_rwkv7_forward_pass.rs, crates/metal-attention-kernels/src/norm.rs
  - **Done when**: All 4 LayerNorm ops execute, outputs normalized
  - **Verify**: Unit test: forward_layer with all norms, verify mean ≈ 0, variance ≈ 1
  - **Commit**: `feat(gpu): wire 4 LayerNorm ops per RWKV-7 layer using RMSNorm`
  - _Requirements: FR-9_
  - _Design: Component A, Data Flow step 2_

- [ ] 1.10 Implement LM head projection
  - **Do**: After final output_norm, dispatch Q4_0 matvec for lm_head: [4096] → [65536]. Write logits to logits_buf. Use output.weight buffer from Rwkv7WeightStore.
  - **Files**: crates/metal-attention/src/gpu_rwkv7_forward_pass.rs
  - **Done when**: LM head produces [65536] logits, all finite
  - **Verify**: Unit test: forward_token returns vec![65536], no NaN/Inf
  - **Commit**: `feat(gpu): implement RWKV-7 LM head projection to logits`
  - _Requirements: FR-4_
  - _Design: Component A, Data Flow step 2_

- [ ] 1.11 Wire greedy argmax sampling
  - **Do**: Add forward_token_greedy() method to dispatch GPU argmax kernel (reuse existing argmax.metal). Readback single u32 token ID. Alternative: CPU fallback with logits readback if GPU argmax fails.
  - **Files**: crates/metal-attention/src/gpu_rwkv7_forward_pass.rs
  - **Done when**: forward_token_greedy() returns token ID in [0, 65536)
  - **Verify**: Unit test: forward_token_greedy with known logits, verify argmax correct
  - **Commit**: `feat(gpu): add greedy argmax for RWKV-7 decode`
  - _Requirements: FR-15_
  - _Design: Component A, Data Flow step 2_

- [ ] 1.12 Integrate RWKV tokenizer from GGUF
  - **Do**: In main.rs run_inference_gpu(), detect RWKV architecture from GGUF. Build GgufTokenizer from metadata (rwkv_vocab_v20230424.txt embedded). Tokenize prompt to token IDs.
  - **Files**: src/main.rs
  - **Done when**: Prompt "Hello" tokenizes to expected RWKV token IDs
  - **Verify**: Manual test: cargo run -- run -m rwkv7.gguf --gpu -p "Hello" --max-tokens 1
  - **Commit**: `feat(cli): integrate RWKV tokenizer from GGUF metadata`
  - _Requirements: FR-13_
  - _Design: Data Flow step 3_

- [ ] 1.13 POC Checkpoint: E2E single token generation
  - **Do**: Run full forward pass: tokenize "Hello" → embed → 32 layers (simplified WKV) → lm_head → argmax → decode. Verify generated text is non-empty, no crashes.
  - **Done when**: `metal-attention run -m rwkv7.gguf --gpu -p "Hello" -n 1` generates 1 token
  - **Verify**: Manual test, inspect stdout for 1 generated token
  - **Commit**: `feat(rwkv7): POC complete - single token generation working`

## Phase 2: Refactoring (Full WKV-7)

Upgrade WKV kernel to full recurrence with delta rule, LoRA projections, GroupNorm, gated output.

- [ ] 2.1 Implement LoRA matvec helper
  - **Do**: Create lora_matvec kernel: compute output = base + (input @ w1) @ w2. Two-stage matvec with intermediate [rank] buffer. Dispatch twice: x @ w1 → intermediate, intermediate @ w2 → output. Support ranks 32-128. Handle Q4_0 or F32 weights.
  - **Files**: crates/metal-attention-kernels/shaders/lora_matvec.metal, crates/metal-attention-kernels/src/lora.rs
  - **Done when**: LoRA matvec produces correct output for 4096 → 64 → 4096
  - **Verify**: Unit test: lora_matvec with known weights, compare vs CPU two-stage matvec
  - **Commit**: `feat(gpu): add LoRA low-rank matvec kernel (two-stage)`
  - _Requirements: FR-12_
  - _Design: Component F (LoRA Small Matvec Kernel)_

- [ ] 2.2 Dispatch 8 LoRA projections per layer
  - **Do**: In forward_layer(), call lora_matvec 8 times: (1-3) decay LoRA w0/w1/w2, (4-6) alpha LoRA a0/a1/a2, (7-9) v-blend LoRA v0/v1/v2, (10-11) gate LoRA g1/g2 (no g0). Compute: w_out = w0 + tanh((xw @ w1) @ w2), a_out = a0 + (xa @ a1) @ a2, etc. Use time_mix LoRA buffers from Rwkv7WeightStore.
  - **Files**: crates/metal-attention/src/gpu_rwkv7_forward_pass.rs
  - **Done when**: All 8 LoRA outputs computed, shapes [4096], finite
  - **Verify**: Unit test: forward_layer with LoRA, verify output shapes + non-zero
  - **Commit**: `feat(gpu): dispatch 8 LoRA projections per RWKV-7 layer`
  - _Requirements: FR-12_
  - _Design: Component A, Data Flow step 2_

- [ ] 2.3 Create WKV-7 kernel with delta rule (part 1: state update)
  - **Do**: Create rwkv_wkv_v7.metal. Upgrade state update: state_new = diag(w)*state + state@ab + v^T*k. Compute ab = (-kk)^T @ (kk * a) where kk = L2_normalize(k * k_k). Keep output = r*state (no GroupNorm yet). Single threadgroup, sequential per head.
  - **Files**: crates/metal-attention-kernels/shaders/rwkv_wkv_v7.metal, crates/metal-attention-kernels/src/rwkv.rs
  - **Done when**: WKV-7 kernel compiles, dispatches, state update matches CPU reference
  - **Verify**: Unit test: dispatch_rwkv_wkv_v7 vs CPU reference, max error < 1e-3
  - **Commit**: `feat(gpu): implement WKV-7 kernel with delta rule state update`
  - _Requirements: FR-10_
  - _Design: Component D (WKV-7 Kernel)_

- [ ] 2.4 Add GroupNorm kernel for WKV output
  - **Do**: Create group_norm.metal. Split [4096] into 64 groups of 64 elements. Per-group: compute mean, variance, normalize x_norm = (x - mean) / sqrt(var + eps). Apply weight + bias: output = x_norm * weight + bias. Use time_mix_ln_weight/bias buffers.
  - **Files**: crates/metal-attention-kernels/shaders/group_norm.metal, crates/metal-attention-kernels/src/norm.rs
  - **Done when**: GroupNorm produces [4096] output with per-group normalization
  - **Verify**: Unit test: group_norm with known input, verify mean=0 var=1 per group
  - **Commit**: `feat(gpu): add GroupNorm kernel for WKV-7 output`
  - _Requirements: FR-11_
  - _Design: Component E (GroupNorm Kernel)_

- [ ] 2.5 Integrate GroupNorm in WKV-7 kernel (part 2: output)
  - **Do**: In rwkv_wkv_v7.metal, after computing r*state, apply GroupNorm. Pass group_norm_weight/bias buffers. Output = GroupNorm(r * state). Update dispatch_rwkv_wkv_v7() to include norm buffers.
  - **Files**: crates/metal-attention-kernels/shaders/rwkv_wkv_v7.metal, crates/metal-attention-kernels/src/rwkv.rs
  - **Done when**: WKV-7 output includes GroupNorm, matches CPU reference
  - **Verify**: Unit test: WKV-7 with GroupNorm vs CPU, max error < 1e-3
  - **Commit**: `feat(gpu): integrate GroupNorm in WKV-7 kernel output`
  - _Requirements: FR-11_
  - _Design: Component D (WKV-7 Kernel)_

- [ ] 2.6 Add gated output projection to WKV-7
  - **Do**: In rwkv_wkv_v7.metal, after GroupNorm, apply gate: output = sigmoid(g) * (output @ O). g comes from gate LoRA g1/g2. O is attn_output projection (already computed in 1.6). Compute sigmoid(g) per element, multiply output.
  - **Files**: crates/metal-attention-kernels/shaders/rwkv_wkv_v7.metal, crates/metal-attention-kernels/src/rwkv.rs
  - **Done when**: Gated output matches CPU reference, finite values
  - **Verify**: Unit test: WKV-7 full vs CPU reference, max error < 1e-3
  - **Commit**: `feat(gpu): add gated output projection to WKV-7 kernel`
  - _Requirements: FR-10_
  - _Design: Component D (WKV-7 Kernel)_

- [ ] 2.7 Add bonus r*k attention term (optional)
  - **Do**: In rwkv_wkv_v7.metal, add bonus term: output += r * k (element-wise). Small correction to WKV output. Validate against RWKV-7 paper.
  - **Files**: crates/metal-attention-kernels/shaders/rwkv_wkv_v7.metal
  - **Done when**: Bonus term added, output matches reference
  - **Verify**: Unit test: compare with/without bonus term vs CPU
  - **Commit**: `feat(gpu): add bonus r*k attention term to WKV-7`
  - _Requirements: FR-10_
  - _Design: Component D (WKV-7 Kernel)_

- [ ] 2.8 Switch forward_layer to use full WKV-7 kernel
  - **Do**: In GpuRwkv7ForwardPass::forward_layer(), replace dispatch_rwkv_wkv() with dispatch_rwkv_wkv_v7(). Pass all 15+ buffers (r, k, v, w0/w1/w2, a0/a1/a2, kk, ka, rk, group_norm_w/b, g, o_proj, state). Remove standalone LoRA dispatch (now inside WKV-7 kernel).
  - **Files**: crates/metal-attention/src/gpu_rwkv7_forward_pass.rs
  - **Done when**: Full WKV-7 executes per layer, generation quality improves
  - **Verify**: Manual test: generate 10 tokens, verify no immediate repetition
  - **Commit**: `feat(gpu): switch to full WKV-7 kernel with all features`
  - _Requirements: FR-10_
  - _Design: Component A_

## Phase 3: Testing (E2E + Quality)

Validate generation quality, add tests, fix edge cases.

- [ ] 3.1 Unit test: embedding lookup correctness
  - **Do**: Test embed_token() for known token IDs. Compare GPU vs GGUF direct lookup (dequantize Q8_0 manually). Verify within 1e-5.
  - **Files**: crates/metal-attention/src/gpu_rwkv7_forward_pass.rs (tests module)
  - **Done when**: Test passes with 10 random tokens
  - **Verify**: cargo test --package metal-attention embed_token
  - **Commit**: `test(gpu): add embedding lookup correctness test`

- [ ] 3.2 Unit test: token shift correctness
  - **Do**: Test token_shift with known current/prev/lerp inputs. Verify output = lerp * current + (1 - lerp) * prev.
  - **Files**: crates/metal-attention-kernels/src/token_shift.rs (tests module)
  - **Done when**: Test passes with edge cases (lerp=0, lerp=1, lerp=0.5)
  - **Verify**: cargo test --package metal-attention-kernels token_shift
  - **Commit**: `test(gpu): add token shift kernel test`

- [ ] 3.3 Unit test: WKV-7 kernel vs CPU reference
  - **Do**: Test dispatch_rwkv_wkv_v7() against cpu_rwkv_wkv_v7() (implement CPU reference first). Test with 4 sequence lengths (1, 4, 16, 64 tokens), 10 random seeds. Verify max error < 1e-3.
  - **Files**: crates/metal-attention-kernels/src/rwkv.rs (tests module)
  - **Done when**: All tests pass, GPU matches CPU within tolerance
  - **Verify**: cargo test --package metal-attention-kernels rwkv_wkv_v7
  - **Commit**: `test(gpu): add WKV-7 kernel vs CPU reference validation`

- [ ] 3.4 Unit test: GroupNorm correctness
  - **Do**: Test group_norm kernel with known input. Verify per-group mean=0, variance=1. Test with different num_groups (1, 16, 64).
  - **Files**: crates/metal-attention-kernels/src/norm.rs (tests module)
  - **Done when**: Test passes with 5 random inputs
  - **Verify**: cargo test --package metal-attention-kernels group_norm
  - **Commit**: `test(gpu): add GroupNorm kernel correctness test`

- [ ] 3.5 Unit test: LoRA matvec correctness
  - **Do**: Test lora_matvec with known weights (rank 32, 64, 128). Verify output = base + (x @ w1) @ w2. Compare vs CPU two-stage matvec.
  - **Files**: crates/metal-attention-kernels/src/lora.rs (tests module)
  - **Done when**: Test passes for all ranks
  - **Verify**: cargo test --package metal-attention-kernels lora_matvec
  - **Commit**: `test(gpu): add LoRA matvec correctness test`

- [ ] 3.6 Integration test: single layer forward pass
  - **Do**: Test forward_layer() with random input + state. Verify output shape [4096], state updated, no NaN/Inf.
  - **Files**: crates/metal-attention/src/gpu_rwkv7_forward_pass.rs (tests module)
  - **Done when**: Test passes with 3 random seeds
  - **Verify**: cargo test --package metal-attention forward_layer
  - **Commit**: `test(gpu): add single layer forward pass integration test`

- [ ] 3.7 Integration test: full 32-layer forward pass
  - **Do**: Test forward_token() through all 32 layers. Verify logits shape [65536], all finite. Test with 5 random token IDs.
  - **Files**: crates/metal-attention/src/gpu_rwkv7_forward_pass.rs (tests module)
  - **Done when**: Test passes, no panics
  - **Verify**: cargo test --package metal-attention forward_token_full
  - **Commit**: `test(gpu): add full 32-layer forward pass test`

- [ ] 3.8 Integration test: autoregressive decode loop
  - **Do**: Test decode loop: start with BOS token, generate 10 tokens. Verify no EOS prematurely, tokens in vocab range, no immediate repetition.
  - **Files**: src/main.rs (tests module or manual test)
  - **Done when**: Decode generates 10 coherent tokens
  - **Verify**: cargo run -- run -m rwkv7.gguf --gpu -p "Once upon a time" -n 10
  - **Commit**: `test(rwkv7): add autoregressive decode loop test`

## Phase 4: Quality Gates

Lint, fmt, docs, benchmarks.

- [ ] 4.1 Local quality check
  - **Do**: Run all quality checks locally: `cargo fmt --check`, `cargo clippy --all-targets`, `cargo test --all`, `cargo doc --no-deps`. Fix all warnings.
  - **Verify**: All commands pass with 0 warnings
  - **Done when**: CI checks will pass
  - **Commit**: `chore(rwkv7): fix clippy warnings and fmt`

- [ ] 4.2 Add benchmark command support
  - **Do**: In main.rs run_bench_gpu(), add RWKV-7 branch. Measure prefill + decode tok/s separately. Output JSONL with --json flag. Test seq_lengths 128, 512. Gen_length 64.
  - **Files**: src/main.rs
  - **Done when**: `metal-attention bench -m rwkv7.gguf --gpu --seq-lengths 128,512 --gen-length 64` runs
  - **Verify**: Manual test, inspect JSONL output for tok/s metrics
  - **Commit**: `feat(bench): add RWKV-7 GPU benchmark support`
  - _Requirements: FR-16, FR-17_

- [ ] 4.3 Document GpuRwkv7ForwardPass API
  - **Do**: Add module-level docs to gpu_rwkv7_forward_pass.rs explaining architecture, usage, limitations. Document public methods with examples. Add safety notes for Metal buffer usage.
  - **Files**: crates/metal-attention/src/gpu_rwkv7_forward_pass.rs
  - **Done when**: `cargo doc --no-deps --open` shows complete docs
  - **Verify**: cargo doc --package metal-attention
  - **Commit**: `docs(gpu): document GpuRwkv7ForwardPass API and architecture`

- [ ] 4.4 Benchmark: measure decode tok/s on M4 Pro
  - **Do**: Run benchmark on M4 Pro with RWKV-7 7.2B Q4_0. Measure decode tok/s (exclude prefill). Compare vs CPU HybridModel path. Document in specs/rwkv7-inference/.progress.md.
  - **Done when**: Tok/s measured, >50 tok/s target achieved
  - **Verify**: cargo run --release -- bench -m rwkv7.gguf --gpu --json
  - **Commit**: `perf(rwkv7): benchmark decode tok/s on M4 Pro`
  - _Requirements: NFR-1_

- [ ] 4.5 Create PR with benchmark data
  - **Do**: Push branch, create PR with gh CLI. Include benchmark results in PR description. Link to spec artifacts (research.md, requirements.md, design.md, tasks.md).
  - **Verify**: `gh pr checks --watch` all green
  - **Done when**: PR ready for review
  - **Commit**: N/A (PR creation)

## Notes

- **POC shortcuts taken**: Simplified WKV in Phase 1 (skips delta rule, LoRA, GroupNorm). Channel mix uses placeholder kernel (may need optimization).
- **Production TODOs**: Full WKV-7 recurrence (Phase 2). Optimize LoRA small matvecs (may need dedicated kernel vs reusing Q4_0 matvec). GroupNorm kernel optimization (threadgroup reduction). Multi-head parallel WKV dispatch (current: sequential per head).
