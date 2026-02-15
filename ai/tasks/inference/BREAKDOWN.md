---
id: inference.BREAKDOWN
module: inference
priority: 5
status: failing
version: 1
origin: spec-workflow
dependsOn: [traits.BREAKDOWN, kernels.BREAKDOWN, gguf.BREAKDOWN, models.BREAKDOWN]
tags: [breakdown]
testRequirements:
  unit:
    required: false
    pattern: "tests/inference/**/*.test.*"
---
# Inference -- Breakdown

## Context

The inference module ties everything together into the public `Engine` and `Model` API. It implements the prefill/decode loop that processes tokens through all layers, the sampling engine for token generation (temperature, top-p, top-k, repetition penalty), and the streaming `TokenStream` iterator. This is where the hybrid architecture advantage materializes -- linear layers use O(D^2) decode steps while attention layers use O(N*D), and the engine dispatches to the right path per layer via `LayerSchedule`.

## Scope

- **Engine API**: `Engine::new()` and `Engine::with_config()` for Metal device initialization + PSO cache setup
- **Model loading**: `Engine::load_model()` orchestrating GGUF parsing -> architecture detection -> weight mapping -> PSO prewarming -> Model construction
- **Prefill phase**: Process all prompt tokens in parallel through all layers: `embed -> [norm -> seq_block -> norm -> ffn] * N -> final_norm -> logits`
- **Decode phase**: Single-token autoregressive generation: `embed -> [norm -> seq_block -> norm -> ffn] * N -> final_norm -> logits -> sample -> next_token`
- **Sampling engine**: Temperature scaling, top-k filtering, top-p (nucleus) filtering, repetition penalty, greedy argmax, softmax + categorical sampling
- **Token generation**: `Model::generate()` returning `TokenStream` (Iterator<Item=Result<Token, Error>>); `Model::generate_all()` collecting to `GenerationResult`
- **Statistics**: `TokenStats` per token (count, elapsed, tok/s), `GenerationStats` summary (prefill/decode split timing, peak memory)
- **Runtime configuration**: `EngineConfig`, `GenerationParams` with builder pattern, `SamplingParams`

## Key Decisions

- **From TECH.md**: Prefill processes all prompt tokens in a single command buffer (sequential within-buffer is fine since layers are dependent). Decode uses triple-buffered command submission for CPU/GPU overlap.
- **From TECH.md**: Per-layer dispatch: `match schedule.types[layer_idx]` -> `LayerType::Linear` calls `linear_impl.prefill_chunked()`/`decode_step()`; `LayerType::Attention` calls `attention_impl.prefill_attention()`/`decode_attention()` with RoPE and GQA remap.
- **From TECH.md**: Sampling is CPU-side: logits readback from GPU -> repetition penalty -> temperature -> top-k -> top-p -> softmax -> sample. Seed-based reproducibility.
- **From UX.md**: `GenerationParams` uses builder pattern with defaults: max_tokens=256, temperature=0.7, top_p=0.9, top_k=40, repeat_penalty=1.1. `deterministic()` helper sets temperature=0.0, seed=42.
- **From UX.md**: `TokenStream` implements `Iterator` yielding `Token` structs with `text`, `id`, and `stats` fields. Enables streaming output in CLI.
- **From PM.md**: P0-6 requires token generation loop. P0-7 requires RWKV-7 end-to-end inference. Target: <500ms first token latency, <5s model load.

## Acceptance Criteria

1. `Engine::new()` successfully initializes Metal device and PSO cache
2. `Engine::load_model(path)` loads a GGUF model and returns a `Model` with correct `ModelInfo`
3. `Model::generate()` returns a `TokenStream` that yields valid `Token` structs
4. `Model::generate_all()` returns a `GenerationResult` with non-empty text
5. Prefill processes a multi-token prompt and produces valid logits (no NaN, correct vocab size)
6. Decode generates at least 10 tokens autoregressively without error
7. Sampling with temperature=0.0 (greedy) produces deterministic output for the same seed
8. Sampling with temperature=0.7, top_p=0.9, top_k=40 produces valid token IDs
9. Repetition penalty modifies logits correctly (positive scores divided, negative scores multiplied)
10. `TokenStream` reports correct cumulative token count and tokens_per_second
11. `GenerationStats` contains valid prefill_time, decode_time, peak_memory_bytes
12. PSO prewarming at model load time compiles all required kernel variants

## Technical Notes

- From TECH.md: Embedding lookup converts token IDs to `[seq_len, hidden_size]` tensor. Output projection converts final hidden state to `[vocab_size]` logits.
- From TECH.md: Residual connections: `x = x + projected_output` after each seq_block and FFN.
- From TECH.md: For hybrid models, linear layers during decode are O(D^2) -- matrix-vector multiply with fixed state. Attention layers are O(N*D) -- query against full KV cache. The ratio (e.g., 7:1) determines how often the expensive O(N*D) path is hit.
- From QA.md: End-to-end test: load model -> prefill("Hello") -> decode 10 tokens -> verify all tokens are valid vocab IDs and no NaN logits.
- From QA.md: Performance targets: first token latency <500ms (512-token prompt), model load <5s (7B Q4 GGUF).
- From UX.md: `GenerationParams::new()` returns sensible defaults. Builder methods: `.max_tokens()`, `.temperature()`, `.top_p()`, `.top_k()`, `.seed()`, `.deterministic()`.
