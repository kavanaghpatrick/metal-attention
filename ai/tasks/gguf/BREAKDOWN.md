---
id: gguf.BREAKDOWN
module: gguf
priority: 3
status: failing
version: 1
origin: spec-workflow
dependsOn: [devops.BREAKDOWN]
tags: [breakdown]
testRequirements:
  unit:
    required: false
    pattern: "tests/gguf/**/*.test.*"
---
# GGUF -- Breakdown

## Context

GGUF (GGML Universal File) is the de facto standard format for local model deployment. Every model metal-attention supports will be loaded from GGUF files. The parser must handle the binary format with mmap for zero-copy access, extract architecture metadata to auto-detect model type, map tensor names to layer roles, and extract embedded tokenizer data. This crate has no Metal dependency -- it produces byte slices that the kernels crate wraps in Metal buffers.

## Scope

- **Binary parser**: `GgufFile` struct with mmap-based zero-copy access to header, metadata KVs, tensor info, and tensor data
- **Metadata accessor**: Typed getters for string, u32, u64, f32, bool, and array metadata values
- **Tensor info**: `GgufTensorInfo` with name, dimensions, quantization type, byte offset; `tensor_data()` returning `&[u8]` slices
- **Quantization types**: Enum covering Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M, Q6_K, F16, F32, BF16
- **Architecture detection**: `detect_architecture()` function using `general.architecture` metadata + fallback tensor name pattern inference
- **Weight mapping**: `map_tensor_name()` function mapping GGUF tensor names to `(layer_index, WeightRole)` tuples for all supported architectures (llama, jamba, rwkv, griffin, zamba, mamba)
- **Tokenizer**: Minimal BPE/SentencePiece tokenizer from GGUF-embedded `tokenizer.ggml.*` metadata fields; optional `tokenizers` crate behind feature flag

## Key Decisions

- **From TECH.md**: GGUF header is little-endian: 4-byte magic "GGUF", uint32 version, uint64 tensor_count, uint64 metadata_kv_count. Tensor data aligned to `general.alignment` (default 32 bytes).
- **From TECH.md**: Architecture detection order: (1) check `general.architecture` metadata key, (2) match known values (llama, jamba, rwkv, griffin, recurrentgemma, zamba, mamba), (3) fallback to tensor name pattern inference.
- **From TECH.md**: Weight roles: QueryProj, KeyProj, ValueProj, OutputProj, GateProj, UpProj, DownProj, AttnNorm, FFNNorm, SSMIn, SSMOut, SSMConv1d, SSMA/B/C/D/Dt, TimeMix, ChannelMix, TokenEmbedding, OutputWeight, OutputNorm.
- **From TECH.md**: Tokenizer decision -- implement minimal GGUF tokenizer reader for core path (avoids ~350 crate dependency tree of `tokenizers`). Support `tokenizer.json` sidecar via optional feature flag.
- **From PM.md**: P0-5 requires GGUF model loading (weights, tokenizer, architecture metadata). Without it, no models can run.
- **From UX.md**: Architecture auto-detection from GGUF metadata is a key UX decision -- users should never need to specify model type.

## Acceptance Criteria

1. `GgufFile::open()` successfully mmap's a real GGUF file and parses header
2. `get_string("general.architecture")` returns correct architecture name
3. `get_u32()`, `get_f32()`, `get_string()` return correct typed metadata values
4. `get_tensor()` returns correct tensor info (name, shape, quantization type) and valid byte slice
5. `tensors()` iterator yields all tensors in the file
6. `detect_architecture()` correctly identifies at least llama and rwkv from GGUF metadata
7. `map_tensor_name()` maps standard tensor names (e.g., `blk.0.attn_q.weight`) to correct `(layer_index, WeightRole)` pairs
8. Tokenizer encodes and decodes a simple string round-trip correctly
9. Parser handles GGUF v3 format (current standard)
10. Mmap'd data alignment is preserved for downstream zero-copy Metal buffer creation

## Technical Notes

- From TECH.md: GGUF metadata typed values use enum discriminants: uint8=0, int8=1, uint16=2, ..., string=8, array=9. Parser must handle all types.
- From TECH.md: Hybrid model metadata lives under architecture-specific prefixes: `jamba.ssm.*`, `jamba.attention.*`, `rwkv.*`, `llama.attention.*`. Layer type inferred from tensor name patterns: `blk.N.attn.*` vs `blk.N.ssm.*` vs `blk.N.channel_mixing.*`.
- From TECH.md: Tensor data section starts at `data_offset`, computed after header + metadata + tensor info array. Each tensor's byte offset is relative to data section start.
- From QA.md: GGUF parsing tests are CPU-only (no GPU needed), suitable for CI Tier 1. Test with real small GGUF files or carefully crafted test fixtures.
- From PM.md: Risk "GGUF format lacks hybrid architecture metadata" (Medium likelihood, Medium impact) -- mitigate with custom metadata keys or layer name heuristics.
- From UX.md: Model path resolution searches CWD, env var, config file, `~/models/`, `~/.cache/metal-attention/models/` in order.
