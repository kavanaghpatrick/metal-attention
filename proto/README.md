# Prototype Validation

This directory contains the 8 GPU kernel prototypes that validated every design decision in the [PRD](../PRD.md). These are **validation experiments, not production code** — but the Metal shaders and benchmark data are the empirical foundation of this project.

## Running the prototypes

```bash
# Requires Apple Silicon Mac, Rust toolchain, Xcode CLT
cd proto
cargo test
cargo bench
```

## Prototypes

| File | What it validates | Key finding |
|------|-------------------|-------------|
| `proto1_flash.rs` | Flash attention baseline | 0.16 TFLOPS at N=2048, D=64 |
| `proto2_stitch.rs` | Function stitching overhead | `noinline` costs +39%, `always_inline` is 0% |
| `proto3_paged.rs` | PagedAttention V2 viability | ~9% overhead, max page_size=32 for D=64 |
| `proto4_constants.rs` | Function constant dispatch | 0% GPU overhead, 178ns cache hit |
| `proto5_cubecl.rs` | CubeCL/wgpu viability | 58-70% of hand-written MSL (not viable) |
| `proto6_fla.rs` | Linear attention performance | **0.12x wall-clock vs flash at N=1024** |
| `proto7_variants.rs` | RoPE/ALiBi/GQA overhead | All <0.1% of base attention |
| `proto8_burn.rs` | Burn framework integration | Works without forking, 2-17us bridge |

## Metal shaders

The `shaders/` directory contains hand-written MSL kernels that will be ported to production:

- `flash_attention.metal` — tiled softmax attention with simdgroup_matrix
- `linear_attention.metal` — FLA chunk-based kernels (chunk_h, chunk_o)
- `paged_attention.metal` + `paged_reduce.metal` — PagedAttention V2
- `rope.metal` — rotary position embeddings
- `gqa_remap.metal` — grouped-query attention remapping
- `types.h` — shared GPU type definitions

## Data files

- `findings.jsonl` — 58 knowledge base findings generated during investigation
- `bench.json` — raw criterion benchmark results
- `bench_stderr.txt` — benchmark output log

## Shared infrastructure

- `device.rs` — Metal device initialization
- `pipeline.rs` — PsoCache with function constant specialization
- `encode.rs` — command encoder helpers
- `timing.rs` — GPU timing via MTLCommandBuffer timestamps
- `types.rs` — shared Rust/Metal type definitions

See [SYNTHESIS.md](../SYNTHESIS.md) for the full analysis of these results.
