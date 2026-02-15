//! Criterion benchmarks for real-GGUF GPU vs CPU decode throughput.
//!
//! Loads SmolLM-135M.Q4_0.gguf and measures tok/s for both GpuForwardPass
//! (Metal compute kernels) and HybridModel (CPU F32 path).
//!
//! Skips gracefully if the model file is absent.

use std::path::{Path, PathBuf};
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};

use metal_attention::gpu_forward_pass::GpuForwardPass;
use metal_attention::model::HybridModel;
use metal_attention_kernels::device::GpuDevice;
use metal_attention_kernels::pipeline::PsoCache;

const MODEL_RELPATH: &str = "models/SmolLM-135M.Q4_0.gguf";
const WARMUP_TOKENS: usize = 3;
const DECODE_TOKENS: usize = 100;
const PREFILL_TOKENS: &[u32] = &[1, 2, 3];

/// Resolve model path relative to the workspace root (Cargo.toml dir).
fn model_path() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest_dir).join(MODEL_RELPATH)
}

/// Greedy argmax over a logit vector.
fn argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}

/// Benchmark GPU decode throughput with real GGUF weights.
///
/// Loads the model once, then for each Criterion iteration: resets KV cache,
/// prefills with tokens [1, 2, 3], and benchmarks 100 decode tokens.
fn bench_gpu_decode(c: &mut Criterion) {
    let path = model_path();
    if !path.exists() {
        eprintln!(
            "SKIP bench_gpu_decode: model not found at {}",
            path.display()
        );
        return;
    }

    // Load model outside the benchmark loop (one-time cost)
    let mut gpu = GpuForwardPass::from_gguf(&path).expect("Failed to load GPU model");

    // Warmup: run a few tokens to trigger PSO compilation and GPU initialization
    for &tok in PREFILL_TOKENS {
        let _ = gpu.forward_token(tok).expect("GPU warmup prefill failed");
    }
    for _ in 0..WARMUP_TOKENS {
        let logits = gpu.forward_token(1).expect("GPU warmup failed");
        let _ = argmax(&logits);
    }

    let mut group = c.benchmark_group("decode_throughput");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));
    group.throughput(criterion::Throughput::Elements(DECODE_TOKENS as u64));

    group.bench_function("gpu_decode_100tok", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                // Reset KV cache so each iteration starts from a clean state
                gpu.reset();
                // Re-prefill after reset (not timed)
                let mut logits = Vec::new();
                for &tok in PREFILL_TOKENS {
                    logits = gpu.forward_token(tok).expect("GPU prefill failed");
                }
                // Decode 100 tokens (timed)
                let start = std::time::Instant::now();
                for _ in 0..DECODE_TOKENS {
                    let next = argmax(&logits);
                    logits = gpu.forward_token(next).expect("GPU decode failed");
                }
                total += start.elapsed();
            }
            total
        })
    });

    group.finish();

    eprintln!("GPU decode benchmark complete ({DECODE_TOKENS} tokens/iteration)");
}

/// Benchmark CPU decode throughput with real GGUF weights.
///
/// Loads the model once via HybridModel, then for each Criterion iteration:
/// creates fresh state, prefills with tokens [1, 2, 3], and benchmarks 100
/// decode tokens.
fn bench_cpu_decode(c: &mut Criterion) {
    let path = model_path();
    if !path.exists() {
        eprintln!(
            "SKIP bench_cpu_decode: model not found at {}",
            path.display()
        );
        return;
    }

    // Load model (needs GPU device for Q8_0 embed dequantization)
    let device = GpuDevice::new();
    let mut pso_cache = PsoCache::new(device.library.clone());
    let cpu_model = HybridModel::from_gguf(&path, Some(&device), Some(&mut pso_cache))
        .expect("Failed to load CPU model");

    // Warmup: run a few tokens to warm up CPU caches
    {
        let mut state = cpu_model.init_state();
        for &tok in PREFILL_TOKENS {
            let _ = cpu_model.forward_token(tok, &mut state);
        }
        for _ in 0..WARMUP_TOKENS {
            let logits = cpu_model.forward_token(1, &mut state);
            let _ = argmax(&logits);
        }
    }

    let mut group = c.benchmark_group("decode_throughput");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60));
    group.throughput(criterion::Throughput::Elements(DECODE_TOKENS as u64));

    group.bench_function("cpu_decode_100tok", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                // Fresh state each iteration (clean KV cache)
                let mut state = cpu_model.init_state();
                let mut logits = Vec::new();
                for &tok in PREFILL_TOKENS {
                    logits = cpu_model.forward_token(tok, &mut state);
                }
                // Decode 100 tokens (timed)
                let start = std::time::Instant::now();
                for _ in 0..DECODE_TOKENS {
                    let next = argmax(&logits);
                    logits = cpu_model.forward_token(next, &mut state);
                }
                total += start.elapsed();
            }
            total
        })
    });

    group.finish();

    eprintln!("CPU decode benchmark complete ({DECODE_TOKENS} tokens/iteration)");
}

criterion_group!(benches, bench_gpu_decode, bench_cpu_decode);
criterion_main!(benches);
