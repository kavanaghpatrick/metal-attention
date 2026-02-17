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

/// Benchmark GPU batched prefill throughput with forward_prompt.
///
/// Measures tok/s for multi-token batched prefill at various prompt lengths.
/// Uses forward_prompt which batches matvec operations for SLC cache reuse.
fn bench_gpu_prefill(c: &mut Criterion) {
    let path = model_path();
    if !path.exists() {
        eprintln!(
            "SKIP bench_gpu_prefill: model not found at {}",
            path.display()
        );
        return;
    }

    let mut gpu = GpuForwardPass::from_gguf(&path).expect("Failed to load GPU model");

    // Warmup
    for &tok in PREFILL_TOKENS {
        let _ = gpu.forward_token(tok).expect("GPU warmup prefill failed");
    }
    gpu.reset();

    let mut group = c.benchmark_group("prefill_throughput");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));

    for &prompt_len in &[16u64, 32, 64, 128, 256] {
        // Generate a synthetic prompt of the given length
        let prompt: Vec<u32> = (1..=prompt_len as u32).collect();

        group.throughput(criterion::Throughput::Elements(prompt_len));
        group.bench_function(format!("gpu_prefill_{prompt_len}tok"), |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    gpu.reset();
                    let start = std::time::Instant::now();
                    let _ = gpu.forward_prompt(&prompt).expect("GPU prefill failed");
                    total += start.elapsed();
                }
                total
            })
        });
    }

    group.finish();
    eprintln!("GPU prefill benchmark complete");
}

const MISTRAL_MODEL_RELPATH: &str = "models/mistral-7b-v0.1.Q4_0.gguf";

fn mistral_model_path() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest_dir).join(MISTRAL_MODEL_RELPATH)
}

/// Benchmark Mistral-7B batch prefill throughput with forward_prompt.
///
/// Measures tok/s for 128-token batched prefill on Mistral-7B Q4_0.
/// Target: >= 100 tok/s. Tests GQA (32Q/8KV), Q6_K lm_head, and
/// multi_token_matvec_q4_0 kernels at large dimensions.
fn bench_prefill_mistral(c: &mut Criterion) {
    let path = mistral_model_path();
    if !path.exists() {
        eprintln!(
            "SKIP bench_prefill_mistral: model not found at {}",
            path.display()
        );
        return;
    }

    let mut gpu = GpuForwardPass::from_gguf(&path).expect("Failed to load Mistral-7B");

    // Warmup: one prefill pass
    let warmup_prompt: Vec<u32> = (1..=16).collect();
    let _ = gpu.forward_prompt(&warmup_prompt).expect("warmup failed");
    gpu.reset();

    let mut group = c.benchmark_group("mistral_prefill");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    for &prompt_len in &[32u64, 64, 128] {
        let prompt: Vec<u32> = (1..=prompt_len as u32).collect();

        group.throughput(criterion::Throughput::Elements(prompt_len));
        group.bench_function(format!("mistral_prefill_{prompt_len}tok"), |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    gpu.reset();
                    let start = std::time::Instant::now();
                    let _ = gpu.forward_prompt(&prompt).expect("Mistral prefill failed");
                    total += start.elapsed();
                }
                total
            })
        });
    }

    group.finish();
    eprintln!("Mistral-7B prefill benchmark complete");
}

/// Benchmark Mistral-7B decode throughput (single-token autoregressive).
///
/// Loads Mistral-7B Q4_0, prefills with 3 tokens, then measures 100 decode
/// tokens. Reports tok/s. Baseline: 42 tok/s. Target with Q6_K: >= 47 tok/s.
fn bench_decode_mistral(c: &mut Criterion) {
    let path = mistral_model_path();
    if !path.exists() {
        eprintln!(
            "SKIP bench_decode_mistral: model not found at {}",
            path.display()
        );
        return;
    }

    let mut gpu = GpuForwardPass::from_gguf(&path).expect("Failed to load Mistral-7B");

    // Warmup: prefill + a few decode tokens
    for &tok in PREFILL_TOKENS {
        let _ = gpu.forward_token(tok).expect("warmup prefill failed");
    }
    for _ in 0..3 {
        let logits = gpu.forward_token(1).expect("warmup decode failed");
        let _ = argmax(&logits);
    }

    let mut group = c.benchmark_group("mistral_decode");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60));
    group.throughput(criterion::Throughput::Elements(DECODE_TOKENS as u64));

    group.bench_function("mistral_decode_100tok", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                gpu.reset();
                let mut logits = Vec::new();
                for &tok in PREFILL_TOKENS {
                    logits = gpu.forward_token(tok).expect("prefill failed");
                }
                let start = std::time::Instant::now();
                for _ in 0..DECODE_TOKENS {
                    let next = argmax(&logits);
                    logits = gpu.forward_token(next).expect("decode failed");
                }
                total += start.elapsed();
            }
            total
        })
    });

    group.finish();
    eprintln!("Mistral-7B decode benchmark complete ({DECODE_TOKENS} tokens/iteration)");
}

/// Benchmark EAGLE decode throughput with random draft head weights.
///
/// Uses EagleDecoder with random weights on Mistral-7B. Random weights yield
/// ~0% acceptance rate (overhead-only), but validates benchmark infrastructure.
/// Reports effective tok/s.
fn bench_eagle_decode(c: &mut Criterion) {
    let path = mistral_model_path();
    if !path.exists() {
        eprintln!(
            "SKIP bench_eagle_decode: model not found at {}",
            path.display()
        );
        return;
    }

    const EAGLE_PROMPT: &[u32] = &[1, 2, 3, 4, 5];
    const EAGLE_TOKENS: usize = 100;
    const N_DRAFT: usize = 6;

    // Load EagleDecoder with random weights (one-time cost)
    let mut decoder =
        metal_attention::EagleDecoder::new_random(&path, N_DRAFT).expect("Failed to load EagleDecoder");

    // Warmup: one short generation
    let _ = decoder.generate(EAGLE_PROMPT, 5, |_| {});

    let mut group = c.benchmark_group("eagle_decode");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));
    group.throughput(criterion::Throughput::Elements(EAGLE_TOKENS as u64));

    group.bench_function("eagle_random_decode_100tok", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let start = std::time::Instant::now();
                let _ = decoder
                    .generate(EAGLE_PROMPT, EAGLE_TOKENS, |_| {})
                    .expect("EAGLE decode failed");
                total += start.elapsed();
            }
            total
        })
    });

    group.bench_function("baseline_decode_100tok", |b| {
        // Use a plain GpuForwardPass for target-only baseline
        let mut gpu =
            metal_attention::GpuForwardPass::from_gguf(&path).expect("Failed to load baseline model");

        // Warmup baseline
        for &tok in EAGLE_PROMPT {
            let _ = gpu.forward_token(tok).expect("baseline warmup failed");
        }
        gpu.reset();

        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                gpu.reset();
                // Prefill (not timed)
                let mut logits = Vec::new();
                for &tok in EAGLE_PROMPT {
                    logits = gpu.forward_token(tok).expect("baseline prefill failed");
                }
                // Decode 100 tokens (timed)
                let start = std::time::Instant::now();
                for _ in 0..EAGLE_TOKENS {
                    let next = argmax(&logits);
                    logits = gpu.forward_token(next).expect("baseline decode failed");
                }
                total += start.elapsed();
            }
            total
        })
    });

    group.finish();

    eprintln!(
        "EAGLE decode benchmark complete ({EAGLE_TOKENS} tokens/iteration, n_draft={N_DRAFT})"
    );
}

criterion_group!(
    benches,
    bench_gpu_decode,
    bench_cpu_decode,
    bench_gpu_prefill,
    bench_prefill_mistral,
    bench_decode_mistral,
    bench_eagle_decode
);
criterion_main!(benches);
