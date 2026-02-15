//! Criterion benchmarks for flash and linear attention kernels.
//!
//! Benchmarks various sequence lengths and head dimensions to measure
//! throughput in TFLOPS for both attention kernel types.

use criterion::{criterion_group, criterion_main, Criterion};

fn bench_flash_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("flash_attention");
    for &seq_len in &[64, 128, 256, 512, 1024] {
        group.bench_function(format!("N={seq_len}_D=64"), |b| {
            b.iter(|| {
                // TODO: Wire up actual flash attention dispatch
                // For now, measure overhead of empty iteration
                std::hint::black_box(seq_len);
            });
        });
    }
    group.finish();
}

fn bench_linear_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_attention");
    for &seq_len in &[64, 128, 256, 512, 1024] {
        group.bench_function(format!("N={seq_len}_D=64"), |b| {
            b.iter(|| {
                // TODO: Wire up actual linear attention dispatch
                std::hint::black_box(seq_len);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_flash_attention, bench_linear_attention);
criterion_main!(benches);
