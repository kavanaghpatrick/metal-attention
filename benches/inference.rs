//! Criterion benchmarks for end-to-end inference (prefill + decode).
//!
//! Measures tok/s for various model sizes using synthetic random weights.

use criterion::{criterion_group, criterion_main, Criterion};

fn bench_prefill(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefill");
    for &seq_len in &[128, 256, 512] {
        group.bench_function(format!("seq={seq_len}"), |b| {
            b.iter(|| {
                // TODO: Wire up HybridModel::random + prefill
                std::hint::black_box(seq_len);
            });
        });
    }
    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode");
    for &gen_len in &[32, 64, 128] {
        group.bench_function(format!("gen={gen_len}"), |b| {
            b.iter(|| {
                // TODO: Wire up HybridModel::random + decode loop
                std::hint::black_box(gen_len);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_prefill, bench_decode);
criterion_main!(benches);
