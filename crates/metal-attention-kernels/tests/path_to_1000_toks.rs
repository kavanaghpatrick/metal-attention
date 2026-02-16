//! Path to 1000+ tok/s — Experiment Suite
//!
//! Identifies the highest-impact optimizations for reaching 1000+ tok/s
//! on SmolLM-135M Q4_0 (Apple Silicon M4, 120 GB/s bandwidth).
//!
//! Current: ~460 tok/s. Theoretical max: ~1,400 tok/s (85.8 MB weight reads).
//!
//! Experiments:
//!   1. Peak achievable bandwidth (ceiling)
//!   2. Dispatch overhead isolation (how much do 302 dispatches cost?)
//!   3. Dispatch scaling (what dispatch count gets us to 1000 tok/s?)
//!   4. Small vs large matvec bandwidth utilization
//!   5. Megakernel fusion (rmsnorm+matvec in one dispatch)
//!   6. Multi-token batching (amortize dispatch + weight reads)
//!   7. Real forward pass time breakdown
//!
//! Run: cargo test --release -p metal-attention-kernels --test path_to_1000_toks -- --nocapture --ignored

use metal_attention_kernels::buffer::{alloc_buffer, alloc_buffer_with_data};
use metal_attention_kernels::device::GpuDevice;
use metal_attention_kernels::dispatch::{set_buffer, set_bytes};
use metal_attention_kernels::pipeline::{PsoCache, PsoKey};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
};
use std::time::Instant;

const SMOLLM_HIDDEN: u32 = 576;
const SMOLLM_FFN: u32 = 1536;
const SMOLLM_HEADS: u32 = 9;
const SMOLLM_KV_HEADS: u32 = 3;
const SMOLLM_HEAD_DIM: u32 = 64;
const SMOLLM_VOCAB: u32 = 49152;
const SMOLLM_LAYERS: u32 = 30;
const SMOLLM_WEIGHT_BYTES: usize = 85_800_000; // ~85.8 MB total weight reads per token
const M4_BW_GBS: f64 = 120.0;

fn theoretical_max_toks() -> f64 {
    M4_BW_GBS * 1e9 / SMOLLM_WEIGHT_BYTES as f64
}

#[test]
#[ignore]
fn experiment_suite() {
    let gpu = GpuDevice::new();
    let mut pso_cache = PsoCache::new(gpu.library.clone());

    pso_cache.prewarm(&[
        PsoKey::simple("bandwidth_read_f32"),
        PsoKey::simple("bandwidth_read_high_occupancy"),
        PsoKey::simple("dispatch_overhead_noop"),
        PsoKey::simple("matvec_q4_0"),
        PsoKey::simple("matvec_q4_0_batched"),
        PsoKey::simple("rmsnorm"),
        PsoKey::simple("megakernel_rmsnorm_matvec"),
        PsoKey::simple("bench_multi_token_matvec_q4_0"),
    ]);

    eprintln!("\n{}", "=".repeat(80));
    eprintln!("  PATH TO 1000+ TOK/S — EXPERIMENT SUITE");
    eprintln!("  SmolLM-135M Q4_0 | M4 120 GB/s | Current ~460 tok/s");
    eprintln!("  Theoretical max: {:.0} tok/s (bandwidth-limited)", theoretical_max_toks());
    eprintln!("{}\n", "=".repeat(80));

    exp1_peak_bandwidth(&gpu, &pso_cache);
    exp2_dispatch_overhead(&gpu, &pso_cache);
    exp3_dispatch_scaling(&gpu, &pso_cache);
    exp4_matvec_size_scaling(&gpu, &pso_cache);
    exp5_megakernel_fusion(&gpu, &pso_cache);
    exp6_multi_token_batch(&gpu, &pso_cache);
    exp7_forward_pass_breakdown(&gpu, &pso_cache);

    eprintln!("\n{}", "=".repeat(80));
    eprintln!("  SUMMARY — PRIORITY-ORDERED OPTIMIZATION TARGETS");
    eprintln!("{}", "=".repeat(80));
}

// =========================================================================
// Experiment 1: What's the peak achievable bandwidth on this GPU?
// This is the absolute ceiling — even perfect code can't exceed this.
// =========================================================================
fn exp1_peak_bandwidth(gpu: &GpuDevice, pso_cache: &PsoCache) {
    eprintln!("━━━ Exp 1: Peak Achievable Bandwidth ━━━");
    eprintln!("  (Sets the ceiling for any bandwidth-bound approach)\n");

    let pso = pso_cache.get(&PsoKey::simple("bandwidth_read_high_occupancy")).unwrap();

    for size_mb in [8, 16, 32, 64, 86, 128, 256] {
        let n_bytes = size_mb * 1024 * 1024;
        let n_float4 = n_bytes / 16;
        let data: Vec<f32> = (0..n_bytes / 4).map(|i| (i % 1000) as f32 * 0.001).collect();
        let data_buf = alloc_buffer_with_data(&gpu.device, &data);
        let out_buf = alloc_buffer(&gpu.device, 1024 * 4);
        let n_float4_u32 = n_float4 as u32;

        // Warmup
        for _ in 0..5 {
            let cmd = gpu.command_queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(pso);
            set_buffer(&enc, &data_buf, 0, 0);
            set_buffer(&enc, &out_buf, 0, 1);
            set_bytes(&enc, &n_float4_u32, 2);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: 1024, height: 1, depth: 1 },
                MTLSize { width: 256, height: 1, depth: 1 },
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }

        // Timed
        let iterations = 30;
        let start = Instant::now();
        for _ in 0..iterations {
            let cmd = gpu.command_queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(pso);
            set_buffer(&enc, &data_buf, 0, 0);
            set_buffer(&enc, &out_buf, 0, 1);
            set_bytes(&enc, &n_float4_u32, 2);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: 1024, height: 1, depth: 1 },
                MTLSize { width: 256, height: 1, depth: 1 },
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }
        let elapsed = start.elapsed();
        let avg_us = elapsed.as_micros() as f64 / iterations as f64;
        let bw_gbs = (n_bytes as f64 / 1e9) / (avg_us / 1e6);
        let utilization = bw_gbs / M4_BW_GBS * 100.0;

        eprintln!(
            "  {:>4} MB:  {:>8.1} us  {:>6.1} GB/s  ({:.0}% of {:.0} GB/s)",
            size_mb, avg_us, bw_gbs, utilization, M4_BW_GBS
        );
    }
    eprintln!();
}

// =========================================================================
// Experiment 2: How much do 302 dispatches cost in absolute time?
// Isolates dispatch overhead from actual compute/bandwidth work.
// =========================================================================
fn exp2_dispatch_overhead(gpu: &GpuDevice, pso_cache: &PsoCache) {
    eprintln!("━━━ Exp 2: Dispatch Overhead Isolation ━━━");
    eprintln!("  (Noop kernel — pure dispatch + pipeline + barrier cost)\n");

    let pso = pso_cache.get(&PsoKey::simple("dispatch_overhead_noop")).unwrap();
    let out_buf = alloc_buffer(&gpu.device, 4);

    for n_dispatches in [1, 10, 30, 100, 150, 200, 302, 484, 1000] {
        // Warmup
        for _ in 0..5 {
            let cmd = gpu.command_queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            for _ in 0..n_dispatches {
                enc.setComputePipelineState(pso);
                set_buffer(&enc, &out_buf, 0, 0);
                enc.dispatchThreads_threadsPerThreadgroup(
                    MTLSize { width: 1, height: 1, depth: 1 },
                    MTLSize { width: 1, height: 1, depth: 1 },
                );
            }
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }

        let iterations = 100;
        let start = Instant::now();
        for _ in 0..iterations {
            let cmd = gpu.command_queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            for _ in 0..n_dispatches {
                enc.setComputePipelineState(pso);
                set_buffer(&enc, &out_buf, 0, 0);
                enc.dispatchThreads_threadsPerThreadgroup(
                    MTLSize { width: 1, height: 1, depth: 1 },
                    MTLSize { width: 1, height: 1, depth: 1 },
                );
            }
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }
        let avg_us = start.elapsed().as_micros() as f64 / iterations as f64;
        let per_dispatch = avg_us / n_dispatches as f64;
        let equiv_toks = 1_000_000.0 / avg_us;

        eprintln!(
            "  {:>4} dispatches: {:>8.1} us total  ({:>5.2} us/each)  overhead alone = {:.0} tok/s ceiling",
            n_dispatches, avg_us, per_dispatch, equiv_toks
        );
    }

    eprintln!("\n  KEY: At 302 dispatches, dispatch overhead ALONE limits throughput.");
    eprintln!("  If overhead > 714 us, 1000+ tok/s is impossible without fewer dispatches.\n");
}

// =========================================================================
// Experiment 3: What dispatch count gets us to 1000 tok/s?
// Given real matvec work + dispatch overhead, find the crossover.
// =========================================================================
fn exp3_dispatch_scaling(gpu: &GpuDevice, pso_cache: &PsoCache) {
    eprintln!("━━━ Exp 3: Dispatch Count vs Throughput (with real Q4_0 work) ━━━");
    eprintln!("  (Each dispatch does real Q4_0 matvec 576→72 rows)\n");

    let pso = pso_cache.get(&PsoKey::simple("matvec_q4_0")).unwrap();

    // Create realistic weight buffer: 576 input, 72 output rows per dispatch
    // This simulates splitting a layer's work across N dispatches
    let in_dim: u32 = SMOLLM_HIDDEN;
    let rows_per_dispatch: u32 = 72; // 576/8 = 72 rows per dispatch
    let n_blocks_per_row = in_dim / 32;
    let total_rows: u32 = SMOLLM_HIDDEN; // one full projection

    // Weight data (Q4_0 blocks)
    let total_blocks = (total_rows * n_blocks_per_row) as usize;
    let weight_data: Vec<u8> = (0..total_blocks * 18).map(|i| (i % 256) as u8).collect();
    let weight_buf = alloc_buffer_with_data(&gpu.device, &weight_data);

    // Input/output
    let input_data: Vec<f32> = (0..in_dim as usize).map(|i| (i as f32) * 0.01).collect();
    let input_buf = alloc_buffer_with_data(&gpu.device, &input_data);
    let output_buf = alloc_buffer(&gpu.device, total_rows as usize * 4);

    let weight_bytes = total_blocks * 18;

    for n_dispatches in [1, 2, 4, 8, 16, 32, 72] {
        let rows_each = total_rows / n_dispatches;
        let blocks_per_slice = (rows_each * n_blocks_per_row) as usize;

        // Warmup
        for _ in 0..3 {
            let cmd = gpu.command_queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            for d in 0..n_dispatches {
                let offset = d as usize * blocks_per_slice * 18;
                enc.setComputePipelineState(pso);
                set_buffer(&enc, &weight_buf, offset, 0);
                set_buffer(&enc, &input_buf, 0, 1);
                set_buffer(&enc, &output_buf, d as usize * rows_each as usize * 4, 2);
                set_bytes(&enc, &rows_each, 3);
                set_bytes(&enc, &in_dim, 4);
                let n_groups = (rows_each + 7) / 8;
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: n_groups as usize, height: 1, depth: 1 },
                    MTLSize { width: 256, height: 1, depth: 1 },
                );
            }
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }

        let iterations = 100;
        let start = Instant::now();
        for _ in 0..iterations {
            let cmd = gpu.command_queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            for d in 0..n_dispatches {
                let offset = d as usize * blocks_per_slice * 18;
                enc.setComputePipelineState(pso);
                set_buffer(&enc, &weight_buf, offset, 0);
                set_buffer(&enc, &input_buf, 0, 1);
                set_buffer(&enc, &output_buf, d as usize * rows_each as usize * 4, 2);
                set_bytes(&enc, &rows_each, 3);
                set_bytes(&enc, &in_dim, 4);
                let n_groups = (rows_each + 7) / 8;
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: n_groups as usize, height: 1, depth: 1 },
                    MTLSize { width: 256, height: 1, depth: 1 },
                );
            }
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }
        let avg_us = start.elapsed().as_micros() as f64 / iterations as f64;
        let bw = (weight_bytes as f64 / 1e9) / (avg_us / 1e6);

        eprintln!(
            "  {:>2} dispatches ({}×{} rows): {:>7.1} us  {:.1} GB/s  ({:.0}% util)",
            n_dispatches,
            n_dispatches,
            rows_each,
            avg_us,
            bw,
            bw / M4_BW_GBS * 100.0
        );
    }
    eprintln!();
}

// =========================================================================
// Experiment 4: How does matvec bandwidth scale with matrix size?
// Tests if small matrices (576×192) are fundamentally slower than large (576×49152).
// =========================================================================
fn exp4_matvec_size_scaling(gpu: &GpuDevice, pso_cache: &PsoCache) {
    eprintln!("━━━ Exp 4: Matvec Bandwidth vs Matrix Size ━━━");
    eprintln!("  (Single dispatch, varying output dimension)\n");

    let pso = pso_cache.get(&PsoKey::simple("matvec_q4_0")).unwrap();
    let in_dim: u32 = SMOLLM_HIDDEN; // 576

    // Test matrix sizes matching our actual projections
    let test_cases: Vec<(u32, &str)> = vec![
        (SMOLLM_KV_HEADS * SMOLLM_HEAD_DIM, "K/V proj (192)"),
        (SMOLLM_HEADS * SMOLLM_HEAD_DIM, "Q/O proj (576)"),
        (SMOLLM_FFN, "FFN gate/up (1536)"),
        (SMOLLM_FFN * 2, "2× FFN (3072)"),
        (SMOLLM_VOCAB, "lm_head (49152)"),
    ];

    for (out_dim, label) in &test_cases {
        let n_blocks_per_row = in_dim / 32;
        let total_blocks = (*out_dim * n_blocks_per_row) as usize;
        let weight_bytes = total_blocks * 18;

        let weight_data: Vec<u8> = (0..weight_bytes).map(|i| (i % 256) as u8).collect();
        let weight_buf = alloc_buffer_with_data(&gpu.device, &weight_data);
        let input_data: Vec<f32> = (0..in_dim as usize).map(|i| (i as f32) * 0.01).collect();
        let input_buf = alloc_buffer_with_data(&gpu.device, &input_data);
        let output_buf = alloc_buffer(&gpu.device, *out_dim as usize * 4);

        let n_groups = (*out_dim + 7) / 8;

        // Warmup
        for _ in 0..5 {
            let cmd = gpu.command_queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(pso);
            set_buffer(&enc, &weight_buf, 0, 0);
            set_buffer(&enc, &input_buf, 0, 1);
            set_buffer(&enc, &output_buf, 0, 2);
            set_bytes(&enc, out_dim, 3);
            set_bytes(&enc, &in_dim, 4);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: n_groups as usize, height: 1, depth: 1 },
                MTLSize { width: 256, height: 1, depth: 1 },
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }

        let iterations = 100;
        let start = Instant::now();
        for _ in 0..iterations {
            let cmd = gpu.command_queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(pso);
            set_buffer(&enc, &weight_buf, 0, 0);
            set_buffer(&enc, &input_buf, 0, 1);
            set_buffer(&enc, &output_buf, 0, 2);
            set_bytes(&enc, out_dim, 3);
            set_bytes(&enc, &in_dim, 4);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: n_groups as usize, height: 1, depth: 1 },
                MTLSize { width: 256, height: 1, depth: 1 },
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }
        let avg_us = start.elapsed().as_micros() as f64 / iterations as f64;
        let bw = (weight_bytes as f64 / 1e9) / (avg_us / 1e6);

        eprintln!(
            "  {:>20}: {:>5}×{:<4}  {:>6.1} KB  {:>7.1} us  {:>5.1} GB/s  ({:.0}%)",
            label,
            out_dim,
            in_dim,
            weight_bytes as f64 / 1024.0,
            avg_us,
            bw,
            bw / M4_BW_GBS * 100.0
        );
    }
    eprintln!();
}

// =========================================================================
// Experiment 5: Megakernel fusion — rmsnorm + matvec in one dispatch
// Measures: Does eliminating 1 dispatch save meaningful time?
// =========================================================================
fn exp5_megakernel_fusion(gpu: &GpuDevice, pso_cache: &PsoCache) {
    eprintln!("━━━ Exp 5: Megakernel Fusion (RMSNorm + Q4_0 Matvec) ━━━");
    eprintln!("  (Does fusing 2 kernels into 1 dispatch save time?)\n");

    let pso_rmsnorm = pso_cache.get(&PsoKey::simple("rmsnorm")).unwrap();
    let pso_matvec = pso_cache.get(&PsoKey::simple("matvec_q4_0")).unwrap();
    let pso_mega = pso_cache.get(&PsoKey::simple("megakernel_rmsnorm_matvec")).unwrap();

    let hidden = SMOLLM_HIDDEN;
    let out_dim = hidden; // Q projection: 576→576
    let eps: f32 = 1e-5;
    let n_blocks_per_row = hidden / 32;
    let total_blocks = (out_dim * n_blocks_per_row) as usize;
    let weight_bytes = total_blocks * 18;

    // Allocate buffers
    let input_data: Vec<f32> = (0..hidden as usize).map(|i| (i as f32) * 0.01).collect();
    let input_buf = alloc_buffer_with_data(&gpu.device, &input_data);
    let norm_weight: Vec<f32> = vec![1.0; hidden as usize];
    let norm_buf = alloc_buffer_with_data(&gpu.device, &norm_weight);
    let norm_out = alloc_buffer(&gpu.device, hidden as usize * 4);
    let mat_data: Vec<u8> = (0..weight_bytes).map(|i| (i % 256) as u8).collect();
    let mat_buf = alloc_buffer_with_data(&gpu.device, &mat_data);
    let output_buf = alloc_buffer(&gpu.device, out_dim as usize * 4);

    // --- Separate: rmsnorm then matvec (2 dispatches) ---
    let iterations = 200;
    // Warmup
    for _ in 0..10 {
        let cmd = gpu.command_queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        // rmsnorm
        enc.setComputePipelineState(pso_rmsnorm);
        set_buffer(&enc, &input_buf, 0, 0);
        set_buffer(&enc, &norm_buf, 0, 1);
        set_buffer(&enc, &norm_out, 0, 2);
        set_bytes(&enc, &hidden, 3);
        set_bytes(&enc, &eps, 4);
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: 1, height: 1, depth: 1 },
            MTLSize { width: 256, height: 1, depth: 1 },
        );
        // matvec
        enc.setComputePipelineState(pso_matvec);
        set_buffer(&enc, &mat_buf, 0, 0);
        set_buffer(&enc, &norm_out, 0, 1);
        set_buffer(&enc, &output_buf, 0, 2);
        set_bytes(&enc, &out_dim, 3);
        set_bytes(&enc, &hidden, 4);
        let n_groups = (out_dim + 7) / 8;
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: n_groups as usize, height: 1, depth: 1 },
            MTLSize { width: 256, height: 1, depth: 1 },
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let cmd = gpu.command_queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_rmsnorm);
        set_buffer(&enc, &input_buf, 0, 0);
        set_buffer(&enc, &norm_buf, 0, 1);
        set_buffer(&enc, &norm_out, 0, 2);
        set_bytes(&enc, &hidden, 3);
        set_bytes(&enc, &eps, 4);
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: 1, height: 1, depth: 1 },
            MTLSize { width: 256, height: 1, depth: 1 },
        );
        enc.setComputePipelineState(pso_matvec);
        set_buffer(&enc, &mat_buf, 0, 0);
        set_buffer(&enc, &norm_out, 0, 1);
        set_buffer(&enc, &output_buf, 0, 2);
        set_bytes(&enc, &out_dim, 3);
        set_bytes(&enc, &hidden, 4);
        let n_groups = (out_dim + 7) / 8;
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: n_groups as usize, height: 1, depth: 1 },
            MTLSize { width: 256, height: 1, depth: 1 },
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
    }
    let separate_us = start.elapsed().as_micros() as f64 / iterations as f64;

    // --- Fused: megakernel (1 dispatch) ---
    for _ in 0..10 {
        let cmd = gpu.command_queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_mega);
        set_buffer(&enc, &input_buf, 0, 0);
        set_buffer(&enc, &norm_buf, 0, 1);
        set_buffer(&enc, &mat_buf, 0, 2);
        set_buffer(&enc, &output_buf, 0, 3);
        set_bytes(&enc, &hidden, 4);
        set_bytes(&enc, &eps, 5);
        set_bytes(&enc, &out_dim, 6);
        let n_groups = (out_dim + 7) / 8;
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: n_groups as usize, height: 1, depth: 1 },
            MTLSize { width: 256, height: 1, depth: 1 },
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let cmd = gpu.command_queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso_mega);
        set_buffer(&enc, &input_buf, 0, 0);
        set_buffer(&enc, &norm_buf, 0, 1);
        set_buffer(&enc, &mat_buf, 0, 2);
        set_buffer(&enc, &output_buf, 0, 3);
        set_bytes(&enc, &hidden, 4);
        set_bytes(&enc, &eps, 5);
        set_bytes(&enc, &out_dim, 6);
        let n_groups = (out_dim + 7) / 8;
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: n_groups as usize, height: 1, depth: 1 },
            MTLSize { width: 256, height: 1, depth: 1 },
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
    }
    let fused_us = start.elapsed().as_micros() as f64 / iterations as f64;

    let savings = separate_us - fused_us;
    let pct = savings / separate_us * 100.0;

    eprintln!("  Separate (rmsnorm + matvec):  {:>7.1} us", separate_us);
    eprintln!("  Fused (megakernel):           {:>7.1} us", fused_us);
    eprintln!("  Savings per fusion:           {:>7.1} us ({:.1}%)", savings, pct);
    eprintln!(
        "  If applied to all 7 rmsnorm+matvec pairs/layer × 30 layers = {} fusions:",
        7 * 30
    );
    eprintln!(
        "  Estimated total savings:      {:>7.1} us",
        savings * (7.0 * 30.0)
    );
    let current_token_us = 1_000_000.0 / 460.0;
    let projected_us = current_token_us - savings * (7.0 * 30.0);
    eprintln!(
        "  Projected tok/s:              {:.0} (from 460)",
        1_000_000.0 / projected_us
    );
    eprintln!();
}

// =========================================================================
// Experiment 6: Multi-token batching (process N tokens per forward pass)
// Tests if cache reuse across tokens amortizes weight reads.
// =========================================================================
fn exp6_multi_token_batch(gpu: &GpuDevice, pso_cache: &PsoCache) {
    eprintln!("━━━ Exp 6: Multi-Token Batching (weight cache reuse) ━━━");
    eprintln!("  (Same weights read once, applied to N token vectors)\n");

    let pso_single = pso_cache.get(&PsoKey::simple("matvec_q4_0")).unwrap();
    let pso_batch = pso_cache.get(&PsoKey::simple("bench_multi_token_matvec_q4_0")).unwrap();

    let in_dim: u32 = SMOLLM_HIDDEN;
    let out_dim: u32 = SMOLLM_HIDDEN; // 576→576

    let n_blocks_per_row = in_dim / 32;
    let total_blocks = (out_dim * n_blocks_per_row) as usize;
    let weight_bytes = total_blocks * 18;

    let weight_data: Vec<u8> = (0..weight_bytes).map(|i| (i % 256) as u8).collect();
    let weight_buf = alloc_buffer_with_data(&gpu.device, &weight_data);

    for batch_size in [1u32, 2, 4, 8, 16] {
        let input_data: Vec<f32> =
            (0..(batch_size * in_dim) as usize).map(|i| (i as f32) * 0.001).collect();
        let input_buf = alloc_buffer_with_data(&gpu.device, &input_data);
        let output_buf = alloc_buffer(&gpu.device, (batch_size * out_dim) as usize * 4);

        if batch_size == 1 {
            // Use standard single-token matvec for batch=1
            let n_groups = (out_dim + 7) / 8;

            for _ in 0..10 {
                let cmd = gpu.command_queue.commandBuffer().unwrap();
                let enc = cmd.computeCommandEncoder().unwrap();
                enc.setComputePipelineState(pso_single);
                set_buffer(&enc, &weight_buf, 0, 0);
                set_buffer(&enc, &input_buf, 0, 1);
                set_buffer(&enc, &output_buf, 0, 2);
                set_bytes(&enc, &out_dim, 3);
                set_bytes(&enc, &in_dim, 4);
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: n_groups as usize, height: 1, depth: 1 },
                    MTLSize { width: 256, height: 1, depth: 1 },
                );
                enc.endEncoding();
                cmd.commit();
                cmd.waitUntilCompleted();
            }

            let iterations = 200;
            let start = Instant::now();
            for _ in 0..iterations {
                let cmd = gpu.command_queue.commandBuffer().unwrap();
                let enc = cmd.computeCommandEncoder().unwrap();
                enc.setComputePipelineState(pso_single);
                set_buffer(&enc, &weight_buf, 0, 0);
                set_buffer(&enc, &input_buf, 0, 1);
                set_buffer(&enc, &output_buf, 0, 2);
                set_bytes(&enc, &out_dim, 3);
                set_bytes(&enc, &in_dim, 4);
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: n_groups as usize, height: 1, depth: 1 },
                    MTLSize { width: 256, height: 1, depth: 1 },
                );
                enc.endEncoding();
                cmd.commit();
                cmd.waitUntilCompleted();
            }
            let avg_us = start.elapsed().as_micros() as f64 / iterations as f64;
            let per_token = avg_us;
            let bw = (weight_bytes as f64 / 1e9) / (per_token / 1e6);

            eprintln!(
                "  batch={:>2}: {:>7.1} us total  {:>7.1} us/token  {:>5.1} GB/s",
                batch_size, avg_us, per_token, bw
            );
        } else {
            // Multi-token kernel
            let n_groups = (out_dim + 7) / 8;

            for _ in 0..10 {
                let cmd = gpu.command_queue.commandBuffer().unwrap();
                let enc = cmd.computeCommandEncoder().unwrap();
                enc.setComputePipelineState(pso_batch);
                set_buffer(&enc, &weight_buf, 0, 0);
                set_buffer(&enc, &input_buf, 0, 1);
                set_buffer(&enc, &output_buf, 0, 2);
                set_bytes(&enc, &out_dim, 3);
                set_bytes(&enc, &in_dim, 4);
                set_bytes(&enc, &batch_size, 5);
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: n_groups as usize, height: 1, depth: 1 },
                    MTLSize { width: 256, height: 1, depth: 1 },
                );
                enc.endEncoding();
                cmd.commit();
                cmd.waitUntilCompleted();
            }

            let iterations = 200;
            let start = Instant::now();
            for _ in 0..iterations {
                let cmd = gpu.command_queue.commandBuffer().unwrap();
                let enc = cmd.computeCommandEncoder().unwrap();
                enc.setComputePipelineState(pso_batch);
                set_buffer(&enc, &weight_buf, 0, 0);
                set_buffer(&enc, &input_buf, 0, 1);
                set_buffer(&enc, &output_buf, 0, 2);
                set_bytes(&enc, &out_dim, 3);
                set_bytes(&enc, &in_dim, 4);
                set_bytes(&enc, &batch_size, 5);
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: n_groups as usize, height: 1, depth: 1 },
                    MTLSize { width: 256, height: 1, depth: 1 },
                );
                enc.endEncoding();
                cmd.commit();
                cmd.waitUntilCompleted();
            }
            let avg_us = start.elapsed().as_micros() as f64 / iterations as f64;
            let per_token = avg_us / batch_size as f64;
            let bw = (weight_bytes as f64 / 1e9) / (per_token / 1e6);

            eprintln!(
                "  batch={:>2}: {:>7.1} us total  {:>7.1} us/token  {:>5.1} GB/s  ({:.1}x vs batch=1)",
                batch_size, avg_us, per_token, bw,
                avg_us / per_token / batch_size as f64 * batch_size as f64
            );
        }
    }
    eprintln!("\n  KEY: If batch>1 shows near-linear scaling, speculative decode is viable.\n");
}

// =========================================================================
// Experiment 7: Time breakdown of actual forward pass operations
// Measures each kernel type individually to find the biggest time sinks.
// =========================================================================
fn exp7_forward_pass_breakdown(gpu: &GpuDevice, pso_cache: &PsoCache) {
    eprintln!("━━━ Exp 7: Forward Pass Time Breakdown (per operation type) ━━━");
    eprintln!("  (What percentage of time does each kernel type take?)\n");

    let hidden = SMOLLM_HIDDEN;
    let ffn = SMOLLM_FFN;
    let heads = SMOLLM_HEADS;
    let kv_heads = SMOLLM_KV_HEADS;
    let head_dim = SMOLLM_HEAD_DIM;
    let eps: f32 = 1e-5;

    // Allocate shared buffers
    let input_buf = alloc_buffer_with_data(
        &gpu.device,
        &vec![0.01f32; hidden as usize],
    );
    let norm_weight = alloc_buffer_with_data(&gpu.device, &vec![1.0f32; hidden as usize]);
    let norm_out = alloc_buffer(&gpu.device, hidden as usize * 4);

    // Q4_0 weight buffers for different sizes
    let make_q4_weight = |out_dim: u32| {
        let n_blocks = (out_dim * hidden / 32) as usize;
        let data: Vec<u8> = (0..n_blocks * 18).map(|i| (i % 256) as u8).collect();
        alloc_buffer_with_data(&gpu.device, &data)
    };

    let qkv_weight = make_q4_weight(hidden + 2 * kv_heads * head_dim); // batched
    let o_weight = make_q4_weight(hidden);
    let gate_up_weight = make_q4_weight(2 * ffn); // batched gate+up
    let down_weight = make_q4_weight(hidden); // down: ffn→hidden, but weight is hidden×ffn
    let lm_head_weight = make_q4_weight(SMOLLM_VOCAB);

    let scratch_q = alloc_buffer(&gpu.device, (heads * head_dim) as usize * 4);
    let scratch_kv = alloc_buffer(&gpu.device, (kv_heads * head_dim * 2) as usize * 4);
    let scratch_gate = alloc_buffer(&gpu.device, ffn as usize * 4);
    let scratch_up = alloc_buffer(&gpu.device, ffn as usize * 4);
    let scratch_silu = alloc_buffer(&gpu.device, ffn as usize * 4);
    let output_buf = alloc_buffer(&gpu.device, hidden as usize * 4);
    let logits_buf = alloc_buffer(&gpu.device, SMOLLM_VOCAB as usize * 4);

    let pso_rmsnorm = pso_cache.get(&PsoKey::simple("rmsnorm")).unwrap();
    let pso_matvec = pso_cache.get(&PsoKey::simple("matvec_q4_0")).unwrap();
    let pso_batched = pso_cache.get(&PsoKey::simple("matvec_q4_0_batched")).unwrap();

    let iterations = 200;

    // Helper closure to time a specific operation pattern
    struct OpTime {
        name: &'static str,
        per_layer: u32,
        total_us: f64,
    }

    let mut results: Vec<OpTime> = Vec::new();

    // --- RMSNorm (2 per layer + 1 final = 61 total) ---
    {
        // Warmup
        for _ in 0..10 {
            let cmd = gpu.command_queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(pso_rmsnorm);
            set_buffer(&enc, &input_buf, 0, 0);
            set_buffer(&enc, &norm_weight, 0, 1);
            set_buffer(&enc, &norm_out, 0, 2);
            set_bytes(&enc, &hidden, 3);
            set_bytes(&enc, &eps, 4);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: 1, height: 1, depth: 1 },
                MTLSize { width: 256, height: 1, depth: 1 },
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }

        let start = Instant::now();
        for _ in 0..iterations {
            let cmd = gpu.command_queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            // 61 rmsnorm dispatches (2/layer × 30 + 1 final)
            for _ in 0..61 {
                enc.setComputePipelineState(pso_rmsnorm);
                set_buffer(&enc, &input_buf, 0, 0);
                set_buffer(&enc, &norm_weight, 0, 1);
                set_buffer(&enc, &norm_out, 0, 2);
                set_bytes(&enc, &hidden, 3);
                set_bytes(&enc, &eps, 4);
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: 1, height: 1, depth: 1 },
                    MTLSize { width: 256, height: 1, depth: 1 },
                );
            }
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }
        let total = start.elapsed().as_micros() as f64 / iterations as f64;
        results.push(OpTime { name: "RMSNorm (61 dispatches)", per_layer: 2, total_us: total });
    }

    // --- Batched QKV matvec (1 per layer = 30) ---
    {
        let total_out = hidden + 2 * kv_heads * head_dim;
        let n_groups = (total_out + 7) / 8;

        for _ in 0..10 {
            let cmd = gpu.command_queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(pso_batched);
            set_buffer(&enc, &qkv_weight, 0, 0);
            set_buffer(&enc, &norm_out, 0, 1);
            set_buffer(&enc, &scratch_q, 0, 2);
            set_bytes(&enc, &total_out, 3);
            set_bytes(&enc, &hidden, 4);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: n_groups as usize, height: 1, depth: 1 },
                MTLSize { width: 256, height: 1, depth: 1 },
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }

        let start = Instant::now();
        for _ in 0..iterations {
            let cmd = gpu.command_queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            for _ in 0..30 {
                enc.setComputePipelineState(pso_batched);
                set_buffer(&enc, &qkv_weight, 0, 0);
                set_buffer(&enc, &norm_out, 0, 1);
                set_buffer(&enc, &scratch_q, 0, 2);
                set_bytes(&enc, &total_out, 3);
                set_bytes(&enc, &hidden, 4);
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: n_groups as usize, height: 1, depth: 1 },
                    MTLSize { width: 256, height: 1, depth: 1 },
                );
            }
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }
        let total = start.elapsed().as_micros() as f64 / iterations as f64;
        results.push(OpTime { name: "Batched QKV matvec (30 dispatches)", per_layer: 1, total_us: total });
    }

    // --- O projection + accumulate (1 per layer = 30) ---
    {
        let n_groups = (hidden + 7) / 8;
        let pso_accum = pso_cache.get(&PsoKey::simple("matvec_q4_0")).unwrap();

        for _ in 0..10 {
            let cmd = gpu.command_queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(pso_accum);
            set_buffer(&enc, &o_weight, 0, 0);
            set_buffer(&enc, &scratch_q, 0, 1);
            set_buffer(&enc, &output_buf, 0, 2);
            set_bytes(&enc, &hidden, 3);
            set_bytes(&enc, &(heads * head_dim), 4);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: n_groups as usize, height: 1, depth: 1 },
                MTLSize { width: 256, height: 1, depth: 1 },
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }

        let start = Instant::now();
        for _ in 0..iterations {
            let cmd = gpu.command_queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            for _ in 0..30 {
                enc.setComputePipelineState(pso_accum);
                set_buffer(&enc, &o_weight, 0, 0);
                set_buffer(&enc, &scratch_q, 0, 1);
                set_buffer(&enc, &output_buf, 0, 2);
                set_bytes(&enc, &hidden, 3);
                set_bytes(&enc, &(heads * head_dim), 4);
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: n_groups as usize, height: 1, depth: 1 },
                    MTLSize { width: 256, height: 1, depth: 1 },
                );
            }
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }
        let total = start.elapsed().as_micros() as f64 / iterations as f64;
        results.push(OpTime { name: "O proj matvec (30 dispatches)", per_layer: 1, total_us: total });
    }

    // --- FFN gate+up batched (1 per layer = 30) ---
    {
        let total_out = 2 * ffn;
        let n_groups = (total_out + 7) / 8;

        for _ in 0..10 {
            let cmd = gpu.command_queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(pso_batched);
            set_buffer(&enc, &gate_up_weight, 0, 0);
            set_buffer(&enc, &norm_out, 0, 1);
            set_buffer(&enc, &scratch_gate, 0, 2);
            set_bytes(&enc, &total_out, 3);
            set_bytes(&enc, &hidden, 4);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: n_groups as usize, height: 1, depth: 1 },
                MTLSize { width: 256, height: 1, depth: 1 },
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }

        let start = Instant::now();
        for _ in 0..iterations {
            let cmd = gpu.command_queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            for _ in 0..30 {
                enc.setComputePipelineState(pso_batched);
                set_buffer(&enc, &gate_up_weight, 0, 0);
                set_buffer(&enc, &norm_out, 0, 1);
                set_buffer(&enc, &scratch_gate, 0, 2);
                set_bytes(&enc, &total_out, 3);
                set_bytes(&enc, &hidden, 4);
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: n_groups as usize, height: 1, depth: 1 },
                    MTLSize { width: 256, height: 1, depth: 1 },
                );
            }
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }
        let total = start.elapsed().as_micros() as f64 / iterations as f64;
        results.push(OpTime { name: "FFN gate+up batched (30 dispatches)", per_layer: 1, total_us: total });
    }

    // --- Down proj + accumulate (1 per layer = 30) ---
    // Note: down proj is hidden×ffn (reads ffn input, produces hidden output)
    {
        let n_groups = (hidden + 7) / 8;
        let in_dim_down = ffn;

        // Need weight sized for hidden×ffn
        let n_blocks_down = (hidden * in_dim_down / 32) as usize;
        let down_data: Vec<u8> = (0..n_blocks_down * 18).map(|i| (i % 256) as u8).collect();
        let down_buf = alloc_buffer_with_data(&gpu.device, &down_data);

        for _ in 0..10 {
            let cmd = gpu.command_queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(pso_matvec);
            set_buffer(&enc, &down_buf, 0, 0);
            set_buffer(&enc, &scratch_silu, 0, 1);
            set_buffer(&enc, &output_buf, 0, 2);
            set_bytes(&enc, &hidden, 3);
            set_bytes(&enc, &in_dim_down, 4);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: n_groups as usize, height: 1, depth: 1 },
                MTLSize { width: 256, height: 1, depth: 1 },
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }

        let start = Instant::now();
        for _ in 0..iterations {
            let cmd = gpu.command_queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            for _ in 0..30 {
                enc.setComputePipelineState(pso_matvec);
                set_buffer(&enc, &down_buf, 0, 0);
                set_buffer(&enc, &scratch_silu, 0, 1);
                set_buffer(&enc, &output_buf, 0, 2);
                set_bytes(&enc, &hidden, 3);
                set_bytes(&enc, &in_dim_down, 4);
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: n_groups as usize, height: 1, depth: 1 },
                    MTLSize { width: 256, height: 1, depth: 1 },
                );
            }
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }
        let total = start.elapsed().as_micros() as f64 / iterations as f64;
        results.push(OpTime { name: "Down proj matvec (30 dispatches)", per_layer: 1, total_us: total });
    }

    // --- lm_head (1 dispatch) ---
    {
        let n_groups = (SMOLLM_VOCAB + 7) / 8;

        for _ in 0..10 {
            let cmd = gpu.command_queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(pso_matvec);
            set_buffer(&enc, &lm_head_weight, 0, 0);
            set_buffer(&enc, &norm_out, 0, 1);
            set_buffer(&enc, &logits_buf, 0, 2);
            set_bytes(&enc, &SMOLLM_VOCAB, 3);
            set_bytes(&enc, &hidden, 4);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: n_groups as usize, height: 1, depth: 1 },
                MTLSize { width: 256, height: 1, depth: 1 },
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }

        let start = Instant::now();
        for _ in 0..iterations {
            let cmd = gpu.command_queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(pso_matvec);
            set_buffer(&enc, &lm_head_weight, 0, 0);
            set_buffer(&enc, &norm_out, 0, 1);
            set_buffer(&enc, &logits_buf, 0, 2);
            set_bytes(&enc, &SMOLLM_VOCAB, 3);
            set_bytes(&enc, &hidden, 4);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: n_groups as usize, height: 1, depth: 1 },
                MTLSize { width: 256, height: 1, depth: 1 },
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }
        let total = start.elapsed().as_micros() as f64 / iterations as f64;
        results.push(OpTime { name: "lm_head (1 dispatch, 49152 rows)", per_layer: 0, total_us: total });
    }

    // --- Print results ---
    let grand_total: f64 = results.iter().map(|r| r.total_us).sum();
    eprintln!("  {:>45}  {:>8}  {:>6}", "Operation", "Time(us)", "Share");
    eprintln!("  {}", "-".repeat(67));
    for r in &results {
        let share = r.total_us / grand_total * 100.0;
        eprintln!(
            "  {:>45}  {:>8.1}  {:>5.1}%",
            r.name, r.total_us, share
        );
    }
    eprintln!("  {}", "-".repeat(67));
    eprintln!("  {:>45}  {:>8.1}  100.0%", "TOTAL (measured)", grand_total);
    eprintln!("\n  Note: Excludes attention, RoPE, SiLU, KV append, argmax.");
    eprintln!("  These are small compared to matvec but add dispatch overhead.");
    eprintln!(
        "\n  Measured matvec total: {:.1} us = {:.0} tok/s (matvec only)",
        grand_total,
        1_000_000.0 / grand_total
    );
    eprintln!(
        "  Actual forward pass: ~{:.0} us = ~460 tok/s",
        1_000_000.0 / 460.0
    );
    eprintln!(
        "  Overhead (attention + RoPE + SiLU + KV + dispatch): ~{:.0} us ({:.0}%)",
        1_000_000.0 / 460.0 - grand_total,
        (1_000_000.0 / 460.0 - grand_total) / (1_000_000.0 / 460.0) * 100.0
    );
    eprintln!();
}
