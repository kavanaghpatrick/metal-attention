//! Quick A/B/C/D benchmark: 4 matvec_q4_0 kernel variants head-to-head.
//!
//! Run: cargo test --release --test matvec_experiment -- --nocapture --ignored

use metal_attention_kernels::buffer::{alloc_buffer, alloc_buffer_with_data, read_buffer_slice};
use metal_attention_kernels::device::GpuDevice;
use metal_attention_kernels::dispatch::{set_buffer, set_bytes};
use metal_attention_kernels::pipeline::{PsoCache, PsoKey};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
};
use std::time::Instant;

/// Q4_0 block: 18 bytes = 2 (fp16 scale) + 16 (packed nibbles)
const Q4_0_BLOCK_SIZE: usize = 32;
const Q4_0_BYTES_PER_BLOCK: usize = 18;

fn encode_q4_0_block(scale: f32, values: &[i8; 32]) -> [u8; 18] {
    let mut block = [0u8; 18];
    let scale_f16 = half::f16::from_f32(scale);
    let scale_bytes = scale_f16.to_le_bytes();
    block[0] = scale_bytes[0];
    block[1] = scale_bytes[1];
    for i in 0..16 {
        let lo = (values[i] + 8) as u8 & 0x0F;
        let hi = (values[i + 16] + 8) as u8 & 0x0F;
        block[2 + i] = lo | (hi << 4);
    }
    block
}

fn generate_weights(out_dim: usize, in_dim: usize) -> Vec<u8> {
    let n_blocks_per_row = in_dim / Q4_0_BLOCK_SIZE;
    let mut weight_bytes = Vec::with_capacity(out_dim * n_blocks_per_row * Q4_0_BYTES_PER_BLOCK);
    for row in 0..out_dim {
        for b in 0..n_blocks_per_row {
            let scale = 0.01 * ((row * n_blocks_per_row + b) % 100 + 1) as f32;
            let mut values = [0i8; 32];
            for i in 0..32 {
                values[i] = ((row + b + i) % 15) as i8 - 7;
            }
            let block = encode_q4_0_block(scale, &values);
            weight_bytes.extend_from_slice(&block);
        }
    }
    weight_bytes
}

fn generate_input(in_dim: usize) -> Vec<f32> {
    (0..in_dim).map(|i| 0.1 * ((i % 10) as f32 - 5.0)).collect()
}

struct KernelConfig {
    name: &'static str,
    pso_name: &'static str,
    threads_per_group: usize,
    /// How many rows each threadgroup handles (for grid size calculation)
    rows_per_group: usize,
}

fn bench_kernel(
    gpu: &GpuDevice,
    pso_cache: &mut PsoCache,
    config: &KernelConfig,
    weight_buf: &objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    input_buf: &objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    output_buf: &objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    out_dim: usize,
    in_dim: usize,
    warmup: usize,
    iterations: usize,
) -> (f64, Vec<f32>) {
    let pso_key = PsoKey::simple(config.pso_name);
    let pso = pso_cache.get_or_compile(&pso_key);

    let out_dim_u32 = out_dim as u32;
    let in_dim_u32 = in_dim as u32;

    let n_groups = (out_dim + config.rows_per_group - 1) / config.rows_per_group;
    let grid = MTLSize {
        width: n_groups,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: config.threads_per_group,
        height: 1,
        depth: 1,
    };

    // Warmup
    for _ in 0..warmup {
        let cmd = gpu.command_queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso);
        set_buffer(&enc, weight_buf, 0, 0);
        set_buffer(&enc, input_buf, 0, 1);
        set_buffer(&enc, output_buf, 0, 2);
        set_bytes(&enc, &out_dim_u32, 3);
        set_bytes(&enc, &in_dim_u32, 4);
        enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
    }

    // Timed iterations
    let start = Instant::now();
    for _ in 0..iterations {
        let cmd = gpu.command_queue.commandBuffer().unwrap();
        let enc = cmd.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(pso);
        set_buffer(&enc, weight_buf, 0, 0);
        set_buffer(&enc, input_buf, 0, 1);
        set_buffer(&enc, output_buf, 0, 2);
        set_bytes(&enc, &out_dim_u32, 3);
        set_bytes(&enc, &in_dim_u32, 4);
        enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
    }
    let elapsed = start.elapsed();
    let avg_us = elapsed.as_micros() as f64 / iterations as f64;

    // Read output for correctness check
    let result = unsafe { read_buffer_slice(output_buf, out_dim) };

    (avg_us, result)
}

#[test]
#[ignore] // Run with: cargo test --release --test matvec_experiment -- --nocapture --ignored
fn matvec_q4_0_kernel_shootout() {
    let gpu = GpuDevice::new();
    let mut pso_cache = PsoCache::new(gpu.library.clone());

    let configs = vec![
        KernelConfig {
            name: "v1_baseline (32t, scalar)",
            pso_name: "matvec_q4_0",
            threads_per_group: 32,
            rows_per_group: 1,
        },
        KernelConfig {
            name: "v2_vec4    (32t, float4)",
            pso_name: "matvec_q4_0_v2_vec4",
            threads_per_group: 32,
            rows_per_group: 1,
        },
        KernelConfig {
            name: "v3_multirow(256t, 4row)",
            pso_name: "matvec_q4_0_v3_multirow",
            threads_per_group: 256,
            rows_per_group: 4,
        },
        KernelConfig {
            name: "v4_simdgrp (256t, 32row)",
            pso_name: "matvec_q4_0_v4_simdgroup",
            threads_per_group: 256,
            rows_per_group: 32,
        },
    ];

    // Test all SmolLM-135M dimensions
    let dimensions = vec![
        (576, 576, "Q/O proj (576→576)"),
        (192, 576, "K/V proj (576→192)"),
        (1536, 576, "gate/up  (576→1536)"),
        (576, 1536, "down     (1536→576)"),
        (49152, 576, "lm_head  (576→49152)"),
    ];

    let warmup = 10;
    let iterations = 100;

    eprintln!("\n{}", "=".repeat(80));
    eprintln!("  MATVEC Q4_0 KERNEL SHOOTOUT");
    eprintln!("  Warmup: {warmup}, Iterations: {iterations}");
    eprintln!("{}\n", "=".repeat(80));

    for (out_dim, in_dim, label) in &dimensions {
        let out_dim = *out_dim;
        let in_dim = *in_dim;

        let weight_bytes = generate_weights(out_dim, in_dim);
        let input = generate_input(in_dim);

        let weight_buf = alloc_buffer_with_data(&gpu.device, &weight_bytes);
        let input_buf = alloc_buffer_with_data(&gpu.device, &input);
        let output_buf = alloc_buffer(&gpu.device, out_dim * 4);

        eprintln!("--- {label} ({out_dim}×{in_dim}) ---");

        let mut baseline_output: Option<Vec<f32>> = None;

        for config in &configs {
            let (avg_us, result) = bench_kernel(
                &gpu,
                &mut pso_cache,
                config,
                &weight_buf,
                &input_buf,
                &output_buf,
                out_dim,
                in_dim,
                warmup,
                iterations,
            );

            // Check correctness against baseline
            let correct = if let Some(ref baseline) = baseline_output {
                let max_diff = result
                    .iter()
                    .zip(baseline.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                if max_diff > 0.01 {
                    format!("MISMATCH max_diff={:.6}", max_diff)
                } else {
                    format!("OK (max_diff={:.6})", max_diff)
                }
            } else {
                baseline_output = Some(result.clone());
                "baseline".to_string()
            };

            // Estimate tok/s contribution (210 matvec dispatches per token for SmolLM)
            // This is a rough estimate: if matvec takes X us per dispatch,
            // and there are 210 per token, total matvec time = 210 * X us
            // But dimensions vary, so compute weighted contribution
            let tok_s_if_all = 1_000_000.0 / (avg_us * 210.0);

            eprintln!(
                "  {:<28}  {:>8.1} us  ({:>6.0} tok/s equiv)  {}",
                config.name, avg_us, tok_s_if_all, correct
            );
        }
        eprintln!();
    }

    // Summary: estimate total forward pass matvec time
    eprintln!("--- ESTIMATED TOTAL MATVEC TIME PER TOKEN ---");
    eprintln!("(SmolLM-135M: 7 matvec/layer × 30 layers = 210 dispatches)\n");
    eprintln!("Per-layer breakdown:");
    eprintln!("  3× Q/K/V proj (in=576):  Q(576), K(192), V(192)");
    eprintln!("  1× O proj (576→576)");
    eprintln!("  2× gate/up (576→1536)");
    eprintln!("  1× down (1536→576)");
    eprintln!("  Final: 1× lm_head (576→49152)\n");

    // Re-run each config to get total per-token estimate
    let layer_dims = vec![
        (576, 576, 1),  // Q proj
        (192, 576, 1),  // K proj
        (192, 576, 1),  // V proj
        (576, 576, 1),  // O proj
        (1536, 576, 1), // gate
        (1536, 576, 1), // up
        (576, 1536, 1), // down
    ];

    for config in &configs {
        let mut total_layer_us = 0.0f64;

        for (out_dim, in_dim, count) in &layer_dims {
            let weight_bytes = generate_weights(*out_dim, *in_dim);
            let input = generate_input(*in_dim);
            let weight_buf = alloc_buffer_with_data(&gpu.device, &weight_bytes);
            let input_buf = alloc_buffer_with_data(&gpu.device, &input);
            let output_buf = alloc_buffer(&gpu.device, *out_dim * 4);

            let (avg_us, _) = bench_kernel(
                &gpu,
                &mut pso_cache,
                config,
                &weight_buf,
                &input_buf,
                &output_buf,
                *out_dim,
                *in_dim,
                5,
                50,
            );
            total_layer_us += avg_us * *count as f64;
        }

        // lm_head
        let weight_bytes = generate_weights(49152, 576);
        let input = generate_input(576);
        let weight_buf = alloc_buffer_with_data(&gpu.device, &weight_bytes);
        let input_buf = alloc_buffer_with_data(&gpu.device, &input);
        let output_buf = alloc_buffer(&gpu.device, 49152 * 4);
        let (lm_head_us, _) = bench_kernel(
            &gpu,
            &mut pso_cache,
            config,
            &weight_buf,
            &input_buf,
            &output_buf,
            49152,
            576,
            5,
            50,
        );

        let total_us = total_layer_us * 30.0 + lm_head_us;
        let estimated_tok_s = 1_000_000.0 / total_us;

        eprintln!(
            "  {:<28}  layer: {:>6.1} us × 30 = {:>8.1} us  lm_head: {:>6.1} us  TOTAL: {:>8.1} us  ({:.0} tok/s matvec-only)",
            config.name,
            total_layer_us,
            total_layer_us * 30.0,
            lm_head_us,
            total_us,
            estimated_tok_s
        );
    }

    eprintln!("\nNote: actual tok/s will be lower due to attention, rmsnorm, rope, etc.");
    eprintln!("Current total forward pass: ~3.4ms (292 tok/s). Matvec is ~60-70% of that.");
}
