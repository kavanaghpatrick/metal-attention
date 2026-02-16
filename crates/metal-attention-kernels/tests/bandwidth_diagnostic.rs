//! GPU Bandwidth & Dispatch Overhead Diagnostic
//!
//! Measures:
//! 1. Achievable memory bandwidth (theoretical ceiling for our workload)
//! 2. Per-dispatch overhead within a single command buffer
//! 3. Per-dispatch overhead with separate command buffers
//! 4. Bandwidth at different occupancy levels (32 vs 256 threads/group)
//!
//! Run: cargo test --release -p metal-attention-kernels --test bandwidth_diagnostic -- --nocapture --ignored

use metal_attention_kernels::buffer::{alloc_buffer, alloc_buffer_with_data};
use metal_attention_kernels::device::GpuDevice;
use metal_attention_kernels::dispatch::{set_buffer, set_bytes};
use metal_attention_kernels::pipeline::{PsoCache, PsoKey};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
};
use std::time::Instant;

#[test]
#[ignore]
fn bandwidth_and_dispatch_diagnostic() {
    let gpu = GpuDevice::new();
    let mut pso_cache = PsoCache::new(gpu.library.clone());

    // Prewarm all PSOs upfront so we can use immutable get() later
    pso_cache.prewarm(&[
        PsoKey::simple("bandwidth_read_f32"),
        PsoKey::simple("bandwidth_read_q4_0"),
        PsoKey::simple("dispatch_overhead_noop"),
        PsoKey::simple("bandwidth_read_high_occupancy"),
        PsoKey::simple("bandwidth_read_f16"),
    ]);

    eprintln!("\n{}", "=".repeat(80));
    eprintln!("  GPU BANDWIDTH & DISPATCH OVERHEAD DIAGNOSTIC");
    eprintln!("{}\n", "=".repeat(80));

    // =========================================================================
    // Test 1: Pure memory bandwidth (large sequential read)
    // =========================================================================
    eprintln!("--- Test 1: Peak Memory Bandwidth (sequential float4 reads) ---");
    {
        let data_sizes_mb: Vec<usize> = vec![16, 32, 64, 128];
        let pso = pso_cache
            .get(&PsoKey::simple("bandwidth_read_f32"))
            .unwrap();

        for size_mb in &data_sizes_mb {
            let n_bytes = size_mb * 1024 * 1024;
            let n_float4 = n_bytes / 16;
            let data: Vec<f32> = (0..n_bytes / 4)
                .map(|i| (i % 1000) as f32 * 0.001)
                .collect();
            let data_buf = alloc_buffer_with_data(&gpu.device, &data);
            let out_buf = alloc_buffer(&gpu.device, 4);

            let n_float4_u32 = n_float4 as u32;

            // Warmup
            for _ in 0..3 {
                let cmd = gpu.command_queue.commandBuffer().unwrap();
                let enc = cmd.computeCommandEncoder().unwrap();
                enc.setComputePipelineState(pso);
                set_buffer(&enc, &data_buf, 0, 0);
                set_buffer(&enc, &out_buf, 0, 1);
                set_bytes(&enc, &n_float4_u32, 2);
                enc.dispatchThreads_threadsPerThreadgroup(
                    MTLSize {
                        width: 65536,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 256,
                        height: 1,
                        depth: 1,
                    },
                );
                enc.endEncoding();
                cmd.commit();
                cmd.waitUntilCompleted();
            }

            // Timed
            let iterations = 20;
            let start = Instant::now();
            for _ in 0..iterations {
                let cmd = gpu.command_queue.commandBuffer().unwrap();
                let enc = cmd.computeCommandEncoder().unwrap();
                enc.setComputePipelineState(pso);
                set_buffer(&enc, &data_buf, 0, 0);
                set_buffer(&enc, &out_buf, 0, 1);
                set_bytes(&enc, &n_float4_u32, 2);
                enc.dispatchThreads_threadsPerThreadgroup(
                    MTLSize {
                        width: 65536,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 256,
                        height: 1,
                        depth: 1,
                    },
                );
                enc.endEncoding();
                cmd.commit();
                cmd.waitUntilCompleted();
            }
            let elapsed = start.elapsed();
            let avg_us = elapsed.as_micros() as f64 / iterations as f64;
            let bandwidth_gbs = (n_bytes as f64 / 1e9) / (avg_us / 1e6);

            eprintln!(
                "  {:>4} MB:  {:>8.1} us  ({:>6.1} GB/s)",
                size_mb, avg_us, bandwidth_gbs
            );
        }
    }
    eprintln!();

    // =========================================================================
    // Test 2: Dispatch overhead (noop kernel, single vs batched command buffers)
    // =========================================================================
    eprintln!("--- Test 2: Dispatch Overhead ---");
    {
        let pso = pso_cache
            .get(&PsoKey::simple("dispatch_overhead_noop"))
            .unwrap();
        let out_buf = alloc_buffer(&gpu.device, 4);

        // 2a: Separate command buffers (our benchmark pattern)
        let iterations = 500;
        let start = Instant::now();
        for _ in 0..iterations {
            let cmd = gpu.command_queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(pso);
            set_buffer(&enc, &out_buf, 0, 0);
            enc.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }
        let elapsed = start.elapsed();
        let avg_us_separate = elapsed.as_micros() as f64 / iterations as f64;
        eprintln!(
            "  Separate cmd buffers (commit+wait each): {:>6.1} us/dispatch",
            avg_us_separate
        );

        // 2b: Single command buffer, N dispatches encoded, single commit+wait
        for n_dispatches in [10, 50, 100, 200, 484] {
            let start = Instant::now();
            let iterations = 50;
            for _ in 0..iterations {
                let cmd = gpu.command_queue.commandBuffer().unwrap();
                let enc = cmd.computeCommandEncoder().unwrap();
                for _ in 0..n_dispatches {
                    enc.setComputePipelineState(pso);
                    set_buffer(&enc, &out_buf, 0, 0);
                    enc.dispatchThreads_threadsPerThreadgroup(
                        MTLSize {
                            width: 1,
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: 1,
                            height: 1,
                            depth: 1,
                        },
                    );
                }
                enc.endEncoding();
                cmd.commit();
                cmd.waitUntilCompleted();
            }
            let elapsed = start.elapsed();
            let avg_us_total = elapsed.as_micros() as f64 / iterations as f64;
            let per_dispatch = avg_us_total / n_dispatches as f64;

            eprintln!(
                "  Single cmd buf, {:>3} dispatches: {:>8.1} us total ({:>5.1} us/dispatch)",
                n_dispatches, avg_us_total, per_dispatch
            );
        }
    }
    eprintln!();

    // =========================================================================
    // Test 3: Bandwidth at different occupancy (32 vs 256 threads/group)
    // =========================================================================
    eprintln!("--- Test 3: Bandwidth vs Occupancy (70MB read, simulating weight load) ---");
    {
        let pso_high = pso_cache
            .get(&PsoKey::simple("bandwidth_read_high_occupancy"))
            .unwrap();
        let pso_low = pso_cache
            .get(&PsoKey::simple("bandwidth_read_f32"))
            .unwrap();

        // 70MB to match SmolLM weight size
        let n_bytes: usize = 70 * 1024 * 1024;
        let n_float4 = n_bytes / 16;
        let data: Vec<f32> = (0..n_bytes / 4)
            .map(|i| (i % 1000) as f32 * 0.001)
            .collect();
        let data_buf = alloc_buffer_with_data(&gpu.device, &data);
        let out_buf = alloc_buffer(&gpu.device, 1024 * 4);
        let n_float4_u32 = n_float4 as u32;

        // High occupancy: 1024 groups × 256 threads = 262K threads
        let iterations = 20;
        // Warmup
        for _ in 0..3 {
            let cmd = gpu.command_queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(pso_high);
            set_buffer(&enc, &data_buf, 0, 0);
            set_buffer(&enc, &out_buf, 0, 1);
            set_bytes(&enc, &n_float4_u32, 2);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: 1024,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }

        let start = Instant::now();
        for _ in 0..iterations {
            let cmd = gpu.command_queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(pso_high);
            set_buffer(&enc, &data_buf, 0, 0);
            set_buffer(&enc, &out_buf, 0, 1);
            set_bytes(&enc, &n_float4_u32, 2);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: 1024,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }
        let elapsed = start.elapsed();
        let avg_us_high = elapsed.as_micros() as f64 / iterations as f64;
        let bw_high = (n_bytes as f64 / 1e9) / (avg_us_high / 1e6);

        // Low occupancy: 65536 threads total, 32/group = 2048 groups
        // (similar to our actual matvec dispatch pattern)
        for _ in 0..3 {
            let cmd = gpu.command_queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(pso_low);
            set_buffer(&enc, &data_buf, 0, 0);
            set_buffer(&enc, &out_buf, 0, 1);
            set_bytes(&enc, &n_float4_u32, 2);
            enc.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: 65536,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }

        let start = Instant::now();
        for _ in 0..iterations {
            let cmd = gpu.command_queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(pso_low);
            set_buffer(&enc, &data_buf, 0, 0);
            set_buffer(&enc, &out_buf, 0, 1);
            set_bytes(&enc, &n_float4_u32, 2);
            enc.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: 65536,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }
        let elapsed = start.elapsed();
        let avg_us_low = elapsed.as_micros() as f64 / iterations as f64;
        let bw_low = (n_bytes as f64 / 1e9) / (avg_us_low / 1e6);

        eprintln!(
            "  High occupancy (1024 groups × 256 threads): {:>8.1} us  ({:.1} GB/s)",
            avg_us_high, bw_high
        );
        eprintln!(
            "  Low  occupancy (2048 groups × 32 threads):  {:>8.1} us  ({:.1} GB/s)",
            avg_us_low, bw_low
        );
        eprintln!("  Ratio: {:.2}x", bw_high / bw_low);
    }
    eprintln!();

    // =========================================================================
    // Test 4: Simulated forward pass dispatch pattern
    // =========================================================================
    eprintln!("--- Test 4: Simulated Forward Pass (484 dispatches, real work) ---");
    {
        // Simulate 484 matvec dispatches with real Q4_0 weight reads
        // Each dispatch reads a different slice of a large weight buffer
        let pso = pso_cache
            .get(&PsoKey::simple("bandwidth_read_q4_0"))
            .unwrap();

        // ~70MB of Q4_0 blocks
        let n_blocks_total: usize = 70 * 1024 * 1024 / 18;
        let blocks_data: Vec<u8> = (0..n_blocks_total * 18).map(|i| (i % 256) as u8).collect();
        let blocks_buf = alloc_buffer_with_data(&gpu.device, &blocks_data);
        let out_buf = alloc_buffer(&gpu.device, 484 * 4);

        // Split into 484 "dispatches" each reading n_blocks_total/484 blocks
        let blocks_per_dispatch = (n_blocks_total / 484) as u32;

        // Single command buffer, 484 dispatches
        let iterations = 20;
        // Warmup
        for _ in 0..3 {
            let cmd = gpu.command_queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            for d in 0..484u32 {
                enc.setComputePipelineState(pso);
                set_buffer(
                    &enc,
                    &blocks_buf,
                    (d * blocks_per_dispatch * 18) as usize,
                    0,
                );
                set_buffer(&enc, &out_buf, (d * 4) as usize, 1);
                set_bytes(&enc, &blocks_per_dispatch, 2);
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 32,
                        height: 1,
                        depth: 1,
                    },
                );
            }
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }

        let start = Instant::now();
        for _ in 0..iterations {
            let cmd = gpu.command_queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            for d in 0..484u32 {
                enc.setComputePipelineState(pso);
                set_buffer(
                    &enc,
                    &blocks_buf,
                    (d * blocks_per_dispatch * 18) as usize,
                    0,
                );
                set_buffer(&enc, &out_buf, (d * 4) as usize, 1);
                set_bytes(&enc, &blocks_per_dispatch, 2);
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 32,
                        height: 1,
                        depth: 1,
                    },
                );
            }
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }
        let elapsed = start.elapsed();
        let avg_us = elapsed.as_micros() as f64 / iterations as f64;
        let total_bytes_read = n_blocks_total * 18;
        let bandwidth = (total_bytes_read as f64 / 1e9) / (avg_us / 1e6);

        eprintln!(
            "  484 dispatches, 70MB total Q4_0: {:>8.1} us  ({:.1} GB/s, {:.0} tok/s equiv)",
            avg_us,
            bandwidth,
            1_000_000.0 / avg_us
        );

        // Now same data, ONE dispatch reading everything
        let n_blocks_total_u32 = n_blocks_total as u32;
        for _ in 0..3 {
            let cmd = gpu.command_queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(pso);
            set_buffer(&enc, &blocks_buf, 0, 0);
            set_buffer(&enc, &out_buf, 0, 1);
            set_bytes(&enc, &n_blocks_total_u32, 2);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }

        let start = Instant::now();
        for _ in 0..iterations {
            let cmd = gpu.command_queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(pso);
            set_buffer(&enc, &blocks_buf, 0, 0);
            set_buffer(&enc, &out_buf, 0, 1);
            set_bytes(&enc, &n_blocks_total_u32, 2);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }
        let elapsed = start.elapsed();
        let avg_us_single = elapsed.as_micros() as f64 / iterations as f64;
        let bandwidth_single = (total_bytes_read as f64 / 1e9) / (avg_us_single / 1e6);

        eprintln!(
            "  1 dispatch, 70MB total Q4_0:     {:>8.1} us  ({:.1} GB/s)",
            avg_us_single, bandwidth_single
        );
        eprintln!(
            "  Dispatch overhead cost: {:>8.1} us ({:.1}% of total time)",
            avg_us - avg_us_single,
            (avg_us - avg_us_single) / avg_us * 100.0
        );
    }
    eprintln!();

    // =========================================================================
    // Test 5: PSO switch overhead
    // =========================================================================
    eprintln!("--- Test 5: PSO Switch Cost ---");
    {
        let pso_a = pso_cache
            .get(&PsoKey::simple("dispatch_overhead_noop"))
            .unwrap();
        let pso_b = pso_cache
            .get(&PsoKey::simple("bandwidth_read_f32"))
            .unwrap();
        let out_buf = alloc_buffer(&gpu.device, 4);
        let n: u32 = 1;

        // Same PSO repeated
        let iterations = 100;
        let start = Instant::now();
        for _ in 0..iterations {
            let cmd = gpu.command_queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            for _ in 0..100 {
                enc.setComputePipelineState(pso_a);
                set_buffer(&enc, &out_buf, 0, 0);
                enc.dispatchThreads_threadsPerThreadgroup(
                    MTLSize {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                );
            }
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }
        let same_pso_us = start.elapsed().as_micros() as f64 / iterations as f64;

        // Alternating PSOs
        let start = Instant::now();
        for _ in 0..iterations {
            let cmd = gpu.command_queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            for i in 0..100 {
                if i % 2 == 0 {
                    enc.setComputePipelineState(pso_a);
                } else {
                    enc.setComputePipelineState(pso_b);
                    set_bytes(&enc, &n, 2);
                }
                set_buffer(&enc, &out_buf, 0, 0);
                set_buffer(&enc, &out_buf, 0, 1);
                enc.dispatchThreads_threadsPerThreadgroup(
                    MTLSize {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                );
            }
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }
        let alt_pso_us = start.elapsed().as_micros() as f64 / iterations as f64;

        eprintln!(
            "  100 dispatches, same PSO:        {:>6.1} us ({:.2} us/dispatch)",
            same_pso_us,
            same_pso_us / 100.0
        );
        eprintln!(
            "  100 dispatches, alternating PSO:  {:>6.1} us ({:.2} us/dispatch)",
            alt_pso_us,
            alt_pso_us / 100.0
        );
        eprintln!(
            "  PSO switch penalty: {:.2} us/switch",
            (alt_pso_us - same_pso_us) / 50.0 // 50 switches out of 100 dispatches
        );
    }

    eprintln!("\n{}", "=".repeat(80));
    eprintln!("  ANALYSIS");
    eprintln!("{}", "=".repeat(80));
    eprintln!("  SmolLM-135M weights: ~70 MB");
    eprintln!("  M4 theoretical bandwidth: 120 GB/s");
    eprintln!("  Theoretical minimum: 70MB / 120 GB/s = 583 us = 1,715 tok/s");
    eprintln!("  Current actual: 3,420 us = 292 tok/s (17% utilization)");
    eprintln!("  Gap to explain: 2,837 us\n");
}
