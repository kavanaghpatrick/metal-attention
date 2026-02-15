//! Unit tests for the GPU argmax two-stage parallel reduction kernel.
//!
//! Tests run the argmax_reduce + argmax_final Metal kernels directly on
//! synthetic input vectors and compare against CPU argmax. Does NOT require
//! a model file — only needs the metallib with argmax shaders.

use metal_attention_kernels::buffer::{alloc_buffer, alloc_buffer_private, alloc_buffer_with_data, read_buffer_slice};
use metal_attention_kernels::device::GpuDevice;
use metal_attention_kernels::dispatch::{set_buffer, set_bytes};
use metal_attention_kernels::pipeline::{PsoCache, PsoKey};
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize};

/// Run GPU argmax on the given f32 slice; returns the index of the max element.
fn gpu_argmax(data: &[f32]) -> u32 {
    let vocab_size = data.len();
    assert!(vocab_size > 0, "data must be non-empty");

    let device = GpuDevice::new();
    let mut pso_cache = PsoCache::new(device.library.clone());
    pso_cache.prewarm(&[
        PsoKey::simple("argmax_reduce"),
        PsoKey::simple("argmax_final"),
    ]);

    // Allocate buffers
    let logits_buf = alloc_buffer_with_data(&device.device, data);
    let num_groups = (vocab_size + 256 * 4 - 1) / (256 * 4);
    let partial_vals = alloc_buffer_private(&device.device, num_groups * std::mem::size_of::<f32>());
    let partial_idxs = alloc_buffer_private(&device.device, num_groups * std::mem::size_of::<u32>());
    let result_buf = alloc_buffer(&device.device, std::mem::size_of::<u32>());

    // Create command buffer + compute encoder
    let cmd_buf = device
        .command_queue
        .commandBuffer()
        .expect("Failed to create command buffer");
    let encoder = cmd_buf
        .computeCommandEncoder()
        .expect("Failed to create compute encoder");

    let vocab_size_u32 = vocab_size as u32;
    let num_groups_u32 = num_groups as u32;

    // Stage 1: argmax_reduce
    let pso_reduce = pso_cache
        .get(&PsoKey::simple("argmax_reduce"))
        .expect("argmax_reduce PSO missing");
    encoder.setComputePipelineState(pso_reduce);
    set_buffer(&encoder, &logits_buf, 0, 0);
    set_bytes(&encoder, &vocab_size_u32, 1);
    set_buffer(&encoder, &partial_vals, 0, 2);
    set_buffer(&encoder, &partial_idxs, 0, 3);

    let grid = MTLSize { width: num_groups, height: 1, depth: 1 };
    let tg = MTLSize { width: 256, height: 1, depth: 1 };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);

    // Stage 2: argmax_final
    let pso_final = pso_cache
        .get(&PsoKey::simple("argmax_final"))
        .expect("argmax_final PSO missing");
    encoder.setComputePipelineState(pso_final);
    set_buffer(&encoder, &partial_vals, 0, 0);
    set_buffer(&encoder, &partial_idxs, 0, 1);
    set_bytes(&encoder, &num_groups_u32, 2);
    set_buffer(&encoder, &result_buf, 0, 3);

    let grid_final = MTLSize { width: 1, height: 1, depth: 1 };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_final, tg);

    encoder.endEncoding();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    let result: Vec<u32> = unsafe { read_buffer_slice(&result_buf, 1) };
    result[0]
}

/// CPU reference argmax: returns index of the maximum finite value.
/// Skips NaN values (consistent with GPU `>` comparison semantics where NaN > x is false).
fn cpu_argmax(data: &[f32]) -> u32 {
    let mut max_val = f32::NEG_INFINITY;
    let mut max_idx: u32 = 0;
    for (i, &v) in data.iter().enumerate() {
        if v > max_val {
            max_val = v;
            max_idx = i as u32;
        }
    }
    max_idx
}

// ---------------------------------------------------------------------------
// Test cases
// ---------------------------------------------------------------------------

#[test]
fn argmax_known_vector() {
    // Simple vector where the max is at a known position
    let data = vec![0.1, 0.5, 0.3, 0.9, 0.2, 0.7, 0.4, 0.8, 0.6, 0.0];
    let result = gpu_argmax(&data);
    assert_eq!(result, 3, "argmax of [0.1, 0.5, 0.3, 0.9, ...] should be index 3");
}

#[test]
fn argmax_all_same_values() {
    // All elements equal -- GPU reduction uses `>` (strict), so first element wins
    // because initial max is -INFINITY, and all values tie after the first.
    // The kernel iterates forward, so index 0 gets the first `> -INF` comparison.
    let data = vec![1.0; 1024];
    let result = gpu_argmax(&data);
    assert_eq!(result, 0, "all-same values should return index 0 (first wins with strict >)");
}

#[test]
fn argmax_max_at_last_index() {
    let n = 2048;
    let mut data = vec![0.0f32; n];
    data[n - 1] = 999.0;
    let result = gpu_argmax(&data);
    assert_eq!(
        result,
        (n - 1) as u32,
        "max at last index should be found"
    );
}

#[test]
fn argmax_nan_not_selected() {
    // NaN values should not be selected because NaN > x is false in IEEE754.
    // Put NaN at several positions, real max at index 5.
    let mut data = vec![0.0f32; 32];
    data[0] = f32::NAN;
    data[5] = 10.0;
    data[10] = f32::NAN;
    data[15] = f32::NAN;
    data[20] = 5.0;
    let result = gpu_argmax(&data);
    assert_eq!(result, 5, "NaN should not be selected; max is at index 5");
}

#[test]
fn argmax_inf_handling() {
    // +inf should win over all finite values; -inf should lose.
    let mut data = vec![100.0f32; 64];
    data[10] = f32::NEG_INFINITY;
    data[30] = f32::INFINITY;
    data[50] = f32::NEG_INFINITY;
    let result = gpu_argmax(&data);
    assert_eq!(result, 30, "+inf at index 30 should be selected");
}

#[test]
fn argmax_full_vocab_49152() {
    // Full SmolLM vocab size: 49152 elements.
    // Use a pattern with a unique, clear maximum to avoid tie-breaking ambiguity.
    // Place a unique spike at a known position well into the array.
    let vocab_size: usize = 49152;
    let spike_idx: usize = 31337;
    let mut data: Vec<f32> = (0..vocab_size)
        .map(|i| {
            // Linear ramp from -1.0 to 1.0, guarantees no ties at the spike
            -1.0 + (i as f32 / vocab_size as f32) * 2.0
        })
        .collect();
    // Place spike well above the ramp's max (~1.0)
    data[spike_idx] = 100.0;

    let gpu_result = gpu_argmax(&data);

    assert_eq!(
        gpu_result, spike_idx as u32,
        "GPU argmax ({}) != expected spike index ({}) for vocab_size=49152",
        gpu_result, spike_idx
    );
    assert!(
        (gpu_result as usize) < vocab_size,
        "result token_id ({}) must be < vocab_size ({})",
        gpu_result, vocab_size
    );
}

#[test]
fn argmax_full_vocab_49152_random() {
    // Full vocab with pseudo-random data from a simple LCG.
    // Verifies GPU matches CPU argmax when there is a unique maximum.
    let vocab_size: usize = 49152;
    let mut rng_state: u64 = 12345;
    let data: Vec<f32> = (0..vocab_size)
        .map(|_| {
            // Simple LCG for deterministic pseudo-random f32
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            // Map to [-1, 1] range
            ((rng_state >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0
        })
        .collect();

    let gpu_result = gpu_argmax(&data);
    let cpu_result = cpu_argmax(&data);

    // With random data, ties are astronomically unlikely (f32 has ~7 digits of precision)
    assert_eq!(
        gpu_result, cpu_result,
        "GPU argmax ({}) != CPU argmax ({}) for random vocab_size=49152",
        gpu_result, cpu_result
    );
    assert!(
        (gpu_result as usize) < vocab_size,
        "result token_id ({}) must be < vocab_size ({})",
        gpu_result, vocab_size
    );
}

#[test]
fn argmax_property_result_in_range() {
    // Property test: for various sizes, result should always be < vocab_size.
    for &size in &[1, 7, 255, 256, 257, 1023, 1024, 1025, 4096, 10000, 49152] {
        let data: Vec<f32> = (0..size).map(|i| (i as f32 * 0.37).cos()).collect();
        let result = gpu_argmax(&data);
        assert!(
            (result as usize) < size,
            "result {} out of range for size {}",
            result, size
        );
        // Also verify matches CPU
        let cpu = cpu_argmax(&data);
        assert_eq!(
            result, cpu,
            "GPU ({}) != CPU ({}) for size {}",
            result, cpu, size
        );
    }
}

#[test]
fn argmax_single_element() {
    let data = vec![42.0f32];
    let result = gpu_argmax(&data);
    assert_eq!(result, 0, "single element should return index 0");
}

#[test]
fn argmax_two_elements_max_second() {
    let data = vec![1.0f32, 2.0];
    let result = gpu_argmax(&data);
    assert_eq!(result, 1, "max at index 1 of 2-element vector");
}

#[test]
fn argmax_negative_values() {
    // All negative: most negative values, max is the least negative
    let data = vec![-10.0, -5.0, -100.0, -1.0, -50.0];
    let result = gpu_argmax(&data);
    assert_eq!(result, 3, "argmax of all-negative should find -1.0 at index 3");
}
