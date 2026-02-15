//! SSM (Selective State Space Model) dispatch: GPU kernel host code.
//!
//! Dispatches the `ssm_scan` Metal kernel for Mamba-style selective scan.
//! The kernel processes tokens sequentially, updating the [d_model, d_state]
//! hidden state with input-dependent A, B, C gates.

use crate::buffer::{alloc_buffer, alloc_buffer_with_data, read_buffer_slice};
use crate::device::GpuDevice;
use crate::pipeline::{PsoCache, PsoKey};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLSize,
};

/// Run the Mamba selective scan on the GPU.
///
/// Implements the recurrence:
///   h_t = A_t * h_{t-1} + B_t * x_t
///   y_t = C_t * h_t + D * x_t
///
/// # Arguments
/// - `device`: GpuDevice with Metal device, command queue, and shader library
/// - `pso_cache`: PSO cache for kernel compilation
/// - `x`: Input vectors [seq_len, d_model]
/// - `a`: Decay factors [seq_len, d_model], values in (0, 1)
/// - `b`: Input gate [seq_len, d_state]
/// - `c`: Output gate [seq_len, d_state]
/// - `d_param`: Skip connection scalar
/// - `state`: Initial hidden state [d_model, d_state]
/// - `seq_len`: Number of tokens to process
/// - `d_model`: Model dimension
/// - `d_state`: State dimension
///
/// # Returns
/// Tuple of (output [seq_len, d_model], final_state [d_model, d_state])
#[allow(clippy::too_many_arguments)]
pub fn dispatch_ssm_scan(
    device: &GpuDevice,
    pso_cache: &mut PsoCache,
    x: &[f32],
    a: &[f32],
    b: &[f32],
    c: &[f32],
    d_param: f32,
    state: &[f32],
    seq_len: usize,
    d_model: usize,
    d_state: usize,
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(x.len(), seq_len * d_model, "x length mismatch");
    assert_eq!(a.len(), seq_len * d_model, "A length mismatch");
    assert_eq!(b.len(), seq_len * d_state, "B length mismatch");
    assert_eq!(c.len(), seq_len * d_state, "C length mismatch");
    assert_eq!(state.len(), d_model * d_state, "State length mismatch");

    let out_len = seq_len * d_model;

    // Allocate Metal buffers
    let x_buf = alloc_buffer_with_data(&device.device, x);
    let a_buf = alloc_buffer_with_data(&device.device, a);
    let b_buf = alloc_buffer_with_data(&device.device, b);
    let c_buf = alloc_buffer_with_data(&device.device, c);
    let d_buf = alloc_buffer_with_data(&device.device, std::slice::from_ref(&d_param));
    let state_buf = alloc_buffer_with_data(&device.device, state);
    let o_buf = alloc_buffer(&device.device, out_len * std::mem::size_of::<f32>());

    // Scalar parameters
    let seq_len_u32 = seq_len as u32;
    let d_model_u32 = d_model as u32;
    let d_state_u32 = d_state as u32;
    let seq_len_buf = alloc_buffer_with_data(&device.device, std::slice::from_ref(&seq_len_u32));
    let d_model_buf = alloc_buffer_with_data(&device.device, std::slice::from_ref(&d_model_u32));
    let d_state_buf = alloc_buffer_with_data(&device.device, std::slice::from_ref(&d_state_u32));

    // Compile PSO (no function constants needed for this simple kernel)
    let pso_key = PsoKey::simple("ssm_scan");
    let pso = pso_cache.get_or_compile(&pso_key);

    // Determine threadgroup size
    let max_threads = pso.maxTotalThreadsPerThreadgroup();
    let total_state = d_model * d_state;
    let threads_per_tg = total_state.min(max_threads);

    // Create command buffer and compute encoder
    let command_buffer = device
        .command_queue
        .commandBuffer()
        .expect("Failed to create command buffer for ssm_scan");
    let encoder = command_buffer
        .computeCommandEncoder()
        .expect("Failed to create compute encoder for ssm_scan");

    encoder.setComputePipelineState(pso);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(&*x_buf), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&*a_buf), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&*b_buf), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(&*c_buf), 0, 3);
        encoder.setBuffer_offset_atIndex(Some(&*d_buf), 0, 4);
        encoder.setBuffer_offset_atIndex(Some(&*state_buf), 0, 5);
        encoder.setBuffer_offset_atIndex(Some(&*o_buf), 0, 6);
        encoder.setBuffer_offset_atIndex(Some(&*seq_len_buf), 0, 7);
        encoder.setBuffer_offset_atIndex(Some(&*d_model_buf), 0, 8);
        encoder.setBuffer_offset_atIndex(Some(&*d_state_buf), 0, 9);
    }

    // Single threadgroup -- kernel processes all tokens sequentially
    let threadgroups = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    let tg_size = MTLSize {
        width: threads_per_tg,
        height: 1,
        depth: 1,
    };

    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, tg_size);
    encoder.endEncoding();

    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    let status = command_buffer.status();
    assert_eq!(
        status,
        objc2_metal::MTLCommandBufferStatus::Completed,
        "ssm_scan command buffer failed with status {:?}. Error: {:?}",
        status,
        command_buffer.error()
    );

    // Read back output and final state
    let output = unsafe { read_buffer_slice(&o_buf, out_len) };
    let final_state = unsafe { read_buffer_slice(&state_buf, d_model * d_state) };

    (output, final_state)
}

/// CPU reference implementation of the Mamba selective scan for testing.
///
/// Implements:
///   h_t = A_t * h_{t-1} + B_t * x_t
///   y_t = C_t * h_t + D * x_t
#[allow(clippy::too_many_arguments)]
pub fn cpu_ssm_scan(
    x: &[f32],
    a: &[f32],
    b: &[f32],
    c: &[f32],
    d_param: f32,
    state: &mut [f32],
    seq_len: usize,
    d_model: usize,
    d_state: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; seq_len * d_model];

    for t in 0..seq_len {
        let x_off = t * d_model;
        let a_off = t * d_model;
        let b_off = t * d_state;
        let c_off = t * d_state;

        // Update state: h[m][s] = A[m] * h[m][s] + B[s] * x[m]
        for m in 0..d_model {
            for s in 0..d_state {
                state[m * d_state + s] =
                    a[a_off + m] * state[m * d_state + s] + b[b_off + s] * x[x_off + m];
            }
        }

        // Compute output: y[m] = sum_s(C[s] * h[m][s]) + D * x[m]
        for m in 0..d_model {
            let mut acc = 0.0f32;
            for s in 0..d_state {
                acc += c[c_off + s] * state[m * d_state + s];
            }
            output[x_off + m] = acc + d_param * x[x_off + m];
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::GpuDevice;
    use crate::pipeline::PsoCache;

    /// Simple deterministic pseudo-random number generator.
    struct SimpleRng {
        state: u64,
    }

    impl SimpleRng {
        fn new(seed: u64) -> Self {
            Self {
                state: seed.wrapping_add(1),
            }
        }

        fn next_u64(&mut self) -> u64 {
            self.state ^= self.state << 13;
            self.state ^= self.state >> 7;
            self.state ^= self.state << 17;
            self.state
        }

        fn next_f32(&mut self) -> f32 {
            (self.next_u64() & 0xFFFFFF) as f32 / 0xFFFFFF as f32
        }

        fn next_f32_range(&mut self, lo: f32, hi: f32) -> f32 {
            lo + self.next_f32() * (hi - lo)
        }
    }

    /// Test: GPU SSM scan matches CPU reference implementation.
    #[test]
    fn test_ssm_scan_gpu_vs_cpu() {
        let d_model = 16;
        let d_state = 4;
        let seq_len = 4;
        let mut rng = SimpleRng::new(42);

        // Generate random inputs
        let x: Vec<f32> = (0..seq_len * d_model)
            .map(|_| rng.next_f32_range(-0.5, 0.5))
            .collect();

        // A (decay) in (0, 1) -- use sigmoid of random values
        let a: Vec<f32> = (0..seq_len * d_model)
            .map(|_| {
                let v = rng.next_f32_range(-2.0, 2.0);
                1.0 / (1.0 + (-v).exp())
            })
            .collect();

        let b: Vec<f32> = (0..seq_len * d_state)
            .map(|_| rng.next_f32_range(-0.5, 0.5))
            .collect();
        let c: Vec<f32> = (0..seq_len * d_state)
            .map(|_| rng.next_f32_range(-0.5, 0.5))
            .collect();

        let d_param = 1.0f32;
        let state = vec![0.0f32; d_model * d_state];

        // CPU reference
        let mut cpu_state = state.clone();
        let cpu_output = cpu_ssm_scan(
            &x,
            &a,
            &b,
            &c,
            d_param,
            &mut cpu_state,
            seq_len,
            d_model,
            d_state,
        );

        // GPU dispatch
        let gpu = GpuDevice::new();
        let mut pso_cache = PsoCache::new(gpu.library.clone());
        let (gpu_output, gpu_state) = dispatch_ssm_scan(
            &gpu,
            &mut pso_cache,
            &x,
            &a,
            &b,
            &c,
            d_param,
            &state,
            seq_len,
            d_model,
            d_state,
        );

        // Compare outputs
        assert_eq!(cpu_output.len(), gpu_output.len());
        let atol = 1e-3;
        for i in 0..cpu_output.len() {
            let diff = (cpu_output[i] - gpu_output[i]).abs();
            assert!(
                diff < atol,
                "Output mismatch at index {}: CPU={}, GPU={}, diff={}",
                i,
                cpu_output[i],
                gpu_output[i],
                diff
            );
        }

        // Compare final states
        assert_eq!(cpu_state.len(), gpu_state.len());
        for i in 0..cpu_state.len() {
            let diff = (cpu_state[i] - gpu_state[i]).abs();
            assert!(
                diff < atol,
                "State mismatch at index {}: CPU={}, GPU={}, diff={}",
                i,
                cpu_state[i],
                gpu_state[i],
                diff
            );
        }
    }

    /// Test: SSM scan with zero initial state and D=0.
    #[test]
    fn test_ssm_scan_d_zero() {
        let d_model = 8;
        let d_state = 2;
        let seq_len = 2;

        // All ones input, A=0.5, B=1.0, C=1.0, D=0
        let x = vec![1.0f32; seq_len * d_model];
        let a = vec![0.5f32; seq_len * d_model];
        let b = vec![1.0f32; seq_len * d_state];
        let c = vec![1.0f32; seq_len * d_state];
        let d_param = 0.0f32;

        let mut state = vec![0.0f32; d_model * d_state];
        let output = cpu_ssm_scan(
            &x, &a, &b, &c, d_param, &mut state, seq_len, d_model, d_state,
        );

        // After t=0: h[m][s] = 0*0 + 1*1 = 1 for all m,s
        // y[m] = sum_s(1 * 1) = d_state = 2
        // After t=1: h[m][s] = 0.5*1 + 1*1 = 1.5
        // y[m] = sum_s(1 * 1.5) = d_state * 1.5 = 3.0
        for m in 0..d_model {
            assert!(
                (output[m] - d_state as f32).abs() < 1e-5,
                "t=0 output[{}] = {}, expected {}",
                m,
                output[m],
                d_state
            );
            assert!(
                (output[d_model + m] - d_state as f32 * 1.5).abs() < 1e-5,
                "t=1 output[{}] = {}, expected {}",
                m,
                output[d_model + m],
                d_state as f32 * 1.5
            );
        }
    }
}
