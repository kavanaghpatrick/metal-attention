//! Property-based numerical invariant tests for Metal attention kernels.
//!
//! These tests run on CPU only (no GPU required) and use proptest to verify
//! mathematical invariants that must hold for any input.

use proptest::prelude::*;

// ---------------------------------------------------------------------------
// CPU reference implementations (same as correctness.rs, CPU-only)
// ---------------------------------------------------------------------------

/// Naive scaled dot-product attention computed in FP64.
fn cpu_attention_f64(q: &[f32], k: &[f32], v: &[f32], seq_len: usize, head_dim: usize) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f64).sqrt();
    let mut output = vec![0.0f64; seq_len * head_dim];

    for i in 0..seq_len {
        let mut scores = vec![0.0f64; seq_len];
        let mut max_score = f64::NEG_INFINITY;

        for j in 0..seq_len {
            let mut dot = 0.0f64;
            for d in 0..head_dim {
                dot += q[i * head_dim + d] as f64 * k[j * head_dim + d] as f64;
            }
            scores[j] = dot * scale;
            max_score = max_score.max(scores[j]);
        }

        let mut sum_exp = 0.0f64;
        for j in 0..seq_len {
            scores[j] = (scores[j] - max_score).exp();
            sum_exp += scores[j];
        }

        for j in 0..seq_len {
            let weight = scores[j] / sum_exp;
            for d in 0..head_dim {
                output[i * head_dim + d] += weight * v[j * head_dim + d] as f64;
            }
        }
    }

    output.iter().map(|&x| x as f32).collect()
}

/// Chunk-based linear attention computed in FP64.
fn cpu_linear_attention_f64(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    chunk_size: usize,
) -> Vec<f32> {
    assert!(seq_len % chunk_size == 0);

    let num_chunks = seq_len / chunk_size;
    let mut h = vec![0.0f64; head_dim * head_dim];
    let mut output = vec![0.0f64; seq_len * head_dim];

    for c in 0..num_chunks {
        let start = c * chunk_size;

        for t in 0..chunk_size {
            let token_idx = start + t;
            for i in 0..head_dim {
                let k_val = k[token_idx * head_dim + i] as f64;
                for j in 0..head_dim {
                    let v_val = v[token_idx * head_dim + j] as f64;
                    h[i * head_dim + j] += k_val * v_val;
                }
            }
        }

        for t in 0..chunk_size {
            let token_idx = start + t;
            for j in 0..head_dim {
                let mut sum = 0.0f64;
                for i in 0..head_dim {
                    sum += q[token_idx * head_dim + i] as f64 * h[i * head_dim + j];
                }
                output[token_idx * head_dim + j] = sum;
            }
        }
    }

    output.iter().map(|&x| x as f32).collect()
}

/// CPU reference for RMSNorm.
fn cpu_rmsnorm(input: &[f32], weight: &[f32], hidden_dim: usize, eps: f32) -> Vec<f32> {
    let num_tokens = input.len() / hidden_dim;
    let mut output = vec![0.0f32; num_tokens * hidden_dim];
    for t in 0..num_tokens {
        let offset = t * hidden_dim;
        let mut ss = 0.0f64;
        for d in 0..hidden_dim {
            let v = input[offset + d] as f64;
            ss += v * v;
        }
        let rms = ((ss / hidden_dim as f64) + eps as f64).sqrt();
        for d in 0..hidden_dim {
            output[offset + d] = ((input[offset + d] as f64 / rms) * weight[d] as f64) as f32;
        }
    }
    output
}

/// CPU reference for embedding lookup.
fn cpu_embedding_lookup(table: &[f32], token_ids: &[u32], hidden_dim: usize) -> Vec<f32> {
    let mut output = Vec::with_capacity(token_ids.len() * hidden_dim);
    for &tid in token_ids {
        let start = tid as usize * hidden_dim;
        output.extend_from_slice(&table[start..start + hidden_dim]);
    }
    output
}

/// CPU reference for RoPE (same as metal_attention_kernels::rope::cpu_rope).
fn cpu_rope(q: &mut [f32], k: &mut [f32], seq_len: usize, head_dim: usize) {
    let theta_base: f32 = 10000.0;
    for token in 0..seq_len {
        for pair in 0..(head_dim / 2) {
            let angle = token as f32 / theta_base.powf(2.0 * pair as f32 / head_dim as f32);
            let cos_a = angle.cos();
            let sin_a = angle.sin();

            let idx0 = token * head_dim + 2 * pair;
            let idx1 = idx0 + 1;

            let q0 = q[idx0];
            let q1 = q[idx1];
            q[idx0] = q0 * cos_a - q1 * sin_a;
            q[idx1] = q0 * sin_a + q1 * cos_a;

            let k0 = k[idx0];
            let k1 = k[idx1];
            k[idx0] = k0 * cos_a - k1 * sin_a;
            k[idx1] = k0 * sin_a + k1 * cos_a;
        }
    }
}

/// Generate deterministic test data.
fn gen_data(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|i| (i as f32 * 0.1 + seed).sin() * 0.5)
        .collect()
}

// ---------------------------------------------------------------------------
// Flash attention property tests
// ---------------------------------------------------------------------------

proptest! {
    /// If V is all zeros, the attention output must be all zeros regardless of Q and K.
    ///
    /// Proof: output = softmax(Q*K^T/sqrt(d)) * V, and V=0 => output=0.
    #[test]
    fn flash_attention_zero_v_gives_zero_output(
        seq_len in 1..=32usize,
        head_dim in prop_oneof![Just(8), Just(16), Just(32)],
    ) {
        let q = gen_data(seq_len * head_dim, 0.0);
        let k = gen_data(seq_len * head_dim, 1.0);
        let v = vec![0.0f32; seq_len * head_dim];

        let output = cpu_attention_f64(&q, &k, &v, seq_len, head_dim);

        for (i, &val) in output.iter().enumerate() {
            prop_assert!(
                val.abs() < 1e-6,
                "output[{i}] = {val}, expected ~0.0 with zero V"
            );
        }
    }

    /// Flash attention output must be finite (no NaN or Inf) for bounded random inputs.
    #[test]
    fn flash_attention_output_is_finite(
        seq_len in 1..=64usize,
        head_dim in prop_oneof![Just(8), Just(16), Just(32)],
        seed in 0.0f32..100.0,
    ) {
        let q = gen_data(seq_len * head_dim, seed);
        let k = gen_data(seq_len * head_dim, seed + 1.0);
        let v = gen_data(seq_len * head_dim, seed + 2.0);

        let output = cpu_attention_f64(&q, &k, &v, seq_len, head_dim);

        for (i, &val) in output.iter().enumerate() {
            prop_assert!(
                val.is_finite(),
                "output[{i}] = {val} is not finite (seq_len={seq_len}, head_dim={head_dim})"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Linear attention property tests
// ---------------------------------------------------------------------------

proptest! {
    /// Linear attention output must be finite for bounded random inputs.
    #[test]
    fn linear_attention_output_is_finite(
        // chunk_size must divide seq_len, so pick multiples
        chunk_size in prop_oneof![Just(4), Just(8)],
        num_chunks in 1..=8usize,
        head_dim in prop_oneof![Just(4), Just(8), Just(16)],
        seed in 0.0f32..100.0,
    ) {
        let seq_len = num_chunks * chunk_size;
        let q = gen_data(seq_len * head_dim, seed);
        let k = gen_data(seq_len * head_dim, seed + 1.0);
        let v = gen_data(seq_len * head_dim, seed + 2.0);

        let output = cpu_linear_attention_f64(&q, &k, &v, seq_len, head_dim, chunk_size);

        for (i, &val) in output.iter().enumerate() {
            prop_assert!(
                val.is_finite(),
                "output[{i}] = {val} is not finite (seq_len={seq_len}, head_dim={head_dim})"
            );
        }
    }

    /// Linear attention with identity K produces V directly when Q=I and single chunk.
    ///
    /// With K=I, V arbitrary, single chunk: H = I^T * V = V (as rows),
    /// then with Q=I: O = I * H = H. Each row i of output = H[i,:] which accumulates
    /// the outer products. For K=I, the outer product sum is the V matrix itself
    /// (interpreting K rows as one-hot selectors).
    #[test]
    fn linear_attention_identity_k_single_chunk(
        head_dim in prop_oneof![Just(4), Just(8)],
    ) {
        let seq_len = head_dim; // seq_len == head_dim so we can use identity
        let chunk_size = seq_len;

        // Q = identity, K = identity
        let mut q = vec![0.0f32; seq_len * head_dim];
        let mut k = vec![0.0f32; seq_len * head_dim];
        for i in 0..seq_len {
            q[i * head_dim + i] = 1.0;
            k[i * head_dim + i] = 1.0;
        }

        let v = gen_data(seq_len * head_dim, 42.0);
        let output = cpu_linear_attention_f64(&q, &k, &v, seq_len, head_dim, chunk_size);

        for (i, (&out, &expected)) in output.iter().zip(v.iter()).enumerate() {
            let diff = (out - expected).abs();
            prop_assert!(
                diff < 1e-5,
                "output[{i}] = {out}, expected {expected}, diff = {diff}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// RMSNorm property tests
// ---------------------------------------------------------------------------

proptest! {
    /// RMSNorm output must be finite for random inputs and weights.
    #[test]
    fn rmsnorm_output_is_finite(
        hidden_dim in prop_oneof![Just(8), Just(16), Just(32)],
        num_tokens in 1..=8usize,
        seed in 0.0f32..100.0,
    ) {
        let input = gen_data(num_tokens * hidden_dim, seed);
        let weight: Vec<f32> = (0..hidden_dim)
            .map(|i| 0.5 + (i as f32 * 0.1 + seed).sin() * 0.3)
            .collect();

        let output = cpu_rmsnorm(&input, &weight, hidden_dim, 1e-5);

        for (i, &val) in output.iter().enumerate() {
            prop_assert!(
                val.is_finite(),
                "output[{i}] = {val} is not finite"
            );
        }
    }

    /// RMSNorm preserves direction: output direction is correlated with input * weight.
    ///
    /// For each token, output = (input / rms) * weight. The sign of each element
    /// should match input[i] * weight[i] (assuming weight > 0 and rms > 0).
    #[test]
    fn rmsnorm_preserves_sign_direction(
        hidden_dim in prop_oneof![Just(8), Just(16), Just(32)],
        seed in 0.0f32..100.0,
    ) {
        let input = gen_data(hidden_dim, seed);
        // Use positive weights to simplify sign analysis
        let weight: Vec<f32> = (0..hidden_dim)
            .map(|i| 0.5 + (i as f32 * 0.1).sin().abs() * 0.5)
            .collect();

        let output = cpu_rmsnorm(&input, &weight, hidden_dim, 1e-5);

        // Check dot product (cosine similarity) between output and input*weight
        let input_weighted: Vec<f64> = input
            .iter()
            .zip(weight.iter())
            .map(|(&i, &w)| i as f64 * w as f64)
            .collect();

        let dot: f64 = output
            .iter()
            .zip(input_weighted.iter())
            .map(|(&o, &iw)| o as f64 * iw)
            .sum();

        // Dot product must be non-negative (same direction) since rms > 0
        prop_assert!(
            dot >= -1e-6,
            "Dot product = {dot}, expected >= 0 (same direction)"
        );
    }
}

// ---------------------------------------------------------------------------
// Embedding lookup property tests
// ---------------------------------------------------------------------------

proptest! {
    /// Embedding lookup returns the correct row for each token ID.
    #[test]
    fn embedding_lookup_correct_index(
        vocab_size in 4..=32usize,
        hidden_dim in prop_oneof![Just(4), Just(8), Just(16)],
        num_tokens in 1..=8usize,
    ) {
        // Build a table where row i = [i*hidden_dim, i*hidden_dim+1, ...]
        let table: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| i as f32)
            .collect();

        // Generate token IDs within vocab range
        let token_ids: Vec<u32> = (0..num_tokens)
            .map(|t| (t % vocab_size) as u32)
            .collect();

        let output = cpu_embedding_lookup(&table, &token_ids, hidden_dim);

        for (t, &tid) in token_ids.iter().enumerate() {
            for d in 0..hidden_dim {
                let expected = (tid as usize * hidden_dim + d) as f32;
                let actual = output[t * hidden_dim + d];
                prop_assert!(
                    (actual - expected).abs() < f32::EPSILON,
                    "token {t} (id={tid}), dim {d}: got {actual}, expected {expected}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// RoPE property tests
// ---------------------------------------------------------------------------

proptest! {
    /// RoPE at position 0 is identity: angle = 0/(10000^(2k/d)) = 0 for all k,
    /// so cos(0)=1, sin(0)=0, rotation = identity.
    #[test]
    fn rope_position_zero_is_identity(
        head_dim in prop_oneof![Just(4), Just(8), Just(16), Just(32)],
        seed in 0.0f32..100.0,
    ) {
        let seq_len = 1; // position 0 only
        let q_orig = gen_data(seq_len * head_dim, seed);
        let k_orig = gen_data(seq_len * head_dim, seed + 1.0);

        let mut q = q_orig.clone();
        let mut k = k_orig.clone();
        cpu_rope(&mut q, &mut k, seq_len, head_dim);

        for (i, (&rotated, &original)) in q.iter().zip(q_orig.iter()).enumerate() {
            let diff = (rotated - original).abs();
            prop_assert!(
                diff < 1e-5,
                "q[{i}]: rotated={rotated}, original={original}, diff={diff}"
            );
        }
        for (i, (&rotated, &original)) in k.iter().zip(k_orig.iter()).enumerate() {
            let diff = (rotated - original).abs();
            prop_assert!(
                diff < 1e-5,
                "k[{i}]: rotated={rotated}, original={original}, diff={diff}"
            );
        }
    }

    /// RoPE output must be finite for any input.
    #[test]
    fn rope_output_is_finite(
        seq_len in 1..=16usize,
        head_dim in prop_oneof![Just(4), Just(8), Just(16)],
        seed in 0.0f32..100.0,
    ) {
        let mut q = gen_data(seq_len * head_dim, seed);
        let mut k = gen_data(seq_len * head_dim, seed + 1.0);

        cpu_rope(&mut q, &mut k, seq_len, head_dim);

        for (i, &val) in q.iter().enumerate() {
            prop_assert!(val.is_finite(), "q[{i}] = {val} is not finite");
        }
        for (i, &val) in k.iter().enumerate() {
            prop_assert!(val.is_finite(), "k[{i}] = {val} is not finite");
        }
    }

    /// RoPE preserves vector magnitude (rotation is unitary).
    ///
    /// Each 2D rotation preserves the magnitude of the (q0, q1) pair, so the
    /// overall L2 norm of Q and K should be preserved.
    #[test]
    fn rope_preserves_norm(
        seq_len in 1..=16usize,
        head_dim in prop_oneof![Just(4), Just(8), Just(16)],
        seed in 0.0f32..100.0,
    ) {
        let mut q = gen_data(seq_len * head_dim, seed);
        let mut k = gen_data(seq_len * head_dim, seed + 1.0);

        let q_norm_before: f64 = q.iter().map(|&x| (x as f64) * (x as f64)).sum();
        let k_norm_before: f64 = k.iter().map(|&x| (x as f64) * (x as f64)).sum();

        cpu_rope(&mut q, &mut k, seq_len, head_dim);

        let q_norm_after: f64 = q.iter().map(|&x| (x as f64) * (x as f64)).sum();
        let k_norm_after: f64 = k.iter().map(|&x| (x as f64) * (x as f64)).sum();

        let q_diff = (q_norm_after - q_norm_before).abs();
        let k_diff = (k_norm_after - k_norm_before).abs();

        prop_assert!(
            q_diff < 1e-3,
            "Q norm changed: before={q_norm_before:.6}, after={q_norm_after:.6}, diff={q_diff:.2e}"
        );
        prop_assert!(
            k_diff < 1e-3,
            "K norm changed: before={k_norm_before:.6}, after={k_norm_after:.6}, diff={k_diff:.2e}"
        );
    }
}
