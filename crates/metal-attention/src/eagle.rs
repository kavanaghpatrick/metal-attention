#![allow(dead_code)]
//! EAGLE-3 speculative decoder: chain speculation with a lightweight draft head.
//!
//! `EagleDecoder` replaces the separate draft model used in standard speculative
//! decoding with an EAGLE draft head that reads hidden states from the target
//! model itself. The head fuses features from 3 internal transformer layers
//! (low/mid/high) and predicts draft tokens via a single decoder layer.
//!
//! Flow per speculation round:
//!   1. Draft N tokens autoregressively through the EAGLE head
//!   2. Verify all N tokens in one batched target forward pass
//!   3. Greedy accept/reject with KV cache rollback on mismatch
//!   4. Reset EAGLE head KV cache for next round

use std::path::Path;

use crate::eagle_head::EagleHead;
use crate::gpu_forward_pass::GpuForwardPass;
use crate::sampling::sample_greedy;
use crate::speculative::SpecStats;

use metal_attention_kernels::device::GpuDevice;

/// EAGLE-3 speculative decoder managing a target model and draft head.
///
/// The draft head is a lightweight network (~1.9% of target params) that
/// reads hidden states captured during target model inference to predict
/// draft tokens without a separate draft model.
pub struct EagleDecoder {
    /// Target model with eagle hidden state capture enabled.
    target: GpuForwardPass,
    /// EAGLE-3 draft head (FC fusion + single decoder layer).
    eagle_head: EagleHead,
    /// Number of draft tokens to propose per speculation round.
    n_draft: usize,
}

impl EagleDecoder {
    /// Create an EagleDecoder with random draft head weights (POC).
    ///
    /// Loads the target model from GGUF, enables hidden state capture
    /// at layers 0/16/31 (Mistral-7B configuration), and creates an
    /// EagleHead with random F32 weights for pipeline validation.
    ///
    /// # Arguments
    /// - `target_path`: Path to the target model GGUF file.
    /// - `n_draft`: Number of draft tokens per speculation round (default: 6).
    pub fn new_random(target_path: &Path, n_draft: usize) -> Result<Self, String> {
        let mut target = GpuForwardPass::from_gguf(target_path)?;

        // Enable hidden state capture at layers 0, 16, 31 (Mistral-7B)
        target.enable_eagle_capture(0, 16, 31);

        let device = GpuDevice::shared();
        let eagle_head = EagleHead::new_random(
            device,
            target.hidden_size(),
            target.num_heads(),
            target.num_kv_heads(),
            target.head_dim(),
            target.intermediate_size(),
            target.vocab_size(),
        );

        Ok(Self {
            target,
            eagle_head,
            n_draft,
        })
    }

    /// Create from pre-loaded components (useful for tests).
    pub fn from_parts(
        target: GpuForwardPass,
        eagle_head: EagleHead,
        n_draft: usize,
    ) -> Self {
        Self {
            target,
            eagle_head,
            n_draft,
        }
    }

    /// Reset both target and eagle head (clear KV caches and positions).
    pub fn reset(&mut self) {
        self.target.reset();
        self.eagle_head.reset_kv_cache();
    }

    /// Generate tokens using EAGLE-3 speculative decoding.
    ///
    /// Prefills the target model with `prompt`, then enters the
    /// draft-verify-accept/reject loop until `max_tokens` are generated.
    /// Calls `callback` with each accepted token for streaming output.
    ///
    /// Returns the generated token sequence and speculation statistics.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        max_tokens: usize,
        mut callback: impl FnMut(u32),
    ) -> Result<(Vec<u32>, SpecStats), String> {
        if prompt.is_empty() {
            return Err("prompt must not be empty".to_string());
        }

        // Prefill target with the full prompt.
        // After forward_prompt, target has the full prompt in its KV cache
        // and returns the argmax predicted next token.
        // Eagle capture buffers are populated during the batch prefill's
        // layer loop (captures hidden states at layers 0/16/31).
        let first_token = self.target.forward_prompt(prompt)?;

        let mut last_token = first_token;
        let mut generated = vec![last_token];
        callback(last_token);

        let mut stats = SpecStats {
            tokens_accepted: 0,
            rounds: 0,
            tokens_drafted: 0,
            draft_accepted: 0,
        };
        stats.tokens_accepted += 1;

        while generated.len() < max_tokens {
            let accepted =
                self.speculation_round(last_token, &mut stats, &mut callback)?;
            generated.extend_from_slice(&accepted);

            if accepted.is_empty() {
                break;
            }
            last_token = *accepted.last().unwrap();
        }

        // Trim to max_tokens
        generated.truncate(max_tokens);

        Ok((generated, stats))
    }

    /// Run one EAGLE speculation round: draft N tokens, verify, accept/reject.
    ///
    /// Algorithm:
    /// 1. Reset eagle KV cache (fresh chain each round).
    /// 2. Draft N tokens through the eagle head, using captured features
    ///    from the target model's hidden state buffers.
    /// 3. Build verify_input = [last_token, d1, ..., dn] and call
    ///    target.forward_prompt_logits() for batched verification.
    /// 4. Greedy accept/reject: compare argmax(target_logits[k]) with
    ///    draft_tokens[k]. Accept while matching, reject at first mismatch.
    /// 5. Rollback target KV cache on rejection.
    ///
    /// Returns the accepted tokens from this round (always >= 1).
    fn speculation_round(
        &mut self,
        last_token: u32,
        stats: &mut SpecStats,
        callback: &mut impl FnMut(u32),
    ) -> Result<Vec<u32>, String> {
        stats.rounds += 1;

        let target_pos_before = self.target.position();

        // Phase 1: Draft N tokens through EAGLE head.
        // Reset eagle KV cache for a fresh autoregressive chain.
        self.eagle_head.reset_kv_cache();

        let mut draft_tokens = Vec::with_capacity(self.n_draft);
        let mut current_token = last_token;

        for _ in 0..self.n_draft {
            // Get captured feature buffers from target model.
            // These are populated during forward_token/forward_prompt.
            // If capture is not yet populated (e.g., first round after batch prefill),
            // the buffers contain whatever was last written -- for POC with random
            // weights this is acceptable.
            let feat_low = self
                .target
                .eagle_capture_low()
                .ok_or("EAGLE capture not enabled (low)")?;
            let feat_mid = self
                .target
                .eagle_capture_mid()
                .ok_or("EAGLE capture not enabled (mid)")?;
            let feat_high = self
                .target
                .eagle_capture_high()
                .ok_or("EAGLE capture not enabled (high)")?;

            let draft_token = self.eagle_head.forward_draft_token(
                feat_low,
                feat_mid,
                feat_high,
                current_token,
                self.target.embed(),
                self.target.lm_head(),
                self.target.lm_head_is_f32(),
                self.target.lm_head_q6k(),
                self.target.lm_head_q8(),
            )?;

            draft_tokens.push(draft_token);
            current_token = draft_token;
        }
        stats.tokens_drafted += draft_tokens.len();

        // Phase 2: Verify with target model in one batch.
        // verify_input = [last_token, d1, d2, ..., dn]  (N+1 tokens)
        let mut verify_input = Vec::with_capacity(1 + self.n_draft);
        verify_input.push(last_token);
        verify_input.extend_from_slice(&draft_tokens);
        let target_logits = self.target.forward_prompt_logits(&verify_input)?;

        // Phase 3: Greedy accept/reject.
        // target_logits[k] = logits after seeing [last_token, d1, ..., dk].
        // argmax(target_logits[k]) should match draft_tokens[k] for acceptance.
        let mut accepted = Vec::new();
        let mut first_rejection = None;

        for k in 0..draft_tokens.len() {
            let target_pick = sample_greedy(&target_logits[k]);
            if target_pick == draft_tokens[k] {
                // Target agrees with draft
                accepted.push(draft_tokens[k]);
                stats.draft_accepted += 1;
                callback(draft_tokens[k]);
            } else {
                // Target disagrees -- use target's pick instead
                accepted.push(target_pick);
                callback(target_pick);
                first_rejection = Some(k);
                break;
            }
        }

        // If all draft tokens accepted, add bonus from target
        if first_rejection.is_none() {
            let bonus = sample_greedy(&target_logits[draft_tokens.len()]);
            accepted.push(bonus);
            callback(bonus);
        }

        stats.tokens_accepted += accepted.len();

        // Phase 4: Rollback target KV cache on rejection.
        // After verification, target processed [last_token, d1, ..., dn] (N+1 tokens).
        // Target position is now target_pos_before + N + 1.
        // On rejection at k: keep [last_token, d1, ..., d_{k-1}] + target_pick.
        //   = k+1 new entries from target_pos_before. Rollback to target_pos_before + k + 1.
        // On all-accepted: keep all N+1 entries. No rollback needed.
        if let Some(k) = first_rejection {
            let target_sync_pos = target_pos_before + k + 1;
            self.target.rollback_to(target_sync_pos);
        }
        // else: all accepted + bonus, target already at correct position

        // Always reset eagle KV cache (fresh chain next round)
        self.eagle_head.reset_kv_cache();

        Ok(accepted)
    }

    /// Get the target model's current KV cache position.
    pub fn target_position(&self) -> usize {
        self.target.position()
    }
}
