//! Speculative decoding: use a fast draft model to propose tokens,
//! verified in batch by the target model.
//!
//! Draft model generates N candidate tokens autoregressively, then
//! the target model verifies all N+1 tokens (last_token + drafts) in a
//! single batched `forward_prompt_logits` call. Tokens are accepted
//! greedily (argmax match) with KV cache rollback on rejection.

use std::path::Path;

use crate::gpu_forward_pass::GpuForwardPass;
use crate::sampling::sample_greedy;

/// Statistics from a speculative decoding run.
#[derive(Debug, Clone)]
pub struct SpecStats {
    /// Total tokens accepted (including bonus tokens).
    pub tokens_accepted: usize,
    /// Total draft rounds executed.
    pub rounds: usize,
    /// Total draft tokens proposed across all rounds.
    pub tokens_drafted: usize,
    /// Total draft tokens accepted across all rounds.
    pub draft_accepted: usize,
}

impl SpecStats {
    fn new() -> Self {
        Self {
            tokens_accepted: 0,
            rounds: 0,
            tokens_drafted: 0,
            draft_accepted: 0,
        }
    }

    /// Acceptance rate: fraction of draft tokens accepted.
    pub fn acceptance_rate(&self) -> f64 {
        if self.tokens_drafted == 0 {
            0.0
        } else {
            self.draft_accepted as f64 / self.tokens_drafted as f64
        }
    }
}

/// Speculative decoder managing a draft model and target model.
///
/// The draft model (small, fast) proposes `n_draft` candidate tokens.
/// The target model (large, accurate) verifies them in one batched pass.
/// Accepted tokens are kept; on rejection, KV caches roll back and the
/// target model's token at the rejection point is used instead.
pub struct SpeculativeDecoder {
    draft: GpuForwardPass,
    target: GpuForwardPass,
    n_draft: usize,
}

impl SpeculativeDecoder {
    /// Create a speculative decoder from two GGUF model paths.
    ///
    /// # Arguments
    /// - `draft_path`: Path to the draft model GGUF file (e.g., SmolLM-135M).
    /// - `target_path`: Path to the target model GGUF file (e.g., Mistral-7B).
    /// - `n_draft`: Number of draft tokens to propose per speculation round.
    pub fn new(draft_path: &Path, target_path: &Path, n_draft: usize) -> Result<Self, String> {
        let draft = GpuForwardPass::from_gguf(draft_path)?;
        let target = GpuForwardPass::from_gguf(target_path)?;
        Ok(Self {
            draft,
            target,
            n_draft,
        })
    }

    /// Create from pre-loaded models (useful for tests).
    pub fn from_models(draft: GpuForwardPass, target: GpuForwardPass, n_draft: usize) -> Self {
        Self {
            draft,
            target,
            n_draft,
        }
    }

    /// Reset both models (clear KV caches and positions).
    pub fn reset(&mut self) {
        self.draft.reset();
        self.target.reset();
    }

    /// Generate tokens using speculative decoding.
    ///
    /// Prefills both models with `prompt`, then runs the draft-verify loop
    /// until `max_tokens` are generated. Calls `callback` with each accepted
    /// token (for streaming output).
    ///
    /// Returns the generated token sequence and stats.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        max_tokens: usize,
        mut callback: impl FnMut(u32),
    ) -> Result<(Vec<u32>, SpecStats), String> {
        if prompt.is_empty() {
            return Err("prompt must not be empty".to_string());
        }

        // Prefill both models with the prompt.
        // After forward_prompt, both models have the full prompt in their KV caches
        // at position 0..N-1 and return the argmax predicted next token.
        let _draft_token = self.draft.forward_prompt(prompt)?;
        let target_token = self.target.forward_prompt(prompt)?;

        // Use target's first token (more accurate).
        // Both models' KV caches are identical (full prompt). The next round
        // will feed target_token to both models as part of its normal flow.
        let mut last_token = target_token;
        let mut generated = vec![last_token];
        callback(last_token);

        let mut stats = SpecStats::new();
        stats.tokens_accepted += 1;

        while generated.len() < max_tokens {
            let accepted = self.speculation_round(last_token, &mut stats, &mut callback)?;
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

    /// Run one speculation round: draft N tokens, verify, accept/reject.
    ///
    /// Both models start at the same position P. Neither model has processed
    /// `last_token` as input yet (it was the output of the previous step).
    ///
    /// Algorithm:
    /// 1. Draft generates N tokens: feeds [last_token, d₁, ..., d_{N-1}],
    ///    returns [d₁, d₂, ..., dₙ]. Draft position becomes P + N.
    /// 2. Target verifies by processing [last_token, d₁, ..., dₙ] in one batch.
    ///    target_logits[k] = logits after seeing [last_token, d₁, ..., dₖ].
    ///    argmax(target_logits[k]) should match draft_tokens[k] for acceptance.
    /// 3. On rejection at k: rollback both to P + k + 1 (keep last_token + d₁..dₖ₋₁).
    ///    On all-accepted: sync draft to P + N + 1 (feed dₙ to draft).
    ///    The bonus/replacement token is NOT fed; next round handles it.
    ///
    /// Returns the accepted tokens from this round (always >= 1).
    fn speculation_round(
        &mut self,
        last_token: u32,
        stats: &mut SpecStats,
        callback: &mut impl FnMut(u32),
    ) -> Result<Vec<u32>, String> {
        stats.rounds += 1;

        let draft_pos_before = self.draft.position();
        let target_pos_before = self.target.position();

        // Phase 1: Draft N tokens autoregressively.
        // Feeds: last_token → d₁, d₁ → d₂, ..., d_{N-1} → dₙ.
        // Draft position after: P + N.
        let mut draft_tokens = Vec::with_capacity(self.n_draft);
        let mut current = last_token;
        for _ in 0..self.n_draft {
            current = self.draft.forward_token_greedy(current)?;
            draft_tokens.push(current);
        }
        stats.tokens_drafted += draft_tokens.len();

        // Phase 2: Verify with target model in one batch.
        // Include last_token so the target processes the same sequence as the draft.
        // verify_input = [last_token, d₁, d₂, ..., dₙ]  (N+1 tokens)
        // target_logits[k] = logits after processing verify_input[0..=k]
        //   → argmax(target_logits[k]) = what target predicts after seeing
        //     [last_token, d₁, ..., dₖ]
        //   → Compare with draft_tokens[k] to verify acceptance.
        let mut verify_input = Vec::with_capacity(1 + self.n_draft);
        verify_input.push(last_token);
        verify_input.extend_from_slice(&draft_tokens);
        let target_logits = self.target.forward_prompt_logits(&verify_input)?;

        // Phase 3: Greedy accept/reject.
        // target_logits[k] predicts the token after [last_token, d₁, ..., dₖ].
        // For k=0: predicts after last_token → should match d₁ = draft_tokens[0].
        // For k=j: predicts after [last_token, d₁, ..., dⱼ] → should match draft_tokens[j].
        // For k=N: predicts after all → bonus token (no draft token to compare).
        let mut accepted = Vec::new();
        let mut first_rejection = None;

        for k in 0..draft_tokens.len() {
            let target_pick = sample_greedy(&target_logits[k]);
            if target_pick == draft_tokens[k] {
                // Target agrees with draft's prediction
                accepted.push(draft_tokens[k]);
                stats.draft_accepted += 1;
                callback(draft_tokens[k]);
            } else {
                // Target disagrees — use target's pick instead
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

        // Phase 4: Rollback and sync KV caches.
        // After this phase, both models should have identical KV state.
        // The last accepted token is NOT fed to either model — the next round
        // will include it as last_token in its verify_input/draft generation.
        if let Some(k) = first_rejection {
            // Rejected at position k. Keep KV entries for:
            // [last_token, d₁, ..., d_{k-1}] = k+1 entries from pos_before.
            // (d_k was rejected, target_pick replaces it but isn't fed yet.)
            let sync_pos = draft_pos_before + k + 1;
            self.draft.rollback_to(sync_pos);

            let target_sync_pos = target_pos_before + k + 1;
            self.target.rollback_to(target_sync_pos);
        } else {
            // All N draft tokens accepted + bonus.
            // Target processed [last_token, d₁, ..., dₙ] (N+1 tokens),
            // so target position = P + N + 1.
            // Draft processed [last_token, d₁, ..., d_{N-1}] (N tokens),
            // so draft position = P + N.
            // Feed dₙ to draft to sync both at P + N + 1.
            let _ = self
                .draft
                .forward_token_greedy(*draft_tokens.last().unwrap())?;
            // Now both at position P + N + 1. Bonus is NOT fed.
        }

        Ok(accepted)
    }

    /// Get the target model's current position.
    pub fn target_position(&self) -> usize {
        self.target.position()
    }

    /// Get the draft model's current position.
    pub fn draft_position(&self) -> usize {
        self.draft.position()
    }
}
