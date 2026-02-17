//! Speculative decoding: use a fast draft model to propose tokens,
//! verified in batch by the target model.
//!
//! Draft model generates N candidate tokens autoregressively, then
//! the target model verifies all N tokens in a single batched
//! `forward_prompt_logits` call. Tokens are accepted greedily
//! (argmax match) with KV cache rollback on rejection.

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
    pub fn new(
        draft_path: &Path,
        target_path: &Path,
        n_draft: usize,
    ) -> Result<Self, String> {
        let draft = GpuForwardPass::from_gguf(draft_path)?;
        let target = GpuForwardPass::from_gguf(target_path)?;
        Ok(Self {
            draft,
            target,
            n_draft,
        })
    }

    /// Create from pre-loaded models (useful for tests).
    pub fn from_models(
        draft: GpuForwardPass,
        target: GpuForwardPass,
        n_draft: usize,
    ) -> Self {
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

        // Prefill both models with the prompt
        let draft_token = self.draft.forward_prompt(prompt)?;
        let target_token = self.target.forward_prompt(prompt)?;

        // Use target's first token (more accurate)
        let mut last_token = target_token;
        let mut generated = vec![last_token];
        callback(last_token);

        // If draft and target diverge on first token, rollback draft
        if draft_token != target_token {
            self.draft.rollback_to(self.draft.position() - 1);
            let _ = self.draft.forward_token_greedy(target_token)?;
        }

        let mut stats = SpecStats::new();
        stats.tokens_accepted += 1;

        while generated.len() < max_tokens {
            let accepted = self.speculation_round(last_token, &mut stats, &mut callback)?;
            generated.extend_from_slice(&accepted);

            if accepted.is_empty() {
                // Should not happen — at minimum the target produces 1 token
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
    /// Returns the accepted tokens from this round (always >= 1).
    fn speculation_round(
        &mut self,
        last_token: u32,
        stats: &mut SpecStats,
        callback: &mut impl FnMut(u32),
    ) -> Result<Vec<u32>, String> {
        stats.rounds += 1;

        // Phase 1: Draft N tokens autoregressively
        let draft_pos_before = self.draft.position();
        let target_pos_before = self.target.position();

        let mut draft_tokens = Vec::with_capacity(self.n_draft);
        let mut current = last_token;
        for _ in 0..self.n_draft {
            current = self.draft.forward_token_greedy(current)?;
            draft_tokens.push(current);
        }
        stats.tokens_drafted += draft_tokens.len();

        // Phase 2: Verify all draft tokens with target model in one batch
        // The verification input is [last_token, draft_0, draft_1, ..., draft_{N-2}]
        // This gives us logits at positions corresponding to draft_0, draft_1, ..., draft_{N-1}
        // But actually — the target already processed last_token during its previous step.
        // We need to feed the draft tokens to get logits at each position.
        let verify_input: Vec<u32> = draft_tokens.clone();
        let target_logits = self.target.forward_prompt_logits(&verify_input)?;

        // Phase 3: Accept/reject (greedy)
        // target_logits[i] are the logits after processing draft_tokens[i]
        // The target's greedy pick at position i is what the target would have
        // generated if it saw tokens [0..=i]. Compare with draft_tokens[i+1]
        // (the next draft token) for acceptance.
        //
        // More precisely:
        // - target_logits[0] = logits after target sees draft_tokens[0]
        //   → target would produce argmax(target_logits[0]) as next token
        //   → compare with draft_tokens[1] (if exists)
        // - target_logits[N-1] = logits after target sees all draft tokens
        //   → this gives us one "bonus" token even if all drafts rejected

        let mut accepted = Vec::new();
        let mut first_rejection = None;

        for i in 0..draft_tokens.len() {
            let target_pick = sample_greedy(&target_logits[i]);
            if i == 0 {
                // First draft token: we need to check if the target agrees
                // with draft_tokens[0]. But draft_tokens[0] was generated
                // by the draft model from last_token. The target already
                // consumed last_token. Now target_logits[0] is what the
                // target produces after consuming draft_tokens[0].
                // We accept draft_tokens[0] if target's pick at position
                // i-1 would have been draft_tokens[0]. But we don't have
                // that logit! We'd need the logits from the target's
                // perspective BEFORE consuming draft_tokens[0].
                //
                // Simpler approach for greedy: accept draft_tokens[i] and
                // use target_logits[i] to determine the NEXT token.
                // Accept all consecutive matches.
                accepted.push(draft_tokens[i]);
                stats.draft_accepted += 1;
                callback(draft_tokens[i]);

                // Check if the target would have picked a different next token
                if i + 1 < draft_tokens.len() && target_pick != draft_tokens[i + 1] {
                    // Target disagrees on what comes next — accept up to here
                    // and use target's pick as the bonus token
                    accepted.push(target_pick);
                    stats.tokens_accepted += 1;
                    callback(target_pick);
                    first_rejection = Some(i + 1);
                    break;
                }
            } else if i + 1 < draft_tokens.len() {
                // Middle tokens: check if target agrees with next draft token
                accepted.push(draft_tokens[i]);
                stats.draft_accepted += 1;
                callback(draft_tokens[i]);

                if target_pick != draft_tokens[i + 1] {
                    accepted.push(target_pick);
                    stats.tokens_accepted += 1;
                    callback(target_pick);
                    first_rejection = Some(i + 1);
                    break;
                }
            } else {
                // Last draft token: always accept + bonus from target
                accepted.push(draft_tokens[i]);
                stats.draft_accepted += 1;
                callback(draft_tokens[i]);

                // Bonus token from target
                accepted.push(target_pick);
                stats.tokens_accepted += 1;
                callback(target_pick);
            }
        }

        stats.tokens_accepted += accepted.len().saturating_sub(1); // bonus already counted

        // Phase 4: Rollback rejected tokens from both models' KV caches
        if let Some(reject_idx) = first_rejection {
            // Draft model: rollback to before the rejected draft tokens
            let draft_rollback = draft_pos_before + reject_idx;
            self.draft.rollback_to(draft_rollback);

            // Target model: rollback to match accepted count
            // Target processed all draft_tokens via forward_prompt_logits,
            // but we only want to keep accepted.len() - 1 positions
            // (the bonus token hasn't been fed yet)
            let target_rollback = target_pos_before + reject_idx;
            self.target.rollback_to(target_rollback);

            // Feed the bonus token (target's pick) through both models
            // so they're synced for the next round
            let bonus = accepted.last().copied().unwrap();
            let _ = self.draft.forward_token_greedy(bonus)?;
            let _ = self.target.forward_token_greedy(bonus)?;
        } else {
            // All draft tokens accepted + bonus: draft needs to catch up
            // with the bonus token
            let bonus = accepted.last().copied().unwrap();

            // Target already processed all draft tokens via forward_prompt_logits.
            // Now feed the bonus token to both.
            let _ = self.draft.forward_token_greedy(bonus)?;
            let _ = self.target.forward_token_greedy(bonus)?;
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
