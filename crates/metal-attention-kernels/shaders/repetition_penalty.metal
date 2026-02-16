#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// GPU-side repetition penalty kernel
//
// For each token ID in the history buffer, penalizes the corresponding logit:
//   - if logit > 0: logit /= penalty
//   - if logit <= 0: logit *= penalty
//
// This runs between lm_head (which produces logits) and argmax (which picks
// the best token), keeping the entire decode loop on GPU.
//
// Dispatch: ceil(num_tokens / 256) threadgroups of 256 threads
// ---------------------------------------------------------------------------

kernel void repetition_penalty(
    device float*       logits      [[buffer(0)]],
    device const uint*  token_ids   [[buffer(1)]],
    constant uint&      num_tokens  [[buffer(2)]],
    constant float&     penalty     [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= num_tokens) return;

    uint token_id = token_ids[tid];
    float val = logits[token_id];

    if (val > 0.0f) {
        logits[token_id] = val / penalty;
    } else {
        logits[token_id] = val * penalty;
    }
}
