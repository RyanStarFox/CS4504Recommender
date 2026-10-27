"""Token log-probabilities and the three ranking losses.

SFT masks the prompt and averages cross-entropy over completion tokens.
DPO continues from the SFT LoRA and scores a frozen copy of that LoRA as
the reference. GRPO reads a fixed group of completions; it does not sample
inside the training step. Advantages are group-normalized NDCG.
"""
import torch
import torch.nn.functional as F


def sequence_logprobs(model, input_ids, attention_mask, completion_mask):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    mask = completion_mask[:, 1:].to(dtype=torch.bool)
    if not mask.any():
        raise RuntimeError("Completion mask is empty")
    # A full log_softmax keeps a second [batch, time, vocab] tensor. The loss
    # only needs the target token, so reduce the vocabulary in short blocks.
    token_log_probs = _target_logprobs(logits, targets).masked_fill(~mask, 0.0)
    del logits, outputs
    totals = token_log_probs.sum(dim=-1)
    if not torch.isfinite(totals).all():
        raise RuntimeError("Non-finite completion log-probability")
    return totals, token_log_probs, mask


def _target_logprobs(logits, targets):
    pieces = []
    width = 16
    for start in range(0, logits.size(1), width):
        block = logits[:, start:start + width, :]
        chosen = block.gather(-1, targets[:, start:start + width].unsqueeze(-1)).squeeze(-1)
        pieces.append(chosen.float() - torch.logsumexp(block.float(), dim=-1))
    return torch.cat(pieces, dim=1)


def sft_loss(token_log_probs, mask):
    denom = mask.sum().clamp(min=1)
    return -(token_log_probs.sum()) / denom


def dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta):
    if beta <= 0:
        raise ValueError("DPO beta must be positive")
    logits = (policy_chosen - policy_rejected) - (reference_chosen - reference_rejected)
    return -F.logsigmoid(beta * logits).mean()


def grpo_loss(policy_tokens, reference_tokens, mask, advantages, beta):
    if beta < 0:
        raise ValueError("GRPO KL coefficient must be non-negative")
    denom = mask.sum().clamp(min=1)
    return grpo_terms(policy_tokens, reference_tokens, mask, advantages, beta, denom)


def grpo_terms(policy_tokens, reference_tokens, mask, advantages, beta, denom):
    """GRPO loss whose denominator is supplied by the caller.

    Splitting a group across this function and summing the pieces matches
    ``grpo_loss`` on the packed group, so each completion can be freed
    before the next one is scored.
    """
    weights = advantages.to(policy_tokens.dtype).unsqueeze(1)
    weighted = -(weights * policy_tokens * mask).sum()
    if beta == 0:
        return weighted / denom
    delta = reference_tokens - policy_tokens
    kl = torch.exp(delta) - delta - 1
    return (weighted + beta * (kl * mask).sum()) / denom
