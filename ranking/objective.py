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
    log_probs = F.log_softmax(logits.float(), dim=-1)
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs.masked_fill(~mask, 0.0)
    totals = token_log_probs.sum(dim=-1)
    if not torch.isfinite(totals).all():
        raise RuntimeError("Non-finite completion log-probability")
    return totals, token_log_probs, mask


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
    weights = advantages.to(policy_tokens.dtype).unsqueeze(1)
    denom = mask.sum().clamp(min=1)
    policy_gradient = -(weights * policy_tokens * mask).sum() / denom
    if beta == 0:
        return policy_gradient
    delta = reference_tokens - policy_tokens
    kl = torch.exp(delta) - delta - 1
    return policy_gradient + beta * (kl * mask).sum() / denom
