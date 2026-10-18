"""Optimizer steps for SFT, DPO and offline GRPO.

Reference log-probabilities are computed before the policy forward. Swapping
the frozen SFT LoRA in place after the policy graph has been built would
change weights that autograd still needs. GRPO never calls the generator.
"""
import torch

from ranking.batching import collate, encode_completion, move_batch, render_prompt
from ranking.data import load_sample_file
from ranking.lora import lora_parameters, lora_state, restore_lora
from ranking.objective import dpo_loss, grpo_loss, sequence_logprobs, sft_loss
from ranking.prompts import place_first, ranking_text, user_prompt
from ranking.reward import group_advantages, ndcg_at_k, parse_ranking


def sft_steps(model, tokenizer, examples, titles, steps, lr, batch_size, max_length, grad_clip):
    rows = [
        _encoded(tokenizer, titles, example, _target(example), max_length)
        for example in examples
    ]
    return _optimize(
        model, tokenizer, rows, steps, lr, batch_size, grad_clip, _sft_batch, train_mode=True)


def dpo_steps(model, tokenizer, pairs, titles, steps, lr, batch_size, max_length, grad_clip, beta):
    reference = lora_state(model)
    rows = []
    for example, rejected in pairs:
        rows.append((
            _encoded(tokenizer, titles, example, _target(example), max_length),
            _encoded(
                tokenizer, titles, example,
                ranking_text(place_first(example.candidates, rejected)), max_length),
        ))

    def loss_fn(model, tokenizer, batch):
        return _dpo_batch(model, tokenizer, batch, reference, beta)

    return _optimize(
        model, tokenizer, rows, steps, lr, batch_size, grad_clip, loss_fn, train_mode=False)


def grpo_steps(model, tokenizer, sample_path, steps, lr, max_length, grad_clip, beta, k=10):
    _, groups = load_sample_file(sample_path)
    reference = lora_state(model)
    rows = []
    for group in groups:
        rewards = [
            ndcg_at_k(parse_ranking(text, group["candidates"]), group["ground_truth"], k)
            for text in group["completions"]
        ]
        encoded = [
            encode_completion(tokenizer, group["prompt"], text, max_length)
            for text in group["completions"]
        ]
        rows.append((encoded, group_advantages(rewards)))

    def loss_fn(model, tokenizer, batch):
        return _grpo_batch(model, tokenizer, batch, reference, beta)

    return _optimize(
        model, tokenizer, rows, steps, lr, batch_size=1, grad_clip=grad_clip,
        loss_fn=loss_fn, train_mode=False, one_row=True)


def _sft_batch(model, tokenizer, batch):
    del tokenizer
    _, tokens, mask = sequence_logprobs(
        model, batch["input_ids"], batch["attention_mask"], batch["completion_mask"])
    return sft_loss(tokens, mask)


def _dpo_batch(model, tokenizer, pairs, reference, beta):
    chosen = [pair[0] for pair in pairs]
    rejected = [pair[1] for pair in pairs]
    reference_chosen, reference_rejected = _frozen_pair(model, tokenizer, chosen, rejected, reference)
    policy_chosen, policy_rejected = _pair_logprobs(model, tokenizer, chosen, rejected)
    return dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta)


def _grpo_batch(model, tokenizer, row, reference, beta):
    encoded, advantages = row
    reference_tokens, mask = _frozen_tokens(model, tokenizer, encoded, reference)
    policy_tokens, policy_mask = _token_logprobs(model, tokenizer, encoded)
    if not torch.equal(mask, policy_mask):
        raise RuntimeError("Reference and policy masks differ")
    advantage = torch.tensor(advantages, dtype=torch.float32, device=policy_tokens.device)
    return grpo_loss(policy_tokens, reference_tokens, mask, advantage, beta)


def _frozen_pair(model, tokenizer, chosen, rejected, reference):
    current = lora_state(model)
    restore_lora(model, reference)
    with torch.no_grad():
        chosen_log, rejected_log = _pair_logprobs(model, tokenizer, chosen, rejected)
    restore_lora(model, current)
    return chosen_log, rejected_log


def _frozen_tokens(model, tokenizer, encoded, reference):
    current = lora_state(model)
    restore_lora(model, reference)
    with torch.no_grad():
        tokens, mask = _token_logprobs(model, tokenizer, encoded)
    restore_lora(model, current)
    return tokens, mask


def _pair_logprobs(model, tokenizer, chosen, rejected):
    rows = [item for pair in zip(chosen, rejected) for item in pair]
    totals, _, _ = sequence_logprobs(model, *_packed(model, tokenizer, rows))
    return totals[0::2], totals[1::2]


def _token_logprobs(model, tokenizer, rows):
    _, tokens, mask = sequence_logprobs(model, *_packed(model, tokenizer, rows))
    return tokens, mask


def _packed(model, tokenizer, rows):
    batch = move_batch(collate(rows, tokenizer.pad_token_id), _device(model))
    return batch["input_ids"], batch["attention_mask"], batch["completion_mask"]


def _encoded(tokenizer, titles, example, completion, max_length):
    prompt = render_prompt(tokenizer, user_prompt(example, titles))
    return encode_completion(tokenizer, prompt, completion, max_length)


def _target(example):
    return ranking_text(place_first(example.candidates, example.ground_truth))


def _optimize(model, tokenizer, rows, steps, lr, batch_size, grad_clip, loss_fn,
              train_mode, one_row=False):
    if steps <= 0 or lr <= 0 or batch_size <= 0:
        raise ValueError("steps, lr and batch_size must be positive")
    if not rows:
        raise ValueError("No rows to train")
    optimizer = torch.optim.AdamW(lora_parameters(model), lr=lr, weight_decay=0.0)
    model.train(train_mode)
    last = None
    for step in range(steps):
        batch = _slice(rows, step, batch_size, tokenizer, model, one_row)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model, tokenizer, batch)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step + 1}")
        loss.backward()
        if any(param.grad is not None and not torch.isfinite(param.grad).all()
               for param in lora_parameters(model)):
            raise RuntimeError("Non-finite LoRA gradient")
        torch.nn.utils.clip_grad_norm_(lora_parameters(model), grad_clip)
        optimizer.step()
        last = float(loss.detach().cpu())
        print(f"step {step + 1}/{steps} loss {last:.4f}", flush=True)
    model.eval()
    return last


def _slice(rows, step, batch_size, tokenizer, model, one_row):
    if one_row:
        return rows[step % len(rows)]
    start = (step * batch_size) % len(rows)
    selected = [rows[(start + offset) % len(rows)] for offset in range(batch_size)]
    first = selected[0]
    if isinstance(first, tuple) and isinstance(first[0], tuple):
        return selected
    return move_batch(collate(selected, tokenizer.pad_token_id), _device(model))


def _device(model):
    return next(model.parameters()).device
