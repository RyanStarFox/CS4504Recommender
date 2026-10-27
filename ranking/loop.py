"""Optimizer steps for SFT, DPO and offline GRPO.

Reference log-probabilities are computed before the policy forward. Swapping
the frozen SFT LoRA in place after the policy graph has been built would
change weights that autograd still needs. GRPO never calls the generator.
"""
import torch

from ranking.batching import collate, encode_completion, move_batch, render_prompt
from ranking.data import load_sample_file
from ranking.lora import lora_parameters, lora_state, restore_lora
from ranking.objective import dpo_loss, grpo_terms, sequence_logprobs, sft_loss
from ranking.prompts import place_first, ranking_text, user_prompt
from ranking.reward import group_advantages, ndcg_at_k, parse_ranking


def sft_steps(model, tokenizer, examples, titles, steps, lr, batch_size, max_length, grad_clip,
              on_step=None, start_step=0, total_steps=None):
    rows = [
        _encoded(tokenizer, titles, example, _target(example), max_length)
        for example in examples
    ]
    return _optimize(
        model, tokenizer, rows, steps, lr, batch_size, grad_clip, _sft_batch,
        train_mode=True, on_step=on_step, start_step=start_step, total_steps=total_steps)


def dpo_steps(model, tokenizer, pairs, titles, steps, lr, batch_size, max_length, grad_clip, beta,
              on_step=None, start_step=0, total_steps=None, reference_state=None):
    reference = lora_state(model) if reference_state is None else reference_state
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
        model, tokenizer, rows, steps, lr, batch_size, grad_clip, loss_fn,
        train_mode=False, on_step=on_step, start_step=start_step, total_steps=total_steps)


def grpo_steps(model, tokenizer, sample_path, steps, lr, max_length, grad_clip, beta, k=10,
               on_step=None, start_step=0, total_steps=None, reference_state=None):
    _, groups = load_sample_file(sample_path)
    reference = lora_state(model) if reference_state is None else reference_state
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
        loss_fn=loss_fn, train_mode=False, one_row=True, on_step=on_step,
        start_step=start_step, total_steps=total_steps)


def _sft_batch(model, tokenizer, batch):
    del tokenizer
    _, tokens, mask = sequence_logprobs(
        model, batch["input_ids"], batch["attention_mask"], batch["completion_mask"])
    return sft_loss(tokens, mask)


def _dpo_batch(model, tokenizer, pairs, reference, beta):
    chosen = [pair[0] for pair in pairs]
    rejected = [pair[1] for pair in pairs]
    reference_chosen, reference_rejected = _frozen_pair(model, tokenizer, chosen, rejected, reference)
    _release_mps_cache(model)
    policy_chosen, policy_rejected = _pair_logprobs(model, tokenizer, chosen, rejected)
    return dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta)


def _grpo_batch(model, tokenizer, row, reference, beta):
    """Score one completion at a time.

    A packed pair keeps two full-length graphs, and the reference forward's
    cache is still resident when the policy forward allocates. Releasing
    each graph before the next forward keeps the peak at one sequence.
    """
    encoded, advantages = row
    frozen = _frozen_group(model, tokenizer, encoded, reference)
    group_denom = torch.stack([mask.sum() for _, mask in frozen]).sum().clamp(min=1)
    total = None
    for index, (item, (reference_tokens, reference_mask)) in enumerate(zip(encoded, frozen)):
        policy_tokens, policy_mask = _token_logprobs(model, tokenizer, [item])
        if not torch.equal(reference_mask, policy_mask):
            raise RuntimeError("Reference and policy masks differ")
        advantage = torch.tensor([advantages[index]], dtype=torch.float32, device=policy_tokens.device)
        part = grpo_terms(
            policy_tokens, reference_tokens, policy_mask, advantage, beta, group_denom)
        if not torch.isfinite(part.detach()):
            raise RuntimeError("Non-finite GRPO loss")
        if index == len(encoded) - 1:
            total = part if total is None else total + part
        else:
            part.backward()
            detached = part.detach()
            total = detached if total is None else total + detached
            del part, policy_tokens, policy_mask
            _release_mps_cache(model)
    return total


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


def _frozen_group(model, tokenizer, encoded, reference):
    current = lora_state(model)
    restore_lora(model, reference)
    frozen = []
    with torch.no_grad():
        for item in encoded:
            tokens, mask = _token_logprobs(model, tokenizer, [item])
            frozen.append((tokens.detach(), mask.detach()))
            del tokens, mask
            _release_mps_cache(model)
    restore_lora(model, current)
    _release_mps_cache(model)
    return frozen


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
              train_mode, one_row=False, on_step=None, start_step=0, total_steps=None):
    if steps <= 0 or lr <= 0 or batch_size <= 0 or start_step < 0:
        raise ValueError("steps, lr and batch_size must be positive")
    if not rows:
        raise ValueError("No rows to train")
    total = total_steps or (start_step + steps)
    optimizer = torch.optim.AdamW(lora_parameters(model), lr=lr, weight_decay=0.0)
    model.train(train_mode)
    last = None
    for offset in range(steps):
        index = start_step + offset
        batch = _slice(rows, index, batch_size, tokenizer, model, one_row)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model, tokenizer, batch)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {index + 1}")
        loss.backward()
        if any(param.grad is not None and not torch.isfinite(param.grad).all()
               for param in lora_parameters(model)):
            raise RuntimeError("Non-finite LoRA gradient")
        torch.nn.utils.clip_grad_norm_(lora_parameters(model), grad_clip)
        optimizer.step()
        last = float(loss.detach().cpu())
        del loss
        _release_mps_cache(model)
        print(f"step {index + 1}/{total} loss {last:.4f}", flush=True)
        if on_step is not None:
            on_step(index + 1, model)
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


def _release_mps_cache(model):
    """The MPS allocator keeps freed blocks. Drop them after each step.

    The live weights and optimizer state stay. Only the cache is returned,
    so the footprint does not grow for the rest of the epoch.
    """
    if _device(model).type != "mps":
        return
    torch.mps.empty_cache()
