"""Chat-template rendering and padded completion batches."""
import torch

from ranking.prompts import assert_non_thinking


def render_prompt(tokenizer, user_text):
    messages = [{"role": "user", "content": user_text}]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    return assert_non_thinking(prompt)


def encode_completion(tokenizer, prompt, completion, max_length):
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    eos = tokenizer.eos_token_id
    if eos is None:
        raise RuntimeError("Tokenizer has no eos token")
    if not completion_ids or completion_ids[-1] != eos:
        completion_ids = completion_ids + [eos]
    overflow = len(prompt_ids) + len(completion_ids) - max_length
    if overflow > 0:
        if overflow >= len(prompt_ids):
            raise ValueError("Completion does not fit in max_length")
        prompt_ids = prompt_ids[overflow:]
        if not getattr(encode_completion, "warned", False):
            encode_completion.warned = True
            print(
                f"Truncated {overflow} prompt tokens to fit max_length={max_length}. "
                "Lower --max-history or raise --max-length.",
                flush=True)
    input_ids = prompt_ids + completion_ids
    completion_mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
    return input_ids, completion_mask


def collate(rows, pad_id):
    width = max(len(ids) for ids, _ in rows)
    input_ids, attention_mask, completion_mask = [], [], []
    for ids, mask in rows:
        pad = width - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)
        completion_mask.append(mask + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "completion_mask": torch.tensor(completion_mask, dtype=torch.long),
    }


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}
