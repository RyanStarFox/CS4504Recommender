"""Write one GRPO sample file. Training reads the file and does not generate again.

``hf`` runs the in-memory SFT model and is the day-1 stand-in.
``llama`` shells out to llama.cpp, which is the sampler used for reported results.
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

from ranking.batching import render_prompt
from ranking.data import write_json
from ranking.prompts import user_prompt


def write_samples(path, groups, backend, split="validation"):
    if split != "validation":
        raise ValueError("Only validation candidates are sampled for GRPO training")
    write_json(path, {
        "format_version": 1,
        "backend": backend,
        "split": split,
        "groups": groups,
    })
    return Path(path)


def build_group(user_id, prompt, ground_truth, candidates, completions):
    if len(completions) < 2:
        raise ValueError("A GRPO group needs at least two completions")
    return {
        "user_id": str(user_id),
        "prompt": prompt,
        "ground_truth": int(ground_truth),
        "candidates": list(candidates),
        "completions": list(completions),
    }


def hf_completions(model, tokenizer, prompt, group_size, max_new_tokens, temperature):
    if group_size < 2 or max_new_tokens <= 0 or temperature <= 0:
        raise ValueError("Invalid HF sampling arguments")
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    was_training = model.training
    model.eval()
    generated = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        num_return_sequences=group_size,
        use_cache=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if was_training:
        model.train()
    prompt_width = encoded["input_ids"].shape[1]
    texts = []
    for row in generated:
        text = tokenizer.decode(row[prompt_width:], skip_special_tokens=True).strip()
        if not text:
            text = tokenizer.decode(row[prompt_width:], skip_special_tokens=False).strip()
        texts.append(text or "<no_text>")
    if len(texts) != group_size or any(not text for text in texts):
        raise RuntimeError(f"HF sampling returned {texts!r}")
    return texts


def sample_examples(model, tokenizer, examples, titles, group_size, max_new_tokens, temperature):
    groups = []
    for example in examples:
        prompt = render_prompt(tokenizer, user_prompt(example, titles))
        completions = hf_completions(
            model, tokenizer, prompt, group_size, max_new_tokens, temperature)
        groups.append(build_group(
            example.user_id, prompt, example.ground_truth, example.candidates, completions))
    return groups


def llama_completion(binary, model_path, prompt, max_tokens, temperature, context):
    """Sample one completion. The prompt is already a closed non-thinking chat string.

    Jinja stays off so llama.cpp does not wrap that string in a second template.
    top-k and min-p are disabled; top-p 0.9 matches the Hugging Face stand-in.
    """
    binary_path = binary if Path(binary).is_file() else shutil.which(binary)
    if binary_path is None:
        raise FileNotFoundError(
            f"llama.cpp binary {binary!r} was not found. "
            "Rows 7-9 report generations from llama.cpp after the LoRA is merged and converted to GGUF. "
            "The hf backend only exercises the training loop."
        )
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        handle.write(prompt)
        prompt_file = handle.name
    command = [
        binary_path, "-m", str(model_path), "-f", prompt_file, "--no-escape",
        "--no-jinja", "--simple-io", "--log-disable", "--no-display-prompt",
        "--single-turn", "--reasoning", "off",
        "-n", str(max_tokens), "--temp", str(temperature),
        "--top-k", "0", "--top-p", "0.9", "--min-p", "0", "--repeat-penalty", "1",
        "-c", str(context), "-ngl", "all", "--fit", "off", "-s", "-1",
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True)
    finally:
        Path(prompt_file).unlink(missing_ok=True)
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"llama.cpp exited {completed.returncode}: {detail}")
    text = completed.stdout.decode("utf-8", errors="replace")
    if text.startswith(prompt):
        text = text[len(prompt):]
    text = llama_generation(text)
    if not text.strip():
        raise RuntimeError("llama.cpp returned an empty completion")
    return text


def llama_generation(transcript):
    """Drop the llama-cli banner, echoed prompt and timing line.

    The ranking itself is the last line before ``[ Prompt: ... t/s ]``.
    An empty generation becomes ``none`` so the sample file still has text
    and the training sequence stays a few tokens long.
    """
    text = transcript
    marker = "[ Prompt:"
    if marker in text:
        text = text[:text.rfind(marker)]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    while lines and lines[-1] in {"</think>", "Exiting..."}:
        lines.pop()
    if not lines:
        return "none"
    last = lines[-1]
    if last.startswith("Loading model") or last.startswith("build "):
        return "none"
    return last


def sample_llama(examples, titles, tokenizer, binary, model_path, group_size,
                 max_new_tokens, temperature, context):
    groups = []
    total = len(examples)
    for index, example in enumerate(examples, start=1):
        prompt = render_prompt(tokenizer, user_prompt(example, titles))
        completions = [
            llama_completion(binary, model_path, prompt, max_new_tokens, temperature, context)
            for _ in range(group_size)
        ]
        groups.append(build_group(
            example.user_id, prompt, example.ground_truth, example.candidates, completions))
        if index == 1 or index % 25 == 0 or index == total:
            print(f"llama sample {index}/{total}", flush=True)
    return groups
