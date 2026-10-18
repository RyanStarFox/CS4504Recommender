"""Train and smoke-test the Qwen3.5 reranker.

Day 1, from the repository root, with the project virtualenv::

    .venv/bin/python -m ranking.train smoke

That builds a tiny candidate file in the same shape as ``mock_candidates.json``
and runs one SFT step, one DPO step and one GRPO step. Formal runs only change
the input paths::

    .venv/bin/python -m ranking.train sft --sequences data/sequences.json --candidates data/mock_candidates.json
    .venv/bin/python -m ranking.train dpo --adapter runs/sft
    .venv/bin/python -m ranking.train sample --backend hf --adapter runs/sft --output runs/grpo_samples.json
    .venv/bin/python -m ranking.train grpo --adapter runs/sft --samples runs/grpo_samples.json

DPO and GRPO both start from the SFT adapter. GRPO does not continue from DPO.
Its sample file is written once, from the frozen SFT model, and the training
step only reads that file. Reported test generations use ``sample --backend llama``
after the adapter has been merged and converted to GGUF.
"""
import argparse
import shutil
from pathlib import Path

import torch

from ranking.batching import render_prompt
from ranking.data import dpo_pairs, load_bundle, write_json
from ranking.lora import (
    TARGET_SUFFIXES, apply_lora, load_adapter, lora_fingerprint, lora_state, restore_lora, save_adapter)
from ranking.loop import dpo_steps, grpo_steps, sft_steps
from ranking.model import ROOT, load_ranker, load_tokenizer, pinned_revision, use_trainable_dtype
from ranking.prompts import place_first, ranking_text, user_prompt
from ranking.sample import (
    build_group, hf_completions, sample_examples, sample_llama, write_samples)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("smoke", "sft", "dpo", "grpo", "sample"))
    parser.add_argument("--sequences", default="data/sequences.json")
    parser.add_argument("--candidates", default="data/mock_candidates.json")
    parser.add_argument("--movies", default="ml-1m/movies.dat")
    parser.add_argument("--adapter")
    parser.add_argument("--output")
    parser.add_argument("--samples")
    parser.add_argument("--backend", choices=("hf", "llama"), default="hf")
    parser.add_argument("--llama-bin", default="llama-cli")
    parser.add_argument("--gguf")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--kl", type=float, default=0.04)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--max-history", type=int, default=10)
    parser.add_argument("--max-rejected", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    args = parser.parse_args(argv)
    torch.manual_seed(args.seed)
    if args.command == "smoke":
        smoke(args)
        return
    bundle = _bundle(args)
    if args.command == "sft":
        _run_sft(args, bundle, Path(args.output or "runs/sft"))
    elif args.command == "dpo":
        _run_dpo(args, bundle, Path(args.output or "runs/dpo"))
    elif args.command == "sample":
        _run_sample(args, bundle, Path(args.output or "runs/grpo_samples.json"))
    elif args.command == "grpo":
        _run_grpo(args, Path(args.output or "runs/grpo"))
    else:
        raise RuntimeError(args.command)


def smoke(args):
    """One optimizer step of each loop on a fixture shaped like the shared JSON files."""
    out = ROOT / "runs" / "smoke"
    if out.exists():
        shutil.rmtree(out)
    fixture = out / "fixture"
    _write_fixture(fixture, args.movies)
    bundle = load_bundle(
        fixture / "sequences.json", fixture / "candidates.json", args.movies,
        candidate_k=10, max_history=5, val_ratio=0, seed=args.seed, limit=1)
    print(
        f"smoke data: train={len(bundle.train)} select={len(bundle.select)} test={len(bundle.test)}",
        flush=True)
    model, tokenizer, device, dtype = load_ranker()
    train_dtype = use_trainable_dtype(model, device, dtype)
    print(f"ranker: {device}, load={dtype}, train={train_dtype}, hidden={model.config.hidden_size}", flush=True)
    config = _adapter_config(args, dropout=0.0)
    config["load_dtype"] = str(dtype)
    config["train_dtype"] = str(train_dtype)
    apply_lora(model, rank=config["rank"], alpha=config["alpha"], dropout=0.0)
    before = lora_fingerprint(model)
    sft_loss = sft_steps(
        model, tokenizer, bundle.train, bundle.titles, steps=1, lr=args.lr, batch_size=1,
        max_length=args.max_length, grad_clip=args.grad_clip)
    if torch.equal(before, lora_fingerprint(model)):
        raise RuntimeError("SFT step did not change LoRA weights")
    save_adapter(out / "sft", model, config)
    sft_weights = lora_state(model)
    dpo_loss = dpo_steps(
        model, tokenizer, dpo_pairs(bundle.train, max_rejected=1), bundle.titles,
        steps=1, lr=args.lr, batch_size=1, max_length=args.max_length,
        grad_clip=args.grad_clip, beta=args.beta)
    save_adapter(out / "dpo", model, {**config, "stage": "dpo", "beta": args.beta})
    restore_lora(model, sft_weights)

    example = bundle.train[0]
    prompt = render_prompt(tokenizer, user_prompt(example, bundle.titles))
    perfect = ranking_text(place_first(example.candidates, example.ground_truth))
    worst = ranking_text(tuple(reversed(place_first(example.candidates, example.ground_truth))))
    crafted = out / "crafted_samples.json"
    write_samples(crafted, [build_group(
        example.user_id, prompt, example.ground_truth, example.candidates, [perfect, worst])], "crafted")
    crafted_before = lora_fingerprint(model)
    crafted_loss = grpo_steps(
        model, tokenizer, crafted, steps=1, lr=args.lr, max_length=args.max_length,
        grad_clip=args.grad_clip, beta=args.kl)
    if torch.equal(crafted_before, lora_fingerprint(model)):
        raise RuntimeError("GRPO step did not change LoRA weights")
    restore_lora(model, sft_weights)
    completions = hf_completions(
        model, tokenizer, prompt, group_size=2, max_new_tokens=8, temperature=args.temperature)
    sampled = out / "hf_samples.json"
    write_samples(sampled, [build_group(
        example.user_id, prompt, example.ground_truth, example.candidates, completions)], "hf")
    hf_loss = grpo_steps(
        model, tokenizer, sampled, steps=1, lr=args.lr, max_length=args.max_length,
        grad_clip=args.grad_clip, beta=args.kl)
    save_adapter(out / "grpo", model, {**config, "stage": "grpo", "kl": args.kl})
    print(
        f"smoke passed: sft={sft_loss:.4f} dpo={dpo_loss:.4f} "
        f"grpo_crafted={crafted_loss:.4f} grpo_hf={hf_loss:.4f} device={device}",
        flush=True)


def _run_sft(args, bundle, output):
    model, tokenizer, device, config = _prepare(args)
    apply_lora(model, rank=config["rank"], alpha=config["alpha"], dropout=config["dropout"])
    print(f"sft on {device} {config['train_dtype']}: {len(bundle.train)} train, {len(bundle.select)} select", flush=True)
    loss = sft_steps(
        model, tokenizer, bundle.train, bundle.titles, args.steps, args.lr, args.batch_size,
        args.max_length, args.grad_clip)
    save_adapter(output, model, config)
    print(f"saved {output} loss {loss:.4f}", flush=True)


def _run_dpo(args, bundle, output):
    if not args.adapter:
        raise ValueError("DPO continues from the SFT adapter; pass --adapter")
    model, tokenizer, device, config = _prepare(args)
    loaded = load_adapter(model, args.adapter)
    config = {
        **loaded, "stage": "dpo", "beta": args.beta,
        "load_dtype": config["load_dtype"], "train_dtype": config["train_dtype"],
    }
    print(f"dpo on {device} {config['train_dtype']}: {len(bundle.train)} train users", flush=True)
    loss = dpo_steps(
        model, tokenizer, dpo_pairs(bundle.train, args.max_rejected), bundle.titles,
        args.steps, args.lr, args.batch_size, args.max_length, args.grad_clip, args.beta)
    save_adapter(output, model, config)
    print(f"saved {output} loss {loss:.4f}", flush=True)


def _run_grpo(args, output):
    if not args.adapter or not args.samples:
        raise ValueError("GRPO starts from the SFT adapter and a sample file; pass --adapter and --samples")
    model, tokenizer, device, config = _prepare(args)
    loaded = load_adapter(model, args.adapter)
    print(f"grpo on {device} {config['train_dtype']}, reading {args.samples}", flush=True)
    loss = grpo_steps(
        model, tokenizer, args.samples, args.steps, args.lr, args.max_length, args.grad_clip, args.kl)
    save_adapter(output, model, {
        **loaded, "stage": "grpo", "kl": args.kl,
        "load_dtype": config["load_dtype"], "train_dtype": config["train_dtype"],
    })
    print(f"saved {output} loss {loss:.4f}", flush=True)


def _run_sample(args, bundle, output):
    if args.backend == "llama":
        if not args.gguf:
            raise ValueError("llama sampling needs the merged GGUF path in --gguf")
        tokenizer = load_tokenizer()
        groups = sample_llama(
            bundle.train, bundle.titles, tokenizer, args.llama_bin, args.gguf,
            args.group_size, args.max_new_tokens, args.temperature, args.context)
        write_samples(output, groups, "llama")
    else:
        if not args.adapter:
            raise ValueError("HF sampling needs the frozen SFT adapter; pass --adapter")
        model, tokenizer, device, dtype = load_ranker()
        load_adapter(model, args.adapter)
        print(f"sampling {len(bundle.train)} validation groups on {device} {dtype}", flush=True)
        groups = sample_examples(
            model, tokenizer, bundle.train, bundle.titles, args.group_size,
            args.max_new_tokens, args.temperature)
        write_samples(output, groups, "hf")
    print(f"wrote {output}", flush=True)


def _bundle(args):
    return load_bundle(
        args.sequences, args.candidates, args.movies, candidate_k=args.candidate_k,
        max_history=args.max_history, val_ratio=args.val_ratio, seed=args.seed, limit=args.limit)


def _prepare(args, dropout=None):
    model, tokenizer, device, loaded = load_ranker()
    train_dtype = use_trainable_dtype(model, device, loaded)
    config = _adapter_config(args, dropout=dropout)
    config["load_dtype"] = str(loaded)
    config["train_dtype"] = str(train_dtype)
    return model, tokenizer, device, config


def _adapter_config(args, dropout=None):
    return {
        "format_version": 1,
        "stage": "sft",
        "model_id": "Qwen/Qwen3.5-0.8B",
        "revision": pinned_revision(),
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout if dropout is None else dropout,
        "targets": list(TARGET_SUFFIXES),
    }


def _write_fixture(directory, movies_path):
    """Popularity-shaped stand-in used only until A publishes mock_candidates.json."""
    from ranking.data import read_movies
    ids = sorted(read_movies(movies_path))[:20]
    if len(ids) < 20:
        raise ValueError("Smoke fixture needs at least 20 movies")
    sequences = {
        "1": {"train": ids[0:5], "validation": ids[5], "test": ids[6]},
        "2": {"train": ids[1:6], "validation": ids[12], "test": ids[13]},
    }
    candidates = {
        "validation": {
            "1": [ids[5], *ids[0:4], *ids[14:19]],
            "2": [ids[12], *ids[1:5], *ids[14:19]],
        },
        "test": {
            "1": [ids[6], *ids[7:15], ids[19]],
            "2": [ids[13], *ids[7:12], *ids[16:20]],
        },
    }
    for split, users in candidates.items():
        for user_id, movies in users.items():
            if len(movies) != 10 or len(set(movies)) != 10:
                raise RuntimeError(f"Bad smoke candidates for {split} user {user_id}: {movies}")
    write_json(directory / "sequences.json", sequences)
    write_json(directory / "candidates.json", candidates)


if __name__ == "__main__":
    main()
