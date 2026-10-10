"""Frozen movie features, indexed by original MovieLens IDs.

No chat template, generation, visual encoder, or user attributes are used.
Each tensor has a JSON sidecar containing inputs, weights and execution settings.
"""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Set Hub paths before importing Transformers (which imports Hub constants).
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from device import get_device, get_dtype
import torch
import transformers
from safetensors import safe_open
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, GPT2Model, GPT2Tokenizer, Qwen3_5TextModel

MOVIE_TEMPLATE = "Title: {title}. Genres: {genres}."


def download_qwen(revision="main"):
    """Keep weights AND Hub cache in this project; pin the resolved commit."""
    from huggingface_hub import HfApi, snapshot_download
    model_id = "Qwen/Qwen3.5-0.8B"
    commit = HfApi(token=False).model_info(model_id, revision=revision).sha
    destination = ROOT / "models/Qwen3.5-0.8B"
    snapshot_download(model_id, revision=commit, local_dir=destination,
                      cache_dir=ROOT / ".cache/huggingface/hub", token=False,
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
    (destination / "revision.json").write_text(json.dumps(
        {"model_id": model_id, "revision": commit}, indent=2) + "\n")
    return destination


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_movies(path):
    records = []
    for line in Path(path).read_text(encoding="latin-1").splitlines():
        movie_id, title, genres = line.split("::")
        records.append((int(movie_id), MOVIE_TEMPLATE.format(
            title=title, genres=genres.replace("|", ", "))))
    ids = [row[0] for row in records]
    if not ids or min(ids) <= 0 or len(set(ids)) != len(ids):
        raise ValueError("Movie IDs must be unique positive integers")
    return sorted(records)


def masked_mean_pool(hidden, attention_mask):
    """Accumulate in float32, including when the backbone uses fp16/bf16."""
    mask = attention_mask.unsqueeze(-1).to(torch.float32)
    counts = mask.sum(dim=1)
    if (counts == 0).any():
        raise ValueError("Cannot pool an empty token sequence")
    return (hidden.float() * mask).sum(dim=1) / counts


def load_encoder(encoder, model_path, device):
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    text_config = getattr(config, "text_config", config)
    hidden_size = int(text_config.hidden_size)
    dtype = get_dtype(device)
    # Build GPT-2 byte BPE directly from its vocabulary and merge rules.
    # The optional serialized tokenizer cache is not needed by this loader.
    tokenizer = (GPT2Tokenizer(vocab=str(Path(model_path) / "vocab.json"),
                              merges=str(Path(model_path) / "merges.txt"))
                 if encoder == "gpt2" else
                 AutoTokenizer.from_pretrained(model_path, local_files_only=True))
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if encoder == "gpt2":
        model, info = GPT2Model.from_pretrained(
            model_path, local_files_only=True, dtype=dtype,
            attn_implementation="eager", output_loading_info=True)
        if info["missing_keys"] or info["mismatched_keys"]:
            raise ValueError(f"Incomplete GPT-2 checkpoint: {info}")
    else:
        # Load only text tensors from the multimodal checkpoint. Strict loading
        # prevents accidentally encoding with newly initialized parameters.
        with torch.device("meta"):
            model = Qwen3_5TextModel(text_config)
        model.config._attn_implementation = "eager"
        state = {}
        for shard in sorted(Path(model_path).glob("*.safetensors")):
            with safe_open(shard, framework="pt", device="cpu") as weights:
                for key in weights.keys():
                    prefix = "model.language_model."
                    if key.startswith(prefix):
                        name = key[len(prefix):]
                        if name in state:
                            raise ValueError(f"Duplicate checkpoint key: {name}")
                        state[name] = weights.get_tensor(key).to(dtype)
        model.load_state_dict(state, strict=True, assign=True)
        del state
        # RoPE frequencies are derived nonpersistent buffers, absent from the
        # checkpoint. Recreate them outside the meta context before .to().
        model.rotary_emb = type(model.rotary_emb)(config=text_config, device="cpu")
    model = model.to(device).eval().requires_grad_(False)
    if any("visual" in name for name, _ in model.named_modules()):
        raise AssertionError("Visual encoder must not be loaded")
    return tokenizer, model, hidden_size


@torch.inference_mode()
def encode_texts(texts, tokenizer, model, hidden_size, device, max_length):
    inputs = tokenizer(texts, padding=True, truncation=True,
                       max_length=max_length, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    hidden = model(**inputs, use_cache=False).last_hidden_state
    if hidden.shape != (*inputs["input_ids"].shape, hidden_size):
        raise AssertionError(f"Unexpected hidden shape: {hidden.shape}")
    pooled = masked_mean_pool(hidden, inputs["attention_mask"]).cpu()
    if not torch.isfinite(pooled).all():
        raise ValueError("Non-finite features")
    return pooled


def extract_movie_features(encoder="gpt2", data_path="ml-1m", output=None,
                           batch_size=8, model_path=None, max_length=32,
                           seed=42, smoke_only=False, force=False, threads=4):
    if batch_size <= 0 or max_length < 2 or threads <= 0:
        raise ValueError("Invalid batch size, max length or thread count")
    if encoder not in {"gpt2", "qwen"}:
        raise ValueError(f"Unknown encoder: {encoder}")
    model_path = Path(model_path or ROOT / (
        "models/gpt2" if encoder == "gpt2" else "models/Qwen3.5-0.8B"))
    source = Path(data_path) / "movies.dat"
    records = read_movies(source)
    output = Path(output or ROOT / "data" / f"{encoder}_movie_features.pt")
    sidecar = output.with_suffix(".json")
    device = get_device()
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    weight_files = sorted(model_path.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors weights in {model_path}")
    revision_file = model_path / "revision.json"
    config = {
        "format_version": 1, "encoder": encoder,
        "model_id": "gpt2" if encoder == "gpt2" else "Qwen/Qwen3.5-0.8B",
        "model_path": str(model_path.resolve().relative_to(ROOT))
        if model_path.resolve().is_relative_to(ROOT) else str(model_path.resolve()),
        "revision": json.loads(revision_file.read_text())["revision"] if revision_file.exists() else None,
        "model_files_sha256": {p.name: sha256(p) for p in sorted(model_path.iterdir())
                                if p.is_file() and p.suffix in {".json", ".txt", ".jinja", ".safetensors"}
                                and not (encoder == "gpt2" and p.name in
                                         {"tokenizer.json", "tokenizer_config.json"})},
        "data_path": str(source), "data_sha256": sha256(source),
        "text_template": MOVIE_TEMPLATE, "pooling": "last_hidden_masked_mean_float32",
        "padding_side": "right", "max_length": max_length,
        "tokenizer_source": "vocab.json+merges.txt" if encoder == "gpt2" else "AutoTokenizer",
        "batch_size": batch_size, "seed": seed, "threads": threads,
        "device": str(device), "model_dtype": str(get_dtype(device)),
        "storage_dtype": "torch.float32", "use_user_text": False,
        "use_alignment": False, "use_cache": False,
        "attention_implementation": "eager", "deterministic_algorithms": True,
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    if output.exists() and not smoke_only and not force:
        if not sidecar.exists():
            raise ValueError("Cache has no generation config; use --force to regenerate")
        previous = json.loads(sidecar.read_text())
        if any(previous.get(key) != value for key, value in config.items()):
            raise ValueError("Cache config differs; use another output or --force")
        features = torch.load(output, map_location="cpu", weights_only=True)
        valid = torch.zeros(max(r[0] for r in records) + 1, dtype=torch.bool)
        valid[[r[0] for r in records]] = True
        if (sha256(output) != previous["tensor_sha256"] or
                features.shape != (len(valid), previous["hidden_size"]) or
                features.dtype != torch.float32 or not torch.isfinite(features).all() or
                torch.count_nonzero(features[~valid])):
            raise ValueError("Invalid feature cache")
        print(f"Reused verified cache: {output}", flush=True)
        return features
    tokenizer, model, hidden_size = load_encoder(encoder, model_path, device)
    # Unequal text lengths check padding handling and repeatability on real weights.
    sample_texts = [records[0][1], records[-1][1], records[len(records) // 2][1]]
    sample = encode_texts(sample_texts, tokenizer, model, hidden_size, device, max_length)
    repeated = encode_texts(sample_texts, tokenizer, model, hidden_size, device, max_length)
    alone = encode_texts(sample_texts[:1], tokenizer, model, hidden_size, device, max_length)
    torch.testing.assert_close(sample, repeated, rtol=0, atol=0)
    # Half-precision GEMMs round differently for different batch shapes.
    # Compare relative vector error, rather than relative errors near zero.
    tolerance = {torch.float32: 1e-5, torch.float16: 0.01,
                 torch.bfloat16: 0.03}[get_dtype(device)]
    padding_relative_error = float((sample[:1] - alone).norm() / alone.norm().clamp_min(1e-12))
    if padding_relative_error > tolerance:
        raise AssertionError(f"Padding/batch relative error {padding_relative_error} > {tolerance}")
    config.update(hidden_size=hidden_size, smoke_test={
        "passed": True, "shape": list(sample.shape), "finite": True,
        "frozen": not any(p.requires_grad for p in model.parameters()),
        "repeat_max_abs_diff": float((sample - repeated).abs().max()),
        "padding_max_abs_diff": float((sample[:1] - alone).abs().max()),
        "padding_relative_l2_error": padding_relative_error,
        "padding_relative_l2_tolerance": tolerance,
        "visual_encoder_loaded": False,
    })
    print(f"{encoder} smoke passed: {device}, {get_dtype(device)}, {tuple(sample.shape)}", flush=True)
    if smoke_only:
        return config
    features = torch.zeros(max(r[0] for r in records) + 1, hidden_size, dtype=torch.float32)
    for start in tqdm(range(0, len(records), batch_size), desc=encoder):
        batch = records[start:start + batch_size]
        features[[r[0] for r in batch]] = encode_texts(
            [r[1] for r in batch], tokenizer, model, hidden_size, device, max_length)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".pt.tmp")
    torch.save(features, temporary)
    temporary.replace(output)
    config.update(movie_count=len(records), shape=list(features.shape), tensor_sha256=sha256(output))
    sidecar.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Saved {output}: {tuple(features.shape)}, {features.dtype}", flush=True)
    return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=["gpt2", "qwen"], default="gpt2")
    parser.add_argument("--data-path", default="ml-1m")
    parser.add_argument("--model-path")
    parser.add_argument("--output")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--download-qwen", action="store_true")
    parser.add_argument("--revision", default="main")
    arguments = vars(parser.parse_args())
    download = arguments.pop("download_qwen")
    revision = arguments.pop("revision")
    if download:
        download_qwen(revision)
    extract_movie_features(**arguments)
