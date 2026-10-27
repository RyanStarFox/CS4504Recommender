"""Merge the SFT LoRA into the base checkpoint and convert it with llama.cpp.

The converter has to match the installed llama.cpp (v0.7.0). A newer
converter can write tensor layouts that this binary will not load.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LLAMA_CPP_DIR = Path.home() / "Library" / "Caches" / "llama.cpp-620d2225c"
LLAMA_CPP_COMMIT = "620d2225cc436ab0b001e3200e590e3b286ef9ab"


def lora_deltas(state, alpha, rank):
    """Map LoRA parameter names onto base checkpoint keys.

    Training wraps ``model.*`` after the language-model prefix is removed.
    The saved Hugging Face tensors still use ``model.language_model.*``.
    The forward update is ``inputs @ A.T @ B.T * (alpha / rank)``, so the
    weight delta is ``(alpha / rank) * B @ A``.
    """
    if rank <= 0 or alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    scaling = alpha / rank
    deltas = {}
    for name, factor_a in state.items():
        if not name.endswith(".lora_A"):
            continue
        module = name[: -len(".lora_A")]
        factor_b = state.get(module + ".lora_B")
        if factor_b is None:
            raise KeyError(f"Missing LoRA B for {module}")
        if not module.startswith("model."):
            raise ValueError(f"Unexpected LoRA module {module}")
        key = "model.language_model." + module[len("model."):] + ".weight"
        delta = scaling * (factor_b.float() @ factor_a.float())
        if key in deltas:
            raise RuntimeError(f"Duplicate LoRA delta for {key}")
        deltas[key] = delta
    if not deltas:
        raise RuntimeError("Adapter has no LoRA weights")
    return deltas


def merge_adapter(adapter_dir, source_dir, dest_dir):
    adapter_dir = Path(adapter_dir)
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    config = json.loads((adapter_dir / "adapter.json").read_text(encoding="utf-8"))
    state = torch.load(adapter_dir / "adapter.pt", map_location="cpu", weights_only=True)
    deltas = lora_deltas(state, alpha=config["alpha"], rank=config["rank"])
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True)
    for path in source_dir.iterdir():
        if path.is_file() and not path.name.endswith(".safetensors"):
            shutil.copy2(path, dest_dir / path.name)
    applied = set()
    shards = sorted(source_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No safetensors in {source_dir}")
    for shard in shards:
        tensors = {}
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                if key in deltas:
                    delta = deltas[key]
                    if tuple(delta.shape) != tuple(tensor.shape):
                        raise RuntimeError(
                            f"{key} delta shape {tuple(delta.shape)} != weight {tuple(tensor.shape)}")
                    tensor = (tensor.float() + delta).to(dtype=tensor.dtype)
                    applied.add(key)
                tensors[key] = tensor
        save_file(tensors, dest_dir / shard.name)
    missing = sorted(set(deltas) - applied)
    if missing:
        raise RuntimeError(f"LoRA target missing from the base checkpoint: {missing[0]}")
    return dest_dir


def convert_hf_to_gguf(source_dir, outfile, llama_cpp_dir=LLAMA_CPP_DIR):
    source_dir = Path(source_dir)
    outfile = Path(outfile)
    script = Path(llama_cpp_dir) / "convert_hf_to_gguf.py"
    if not script.is_file():
        raise FileNotFoundError(
            f"llama.cpp converter is not at {script}. "
            f"Check out tag v0.7.0 ({LLAMA_CPP_COMMIT}) there before sampling."
        )
    outfile.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable, str(script), str(source_dir),
            "--outfile", str(outfile), "--outtype", "f16", "--no-mtp",
        ],
        check=True,
        cwd=script.parent,
    )
    if not outfile.is_file():
        raise RuntimeError(f"Converter did not write {outfile}")
    return outfile


def ensure_gguf(adapter_dir, source_dir, work_dir, llama_cpp_dir=LLAMA_CPP_DIR):
    adapter_dir = Path(adapter_dir)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = work_dir / "sft-f16.gguf"
    stamp_path = work_dir / "sft-f16.stamp"
    adapter_file = adapter_dir / "adapter.pt"
    stamp = f"{adapter_file.stat().st_mtime_ns}:{adapter_file.stat().st_size}"
    if gguf_path.is_file() and stamp_path.is_file() and stamp_path.read_text(encoding="utf-8") == stamp:
        return gguf_path
    merged = work_dir / "merged-hf"
    print(f"merging {adapter_dir} into {merged}", flush=True)
    merge_adapter(adapter_dir, source_dir, merged)
    print(f"converting {merged} to {gguf_path}", flush=True)
    convert_hf_to_gguf(merged, gguf_path, llama_cpp_dir)
    stamp_path.write_text(stamp, encoding="utf-8")
    return gguf_path
