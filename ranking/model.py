"""Text-only Qwen3.5-0.8B. The visual encoder and the MTP head are not loaded.

Weights stay in ``models/Qwen3.5-0.8B``, the same directory B uses. The
revision is the one recorded in ``data/qwen_movie_features.json`` when that
sidecar is present, so the ranker and the frozen features share a checkpoint.
"""
import json
from pathlib import Path

from device import get_device, get_dtype
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "Qwen3.5-0.8B"
SIDECAR = ROOT / "data" / "qwen_movie_features.json"


def pinned_revision():
    if not SIDECAR.exists():
        return "main"
    return json.loads(SIDECAR.read_text(encoding="utf-8"))["revision"]


def ensure_weights(revision=None):
    if any(MODEL_DIR.glob("*.safetensors")):
        return MODEL_DIR
    from features import download_qwen
    download_qwen(revision or pinned_revision())
    if not any(MODEL_DIR.glob("*.safetensors")):
        raise FileNotFoundError(f"No safetensors weights in {MODEL_DIR}")
    return MODEL_DIR


def use_trainable_dtype(model, device, loaded_dtype):
    """Keep the shared load dtype, then upcast MPS fp16 before any optimizer step.

    The pure PyTorch DeltaNet backward produces non-finite gradients in fp16
    on MPS. CUDA keeps the dtype from get_dtype. Inference can stay in fp16.
    """
    if device.type == "mps" and loaded_dtype == torch.float16:
        model.float()
        if model.config.tie_word_embeddings:
            model.tie_weights()
            if model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
                model.lm_head.weight = model.model.embed_tokens.weight
        return torch.float32
    return loaded_dtype


def load_tokenizer(model_path=None):
    model_path = Path(model_path or ensure_weights())
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_ranker(model_path=None):
    """Load the causal text model on the shared device and dtype."""
    model_path = Path(model_path or ensure_weights())
    device = get_device()
    dtype = get_dtype(device)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    text_config = config.text_config
    text_config._attn_implementation = "eager"
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(text_config)
    mapped = _text_state(model_path, dtype)
    missing, unexpected = _assign(model, mapped)
    allowed_missing = {"lm_head.weight"} if text_config.tie_word_embeddings else set()
    if set(missing) - allowed_missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch. missing={missing}, unexpected={unexpected}")
    if text_config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
    model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(text_config, device="cpu")
    model = model.to(device).eval()
    if text_config.tie_word_embeddings:
        model.tie_weights()
        if model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
            model.lm_head.weight = model.model.embed_tokens.weight
    _assert_materialized(model)
    if any("visual" in name for name, _ in model.named_modules()):
        raise AssertionError("Visual encoder must not be loaded")
    model.requires_grad_(False)
    model.config.use_cache = False
    return model, load_tokenizer(model_path), device, dtype


def _text_state(model_path, dtype):
    prefix = "model.language_model."
    mapped = {}
    for shard in sorted(Path(model_path).glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as weights:
            for key in weights.keys():
                if key.startswith("model.visual.") or key.startswith("mtp."):
                    continue
                if key == "lm_head.weight":
                    name = key
                elif key.startswith(prefix):
                    name = "model." + key[len(prefix):]
                else:
                    raise RuntimeError(f"Unexpected checkpoint key {key}")
                if name in mapped:
                    raise RuntimeError(f"Duplicate checkpoint key {name}")
                mapped[name] = weights.get_tensor(key).to(dtype)
    if not mapped:
        raise RuntimeError(f"No text weights in {model_path}")
    return mapped


def _assign(model, mapped):
    own = set(model.state_dict())
    unexpected = sorted(set(mapped) - own)
    missing = sorted(own - set(mapped))
    model.load_state_dict(mapped, strict=False, assign=True)
    return missing, unexpected


def _assert_materialized(model):
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if getattr(tensor, "is_meta", False):
            raise RuntimeError(f"{name} is still on the meta device")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise RuntimeError(f"{name} is non-finite")
