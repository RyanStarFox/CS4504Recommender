"""LoRA on the text model's linear projections. The base weights stay frozen."""
import json
import math
from pathlib import Path

import torch
from torch import nn

TARGET_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
)


class LoRALinear(nn.Module):
    def __init__(self, linear, rank, alpha, dropout):
        super().__init__()
        if rank <= 0 or alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        self.linear = linear
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        device = linear.weight.device
        self.lora_A = nn.Parameter(torch.empty(rank, linear.in_features, device=device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, rank, device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, inputs):
        base = self.linear(inputs)
        dropped = self.dropout(inputs).to(self.lora_A.dtype)
        update = dropped @ self.lora_A.T @ self.lora_B.T
        return base + (update * self.scaling).to(base.dtype)


def apply_lora(model, rank=8, alpha=16, dropout=0.05, targets=TARGET_SUFFIXES):
    model.requires_grad_(False)
    replaced = []
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            raise RuntimeError(f"LoRA is already applied at {name}")
        if isinstance(module, nn.Linear) and name.split(".")[-1] in targets:
            _set_module(model, name, LoRALinear(module, rank, alpha, dropout))
            replaced.append(name)
    if not replaced:
        raise RuntimeError("No linear layers matched the LoRA targets")
    return tuple(replaced)


def lora_parameters(model):
    found = [param for name, param in model.named_parameters() if "lora_A" in name or "lora_B" in name]
    if not found or not all(param.requires_grad for param in found):
        raise RuntimeError("LoRA parameters are missing or frozen")
    return found


def lora_state(model):
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if "lora_A" in name or "lora_B" in name
    }


def restore_lora(model, state):
    named = dict(model.named_parameters())
    missing = [name for name in state if name not in named]
    if missing or len(state) != len(lora_state(model)):
        raise RuntimeError("LoRA snapshot does not match this model")
    with torch.no_grad():
        for name, value in state.items():
            named[name].copy_(value.to(device=named[name].device, dtype=named[name].dtype))


def lora_fingerprint(model):
    pieces = [
        param.detach().float().reshape(-1).cpu()
        for name, param in model.named_parameters()
        if "lora_A" in name or "lora_B" in name
    ]
    if not pieces:
        raise RuntimeError("No LoRA parameters")
    return torch.cat(pieces)


def save_adapter(path, model, config):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    tensors = {name: value.detach().cpu() for name, value in lora_state(model).items()}
    torch.save(tensors, path / "adapter.pt")
    (path / "adapter.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def load_adapter(model, path):
    path = Path(path)
    config = json.loads((path / "adapter.json").read_text(encoding="utf-8"))
    if not any(isinstance(module, LoRALinear) for module in model.modules()):
        apply_lora(model, rank=config["rank"], alpha=config["alpha"], dropout=config["dropout"])
    tensors = torch.load(path / "adapter.pt", map_location="cpu", weights_only=True)
    restore_lora(model, tensors)
    return config


def _set_module(root, qualified_name, module):
    parent = root
    parts = qualified_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)
