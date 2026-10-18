"""SFT, DPO and offline GRPO reranking on frozen-base Qwen3.5-0.8B.

The package reads only ``sequences.json`` and a candidates file. Day-1 debugging
points that candidates path at ``data/mock_candidates.json``; the formal file is
``data/candidates.json``. Both use the same fields.

Reported generations for rows 7-9 come from llama.cpp. The Hugging Face sampler
exists so the training loop can run before a GGUF export is available.
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
