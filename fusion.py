"""Trainable text injection and the optional movie-level alignment objective."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class GatedMovieFusion(nn.Module):
    """D's fuse_items interface, shared by history and candidate embeddings.

    Frozen text caches stay in SASRec. Only projection and gate are learned.
    User text is deliberately absent from the 5a/5b/6 path.
    """

    def __init__(self, text_hidden_size, rec_hidden_size, use_movie_text=True):
        super().__init__()
        if text_hidden_size <= 0 or rec_hidden_size <= 0:
            raise ValueError("Hidden sizes must be positive")
        self.use_movie_text = bool(use_movie_text)
        self.projection = nn.Linear(text_hidden_size, rec_hidden_size, bias=False)
        self.normalization = nn.LayerNorm(rec_hidden_size, elementwise_affine=False)
        self.gate = nn.Linear(2 * rec_hidden_size, rec_hidden_size)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def project_text(self, text_vectors):
        return self.normalization(self.projection(text_vectors.detach()))

    def fuse_items(self, id_vectors, text_vectors):
        if not self.use_movie_text:
            return id_vectors
        if id_vectors.shape[:-1] != text_vectors.shape[:-1]:
            raise ValueError("ID and text batch dimensions must match")
        projected = self.project_text(text_vectors)
        gate = torch.sigmoid(self.gate(torch.cat((id_vectors, projected), dim=-1)))
        valid_text = text_vectors.ne(0).any(dim=-1, keepdim=True)
        return id_vectors + gate * projected * valid_text


def alignment_loss(
    id_vectors, projected_text, movie_ids, valid_item_mask, temperature=0.07
):
    """Symmetric InfoNCE; deduplicate movies and exclude padding/missing IDs.

    Pass raw item_embedding vectors and fusion.project_text(cache[ids]). The
    caller adds an explicitly configured weight to recommendation loss.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if id_vectors.ndim != 2 or projected_text.shape != id_vectors.shape:
        raise ValueError("Expected equal [N, rec_hidden_size] representation shapes")
    if movie_ids.dtype != torch.long or movie_ids.shape != id_vectors.shape[:1]:
        raise ValueError("movie_ids must be a rank-1 long tensor with N entries")
    if valid_item_mask.dtype != torch.bool or valid_item_mask.ndim != 1:
        raise ValueError("valid_item_mask must be a rank-1 bool tensor")
    if movie_ids.numel() and (
        movie_ids.min() < 0 or movie_ids.max() >= len(valid_item_mask)
    ):
        raise ValueError("movie_ids outside valid_item_mask")
    keep = movie_ids.ne(0) & valid_item_mask[movie_ids]
    ids = movie_ids[keep]
    order = torch.argsort(ids)
    sorted_ids = ids[order]
    first = torch.ones_like(sorted_ids, dtype=torch.bool)
    first[1:] = sorted_ids[1:] != sorted_ids[:-1]
    rows = order[first]
    left = id_vectors[keep][rows]
    right = projected_text[keep][rows]
    if len(rows) < 2:
        return (left.sum() + right.sum()) * 0.0
    logits = F.normalize(left.float(), dim=-1) @ F.normalize(right.float(), dim=-1).T
    logits = logits / temperature
    labels = torch.arange(len(rows), device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
