"""SASRec sequence encoder used for temporal next-movie recommendation.

MovieLens IDs are used directly.  ID 0 is reserved for padding, so
``item_num`` is the maximum movie ID plus one.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class SASRec(nn.Module):
    """A causal Transformer recommender with optional frozen text features.

    The phase-1 path uses only ``item_embedding``.  In later phases B can pass
    frozen feature tensors and a trainable fusion module.  The fusion module
    must expose ``fuse_items(id_vectors, text_vectors)`` and, when user text is
    enabled, ``fuse_users(sequence_states, user_vectors)``.  Item fusion is
    deliberately shared by history encoding and candidate scoring.
    """

    def __init__(
        self,
        item_num: int,
        rec_hidden_size: int = 64,
        max_seq_len: int = 50,
        num_blocks: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
        movie_features: torch.Tensor | None = None,
        user_features: torch.Tensor | None = None,
        fusion_module: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if item_num <= 1:
            raise ValueError("item_num must include padding and at least one movie")
        if max_seq_len <= 0 or num_blocks <= 0 or num_heads <= 0:
            raise ValueError("max_seq_len, num_blocks and num_heads must be positive")
        if rec_hidden_size <= 0 or rec_hidden_size % num_heads:
            raise ValueError("rec_hidden_size must be positive and divisible by num_heads")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.item_num = int(item_num)
        self.rec_hidden_size = int(rec_hidden_size)
        self.max_seq_len = int(max_seq_len)

        self.item_embedding = nn.Embedding(
            self.item_num, self.rec_hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_len, self.rec_hidden_size)
        self.input_dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=self.rec_hidden_size,
            nhead=num_heads,
            dim_feedforward=4 * self.rec_hidden_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # norm_first is intentional; nested tensors do not support it and only
        # produce a warning rather than an optimization for this workload.
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_blocks, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(self.rec_hidden_size)
        self.fusion_module = fusion_module

        movie_features = self._validate_features(
            movie_features, expected_rows=self.item_num, name="movie_features"
        )
        user_features = self._validate_features(
            user_features, expected_rows=None, name="user_features"
        )
        # Frozen caches follow the model device but are intentionally excluded
        # from state_dict; their sidecars remain the source of provenance.
        self.register_buffer("movie_features", movie_features, persistent=False)
        self.register_buffer("user_features", user_features, persistent=False)
        self.reset_parameters()

    @staticmethod
    def _validate_features(
        features: torch.Tensor | None,
        expected_rows: int | None,
        name: str,
    ) -> torch.Tensor | None:
        if features is None:
            return None
        if features.ndim != 2 or not features.is_floating_point():
            raise ValueError(f"{name} must be a rank-2 floating-point tensor")
        if expected_rows is not None and features.shape[0] != expected_rows:
            raise ValueError(
                f"{name} has {features.shape[0]} rows; expected {expected_rows}"
            )
        if not torch.isfinite(features).all():
            raise ValueError(f"{name} contains non-finite values")
        result = features.detach().to(dtype=torch.float32)
        result.requires_grad_(False)
        return result

    def reset_parameters(self) -> None:
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def _check_item_ids(self, item_ids: torch.Tensor) -> None:
        if item_ids.dtype != torch.long:
            raise TypeError("item IDs must use torch.long")
        if item_ids.numel() and (
            int(item_ids.min()) < 0 or int(item_ids.max()) >= self.item_num
        ):
            raise ValueError(f"item IDs must be in [0, {self.item_num})")

    def item_vectors(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Return the representation shared by histories and candidates."""
        self._check_item_ids(item_ids)
        id_vectors = self.item_embedding(item_ids) * math.sqrt(self.rec_hidden_size)
        if self.movie_features is None:
            return id_vectors
        if self.fusion_module is None or not hasattr(self.fusion_module, "fuse_items"):
            raise RuntimeError(
                "movie_features require fusion_module.fuse_items(id_vectors, text_vectors)"
            )
        text_vectors = self.movie_features[item_ids]
        result = self.fusion_module.fuse_items(id_vectors, text_vectors)
        if result.shape != id_vectors.shape:
            raise ValueError("fuse_items must preserve the ID-vector shape")
        return result

    def _fuse_users(
        self, sequence_states: torch.Tensor, user_ids: torch.Tensor | None
    ) -> torch.Tensor:
        if self.user_features is None:
            return sequence_states
        if user_ids is None:
            raise ValueError("user_ids are required when user_features are enabled")
        if user_ids.dtype != torch.long or user_ids.ndim != 1:
            raise TypeError("user_ids must be a rank-1 torch.long tensor")
        if user_ids.shape[0] != sequence_states.shape[0]:
            raise ValueError("user_ids and sequence batch sizes differ")
        if user_ids.numel() and (
            int(user_ids.min()) < 0 or int(user_ids.max()) >= self.user_features.shape[0]
        ):
            raise ValueError("user ID is outside user_features")
        if self.fusion_module is None or not hasattr(self.fusion_module, "fuse_users"):
            raise RuntimeError(
                "user_features require fusion_module.fuse_users(states, user_vectors)"
            )
        result = self.fusion_module.fuse_users(
            sequence_states, self.user_features[user_ids]
        )
        if result.shape != sequence_states.shape:
            raise ValueError("fuse_users must preserve the sequence-state shape")
        return result

    def forward(
        self, item_sequences: torch.Tensor, user_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Encode left-padded histories and return one state per position."""
        if item_sequences.ndim != 2:
            raise ValueError("item_sequences must have shape [batch, length]")
        if item_sequences.shape[1] > self.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")
        self._check_item_ids(item_sequences)
        valid = item_sequences.ne(0)
        if not valid.any(dim=1).all():
            raise ValueError("every sequence must contain at least one non-padding item")

        # Positions start at zero for each user's first non-padding interaction.
        positions = (valid.long().cumsum(dim=1) - 1).clamp_min(0)
        states = self.item_vectors(item_sequences) + self.position_embedding(positions)
        states = self.input_dropout(states) * valid.unsqueeze(-1)
        length = item_sequences.shape[1]
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=item_sequences.device),
            diagonal=1,
        )
        states = self.encoder(
            states,
            mask=causal_mask,
            src_key_padding_mask=~valid,
        )
        states = self.output_norm(states) * valid.unsqueeze(-1)
        return self._fuse_users(states, user_ids)

    def training_scores(
        self,
        item_sequences: torch.Tensor,
        positive_items: torch.Tensor,
        negative_items: torch.Tensor,
        user_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score sampled positive and negative next items at every position."""
        if positive_items.shape != item_sequences.shape:
            raise ValueError("positive_items must match item_sequences")
        if negative_items.shape != item_sequences.shape:
            raise ValueError("negative_items must match item_sequences")
        states = self(item_sequences, user_ids=user_ids)
        positive_vectors = self.item_vectors(positive_items)
        negative_vectors = self.item_vectors(negative_items)
        positive_scores = (states * positive_vectors).sum(dim=-1)
        negative_scores = (states * negative_vectors).sum(dim=-1)
        valid = positive_items.ne(0)
        return positive_scores, negative_scores, valid

    def sequence_vectors(
        self, item_sequences: torch.Tensor, user_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the state at each history's last non-padding position."""
        states = self(item_sequences, user_ids=user_ids)
        valid = item_sequences.ne(0)
        indices = (
            valid.long()
            * torch.arange(item_sequences.shape[1], device=item_sequences.device)
        ).max(dim=1).values
        rows = torch.arange(item_sequences.shape[0], device=item_sequences.device)
        return states[rows, indices]

    def score_candidates(
        self,
        item_sequences: torch.Tensor,
        candidate_ids: torch.Tensor,
        user_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score shared candidates ``[C]`` or per-user candidates ``[B, C]``."""
        sequence_vectors = self.sequence_vectors(item_sequences, user_ids=user_ids)
        candidate_vectors = self.item_vectors(candidate_ids)
        if candidate_ids.ndim == 1:
            return sequence_vectors @ candidate_vectors.transpose(0, 1)
        if candidate_ids.ndim == 2 and candidate_ids.shape[0] == item_sequences.shape[0]:
            return torch.einsum("bh,bch->bc", sequence_vectors, candidate_vectors)
        raise ValueError("candidate_ids must have shape [C] or [batch, C]")
