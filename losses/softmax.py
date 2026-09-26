"""Normalised-Softmax cross-entropy (no angular/cosine margin).

Equivalent to CosFace with ``m = 0``.  Serves as the **baseline** in
ablation studies — the model learns discriminative embeddings via
standard classification, without any explicit margin enforcement.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxLoss(nn.Module):
    """Scaled cosine-softmax loss (no margin).

    Args:
        embed_dim:   Embedding dimension.
        num_classes: Number of known identities.
        s:           Logit scale (default ``30``).
    """

    def __init__(self, embed_dim: int, num_classes: int, s: float = 30.0):
        super().__init__()
        self.s = s
        self.weight = nn.Parameter(torch.empty(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = self.s * F.linear(embeddings, F.normalize(self.weight))
        return F.cross_entropy(logits, labels)
