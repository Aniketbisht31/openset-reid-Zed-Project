"""CosFace: Large Margin Cosine Loss.

Reference: Wang et al., *CosFace: Large Margin Cosine Loss for Deep Face
Recognition*, CVPR 2018.

Simpler than ArcFace — the margin ``m`` is subtracted directly from the
cosine similarity of the ground-truth class **before** scaling by ``s``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosFaceLoss(nn.Module):
    """CosFace additive cosine-margin loss.

    Args:
        embed_dim:   Embedding dimension.
        num_classes: Number of known identities.
        s:           Logit scale  (default ``30``).
        m:           Cosine margin (default ``0.35``).
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        s: float = 30.0,
        m: float = 0.35,
    ):
        super().__init__()
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.empty(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: ``[B, D]`` L2-normalised.
            labels:     ``[B]`` integer class labels.

        Returns:
            Scalar cross-entropy loss with CosFace margin.
        """
        cosine = F.linear(embeddings, F.normalize(self.weight))

        one_hot = torch.zeros_like(cosine).scatter_(1, labels.unsqueeze(1), 1.0)
        logits  = (cosine - one_hot * self.m) * self.s

        return F.cross_entropy(logits, labels)
