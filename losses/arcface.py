"""ArcFace: Additive Angular Margin Loss for Deep Face Recognition.

Reference: Deng et al., *ArcFace: Additive Angular Margin Loss for Deep
Face Recognition*, CVPR 2019.

The proxy weight matrix ``W ∈ ℝ^{C × D}`` is L2-normalised per-row so
that ``W · x = cos θ``.  For the ground-truth class the logit becomes
``cos(θ + m)`` before scaling by ``s``.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    """ArcFace margin loss.

    Args:
        embed_dim:   Embedding dimension (must match model output).
        num_classes: Number of known identities.
        s:           Logit scale  (default ``30``).
        m:           Angular margin in **radians** (default ``0.50``).
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        s: float = 30.0,
        m: float = 0.50,
    ):
        super().__init__()
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.empty(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

        # Pre-compute constants
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)       # cos(π − m)
        self.mm    = math.sin(math.pi - m) * m   # sin(π − m)·m

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: ``[B, D]`` L2-normalised.
            labels:     ``[B]`` integer class labels.

        Returns:
            Scalar cross-entropy loss with ArcFace margin.
        """
        # cosine similarity  [B, C]
        cosine = F.linear(embeddings, F.normalize(self.weight))
        sine   = torch.sqrt((1.0 - cosine.pow(2)).clamp(min=0))

        # cos(θ + m) = cosθ·cos m − sinθ·sin m
        phi = cosine * self.cos_m - sine * self.sin_m

        # Numerical guard: when cosθ ≤ cos(π − m), fall back
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine).scatter_(1, labels.unsqueeze(1), 1.0)
        logits = one_hot * phi + (1.0 - one_hot) * cosine
        logits *= self.s

        return F.cross_entropy(logits, labels)
