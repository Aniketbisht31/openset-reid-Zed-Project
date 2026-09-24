"""Embedding head and full ReID model wrapper.

The head projects variable-dimension backbone features into a compact
256-D L2-normalised embedding suitable for cosine-similarity retrieval.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingHead(nn.Module):
    """BN → Linear → BN → L2-norm.

    Args:
        in_channels: Feature dimension from the backbone.
        embed_dim:   Output embedding dimension (default 256).
    """

    def __init__(self, in_channels: int, embed_dim: int = 256):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.fc  = nn.Linear(in_channels, embed_dim, bias=False)
        self.bn2 = nn.BatchNorm1d(embed_dim)

        nn.init.kaiming_normal_(self.fc.weight, mode="fan_out")
        nn.init.ones_(self.bn1.weight);  nn.init.zeros_(self.bn1.bias)
        nn.init.ones_(self.bn2.weight);  nn.init.zeros_(self.bn2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, C_in] → [B, embed_dim]`` (L2-normalised)."""
        x = self.bn1(x)
        x = self.fc(x)
        x = self.bn2(x)
        return F.normalize(x, p=2, dim=1)


class ReIDModel(nn.Module):
    """Complete re-identification model: backbone + embedding head.

    Args:
        backbone:  An ``nn.Module`` with an ``out_channels`` attribute.
        embed_dim: Final embedding dimension (default 256).
    """

    def __init__(self, backbone: nn.Module, embed_dim: int = 256):
        super().__init__()
        self.backbone = backbone
        self.head = EmbeddingHead(backbone.out_channels, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, 3, H, W] → [B, embed_dim]``."""
        return self.head(self.backbone(x))
