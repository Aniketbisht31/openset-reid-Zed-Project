"""Batch-Hard Triplet Loss with online hard-negative mining.

Reference: Hermans, Beyer & Leibe, *In Defense of the Triplet Loss for
Person Re-Identification*, arXiv 2017.

For every anchor in the batch the **hardest positive** (farthest
same-identity sample) and the **hardest negative** (closest
different-identity sample) are selected.  This requires a PK-style
sampler so that each identity has ≥ 2 images in the batch.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchHardTripletLoss(nn.Module):
    """Triplet loss with optional online hard-negative mining.

    Args:
        margin: Additive margin (default ``0.3``).
        hard_mining: If True, mine hardest positive and hardest negative per anchor.
                     If False, average over all valid (anchor, pos, neg) triplets in batch.
    """

    def __init__(self, margin: float = 0.3, hard_mining: bool = True):
        super().__init__()
        self.margin = margin
        self.hard_mining = hard_mining

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: ``[B, D]`` L2-normalised.
            labels:     ``[B]`` identity labels.

        Returns:
            Scalar mean triplet loss over valid triplets.
        """
        # Euclidean distance from cosine (embeddings are unit vectors):
        #   ‖a − b‖ = √(2 − 2·cos(a,b))
        dist = 2.0 - 2.0 * torch.mm(embeddings, embeddings.t())
        dist = dist.clamp(min=0.0).sqrt()            # [B, B]

        B = embeddings.size(0)
        labels_col = labels.view(B)

        is_pos = (labels_col.unsqueeze(0) == labels_col.unsqueeze(1)) & (~torch.eye(B, dtype=torch.bool, device=embeddings.device))
        is_neg = labels_col.unsqueeze(0) != labels_col.unsqueeze(1)

        if not self.hard_mining:
            # Batch-all triplet: (dist_ap - dist_an + margin) for all valid (a, p, n)
            # dist: [B, B] -> [B, B, 1] for pos, [B, 1, B] for neg
            triplet_margin = dist.unsqueeze(2) - dist.unsqueeze(1) + self.margin  # [B, B, B]
            triplet_mask = is_pos.unsqueeze(2) & is_neg.unsqueeze(1)              # [B, B, B]
            active_triplets = F.relu(triplet_margin) * triplet_mask.float()
            num_valid = triplet_mask.sum()
            if num_valid > 0:
                return active_triplets.sum() / (active_triplets > 0).sum().clamp(min=1)
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # Hardest positive per anchor (max dist among same-id)
        pos_dist = dist * is_pos.float()
        hardest_pos, _ = pos_dist.max(dim=1)         # [B]

        # Hardest negative per anchor (min dist among diff-id)
        neg_dist = dist + (~is_neg).float() * 1e6    # mask out non-negatives
        hardest_neg, _ = neg_dist.min(dim=1)          # [B]

        # Margin-based loss
        raw = F.relu(hardest_pos - hardest_neg + self.margin)

        # Only anchors with ≥ 1 positive *and* ≥ 1 negative are valid
        valid = (is_pos.sum(1) > 0) & (is_neg.sum(1) > 0)
        if valid.any():
            return raw[valid].mean()
        return raw.mean()


# Alias
TripletLoss = BatchHardTripletLoss
