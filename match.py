"""Match query images / directories against an enrolled gallery.

Given one or more query images the script:
1. Extracts a 256-D embedding.
2. Computes cosine similarity against every enrolled prototype.
3. Applies the per-class (or global) threshold to accept / reject.
4. Optionally applies a calibrator to output confidence ∈ [0, 1].
"""
from __future__ import annotations

import os
import argparse
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
import yaml

from models.backbone import build_backbone
from models.heads import ReIDModel
from data.market1501 import get_transforms


# ═══════════════════════════════════════════════════════════════════════
# Open-Set Matcher
# ═══════════════════════════════════════════════════════════════════════

class OpenSetMatcher:
    """Cosine-similarity prototype matcher with adaptive rejection."""

    def __init__(self, model, gallery, device, calibrator=None):
        self.model      = model
        self.device     = device
        self.calibrator = calibrator

        self.prototypes  = gallery["prototypes"]
        self.thresholds  = gallery["thresholds"]
        self.label2pid   = gallery["label2pid"]

        # Pre-stack for vectorised matching
        self.labels     = sorted(self.prototypes.keys())
        self.proto_mat  = torch.stack(
            [self.prototypes[l] for l in self.labels]
        ).to(device)                                      # [N, D]
        self.thresh_vec = torch.tensor(
            [self.thresholds[l] for l in self.labels]
        ).to(device)                                      # [N]

    # ── single image ─────────────────────────────────────────────────

    @torch.no_grad()
    def extract(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """``[3,H,W] → [D]``."""
        self.model.eval()
        return self.model(img_tensor.unsqueeze(0).to(self.device)).squeeze(0)

    @torch.no_grad()
    def match(self, embedding: torch.Tensor) -> Dict[str, Any]:
        """Match one embedding against gallery.

        Returns dict with keys:
            matched, pred_label, pred_pid, similarity,
            threshold, confidence, all_similarities.
        """
        emb  = embedding.to(self.device)
        sims = self.proto_mat @ emb                        # [N]

        best_sim, best_idx = sims.max(0)
        best_label = self.labels[best_idx.item()]
        th = self.thresh_vec[best_idx].item()
        matched = best_sim.item() >= th

        # Confidence
        if self.calibrator is not None:
            conf = float(self.calibrator.calibrate(best_sim.item()))
        else:
            conf = float(1.0 / (1.0 + np.exp(-12.0 * (best_sim.item() - th))))

        return {
            "matched":          matched,
            "pred_label":       best_label if matched else -1,
            "pred_pid":         self.label2pid.get(best_label, -1) if matched else -1,
            "similarity":       best_sim.item(),
            "threshold":        th,
            "confidence":       np.clip(conf, 0.0, 1.0),
            "all_similarities": sims.cpu(),
        }

    def match_batch(self, embeddings: torch.Tensor) -> List[Dict[str, Any]]:
        return [self.match(e) for e in embeddings]


# ═══════════════════════════════════════════════════════════════════════
# Convenience loader
# ═══════════════════════════════════════════════════════════════════════

def load_matcher(
    config_path: str,
    checkpoint_path: str,
    gallery_path: str,
    calibrator_path: Optional[str] = None,
) -> OpenSetMatcher:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = build_backbone(cfg["model"]["backbone"], pretrained=False)
    model = ReIDModel(backbone, cfg["model"]["embed_dim"]).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    gallery = torch.load(gallery_path, map_location=device, weights_only=False)

    calibrator = None
    if calibrator_path and os.path.exists(calibrator_path):
        calibrator = torch.load(calibrator_path, weights_only=False)

    return OpenSetMatcher(model, gallery, device, calibrator)


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser("match.py — query against enrolled gallery")
    ap.add_argument("--config",     default="configs/default.yaml")
    ap.add_argument("--checkpoint", default="output/best.pth")
    ap.add_argument("--gallery",    default="output/gallery.pth")
    ap.add_argument("--calibrator", default="output/calibrator.pth")
    ap.add_argument("--query", required=True,
                    help="Single image path or directory of .jpg/.png files")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    matcher = load_matcher(args.config, args.checkpoint, args.gallery,
                           args.calibrator)
    transform = get_transforms(cfg["data"]["height"], cfg["data"]["width"],
                               is_train=False)

    # Gather query paths
    if os.path.isfile(args.query):
        paths = [args.query]
    else:
        paths = sorted(
            os.path.join(args.query, f)
            for f in os.listdir(args.query)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )

    print(f"\nMatching {len(paths)} query image(s) …\n")

    for p in paths:
        img = Image.open(p).convert("RGB")
        emb = matcher.extract(transform(img))
        r   = matcher.match(emb)

        tag = "KNOWN  " if r["matched"] else "UNKNOWN"
        pid = f"PID={r['pred_pid']}" if r["matched"] else "rejected"
        print(f"  {os.path.basename(p):>30s}  [{tag}]  {pid:>12s}  "
              f"sim={r['similarity']:.4f}  th={r['threshold']:.4f}  "
              f"conf={r['confidence']:.4f}")


if __name__ == "__main__":
    main()
