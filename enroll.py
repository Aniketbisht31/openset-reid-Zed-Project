"""Enrol gallery identities — build per-class prototypes and thresholds.

Pipeline:
1. Load trained model.
2. Extract embeddings for every image of every *known* identity.
3. Compute per-identity mean prototype (L2-normalised).
4. Derive intra-class similarity statistics.
5. Set per-class adaptive thresholds (or a single global one).
6. Persist ``gallery.pth``.
"""
from __future__ import annotations

import os
import argparse
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from models.backbone import build_backbone
from models.heads import ReIDModel
from data.market1501 import Market1501, get_transforms


# ── Embedding extraction ─────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    embs, labs, paths = [], [], []
    for imgs, labels, _, ps in tqdm(loader, desc="Extracting", leave=False):
        embs.append(model(imgs.to(device, non_blocking=True)).cpu())
        labs.extend(labels.tolist())
        paths.extend(ps)
    return torch.cat(embs), labs, paths


# ── Prototypes ───────────────────────────────────────────────────────

def build_prototypes(embeddings, labels):
    """Mean-centroid prototype per identity.

    Returns:
        prototypes:      ``{label: tensor[D]}``
        per_class_stats: ``{label: {mean_sim, std_sim, min_sim, count}}``
    """
    groups: dict[int, list] = defaultdict(list)
    for emb, lab in zip(embeddings, labels):
        if lab >= 0:
            groups[lab].append(emb)

    prototypes, stats = {}, {}
    for lab, vecs in groups.items():
        S = torch.stack(vecs)
        mu = S.mean(0)
        mu = mu / mu.norm()

        sims = (S @ mu).clamp(-1, 1)
        prototypes[lab] = mu
        stats[lab] = {
            "mean_sim": sims.mean().item(),
            "std_sim":  sims.std().item() if len(vecs) > 1 else 0.0,
            "min_sim":  sims.min().item(),
            "count":    len(vecs),
        }
    return prototypes, stats


# ── Adaptive thresholds ──────────────────────────────────────────────

def compute_adaptive_thresholds(stats, global_th=0.5, alpha=0.8):
    """``th_i = α · (μ_i − 2σ_i) + (1−α) · global_th``."""
    ths = {}
    for lab, s in stats.items():
        per_class = s["mean_sim"] - 2.0 * s["std_sim"]
        ths[lab] = alpha * per_class + (1 - alpha) * global_th
    return ths


# ── Main ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser("enroll.py — build gallery prototypes")
    ap.add_argument("--config",     default="configs/default.yaml")
    ap.add_argument("--checkpoint", default="output/best.pth")
    ap.add_argument("--output",     default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.output or cfg["output"]["dir"]

    # Model
    backbone = build_backbone(cfg["model"]["backbone"], pretrained=False)
    model = ReIDModel(backbone, cfg["model"]["embed_dim"]).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"[enroll] Model loaded from {args.checkpoint}")

    # Splits
    splits = torch.load(os.path.join(out_dir, "splits.pth"), weights_only=False)
    known_pids = set(splits["known_pids"])

    # Gallery dataset (training images of known IDs)
    transform = get_transforms(cfg["data"]["height"], cfg["data"]["width"],
                               is_train=False)
    ds = Market1501(cfg["data"]["root"], split="train",
                    known_pids=known_pids, transform=transform, relabel=True)
    loader = DataLoader(ds, batch_size=128,
                        num_workers=cfg["data"]["num_workers"],
                        pin_memory=True, shuffle=False)

    # Extract & aggregate
    embs, labs, _ = extract_embeddings(model, loader, device)
    prototypes, stats = build_prototypes(embs, labs)

    # Thresholds
    g_th  = cfg["openset"]["global_threshold"]
    alpha = cfg["openset"]["adaptive_alpha"]
    if cfg["openset"]["threshold_mode"] == "adaptive":
        thresholds = compute_adaptive_thresholds(stats, g_th, alpha)
    else:
        thresholds = {lab: g_th for lab in prototypes}

    gallery = {
        "prototypes":      prototypes,
        "thresholds":      thresholds,
        "per_class_stats": stats,
        "pid2label":       ds.pid2label,
        "label2pid":       {v: k for k, v in ds.pid2label.items()},
        "config":          cfg,
    }
    path = os.path.join(out_dir, "gallery.pth")
    torch.save(gallery, path)

    sims = [s["mean_sim"] for s in stats.values()]
    ths  = list(thresholds.values())
    print(f"[enroll] {len(prototypes)} identities enrolled -> {path}")
    print(f"         intra-class sim : {np.mean(sims):.4f} +/- {np.std(sims):.4f}")
    print(f"         threshold range : [{np.min(ths):.4f}, {np.max(ths):.4f}]  "
          f"mean={np.mean(ths):.4f}")


if __name__ == "__main__":
    main()
