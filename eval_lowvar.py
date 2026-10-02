"""Evaluate on the curated *same-clothing low-variance* subset.

This script:
1. Clusters training images by upper/lower body colour histograms.
2. From each cluster builds **same-clothing impostor pairs**
   (different person, similar outfit).
3. Extracts embeddings and computes verification / open-set metrics
   *only* on these hard pairs.
4. Saves results alongside the full-dataset results for comparison.
"""
from __future__ import annotations

import os
import argparse

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import yaml

from models.backbone import build_backbone
from models.heads import ReIDModel
from data.market1501 import get_transforms
from data.low_variance import build_low_variance_subset
from eval_metrics import (
    compute_roc,
    far_tar_curve,
    far_at_tar,
    expected_calibration_error,
)
from eval_full import _append_csv


@torch.no_grad()
def _embed_paths(model, paths, transform, device):
    """Extract embeddings for a list of image paths."""
    model.eval()
    embs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        t = transform(img).unsqueeze(0).to(device)
        embs.append(model(t).cpu().squeeze(0).numpy())
    return np.stack(embs)


def main():
    ap = argparse.ArgumentParser("eval_lowvar.py — same-clothing evaluation")
    ap.add_argument("--config",      default="configs/default.yaml")
    ap.add_argument("--checkpoint",  default="output/best.pth")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--tag",         default="lowvar")
    ap.add_argument("--n-clusters",  type=int, default=10)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.results_dir, exist_ok=True)
    out_dir = cfg["output"]["dir"]

    # ── model ────────────────────────────────────────────────────────
    backbone = build_backbone(cfg["model"]["backbone"], pretrained=False)
    model = ReIDModel(backbone, cfg["model"]["embed_dim"]).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    transform = get_transforms(cfg["data"]["height"], cfg["data"]["width"],
                               is_train=False)

    # ── splits ───────────────────────────────────────────────────────
    splits = torch.load(os.path.join(out_dir, "splits.pth"), weights_only=False)
    known_pids = set(splits["known_pids"])

    # ── build low-variance subset ────────────────────────────────────
    print("[lowvar] Building same-clothing clusters …")
    clusters, impostor_pairs = build_low_variance_subset(
        cfg["data"]["root"],
        known_pids=known_pids,
        n_clusters=args.n_clusters,
        seed=cfg["train"]["seed"],
    )

    if not impostor_pairs:
        print("[lowvar] No impostor pairs found — aborting.")
        return

    # ── extract embeddings for impostor pairs ────────────────────────
    print(f"[lowvar] Extracting embeddings for {len(impostor_pairs)} pairs …")
    pair_paths_a, pair_pids_a = zip(*[p[0] for p in impostor_pairs])
    pair_paths_b, pair_pids_b = zip(*[p[1] for p in impostor_pairs])

    all_paths = list(set(pair_paths_a) | set(pair_paths_b))
    print(f"  Unique images: {len(all_paths)}")

    embs = _embed_paths(model, all_paths, transform, device)
    path2emb = {p: embs[i] for i, p in enumerate(all_paths)}

    # ── compute impostor similarity scores ───────────────────────────
    impostor_scores = np.array([
        float(path2emb[pa] @ path2emb[pb])
        for (pa, _), (pb, _) in impostor_pairs
    ])

    # ── genuine pairs from same-cluster same-PID ─────────────────────
    from collections import defaultdict

    genuine_scores_list = []
    for cid, members in clusters.items():
        pid_groups: dict[int, list] = defaultdict(list)
        for path, pid, _ in members:
            if path in path2emb:
                pid_groups[pid].append(path)
        for pid, paths in pid_groups.items():
            if len(paths) >= 2:
                for i in range(min(len(paths) - 1, 3)):
                    s = float(path2emb[paths[i]] @ path2emb[paths[i + 1]])
                    genuine_scores_list.append(s)

    genuine_scores = np.array(genuine_scores_list) if genuine_scores_list else np.array([1.0])

    print(f"  Genuine pairs : {len(genuine_scores)}")
    print(f"  Impostor pairs: {len(impostor_scores)}")
    print(f"  Genuine  mean sim: {genuine_scores.mean():.4f} ± {genuine_scores.std():.4f}")
    print(f"  Impostor mean sim: {impostor_scores.mean():.4f} ± {impostor_scores.std():.4f}")

    # ── metrics ──────────────────────────────────────────────────────
    fpr, tpr, roc_th, auroc = compute_roc(genuine_scores, impostor_scores)
    far_v, tar_v, ft_th = far_tar_curve(genuine_scores, impostor_scores)
    far_tar_dict = far_at_tar(genuine_scores, impostor_scores)

    # ── persist ──────────────────────────────────────────────────────
    npz_path = os.path.join(args.results_dir, f"{args.tag}_metrics.npz")
    np.savez_compressed(
        npz_path,
        roc_fpr=fpr, roc_tpr=tpr, auroc=np.float64(auroc),
        far_v=far_v, tar_v=tar_v, ft_th=ft_th,
        genuine_scores=genuine_scores, impostor_scores=impostor_scores,
        n_genuine=len(genuine_scores), n_impostor=len(impostor_scores),
    )

    summary = {
        "tag": args.tag,
        "AUROC": f"{auroc:.4f}",
        "n_genuine": str(len(genuine_scores)),
        "n_impostor": str(len(impostor_scores)),
        "impostor_mean_sim": f"{impostor_scores.mean():.4f}",
    }
    summary.update({k: f"{v:.4f}" for k, v in far_tar_dict.items()})

    csv_path = os.path.join(args.results_dir, "results.csv")
    _append_csv(csv_path, summary)

    print(f"\n{'=' * 50}")
    print("  LOW-VARIANCE (SAME-CLOTHING) RESULTS")
    print(f"{'=' * 50}")
    print(f"  AUROC : {auroc:.4f}")
    for k, v in far_tar_dict.items():
        print(f"  {k:18s}: {v:.4f}")
    print(f"  Saved -> {npz_path}")


if __name__ == "__main__":
    main()
