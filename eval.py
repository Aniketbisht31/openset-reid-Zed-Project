"""Evaluate open-set person re-identification.

Metrics reported:
- FAR @ DIR = {50%, 70%, 80%, 90%, 95%}
- OSCR AUC  (Open-Set Classification Rate)
- Latency benchmark (single-image, with CUDA sync)

Also fits a score calibrator (temperature / Platt / isotonic) and
persists ``calibrator.pth``.
"""
from __future__ import annotations

import os
import time
import argparse
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm
import yaml

from models.backbone import build_backbone
from models.heads import ReIDModel
from data.market1501 import Market1501, get_transforms
from match import OpenSetMatcher


# ═══════════════════════════════════════════════════════════════════════
# Score calibrators
# ═══════════════════════════════════════════════════════════════════════

class TemperatureCalibrator:
    """Grid-search temperature scaling."""

    def __init__(self):
        self.temperature = 1.0

    def fit(self, scores, labels):
        best_t, best_nll = 1.0, float("inf")
        for t in np.linspace(0.01, 5.0, 500):
            p = np.clip(1.0 / (1.0 + np.exp(-scores / t)), 1e-7, 1 - 1e-7)
            nll = -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))
            if nll < best_nll:
                best_nll, best_t = nll, t
        self.temperature = best_t
        print(f"  Temperature: T = {self.temperature:.4f}")

    def calibrate(self, score):
        return float(1.0 / (1.0 + np.exp(-score / self.temperature)))


class PlattCalibrator:
    """Platt scaling (logistic regression on raw scores)."""

    def __init__(self):
        self.a = 1.0
        self.b = 0.0

    def fit(self, scores, labels):
        lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=10000)
        lr.fit(scores.reshape(-1, 1), labels)
        self.a, self.b = float(lr.coef_[0][0]), float(lr.intercept_[0])
        print(f"  Platt: a = {self.a:.4f},  b = {self.b:.4f}")

    def calibrate(self, score):
        return float(1.0 / (1.0 + np.exp(-(self.a * score + self.b))))


class IsotonicCalibrator:
    """Non-parametric isotonic regression."""

    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip")

    def fit(self, scores, labels):
        self.iso.fit(scores, labels)
        print("  Isotonic calibration fitted.")

    def calibrate(self, score):
        return float(self.iso.predict([score])[0])


_CALIBRATORS = {
    "temperature": TemperatureCalibrator,
    "platt":       PlattCalibrator,
    "isotonic":    IsotonicCalibrator,
}


# ═══════════════════════════════════════════════════════════════════════
# Embedding extraction
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_all(model, loader, device):
    model.eval()
    E, L, C, P = [], [], [], []
    for imgs, labs, cams, paths in tqdm(loader, desc="Extracting", leave=False):
        E.append(model(imgs.to(device, non_blocking=True)).cpu())
        L.extend(labs.tolist())
        C.extend(cams.tolist())
        P.extend(paths)
    return torch.cat(E), L, C, P


# ═══════════════════════════════════════════════════════════════════════
# Open-set metrics
# ═══════════════════════════════════════════════════════════════════════

def compute_openset_metrics(matcher, q_embs, q_labels, pid2label):
    """Sweep thresholds and compute FAR-vs-DIR + OSCR."""
    known_set = set(pid2label.values())

    scores, is_known, correct = [], [], []
    for i in range(len(q_labels)):
        r = matcher.match(q_embs[i])
        s = r["similarity"]
        lbl = q_labels[i]
        kn = lbl in known_set

        scores.append(s)
        is_known.append(kn)
        correct.append(kn and r["pred_label"] == lbl)

    scores   = np.asarray(scores)
    is_known = np.asarray(is_known)
    correct  = np.asarray(correct)

    n_kn = is_known.sum()
    n_un = (~is_known).sum()

    ths  = np.linspace(0.0, 1.0, 1000)
    fars = np.empty_like(ths)
    dirs = np.empty_like(ths)

    for i, th in enumerate(ths):
        acc = scores >= th
        fars[i] = (acc & ~is_known).sum() / max(n_un, 1)
        dirs[i] = (acc & correct).sum()   / max(n_kn, 1)

    res = {"n_known": int(n_kn), "n_unknown": int(n_un)}
    for target in (0.50, 0.70, 0.80, 0.90, 0.95):
        mask = dirs >= target
        res[f"FAR@DIR={target:.0%}"] = float(fars[mask].min()) if mask.any() else float("nan")

    # OSCR AUC
    order = np.argsort(-scores)
    cum_un  = np.cumsum(~is_known[order])
    cum_cor = np.cumsum(correct[order])
    oscr_far = cum_un  / max(n_un, 1)
    oscr_ccr = cum_cor / max(n_kn, 1)
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    res["OSCR_AUC"] = float(_trapz(oscr_ccr, oscr_far))

    return res


# ═══════════════════════════════════════════════════════════════════════
# Latency benchmark
# ═══════════════════════════════════════════════════════════════════════

def benchmark_latency(model, device, h=256, w=128, warmup=50, iters=200):
    model.eval()
    x = torch.randn(1, 3, h, w, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    t = np.array(times)
    return {
        "mean_ms": float(t.mean()),
        "std_ms":  float(t.std()),
        "p50_ms":  float(np.percentile(t, 50)),
        "p95_ms":  float(np.percentile(t, 95)),
        "p99_ms":  float(np.percentile(t, 99)),
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser("eval.py — open-set re-id evaluation")
    ap.add_argument("--config",     default="configs/default.yaml")
    ap.add_argument("--checkpoint", default="output/best.pth")
    ap.add_argument("--gallery",    default="output/gallery.pth")
    ap.add_argument("--benchmark",  action="store_true",
                    help="Run single-image latency benchmark")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = cfg["output"]["dir"]

    # ── Load model ───────────────────────────────────────────────────
    backbone = build_backbone(cfg["model"]["backbone"], pretrained=False)
    model = ReIDModel(backbone, cfg["model"]["embed_dim"]).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    gallery = torch.load(args.gallery, map_location=device, weights_only=False)
    matcher = OpenSetMatcher(model, gallery, device)

    splits = torch.load(os.path.join(out_dir, "splits.pth"), weights_only=False)
    known_pids   = set(splits["known_pids"])
    unknown_pids = set(splits["unknown_pids"])

    # ── Build eval sets ──────────────────────────────────────────────
    transform = get_transforms(cfg["data"]["height"], cfg["data"]["width"],
                               is_train=False)
    ds_known = Market1501(cfg["data"]["root"], split="train",
                          known_pids=known_pids, transform=transform,
                          relabel=True)
    ds_unknown = Market1501(cfg["data"]["root"], split="train",
                            known_pids=unknown_pids, transform=transform,
                            relabel=True)

    ldr_kn = DataLoader(ds_known, batch_size=128,
                        num_workers=cfg["data"]["num_workers"],
                        pin_memory=True)
    ldr_un = DataLoader(ds_unknown, batch_size=128,
                        num_workers=cfg["data"]["num_workers"],
                        pin_memory=True)

    print("[eval] Extracting known embeddings …")
    emb_kn, lab_kn, _, _ = extract_all(model, ldr_kn, device)
    print("[eval] Extracting unknown embeddings …")
    emb_un, lab_un, _, _ = extract_all(model, ldr_un, device)

    # Unknown labels → −1 for evaluation
    all_embs  = torch.cat([emb_kn, emb_un])
    all_labs  = lab_kn + ([-1] * len(lab_un))

    print(f"[eval] {len(lab_kn)} known + {len(lab_un)} unknown queries\n")

    # ── Score calibration ────────────────────────────────────────────
    method = cfg["calibration"]["method"]
    print(f"[calib] Fitting {method} calibrator …")

    n_cal = min(3000, len(all_embs))
    idx   = np.random.default_rng(42).choice(len(all_embs), n_cal, replace=False)

    cal_scores, cal_y = [], []
    pid2label = ds_known.pid2label
    for i in idx:
        r = matcher.match(all_embs[i])
        cal_scores.append(r["similarity"])
        lab = all_labs[i]
        cal_y.append(1 if (lab >= 0 and r["pred_label"] == lab) else 0)

    calibrator = _CALIBRATORS[method]()
    calibrator.fit(np.array(cal_scores), np.array(cal_y))

    cal_path = os.path.join(out_dir, "calibrator.pth")
    torch.save(calibrator, cal_path)
    matcher.calibrator = calibrator

    # ── Open-set metrics ─────────────────────────────────────────────
    print("\n[eval] Computing open-set metrics …")
    metrics = compute_openset_metrics(matcher, all_embs, all_labs, pid2label)

    print(f"\n{'=' * 56}")
    print(f"  OPEN-SET RE-ID RESULTS")
    print(f"{'=' * 56}")
    print(f"  Known queries  : {metrics['n_known']}")
    print(f"  Unknown queries: {metrics['n_unknown']}")
    for k in ("FAR@DIR=50%", "FAR@DIR=70%", "FAR@DIR=80%",
              "FAR@DIR=90%", "FAR@DIR=95%"):
        print(f"  {k:18s}: {metrics.get(k, float('nan')):.4f}")
    print(f"  {'OSCR AUC':18s}: {metrics['OSCR_AUC']:.4f}")

    # ── Latency ──────────────────────────────────────────────────────
    if args.benchmark:
        print(f"\n{'=' * 56}")
        print("  LATENCY BENCHMARK")
        print(f"{'=' * 56}")
        lat = benchmark_latency(model, device,
                                cfg["data"]["height"], cfg["data"]["width"])
        for k, v in lat.items():
            print(f"  {k:10s}: {v:7.2f} ms")
        ok = lat["p95_ms"] < 10.0
        print(f"\n  Real-time (<10 ms @ P95): {'[YES]' if ok else '[NO]'}")

    # ── Persist ──────────────────────────────────────────────────────
    torch.save(metrics, os.path.join(out_dir, "metrics.pth"))
    print(f"\n[eval] Metrics -> {os.path.join(out_dir, 'metrics.pth')}")
    print(f"[eval] Calibrator -> {cal_path}")


if __name__ == "__main__":
    main()
