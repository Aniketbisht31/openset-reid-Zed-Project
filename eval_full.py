"""Comprehensive open-set re-id evaluation on the full Market-1501 protocol.

Computes and persists:
  1. Open-set   — DIR@FAR, FAR@DIR, OSCR AUC
  2. Verification — ROC, AUROC, FAR@TAR={90,95,99}%
  3. Closed-set — Rank-1 / -5 / -10, mAP (cross-camera within known IDs)
  4. Calibration — ECE, reliability-diagram data
  5. Latency    — single-image GPU latency (optional)

Outputs
-------
``results/<tag>_metrics.npz``  — all curve data (numpy).
``results/results.csv``        — one-row summary appended.
"""
from __future__ import annotations

import os
import csv
import time
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from models.backbone import build_backbone
from models.heads import ReIDModel
from data.market1501 import Market1501, get_transforms
from match import OpenSetMatcher
from eval import _CALIBRATORS
from eval_metrics import (
    generate_pairs,
    compute_roc,
    far_tar_curve,
    far_at_tar,
    compute_cmc_map,
    openset_eval,
    expected_calibration_error,
)


# ── helpers ──────────────────────────────────────────────────────────

@torch.no_grad()
def _extract(model, loader, device):
    model.eval()
    E, L, C = [], [], []
    for imgs, labs, cams, _ in tqdm(loader, desc="  embed", leave=False):
        E.append(model(imgs.to(device, non_blocking=True)).cpu())
        L.extend(labs.tolist())
        C.extend(cams.tolist())
    return torch.cat(E).numpy(), np.array(L), np.array(C)


def _latency(model, device, h, w, warmup=50, iters=200):
    model.eval()
    x = torch.randn(1, 3, h, w, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    t = np.array(ts)
    return {"mean_ms": t.mean(), "p50_ms": np.median(t),
            "p95_ms": np.percentile(t, 95), "p99_ms": np.percentile(t, 99)}


def _append_csv(path, row: dict):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


# ── main ─────────────────────────────────────────────────────────────

def run_eval(
    config_path: str,
    checkpoint_path: str,
    gallery_path: str,
    results_dir: str = "results",
    tag: str = "default",
    do_benchmark: bool = False,
    calibration_method: str | None = None,
    threshold_mode: str | None = None,
):
    """Run full evaluation and return summary dict."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Allow CLI overrides
    if calibration_method:
        cfg["calibration"]["method"] = calibration_method
    if threshold_mode:
        cfg["openset"]["threshold_mode"] = threshold_mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(results_dir, exist_ok=True)
    out_dir = cfg["output"]["dir"]

    # ── model ────────────────────────────────────────────────────────
    backbone = build_backbone(cfg["model"]["backbone"], pretrained=False)
    model = ReIDModel(backbone, cfg["model"]["embed_dim"]).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ── gallery + splits ─────────────────────────────────────────────
    gallery = torch.load(gallery_path, map_location=device, weights_only=False)
    splits = torch.load(os.path.join(out_dir, "splits.pth"), weights_only=False)
    known_pids = set(splits["known_pids"])
    unknown_pids = set(splits["unknown_pids"])
    pid2label = gallery["pid2label"]

    # Apply threshold_mode override to gallery if requested
    if threshold_mode:
        g_th = cfg["openset"]["global_threshold"]
        alpha = cfg["openset"]["adaptive_alpha"]
        if threshold_mode == "global":
            gallery["thresholds"] = {lab: g_th for lab in gallery["prototypes"]}
        elif threshold_mode == "adaptive" and "per_class_stats" in gallery:
            from enroll import compute_adaptive_thresholds
            gallery["thresholds"] = compute_adaptive_thresholds(gallery["per_class_stats"], g_th, alpha)

    tf = get_transforms(cfg["data"]["height"], cfg["data"]["width"], False)
    nw = cfg["data"]["num_workers"]

    # ── extract embeddings ───────────────────────────────────────────
    print(f"[{tag}] Extracting known …")
    ds_kn = Market1501(cfg["data"]["root"], "train", known_pids=known_pids,
                       transform=tf, relabel=True)
    emb_kn, lab_kn, cam_kn = _extract(
        model, DataLoader(ds_kn, 128, num_workers=nw, pin_memory=True), device)

    print(f"[{tag}] Extracting unknown …")
    ds_un = Market1501(cfg["data"]["root"], "train", known_pids=unknown_pids,
                       transform=tf, relabel=True)
    emb_un, lab_un, cam_un = _extract(
        model, DataLoader(ds_un, 128, num_workers=nw, pin_memory=True), device)

    # ── 1. open-set ──────────────────────────────────────────────────
    matcher = OpenSetMatcher(model, gallery, device)

    all_emb = np.vstack([emb_kn, emb_un])
    all_lab = np.concatenate([lab_kn, -np.ones(len(lab_un), dtype=int)])
    is_known = np.concatenate([np.ones(len(lab_kn), dtype=bool),
                               np.zeros(len(lab_un), dtype=bool)])

    print(f"[{tag}] Matching {len(all_emb)} queries …")
    scores_arr, preds_arr = [], []
    for i in tqdm(range(len(all_emb)), desc="  match", leave=False):
        r = matcher.match(torch.from_numpy(all_emb[i]))
        scores_arr.append(r["similarity"])
        preds_arr.append(r["pred_label"])
    scores = np.array(scores_arr)
    preds = np.array(preds_arr)

    os_res = openset_eval(scores, preds, all_lab, is_known)

    # ── 2. verification ──────────────────────────────────────────────
    print(f"[{tag}] Generating verification pairs …")
    gen_s, imp_s = generate_pairs(emb_kn, lab_kn, n_pairs=20_000)
    # extra cross-set impostors
    n_x = min(5000, len(emb_un))
    rng = np.random.RandomState(42)
    xi = [float(emb_kn[rng.randint(len(emb_kn))] @
                emb_un[rng.randint(len(emb_un))]) for _ in range(n_x)]
    imp_all = np.concatenate([imp_s, np.array(xi)])

    fpr, tpr, roc_th, auroc = compute_roc(gen_s, imp_all)
    far_v, tar_v, ft_th = far_tar_curve(gen_s, imp_all)
    far_tar_dict = far_at_tar(gen_s, imp_all)

    # ── 3. closed-set id (cross-camera within known) ─────────────────
    print(f"[{tag}] CMC + mAP …")
    sim_kk = emb_kn @ emb_kn.T
    cmc, mAP = compute_cmc_map(sim_kk, lab_kn, lab_kn, cam_kn, cam_kn)

    # ── 4. calibration ───────────────────────────────────────────────
    print(f"[{tag}] Calibrating ({cfg['calibration']['method']}) …")
    cal_cls = _CALIBRATORS[cfg["calibration"]["method"]]()
    n_cal = min(4000, len(scores))
    ci = rng.choice(len(scores), n_cal, replace=False)
    cal_s = scores[ci]
    cal_y = np.array([(all_lab[j] >= 0 and preds[j] == all_lab[j])
                      for j in ci], dtype=float)
    cal_cls.fit(cal_s, cal_y)

    # confidence for all queries
    confs = np.array([cal_cls.calibrate(s) for s in scores])
    correct_mask = is_known & (preds == all_lab)
    ece, bins = expected_calibration_error(confs, correct_mask.astype(float))
    print(f"  ECE = {ece:.4f}")

    # save calibrator
    torch.save(cal_cls, os.path.join(out_dir, "calibrator.pth"))

    # ── 5. latency ───────────────────────────────────────────────────
    lat = {}
    if do_benchmark:
        print(f"[{tag}] Latency benchmark …")
        lat = _latency(model, device, cfg["data"]["height"], cfg["data"]["width"])

    # ── persist ──────────────────────────────────────────────────────
    npz_path = os.path.join(results_dir, f"{tag}_metrics.npz")
    np.savez_compressed(
        npz_path,
        # ROC
        roc_fpr=fpr, roc_tpr=tpr, roc_th=roc_th, auroc=np.float64(auroc),
        # FAR-TAR
        far_v=far_v, tar_v=tar_v, ft_th=ft_th,
        # open-set
        os_dirs=os_res["dirs"], os_fars=os_res["fars"], os_th=os_res["thresholds"],
        oscr_far=os_res["oscr_far"], oscr_ccr=os_res["oscr_ccr"],
        oscr_auc=np.float64(os_res["OSCR_AUC"]),
        # CMC
        cmc=cmc, mAP=np.float64(mAP),
        # scores
        genuine_scores=gen_s, impostor_scores=imp_all,
        # calibration
        ece=np.float64(ece),
        bin_edges=bins["edges"], bin_accs=bins["accs"],
        bin_confs=bins["confs"], bin_counts=bins["counts"],
    )

    summary = {
        "tag": tag,
        "rank1": f"{cmc[0]:.4f}",
        "rank5": f"{cmc[4]:.4f}" if len(cmc) > 4 else "",
        "rank10": f"{cmc[9]:.4f}" if len(cmc) > 9 else "",
        "mAP": f"{mAP:.4f}",
        "AUROC": f"{auroc:.4f}",
        "OSCR_AUC": f"{os_res['OSCR_AUC']:.4f}",
        "ECE": f"{ece:.4f}",
    }
    summary.update({k: f"{v:.4f}" for k, v in far_tar_dict.items()})
    for k in sorted(k for k in os_res if k.startswith("DIR@") or k.startswith("FAR@DIR")):
        v = os_res[k]
        summary[k] = f"{v:.4f}" if not np.isnan(v) else "nan"
    if lat:
        summary["latency_p95_ms"] = f"{lat['p95_ms']:.2f}"

    csv_path = os.path.join(results_dir, "results.csv")
    _append_csv(csv_path, summary)

    print(f"[{tag}] Saved -> {npz_path}")
    print(f"         Rank-1={cmc[0]:.4f}  mAP={mAP:.4f}  AUROC={auroc:.4f}  "
          f"OSCR={os_res['OSCR_AUC']:.4f}  ECE={ece:.4f}")
    return summary


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser("eval_full.py")
    ap.add_argument("--config",     default="configs/default.yaml")
    ap.add_argument("--checkpoint", default="output/best.pth")
    ap.add_argument("--gallery",    default="output/gallery.pth")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--tag",        default="default")
    ap.add_argument("--benchmark",  action="store_true")
    ap.add_argument("--calibration", default=None,
                    choices=["temperature", "platt", "isotonic"])
    ap.add_argument("--threshold",  default=None,
                    choices=["global", "adaptive"])
    args = ap.parse_args()

    run_eval(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        gallery_path=args.gallery,
        results_dir=args.results_dir,
        tag=args.tag,
        do_benchmark=args.benchmark,
        calibration_method=args.calibration,
        threshold_mode=args.threshold,
    )


if __name__ == "__main__":
    main()
