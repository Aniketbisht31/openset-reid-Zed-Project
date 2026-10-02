"""Ablation study driver.

Trains each ablation variant, enrols gallery, and evaluates.  The six
conditions reported are:

  1. Softmax  — normalised CE baseline, global threshold, no calibration
  2. +ArcFace — ArcFace margin, global threshold
  3. +Triplet — ArcFace + batch-hard triplet (PK hard-mining)
  4. +Adaptive — same model, adaptive per-class thresholds
  5. +Calibrated — same model, + temperature calibration
  6. Low-var  — same best model evaluated on same-clothing subset

Rows 4 – 5 reuse the checkpoint from row 3 and only change the
evaluation settings, so they do **not** retrain.

Results are collected into ``results/results.csv``.
"""
from __future__ import annotations

import os
import sys
import copy
import subprocess
import argparse

import yaml


# ── ablation definitions ─────────────────────────────────────────────

ABLATIONS = [
    {
        "tag": "1_softmax_baseline",
        "config": "configs/ablation_softmax.yaml",
        "needs_train": True,
        "eval_overrides": {"threshold": "global"},
    },
    {
        "tag": "2_plus_arcface",
        "config": "configs/ablation_arcface.yaml",
        "needs_train": True,
        "eval_overrides": {"threshold": "global"},
    },
    {
        "tag": "3_plus_triplet",
        "config": "configs/ablation_arcface_triplet_nohard.yaml",
        "needs_train": True,
        "eval_overrides": {"threshold": "global"},
    },
    {
        "tag": "4_plus_hard_mining",
        "config": "configs/ablation_arcface_triplet.yaml",
        "needs_train": True,
        "eval_overrides": {"threshold": "global"},
    },
    {
        "tag": "5_plus_adaptive_thresholds",
        "config": "configs/ablation_arcface_triplet.yaml",
        "needs_train": False,
        "parent": "4_plus_hard_mining",
        "eval_overrides": {"threshold": "adaptive"},
    },
    {
        "tag": "6_plus_calibration",
        "config": "configs/ablation_arcface_triplet.yaml",
        "needs_train": False,
        "parent": "4_plus_hard_mining",
        "eval_overrides": {"threshold": "adaptive", "calibration": "temperature"},
    },
]

PY = sys.executable


def _run(cmd: list[str], cwd: str):
    """Run a subprocess and stream output."""
    print(f"\n{'-'*60}")
    print(f"$ {' '.join(cmd)}")
    print(f"{'-'*60}")
    proc = subprocess.run(cmd, cwd=cwd)
    if proc.returncode != 0:
        print(f"[WARN] Command exited with code {proc.returncode}")
    return proc.returncode


def main():
    ap = argparse.ArgumentParser("ablation.py — run all ablation experiments")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--skip-train", action="store_true",
                    help="Skip training, only evaluate existing checkpoints")
    ap.add_argument("--benchmark", action="store_true")
    args = ap.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(args.results_dir, exist_ok=True)

    # Clear old CSV
    csv_path = os.path.join(args.results_dir, "results.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)

    for ab in ABLATIONS:
        tag = ab["tag"]
        config = ab["config"]

        with open(os.path.join(project_root, config)) as f:
            cfg = yaml.safe_load(f)
        out_dir = cfg["output"]["dir"]
        ckpt = os.path.join(out_dir, "best.pth")
        gallery_path = os.path.join(out_dir, "gallery.pth")

        # ── train ────────────────────────────────────────────────────
        if ab["needs_train"] and not args.skip_train:
            print(f"\n{'='*60}")
            print(f"  TRAINING: {tag}")
            print(f"{'='*60}")
            _run([PY, "train.py", "--config", config], project_root)

            # Enrol
            _run([PY, "enroll.py", "--config", config,
                  "--checkpoint", ckpt], project_root)

        elif not ab["needs_train"]:
            # Reuse parent checkpoint
            parent = ab.get("parent", "")
            parent_cfg_path = None
            for other in ABLATIONS:
                if other["tag"] == parent:
                    parent_cfg_path = other["config"]
                    break
            if parent_cfg_path:
                with open(os.path.join(project_root, parent_cfg_path)) as f:
                    parent_cfg = yaml.safe_load(f)
                ckpt = os.path.join(parent_cfg["output"]["dir"], "best.pth")
                gallery_path = os.path.join(parent_cfg["output"]["dir"], "gallery.pth")
                out_dir = parent_cfg["output"]["dir"]
                # Point config output to parent for splits.pth etc.
                config = parent_cfg_path

        # ── evaluate ─────────────────────────────────────────────────
        if not os.path.exists(ckpt):
            print(f"[SKIP] {tag}: checkpoint not found at {ckpt}")
            continue

        print(f"\n{'='*60}")
        print(f"  EVALUATING: {tag}")
        print(f"{'='*60}")

        eval_cmd = [
            PY, "eval_full.py",
            "--config", config,
            "--checkpoint", ckpt,
            "--gallery", gallery_path,
            "--results-dir", args.results_dir,
            "--tag", tag,
        ]
        overrides = ab.get("eval_overrides", {})
        if "calibration" in overrides:
            eval_cmd += ["--calibration", overrides["calibration"]]
        if "threshold" in overrides:
            eval_cmd += ["--threshold", overrides["threshold"]]
        if args.benchmark:
            eval_cmd.append("--benchmark")

        _run(eval_cmd, project_root)

    # ── low-variance eval on best model ──────────────────────────────
    best_cfg = "configs/ablation_arcface_triplet.yaml"
    with open(os.path.join(project_root, best_cfg)) as f:
        best = yaml.safe_load(f)
    best_ckpt = os.path.join(best["output"]["dir"], "best.pth")

    if os.path.exists(best_ckpt):
        print(f"\n{'='*60}")
        print(f"  LOW-VARIANCE EVAL")
        print(f"{'='*60}")
        _run([
            PY, "eval_lowvar.py",
            "--config", best_cfg,
            "--checkpoint", best_ckpt,
            "--results-dir", args.results_dir,
            "--tag", "lowvar",
        ], project_root)

    # ── generate plots ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  GENERATING PLOTS + REPORT")
    print(f"{'='*60}")
    _run([PY, "plot_results.py", "--results-dir", args.results_dir],
         project_root)

    print(f"\n{'='*60}")
    print("  ALL ABLATIONS COMPLETE")
    print(f"{'='*60}")
    print(f"  CSV     : {csv_path}")
    print(f"  Figures : {args.results_dir}/")
    print(f"  Report  : REPORT.md")


if __name__ == "__main__":
    main()
