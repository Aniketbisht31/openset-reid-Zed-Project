"""Auto-generate all evaluation figures and the ablation table.

Reads ``results/<tag>_metrics.npz`` files produced by ``eval_full.py`` /
``eval_lowvar.py`` and ``results/results.csv``, then writes:

    results/
    ├── roc_curve.png
    ├── far_tar.png
    ├── dir_far.png
    ├── oscr_curve.png
    ├── cmc_curve.png
    ├── score_distribution.png
    ├── reliability_diagram.png
    ├── ablation_bar.png
    └── ablation_table.png
"""
from __future__ import annotations

import os
import csv
import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
})

COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#7f7f7f"]


def _load_npz(results_dir: str):
    """Load all *_metrics.npz files into a tag→dict mapping."""
    data = {}
    for p in sorted(Path(results_dir).glob("*_metrics.npz")):
        tag = p.stem.replace("_metrics", "")
        data[tag] = dict(np.load(p, allow_pickle=True))
    return data


def _load_csv(results_dir: str):
    path = os.path.join(results_dir, "results.csv")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


# ═════════════════════════════════════════════════════════════════════
# Individual figure generators
# ═════════════════════════════════════════════════════════════════════

def plot_roc(data, out):
    """ROC curves (full + lowvar overlay)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, (tag, d) in enumerate(data.items()):
        if "roc_fpr" not in d:
            continue
        auroc = float(d.get("auroc", 0))
        ax.plot(d["roc_fpr"], d["roc_tpr"],
                color=COLOURS[i % len(COLOURS)],
                label=f"{tag} (AUROC={auroc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, lw=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curve", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(out, "roc_curve.png"))
    plt.close(fig)


def plot_far_tar(data, out):
    """FAR vs TAR curves."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, (tag, d) in enumerate(data.items()):
        if "far_v" not in d:
            continue
        ax.plot(d["far_v"], d["tar_v"],
                color=COLOURS[i % len(COLOURS)], label=tag)
    ax.set(xlabel="False Accept Rate (FAR)",
           ylabel="True Accept Rate (TAR)",
           title="FAR vs TAR")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.savefig(os.path.join(out, "far_tar.png"))
    plt.close(fig)


def plot_dir_far(data, out):
    """DIR vs FAR (open-set)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, (tag, d) in enumerate(data.items()):
        if "os_dirs" not in d:
            continue
        ax.plot(d["os_fars"], d["os_dirs"],
                color=COLOURS[i % len(COLOURS)], label=tag)
    ax.set(xlabel="False Accept Rate (FAR)",
           ylabel="Detection & Identification Rate (DIR)",
           title="DIR vs FAR (Open-Set)")
    ax.set_xscale("log")
    ax.set_xlim(left=1e-4)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.savefig(os.path.join(out, "dir_far.png"))
    plt.close(fig)


def plot_oscr(data, out):
    """OSCR curves."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, (tag, d) in enumerate(data.items()):
        if "oscr_far" not in d:
            continue
        auc = float(d.get("oscr_auc", 0))
        ax.plot(d["oscr_far"], d["oscr_ccr"],
                color=COLOURS[i % len(COLOURS)],
                label=f"{tag} (AUC={auc:.3f})")
    ax.set(xlabel="False Positive Rate",
           ylabel="Correct Classification Rate",
           title="OSCR Curve")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(out, "oscr_curve.png"))
    plt.close(fig)


def plot_cmc(data, out):
    """CMC (rank-k) curves."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, (tag, d) in enumerate(data.items()):
        if "cmc" not in d:
            continue
        cmc = d["cmc"]
        ranks = np.arange(1, len(cmc) + 1)
        ax.plot(ranks, cmc, color=COLOURS[i % len(COLOURS)],
                label=f"{tag} (R1={cmc[0]:.3f})")
    ax.set(xlabel="Rank", ylabel="Matching Rate",
           title="CMC Curve", xlim=(1, min(50, len(cmc))), ylim=(0, 1))
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(out, "cmc_curve.png"))
    plt.close(fig)


def plot_score_dist(data, out):
    """Genuine vs impostor score distributions."""
    fig, axes = plt.subplots(1, min(len(data), 3), figsize=(5 * min(len(data), 3), 4),
                             squeeze=False)
    for i, (tag, d) in enumerate(data.items()):
        if i >= 3 or "genuine_scores" not in d:
            break
        ax = axes[0, i]
        ax.hist(d["genuine_scores"], bins=80, alpha=0.6, density=True,
                color="#2ca02c", label="Genuine")
        ax.hist(d["impostor_scores"], bins=80, alpha=0.6, density=True,
                color="#d62728", label="Impostor")
        ax.set(title=tag, xlabel="Cosine Similarity", ylabel="Density")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Score Distributions", y=1.02)
    fig.savefig(os.path.join(out, "score_distribution.png"))
    plt.close(fig)


def plot_reliability(data, out):
    """Reliability (calibration) diagrams."""
    tags_with_bins = [(t, d) for t, d in data.items() if "bin_accs" in d]
    if not tags_with_bins:
        return
    n = min(len(tags_with_bins), 3)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)

    for i, (tag, d) in enumerate(tags_with_bins[:n]):
        ax = axes[0, i]
        edges = d["bin_edges"]
        accs = d["bin_accs"]
        confs = d["bin_confs"]
        counts = d["bin_counts"]
        ece = float(d.get("ece", 0))

        width = edges[1] - edges[0]
        centres = (edges[:-1] + edges[1:]) / 2

        ax.bar(centres, accs, width=width * 0.85, alpha=0.7,
               color="#1f77b4", edgecolor="white", label="Accuracy")
        ax.bar(centres, confs, width=width * 0.45, alpha=0.5,
               color="#ff7f0e", edgecolor="white", label="Avg Confidence")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1, label="Perfect")
        ax.set(title=f"{tag}  (ECE={ece:.3f})",
               xlabel="Confidence", ylabel="Accuracy",
               xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.2)

    fig.suptitle("Reliability Diagrams", y=1.02)
    fig.savefig(os.path.join(out, "reliability_diagram.png"))
    plt.close(fig)


def plot_ablation_bar(rows, out):
    """Grouped bar chart comparing ablation variants."""
    if not rows:
        return

    tags = [r.get("tag", "?") for r in rows]
    metrics = ["rank1", "mAP", "AUROC", "OSCR_AUC"]
    present = [m for m in metrics if m in rows[0]]
    if not present:
        return

    x = np.arange(len(tags))
    w = 0.8 / len(present)

    fig, ax = plt.subplots(figsize=(max(8, len(tags) * 1.5), 5))
    for j, m in enumerate(present):
        vals = []
        for r in rows:
            try:
                vals.append(float(r.get(m, 0)))
            except (ValueError, TypeError):
                vals.append(0.0)
        ax.bar(x + j * w, vals, w, label=m, alpha=0.85)

    ax.set_xticks(x + w * (len(present) - 1) / 2)
    ax.set_xticklabels(tags, rotation=30, ha="right", fontsize=9)
    ax.set(ylabel="Score", title="Ablation Comparison", ylim=(0, 1.05))
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(os.path.join(out, "ablation_bar.png"))
    plt.close(fig)


def render_ablation_table(rows, out):
    """Render ablation results as a PNG table."""
    if not rows:
        return
    cols = [k for k in rows[0].keys() if k != "tag"]
    cell_text = [[r.get("tag", "")] + [r.get(c, "") for c in cols] for r in rows]
    col_labels = ["Variant"] + cols

    fig, ax = plt.subplots(figsize=(max(10, len(cols) * 1.2), 0.5 + 0.4 * len(rows)))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=col_labels,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#D9E2F3")
    fig.savefig(os.path.join(out, "ablation_table.png"))
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser("plot_results.py")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    data = _load_npz(args.results_dir)
    rows = _load_csv(args.results_dir)

    if not data and not rows:
        print("[plot] No results found — run eval_full.py or ablation.py first.")
        return

    print(f"[plot] Found {len(data)} .npz files, {len(rows)} CSV rows")

    plot_roc(data, args.results_dir)
    plot_far_tar(data, args.results_dir)
    plot_dir_far(data, args.results_dir)
    plot_oscr(data, args.results_dir)
    plot_cmc(data, args.results_dir)
    plot_score_dist(data, args.results_dir)
    plot_reliability(data, args.results_dir)
    plot_ablation_bar(rows, args.results_dir)
    render_ablation_table(rows, args.results_dir)

    figs = list(Path(args.results_dir).glob("*.png"))
    print(f"[plot] Generated {len(figs)} figures in {args.results_dir}/")
    for f in sorted(figs):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
