#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
#  run_all.sh — Reproduce the full open-set re-id experiment pipeline
#  Seeds are fixed in every config (seed=42).  Run from the project root.
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "═══════════════════════════════════════════════"
echo "  Open-Set Re-ID — Full Reproduction Pipeline"
echo "═══════════════════════════════════════════════"

# ── 0. Install ───────────────────────────────────────────────────────
echo ""
echo "▶ [0/6] Installing dependencies …"
pip install -q -r requirements.txt

# ── 1. Download dataset ──────────────────────────────────────────────
echo ""
echo "▶ [1/6] Downloading Market-1501 …"
python -m data.download ./datasets/market1501

# ── 2. Run ablation (trains + enrols + evaluates all variants) ───────
echo ""
echo "▶ [2/6] Running ablation study …"
python ablation.py --results-dir results --benchmark

# ── 3. Full evaluation on best model ────────────────────────────────
echo ""
echo "▶ [3/6] Full evaluation on default config …"
python eval_full.py \
    --config configs/default.yaml \
    --checkpoint output/best.pth \
    --gallery output/gallery.pth \
    --results-dir results \
    --tag full_default \
    --benchmark

# ── 4. Low-variance (same-clothing) evaluation ──────────────────────
echo ""
echo "▶ [4/6] Low-variance subset evaluation …"
python eval_lowvar.py \
    --config configs/ablation_arcface_triplet.yaml \
    --checkpoint output/ablation_arcface_triplet/best.pth \
    --results-dir results \
    --tag lowvar

# ── 5. Generate all figures + report ─────────────────────────────────
echo ""
echo "▶ [5/6] Generating figures …"
python plot_results.py --results-dir results

# ── 6. Summary ──────────────────────────────────────────────────────
echo ""
echo "▶ [6/6] Done!"
echo ""
echo "Outputs:"
echo "  results/results.csv           — all metrics"
echo "  results/*.png                 — figures"
echo "  results/*_metrics.npz         — raw curve data"
echo "  REPORT.md                     — analysis report"
echo ""
echo "Submission checklist:"
echo "  ✓ configs/          — all YAML configs"
echo "  ✓ data/             — dataset + sampling code"
echo "  ✓ models/           — backbone + heads"
echo "  ✓ losses/           — softmax / arcface / cosface / triplet"
echo "  ✓ results/          — figures, CSV, NPZ"
echo "  ✓ output/           — checkpoints + gallery"
echo "  ✓ train / enroll / match / eval / ablation scripts"
echo "  ✓ requirements.txt  — reproducible deps"
echo "  ✓ README.md         — documentation"
echo "  ✓ REPORT.md         — analysis report"
echo ""
echo "To zip for submission:"
echo "  zip -r openset-reid.zip openset-reid/ -x '*/datasets/*' '*/__pycache__/*' '*.pyc'"
