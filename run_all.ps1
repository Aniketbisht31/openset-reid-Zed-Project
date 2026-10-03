# ─────────────────────────────────────────────────────────────────────
#  run_all.ps1 — Reproduce the full open-set re-id experiment pipeline
#  Windows PowerShell version. Seeds fixed (42). Run from project root.
# ─────────────────────────────────────────────────────────────────────
$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ROOT

Write-Host "═══════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Open-Set Re-ID — Full Reproduction Pipeline"  -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════" -ForegroundColor Cyan

# ── 0. Install ───────────────────────────────────────────────────────
Write-Host "`n▶ [0/6] Installing dependencies …" -ForegroundColor Yellow
pip install -r requirements.txt

# ── 1. Download ──────────────────────────────────────────────────────
Write-Host "`n▶ [1/6] Downloading Market-1501 …" -ForegroundColor Yellow
python -m data.download ./datasets/market1501

# ── 2. Ablation ─────────────────────────────────────────────────────
Write-Host "`n▶ [2/6] Running ablation study …" -ForegroundColor Yellow
python ablation.py --results-dir results --benchmark

# ── 3. Full eval on default config ──────────────────────────────────
Write-Host "`n▶ [3/6] Full evaluation (default config) …" -ForegroundColor Yellow
python eval_full.py `
    --config configs/default.yaml `
    --checkpoint output/best.pth `
    --gallery output/gallery.pth `
    --results-dir results `
    --tag full_default `
    --benchmark

# ── 4. Low-variance ─────────────────────────────────────────────────
Write-Host "`n▶ [4/6] Low-variance subset evaluation …" -ForegroundColor Yellow
python eval_lowvar.py `
    --config configs/ablation_arcface_triplet.yaml `
    --checkpoint output/ablation_arcface_triplet/best.pth `
    --results-dir results `
    --tag lowvar

# ── 5. Figures ──────────────────────────────────────────────────────
Write-Host "`n▶ [5/6] Generating figures …" -ForegroundColor Yellow
python plot_results.py --results-dir results

# ── 6. Summary ──────────────────────────────────────────────────────
Write-Host "`n▶ [6/6] Done!" -ForegroundColor Green
Write-Host ""
Write-Host "Outputs:"
Write-Host "  results/results.csv           — all metrics"
Write-Host "  results/*.png                 — figures"
Write-Host "  results/*_metrics.npz         — raw curve data"
Write-Host "  REPORT.md                     — analysis report"
Write-Host ""
Write-Host "Submission checklist:" -ForegroundColor Cyan
Write-Host "  ✓ configs/          — YAML configs"
Write-Host "  ✓ data/             — dataset, PK sampler, low-var builder"
Write-Host "  ✓ models/           — backbone + heads"
Write-Host "  ✓ losses/           — softmax, arcface, cosface, triplet"
Write-Host "  ✓ results/          — figures, CSV, NPZ"
Write-Host "  ✓ output/           — checkpoints + gallery"
Write-Host "  ✓ scripts           — train / enroll / match / eval / ablation"
Write-Host "  ✓ requirements.txt  — pinned dependencies"
Write-Host "  ✓ README.md         — usage guide"
Write-Host "  ✓ REPORT.md         — complete analysis report"
Write-Host ""
Write-Host "To package for submission:" -ForegroundColor Cyan
Write-Host "  Compress-Archive -Path configs, data, models, losses, *.py, *.sh, *.ps1, *.txt, *.md -DestinationPath openset-reid.zip"

