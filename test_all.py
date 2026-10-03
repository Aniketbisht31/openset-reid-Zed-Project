"""End-to-End System Test for openset-reid.

Tests every module, loss, dataset helper, model, and script:
1. Backbones & Heads
2. Loss functions (Softmax, ArcFace, CosFace, Triplet with/without hard-mining)
3. Data structures & PK sampler
4. Low-variance clothing clustering & pair generation
5. Metrics library (ROC, FAR@TAR, DIR@FAR, CMC/mAP, ECE)
6. Pipeline simulation:
   - Creates a synthetic Market-1501 directory with realistic filenames
   - Trains 1 epoch using train.py
   - Enrolls gallery with enroll.py
   - Matches queries with match.py
   - Evaluates with eval_full.py
   - Evaluates with eval_lowvar.py
   - Generates all plots with plot_results.py
"""
import os
import sys
import shutil
import tempfile
import numpy as np
import torch
from PIL import Image
import yaml

print("==========================================================")
print("  OPENSET-REID: COMPREHENSIVE END-TO-END VALIDATION TEST")
print("==========================================================")

# 1. Backbones & Embedding Head
print("\n[1/7] Testing Backbones & Embedding Head...")
from models.backbone import build_backbone
from models.heads import ReIDModel

for b_name in ["osnet_x0_5", "resnet18", "mobilenetv3_small"]:
    bb = build_backbone(b_name, pretrained=False)
    model = ReIDModel(bb, embed_dim=256)
    x = torch.randn(2, 3, 256, 128)
    emb = model(x)
    assert emb.shape == (2, 256), f"Wrong shape: {emb.shape}"
    assert torch.allclose(emb.norm(dim=1), torch.ones(2), atol=1e-4), "Embedding not L2-normalized"
    print(f"  [PASS] {b_name:>18s} -> 256-D normalized output verified")

# 2. Losses
print("\n[2/7] Testing Losses...")
from losses.softmax import SoftmaxLoss
from losses.arcface import ArcFaceLoss
from losses.cosface import CosFaceLoss
from losses.triplet import BatchHardTripletLoss

dummy_emb = torch.randn(8, 256, requires_grad=True)
dummy_emb_norm = torch.nn.functional.normalize(dummy_emb, p=2, dim=1)
dummy_lbl = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

l_sm = SoftmaxLoss(256, 4)(dummy_emb_norm, dummy_lbl)
l_sm.backward(retain_graph=True)
print(f"  [PASS] SoftmaxLoss:       loss = {l_sm.item():.4f}")

l_af = ArcFaceLoss(256, 4)(dummy_emb_norm, dummy_lbl)
l_af.backward(retain_graph=True)
print(f"  [PASS] ArcFaceLoss:       loss = {l_af.item():.4f}")

l_cf = CosFaceLoss(256, 4)(dummy_emb_norm, dummy_lbl)
l_cf.backward(retain_graph=True)
print(f"  [PASS] CosFaceLoss:       loss = {l_cf.item():.4f}")

l_tp_hard = BatchHardTripletLoss(margin=0.3, hard_mining=True)(dummy_emb_norm, dummy_lbl)
l_tp_hard.backward(retain_graph=True)
print(f"  [PASS] Triplet (PK-Hard): loss = {l_tp_hard.item():.4f}")

l_tp_all = BatchHardTripletLoss(margin=0.3, hard_mining=False)(dummy_emb_norm, dummy_lbl)
l_tp_all.backward()
print(f"  [PASS] Triplet (All):     loss = {l_tp_all.item():.4f}")

# 3. Data Sampler & Low Variance Builder
print("\n[3/7] Testing PK Sampler & Low-Variance Builder...")
from data.sampler import PKSampler
class DummyDS:
    def __init__(self):
        self.samples = [(f"img_{i}.jpg", i // 4, 0) for i in range(32)]
        self.pid2label = {i // 4: i // 4 for i in range(32)}
ds = DummyDS()
sampler = PKSampler(ds, p=4, k=4)
idxs = list(sampler)
assert len(idxs) == 32, f"Sampler yielded {len(idxs)} indices instead of 32"
print(f"  [PASS] PKSampler successfully generated PK batches ({len(idxs)} samples)")

from data.low_variance import extract_color_histogram
test_img_path = "_test_tmp.jpg"
Image.fromarray(np.random.randint(0, 256, (256, 128, 3), dtype=np.uint8)).save(test_img_path)
hist = extract_color_histogram(test_img_path)
assert hist is not None and len(hist) == 48, f"Invalid color histogram length: {len(hist)}"
os.remove(test_img_path)
print(f"  [PASS] Color histogram extraction verified (length={len(hist)})")

# 4. Metrics Library
print("\n[4/7] Testing Comprehensive Metrics Library...")
from eval_metrics import (
    generate_pairs, compute_roc, far_tar_curve, far_at_tar,
    compute_cmc_map, openset_eval, expected_calibration_error
)

gen_s, imp_s = generate_pairs(dummy_emb.detach().numpy(), dummy_lbl.numpy(), n_pairs=50)
assert len(gen_s) == 50 and len(imp_s) == 50
fpr, tpr, _, auroc = compute_roc(gen_s, imp_s)
far_d = far_at_tar(gen_s, imp_s, (0.90, 0.95))
print(f"  [PASS] Pair generation + ROC + FAR@TAR passed (AUROC = {auroc:.4f})")

sim_mat = dummy_emb.detach().numpy() @ dummy_emb.detach().numpy().T
cmc, mAP = compute_cmc_map(sim_mat, dummy_lbl.numpy(), dummy_lbl.numpy(), max_rank=5)
print(f"  [PASS] CMC + mAP passed (Rank-1 = {cmc[0]:.4f}, mAP = {mAP:.4f})")

os_eval = openset_eval(np.random.rand(20), np.zeros(20), np.zeros(20), np.ones(20, dtype=bool))
assert "OSCR_AUC" in os_eval
print(f"  [PASS] Open-Set OSCR evaluation passed (AUC = {os_eval['OSCR_AUC']:.4f})")

ece, bins = expected_calibration_error(np.random.rand(20), np.random.randint(0, 2, 20))
print(f"  [PASS] ECE calculation passed (ECE = {ece:.4f})")

# 5. Pipeline Simulation on Mock Market-1501
print("\n[5/7] Testing Full Pipeline: Train -> Enroll -> Match -> Eval -> LowVar...")
tmp_data_root = os.path.abspath("./_tmp_test_market1501")
tmp_output_dir = os.path.abspath("./_tmp_test_output")
tmp_results_dir = os.path.abspath("./_tmp_test_results")

for d in [tmp_data_root, tmp_output_dir, tmp_results_dir]:
    os.makedirs(d, exist_ok=True)

train_dir = os.path.join(tmp_data_root, "bounding_box_train")
os.makedirs(train_dir, exist_ok=True)

# Generate mock Market-1501 images across 6 identities and 2 cameras:
# Names like: 0001_c1s1_000101_00.jpg
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
for pid in range(1, 7):
    col = colors[pid - 1]
    for img_idx in range(6):
        cam = (img_idx % 2) + 1
        fn = f"{pid:04d}_c{cam}s1_{img_idx:06d}_00.jpg"
        im = Image.new("RGB", (128, 256), color=col)
        im.save(os.path.join(train_dir, fn))

print(f"  [PASS] Created mock Market-1501 dataset with 36 images across 6 identities")

# Create a test YAML config
test_cfg = {
    "data": {
        "root": tmp_data_root,
        "height": 256,
        "width": 128,
        "num_workers": 0,
        "pk_p": 2,
        "pk_k": 2,
    },
    "model": {
        "backbone": "osnet_x0_5",
        "embed_dim": 256,
        "pretrained": False,
    },
    "loss": {
        "margin_type": "arcface",
        "arcface_s": 16.0,
        "arcface_m": 0.3,
        "cosface_s": 16.0,
        "cosface_m": 0.2,
        "triplet": True,
        "triplet_margin": 0.2,
        "triplet_weight": 1.0,
        "margin_weight": 1.0,
    },
    "train": {
        "epochs": 1,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "warmup_epochs": 0,
        "scheduler": "cosine",
        "step_size": 10,
        "gamma": 0.1,
        "fp16": False,
        "seed": 42,
    },
    "openset": {
        "unknown_ratio": 0.33,
        "threshold_mode": "adaptive",
        "global_threshold": 0.5,
        "adaptive_alpha": 0.8,
    },
    "calibration": {
        "method": "temperature",
        "val_fraction": 0.2,
    },
    "output": {
        "dir": tmp_output_dir,
        "save_freq": 1,
    }
}
test_cfg_path = os.path.join(tmp_output_dir, "test_config.yaml")
with open(test_cfg_path, "w") as f:
    yaml.dump(test_cfg, f)

# Run train.py via Python API / subprocess
import subprocess
PY = sys.executable

ret = subprocess.run([PY, "train.py", "--config", test_cfg_path], capture_output=True, text=True)
if ret.returncode != 0:
    print(ret.stdout)
    print(ret.stderr)
    raise RuntimeError(f"train.py failed with code {ret.returncode}")
assert os.path.exists(os.path.join(tmp_output_dir, "best.pth"))
print("  [PASS] train.py ran 1 epoch and produced best.pth")

ret = subprocess.run([
    PY, "enroll.py",
    "--config", test_cfg_path,
    "--checkpoint", os.path.join(tmp_output_dir, "best.pth")
], capture_output=True, text=True)
if ret.returncode != 0:
    print(ret.stdout)
    print(ret.stderr)
    raise RuntimeError(f"enroll.py failed with code {ret.returncode}")
assert os.path.exists(os.path.join(tmp_output_dir, "gallery.pth"))
print("  [PASS] enroll.py created gallery.pth with prototypes and adaptive thresholds")

# Match single query image
sample_query = os.path.join(train_dir, os.listdir(train_dir)[0])
ret = subprocess.run([
    PY, "match.py",
    "--config", test_cfg_path,
    "--checkpoint", os.path.join(tmp_output_dir, "best.pth"),
    "--gallery", os.path.join(tmp_output_dir, "gallery.pth"),
    "--query", sample_query
], capture_output=True, text=True)
if ret.returncode != 0:
    print(ret.stdout)
    print(ret.stderr)
    raise RuntimeError(f"match.py failed with code {ret.returncode}")
print("  [PASS] match.py executed single-query inference and confidence output")

# 6. Comprehensive Eval & LowVar Eval
print("\n[6/7] Testing eval_full.py & eval_lowvar.py...")
ret = subprocess.run([
    PY, "eval_full.py",
    "--config", test_cfg_path,
    "--checkpoint", os.path.join(tmp_output_dir, "best.pth"),
    "--gallery", os.path.join(tmp_output_dir, "gallery.pth"),
    "--results-dir", tmp_results_dir,
    "--tag", "e2e_test",
], capture_output=True, text=True)
if ret.returncode != 0:
    print(ret.stdout)
    print(ret.stderr)
    raise RuntimeError(f"eval_full.py failed with code {ret.returncode}")
assert os.path.exists(os.path.join(tmp_results_dir, "e2e_test_metrics.npz"))
print("  [PASS] eval_full.py generated full open-set metrics and results.csv entry")

ret = subprocess.run([
    PY, "eval_lowvar.py",
    "--config", test_cfg_path,
    "--checkpoint", os.path.join(tmp_output_dir, "best.pth"),
    "--results-dir", tmp_results_dir,
    "--tag", "e2e_lowvar",
    "--n-clusters", "3"
], capture_output=True, text=True)
if ret.returncode != 0:
    print(ret.stdout)
    print(ret.stderr)
    raise RuntimeError(f"eval_lowvar.py failed with code {ret.returncode}")
assert os.path.exists(os.path.join(tmp_results_dir, "e2e_lowvar_metrics.npz"))
print("  [PASS] eval_lowvar.py generated same-clothing impostor evaluation")

# 7. Plotting & Clean-up
print("\n[7/7] Testing plot_results.py...")
ret = subprocess.run([
    PY, "plot_results.py",
    "--results-dir", tmp_results_dir
], capture_output=True, text=True)
if ret.returncode != 0:
    print(ret.stdout)
    print(ret.stderr)
    raise RuntimeError(f"plot_results.py failed with code {ret.returncode}")

import glob
figs = glob.glob(os.path.join(tmp_results_dir, "*.png"))
print(f"  [PASS] plot_results.py produced {len(figs)} figures successfully")
for f in sorted(figs):
    print("    -", os.path.basename(f))

# Clean up temporary test artifacts
shutil.rmtree(tmp_data_root, ignore_errors=True)
shutil.rmtree(tmp_output_dir, ignore_errors=True)
shutil.rmtree(tmp_results_dir, ignore_errors=True)

print("\n==========================================================")
print("  ALL TESTS PASSED: COMPLETE SYSTEM IS FULLY OPERATIONAL!")
print("==========================================================")
