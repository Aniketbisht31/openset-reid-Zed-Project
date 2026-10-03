# openset-reid

**Open-Set Person Re-Identification** with low inter-class variance handling.

Minimises **False Accept Rate (FAR) at fixed recall (DIR)** using compact
embeddings, margin-based metric learning, and adaptive per-identity
thresholds.

---

## Architecture overview

```
┌──────────────┐     ┌────────────────┐     ┌───────────────┐
│  Input Image │────▸│    Backbone     │────▸│ Embedding Head│───▸ 256-D L2-norm
│  256 × 128   │     │ (OSNet-x0.5 /  │     │ BN→FC→BN→L2   │
│              │     │  ResNet18 /     │     │               │
│              │     │  MobileNetV3)   │     │               │
└──────────────┘     └────────────────┘     └───────────────┘
                                                    │
                     ┌──────────────────────────────┤
                     ▼                              ▼
            ┌────────────────┐             ┌────────────────┐
            │  ArcFace /     │             │  Batch-Hard    │
            │  CosFace       │             │  Triplet Loss  │
            │  Margin Loss   │             │  (PK sampler)  │
            └────────────────┘             └────────────────┘
```

### Open-set decision pipeline

```
Query embedding  ──▸  cosine sim vs all prototypes  ──▸  best match
                                                           │
                                               ┌──────────┤
                                               ▼          ▼
                                          sim ≥ τ_i?   calibrate
                                           │  │       (temp/Platt/iso)
                                          YES  NO         │
                                           │   │          ▼
                                       ACCEPT REJECT  confidence ∈[0,1]
```

---

## Project structure

```
openset-reid/
├── configs/
│   └── default.yaml          # all hyper-parameters
├── data/
│   ├── download.py            # Market-1501 downloader (gdown)
│   ├── market1501.py          # dataset + open-set splits
│   ├── sampler.py             # PK batch sampler
│   └── low_variance.py        # clothing-colour clustering
├── models/
│   ├── backbone.py            # OSNet-x0.5, ResNet18, MobileNetV3
│   └── heads.py               # BN→FC→BN→L2 embedding head
├── losses/
│   ├── arcface.py             # additive angular margin
│   ├── cosface.py             # additive cosine margin
│   └── triplet.py             # batch-hard triplet
├── train.py                   # training loop
├── enroll.py                  # build gallery prototypes
├── match.py                   # query → accept/reject
├── eval.py                    # FAR@DIR, OSCR, calibration, latency
├── requirements.txt
└── README.md
```

---

## Quick start

### 1. Install

```bash
cd openset-reid
pip install -r requirements.txt
```

### 2. Download Market-1501

```bash
python -m data.download ./datasets/market1501
```

> The script uses **gdown** to pull the official zip from Google Drive and
> extracts it automatically.

### 3. Train

```bash
python train.py --config configs/default.yaml
```

Key config toggles (edit `configs/default.yaml`):

| Key | Options | Default |
|-----|---------|---------|
| `loss.margin_type` | `arcface` · `cosface` · `none` | `arcface` |
| `loss.triplet` | `true` · `false` | `true` |
| `model.backbone` | `osnet_x0_5` · `resnet18` · `mobilenetv3_small` | `osnet_x0_5` |
| `openset.unknown_ratio` | any float in (0, 1) | `0.2` |

### 4. Enrol gallery

```bash
python enroll.py --checkpoint output/best.pth
```

Writes `output/gallery.pth` with per-identity prototypes and adaptive
thresholds.

### 5. Match queries

```bash
# Single image
python match.py --query path/to/query.jpg

# Directory
python match.py --query path/to/query_dir/
```

### 6. Evaluate

```bash
python eval.py --checkpoint output/best.pth --gallery output/gallery.pth --benchmark
```

Reports FAR @ DIR = {50 %, 70 %, 80 %, 90 %, 95 %}, OSCR AUC, and
single-image latency.

### 7. Low-variance subset

```bash
python -m data.low_variance ./datasets/market1501
```

Clusters training images by upper/lower clothing colour and prints
same-clothing impostor pair counts per cluster.

---

## End-to-End Validation Test

Run the automated test suite to verify all modules, losses, metrics, training, enrollment, inference, and plotting in under 15 seconds:

```bash
cd openset-reid
python test_all.py
```

---

## Minimal Smoke Test (interactive)

Verifies that every module imports correctly, the model forward pass
works, and each loss computes a valid gradient:

```bash
cd openset-reid
python -c "
import torch
from models.backbone import build_backbone
from models.heads import ReIDModel
from losses.arcface import ArcFaceLoss
from losses.cosface import CosFaceLoss
from losses.triplet import BatchHardTripletLoss

# Build model
bb = build_backbone('osnet_x0_5', pretrained=False)
model = ReIDModel(bb, embed_dim=256)
x = torch.randn(8, 3, 256, 128)
emb = model(x)
print(f'Embedding: {emb.shape}, norm={emb.norm(dim=1).mean():.4f}')

# ArcFace
labels = torch.arange(8) % 4
af = ArcFaceLoss(256, num_classes=4)
loss_af = af(emb, labels)
loss_af.backward(retain_graph=True)
print(f'ArcFace loss:  {loss_af.item():.4f}')

# CosFace
cf = CosFaceLoss(256, num_classes=4)
loss_cf = cf(emb, labels)
loss_cf.backward(retain_graph=True)
print(f'CosFace loss:  {loss_cf.item():.4f}')

# Triplet
tl = BatchHardTripletLoss(margin=0.3)
loss_tl = tl(emb, labels)
loss_tl.backward()
print(f'Triplet loss:  {loss_tl.item():.4f}')

print('\n✓ Smoke test passed.')
"
```

Expected output (values will vary):

```
Embedding: torch.Size([8, 256]), norm=1.0000
ArcFace loss:  5.xxxx
CosFace loss:  4.xxxx
Triplet loss:  0.xxxx

✓ Smoke test passed.
```

---

## Design rationale

### Why per-identity adaptive thresholds?

A single global threshold treats all classes equally, but intra-class
variance differs: some people always wear the same outfit (tight
cluster → high threshold acceptable), while others change clothing
across cameras (loose cluster → lower threshold needed).  The adaptive
formula

```
τ_i = α · (μ_i − 2σ_i) + (1 − α) · τ_global
```

blends per-class evidence with a global prior, reducing FAR without
sacrificing DIR on well-clustered identities.

### Why score calibration?

Raw cosine similarity is not a well-calibrated probability.
Temperature / Platt / isotonic calibration maps it to a true
confidence ∈ [0, 1] that can be consumed downstream as a posterior
probability.

### Low-variance subset

Real-world re-id failures concentrate on *same-clothing impostor pairs*
(different people in similar outfits).  The colour-histogram clustering
in `data/low_variance.py` curates these hard pairs for targeted
evaluation and potential hard-example training.

---

## Citation

If you use this code, please cite the underlying methods:

- **ArcFace**: Deng et al., CVPR 2019
- **CosFace**: Wang et al., CVPR 2018
- **Batch-Hard Triplet**: Hermans et al., arXiv 2017
- **OSNet**: Zhou et al., ICCV 2019
- **Market-1501**: Zheng et al., ICCV 2015
