# Open-Set Person Re-Identification under Low Inter-Class Variance

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows%20%7C%20macOS-lightgrey.svg)]()
[![Inference](https://img.shields.io/badge/inference-%3C10ms%2Fimage%20%40%20P95-brightgreen.svg)]()
[![Embedding](https://img.shields.io/badge/embedding-256--D%20L2--normalized-blueviolet.svg)]()

---

## 1. Executive Summary & Problem Context

Real-world computer vision deployments (automated access control, campus security, search-and-rescue, retail analytics) require re-identifying individuals as they traverse non-overlapping camera networks. While conventional academic benchmarks treat person re-identification as a **closed-set retrieval problem** (where every probe image is guaranteed to belong to an enrolled gallery subject), deployed systems inevitably encounter an **open-set world**: the vast majority of observed individuals are unenrolled impostors who should be rejected.

The critical engineering failure in real-world open-set re-ID is the **False Accept**:
> *An unenrolled, unknown individual is mistakenly matched to an enrolled identity because they share near-identical appearance traits (e.g., workers in identical uniforms, students in dress codes, or pedestrians wearing generic dark jackets and denim).*

### Key Objective (PS-1)
Design, train, and validate an end-to-end open-set person re-identification framework that **minimizes the False Accept Rate (FAR) at fixed high recall (True Accept Rate / Detection & Identification Rate)**, specifically under **low inter-class appearance variance**.

---

## 2. Technical Architecture & Algorithmic Design

```
+──────────────────────────+
|  Probe Image (256 x 128) |
+──────────────────────────+
              │
              ▼
+──────────────────────────+
|     Compact Backbone     |  <-- OSNet-x0.5 (~1.3M params) / ResNet-18 / MobileNetV3
|  Omni-Scale Feature Ext. |
+──────────────────────────+
              │
              ▼
+──────────────────────────+
|      Embedding Head      |  <-- BatchNorm1d -> Linear(C, 256) -> BatchNorm1d -> L2-Norm
|  256-D Unit-Norm Vector  |
+──────────────────────────+
              │
      ┌───────┴────────────────────────┐
      ▼                                ▼
+───────────────+              +───────────────+
| ArcFace /     |              |  Batch-Hard   |
| CosFace Loss  |              | Triplet Loss  |  <-- PK Batch Sampler (P identities x K images)
| Angular Margin|              | Online Mining |
+───────────────+              +───────────────+
```

### 2.1 Deep Metric Learning & Embedding Formulation
- **Compact Backbones**: Supports lightweight Omni-Scale Network (**OSNet-x0.5**, ~1.3M params, self-contained), **ResNet-18** (11.2M params), and **MobileNetV3-Small** (1.5M params), satisfying real-time inference latency ($<10$ ms/image on GPU).
- **Embedding Head**: Transforms backbone features via `BatchNorm1d -> Linear(C, 256) -> BatchNorm1d -> L2-Normalization`. The resulting representation lies on a 256-dimensional unit hypersphere $\mathbb{S}^{255}$, allowing cosine similarity to be computed as an inner product.

### 2.2 Dual-Objective Loss Formulation
The framework optimizes an objective balancing angular class separation with pairwise metric distance:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{margin}} \mathcal{L}_{\text{margin}} + \lambda_{\text{triplet}} \mathcal{L}_{\text{triplet}}$$

1. **Additive Angular Margin Loss (ArcFace)**:
   Enforces an angular margin $m$ directly in the cosine space between feature embeddings $\mathbf{x}_i$ and class proxies $\mathbf{w}_j$:
   $$\mathcal{L}_{\text{ArcFace}} = -\log \frac{e^{s \cdot \cos(\theta_{y_i} + m)}}{e^{s \cdot \cos(\theta_{y_i} + m)} + \sum_{j \neq y_i} e^{s \cdot \cos \theta_j}}$$
   *(CosFace $\cos \theta - m$ and standard normalized Softmax CE baseline are also togglable via configuration).*

2. **Batch-Hard Triplet Loss with PK Online Mining**:
   Constructs mini-batches using a **PK Sampler** ($P$ identities $\times$ $K$ images/identity). For each anchor $\mathbf{a}$, the hardest positive $\mathbf{p}^*$ (furthest same-class instance) and hardest negative $\mathbf{n}^*$ (closest different-class instance) are mined online:
   $$\mathcal{L}_{\text{triplet}} = \frac{1}{B} \sum_{i=1}^B \max\left(0, \mathcal{D}(\mathbf{a}_i, \mathbf{p}_i^*) - \mathcal{D}(\mathbf{a}_i, \mathbf{n}_i^*) + \alpha\right)$$
   where $\mathcal{D}(\mathbf{u}, \mathbf{v}) = \sqrt{2 - 2(\mathbf{u} \cdot \mathbf{v})}$.

---

## 3. Open-Set Decision & Calibration Engine

```
Query Embedding (q)
        │
        ▼
Cosine Similarity against Enrolled Gallery Prototypes {p_1, p_2, ..., p_N}
        │
        ├─────────────────────────────────────┐
        ▼                                     ▼
Best Match: s* = max_i (q · p_i)      Adaptive Threshold: τ_{i*}
        │                                     │
        ▼                                     ▼
Rejection Decision: ────────────► Is s* >= τ_{i*}?
                                   ├── YES ──► Enrolled Identity i*
                                   └── NO  ──► REJECT AS UNKNOWN
                                                      │
                                                      ▼
Confidence Calibration ─────────────────► Confidence ∈ [0, 1]
(Temperature / Platt / Isotonic)
```

### 3.1 Prototype Representation & Dynamic Enrolment
- **Centroid Prototypes**: Enrolled identities are represented by normalized centroid embeddings:
  $$\mathbf{p}_i = \frac{\sum_{k=1}^{N_i} \mathbf{e}_{i, k}}{\left\|\sum_{k=1}^{N_i} \mathbf{e}_{i, k}\right\|_2}$$
- **Arbitrary Identity Counts**: No fixed gallery size or class count is assumed; prototypes are dynamically stacked into matrix $\mathbf{P} \in \mathbb{R}^{N \times 256}$ for fast matrix-vector retrieval.

### 3.2 Adaptive Per-Identity Threshold Formulation
Global thresholds treat tight and loose identity clusters identically, leading to false accepts on high-variance identities. We formulate a per-class adaptive threshold blending class compactness with a global prior:

$$\tau_i = \alpha \cdot (\mu_i - 2\sigma_i) + (1 - \alpha) \cdot \tau_{\text{global}}$$

where $\mu_i$ and $\sigma_i$ are the sample mean and standard deviation of intra-class pairwise cosine similarities for identity $i$, and $\alpha \in [0, 1]$ controls the blending factor.

### 3.3 Score Calibration & Reliability
Raw cosine similarities do not represent well-calibrated posterior probabilities. The engine fits calibration models on held-out validation splits:
- **Temperature Scaling**: Logit scaling $\sigma(s / T)$ minimizing negative log-likelihood (NLL).
- **Platt Scaling**: Univariate logistic regression $\sigma(a \cdot s + b)$.
- **Isotonic Regression**: Non-parametric piecewise-constant isotonic fit.
- **Evaluation**: Quantified via **Expected Calibration Error (ECE)** and visualized with **Reliability Diagrams**.

---

## 4. Curated Low-Variance (Same-Clothing) Subset

To explicitly benchmark and mitigate false accepts under minimal inter-class appearance variance, the framework provides an automated clothing clustering pipeline ([`data/low_variance.py`](data/low_variance.py)):

1. **Torso & Leg Region Segmentation**: Crops upper torso ($10\%-40\%$ height) and lower body ($50\%-95\%$ height).
2. **Color Histogram Extraction**: Computes joint HSV histograms across 8 bins per channel (48-D feature vector).
3. **K-Means Clustering**: Clusters all training identities into discrete clothing appearance groups.
4. **Hard Impostor Pair Generation**: Pairs images of **different identities belonging to the identical clothing cluster**:
   $$\mathcal{P}_{\text{hard}} = \{(\mathbf{x}_a, \mathbf{x}_b) \mid y_a \neq y_b \wedge \text{Cluster}(\mathbf{x}_a) = \text{Cluster}(\mathbf{x}_b)\}$$

---

## 5. Experimental Protocols & Metrics

The project evaluates both verification and open-set identification:

| Metric Category | Metrics Reported | Formula / Definition |
|---|---|---|
| **Verification** | **ROC & AUROC** | Area under True Positive Rate vs False Positive Rate curve |
| **Verification** | **FAR @ TAR = 90%, 95%, 99%** | Impostor acceptance rate when genuine acceptance is held at target recall |
| **Open-Set Identification** | **DIR @ FAR = 0.1%, 1%, 5%, 10%** | Detection & Identification Rate at fixed low False Accept Rates |
| **Open-Set Identification** | **FAR @ DIR = 50%, 70%, 80%, 90%, 95%** | Impostor leak rate at fixed identification recall levels |
| **Open-Set Identification** | **OSCR AUC** | Area under Open-Set Classification Rate curve |
| **Closed-Set Retrieval** | **Rank-1, Rank-5, Rank-10, mAP** | CMC matching rates and mean Average Precision (cross-camera valid) |
| **Calibration** | **ECE & Reliability Diagrams** | Expected Calibration Error: $\sum_{m=1}^M \frac{\|B_m\|}{N} \|\text{acc}(B_m) - \text{conf}(B_m)\|$ |
| **Efficiency** | **GPU Latency (P50, P95, P99)** | CUDA-synchronized single-image inference latency |

---

## 6. Ablation Matrix

The modular driver ([`ablation.py`](ablation.py)) evaluates 6 isolated conditions:

| # | Condition | Margin Loss | Triplet Loss | Negative Mining | Decision Threshold | Score Calibration |
|---|---|---|---|---|---|---|
| 1 | `1_softmax_baseline` | Softmax CE ($s=30, m=0$) | None | None | Global ($\tau=0.50$) | None |
| 2 | `2_plus_arcface` | ArcFace ($s=30, m=0.50$) | None | None | Global ($\tau=0.50$) | None |
| 3 | `3_plus_triplet` | ArcFace ($s=30, m=0.50$) | Triplet ($\alpha=0.3$) | Random (Batch-All) | Global ($\tau=0.50$) | None |
| 4 | `4_plus_hard_mining` | ArcFace ($s=30, m=0.50$) | Triplet ($\alpha=0.3$) | Online PK Batch-Hard | Global ($\tau=0.50$) | None |
| 5 | `5_plus_adaptive_thresholds` | ArcFace ($s=30, m=0.50$) | Triplet ($\alpha=0.3$) | Online PK Batch-Hard | Adaptive ($\tau_i$) | None |
| 6 | `6_plus_calibration` | ArcFace ($s=30, m=0.50$) | Triplet ($\alpha=0.3$) | Online PK Batch-Hard | Adaptive ($\tau_i$) | Temperature Scaling |

---

## 7. Repository Layout

```
openset-reid/
├── configs/
│   ├── default.yaml                          # Default production hyper-parameters
│   ├── ablation_softmax.yaml                 # Ablation 1: Softmax baseline
│   ├── ablation_arcface.yaml                 # Ablation 2: +ArcFace margin
│   ├── ablation_arcface_triplet_nohard.yaml  # Ablation 3: +Triplet without hard mining
│   └── ablation_arcface_triplet.yaml         # Ablation 4: +PK batch-hard negative mining
├── data/
│   ├── __init__.py                           # Package initialization
│   ├── download.py                           # Automated Market-1501 downloader
│   ├── market1501.py                         # Open-set split & data loader
│   ├── sampler.py                            # PK batch sampler (P x K)
│   └── low_variance.py                       # HSV color clustering & same-clothing pairs
├── models/
│   ├── __init__.py                           # Package initialization
│   ├── backbone.py                           # OSNet-x0.5, ResNet-18, MobileNetV3-Small
│   └── heads.py                              # BN -> FC(256) -> BN -> L2 embedding head
├── losses/
│   ├── __init__.py                           # Package initialization
│   ├── softmax.py                            # Scaled cosine softmax (no margin)
│   ├── arcface.py                            # Additive angular margin loss
│   ├── cosface.py                            # Additive cosine margin loss
│   └── triplet.py                            # Batch-hard & batch-all triplet loss
├── train.py                                  # Training script (FP16, LR warmup, cosine decay)
├── enroll.py                                 # Gallery prototype enrollment & adaptive thresholding
├── match.py                                  # OpenSetMatcher engine for single/batch query inference
├── eval.py                                   # Standalone evaluation & latency benchmarking
├── eval_full.py                              # Full open-set & closed-set evaluation pipeline
├── eval_lowvar.py                            # Same-clothing hard impostor evaluation
├── eval_metrics.py                           # Math library: ROC, FAR@TAR, DIR@FAR, CMC, ECE
├── ablation.py                               # End-to-end ablation runner across all 6 conditions
├── plot_results.py                           # Generates 9 matplotlib figures and formatted tables
├── test_all.py                               # Self-contained integration test suite (<15s)
├── run_all.sh                                # Linux/macOS bash reproduction script
├── run_all.ps1                               # Windows PowerShell reproduction script
├── requirements.txt                          # Pinned dependencies
├── README.md                                 # Technical documentation
└── REPORT.md                                 # Comprehensive scientific report
```

---

## 8. Getting Started

### 8.1 Prerequisites & Installation
```bash
git clone https://github.com/Aniketbisht31/openset-reid-Zed-Project.git
cd openset-reid-Zed-Project
pip install -r requirements.txt
```

### 8.2 End-to-End Validation Test (<15 seconds)
Execute the self-contained validation test to verify all models, heads, losses, metric calculations, synthetic pipeline, and plot generation:
```bash
python test_all.py
```
Expected output:
```
==========================================================
  OPENSET-REID: COMPREHENSIVE END-TO-END VALIDATION TEST
==========================================================
[1/7] Testing Backbones & Embedding Head... [PASS]
[2/7] Testing Losses...                     [PASS]
[3/7] Testing PK Sampler & Low-Variance...  [PASS]
[4/7] Testing Comprehensive Metrics...      [PASS]
[5/7] Testing Full Pipeline Simulation...   [PASS]
[6/7] Testing eval_full.py & LowVar...      [PASS]
[7/7] Testing plot_results.py...            [PASS]
==========================================================
  ALL TESTS PASSED: COMPLETE SYSTEM IS FULLY OPERATIONAL!
==========================================================
```

### 8.3 Download Dataset
Automated download of Market-1501 with recursive zip extraction:
```bash
python -m data.download ./datasets/market1501
```

### 8.4 Training
Train the compact model with mixed precision (`fp16`) and cosine decay:
```bash
python train.py --config configs/default.yaml
```

### 8.5 Gallery Enrollment
Extract embeddings for enrolled identities, construct mean prototypes, and compute per-identity adaptive thresholds:
```bash
python enroll.py --config configs/default.yaml --checkpoint output/best.pth
```

### 8.6 Matching & Inference
Query an image against the enrolled gallery with automatic unknown rejection and calibrated confidence:
```bash
# Single image inference
python match.py --query path/to/probe.jpg

# Batch directory matching
python match.py --query path/to/probe_folder/
```

### 8.7 Evaluation & Latency Benchmarking
Evaluate full open-set metrics (ROC, FAR@TAR, DIR@FAR, OSCR, CMC, ECE) and GPU latency:
```bash
python eval_full.py --benchmark --results-dir results --tag full_default
```

### 8.8 Low-Variance (Same-Clothing) Evaluation
Evaluate verification specifically against hard impostors sharing the same outfit:
```bash
python eval_lowvar.py --results-dir results --tag lowvar --n-clusters 10
```

### 8.9 Full Ablation Suite & Visualization
Run the complete 6-stage ablation matrix and generate all 9 visual figures:
```bash
python ablation.py --results-dir results
```
Generated figures in `results/`:
- `roc_curve.png`: ROC curves for all models & low-variance subset
- `far_tar.png`: Log-scale FAR vs. TAR verification curves
- `dir_far.png`: Open-set DIR vs. FAR identification curves
- `oscr_curve.png`: Open-Set Classification Rate curves
- `cmc_curve.png`: Rank-1 through Rank-50 CMC matching curves
- `score_distribution.png`: Genuine vs. impostor cosine similarity histograms
- `reliability_diagram.png`: Reliability diagrams and confidence calibration
- `ablation_bar.png`: Grouped comparative metric bars
- `ablation_table.png`: Formatted summary table of all ablation variants

---

## 9. One-Click Reproduction Pipeline

To reproduce the entire experimental study with fixed seeds (`seed=42`):

- **Linux / macOS:**
  ```bash
  chmod +x run_all.sh
  ./run_all.sh
  ```

- **Windows PowerShell:**
  ```powershell
  .\run_all.ps1
  ```

---

## 10. Submission Checklist & Packaging

To create a clean distribution archive excluding raw datasets, model weights, and local Python caches:

- **Linux / macOS:**
  ```bash
  zip -r openset-reid.zip . -x "datasets/*" "__pycache__/*" "*.pyc" "output/*.pth" ".git/*"
  ```

- **Windows PowerShell:**
  ```powershell
  Compress-Archive -Path configs, data, models, losses, *.py, *.sh, *.ps1, *.txt, *.md -DestinationPath openset-reid.zip -Force
  ```

---

## 11. References & Citations

1. **OSNet**: Zhou et al., *"Omni-Scale Feature Learning for Person Re-Identification"*, ICCV 2019.
2. **ArcFace**: Deng et al., *"ArcFace: Additive Angular Margin Loss for Deep Face Recognition"*, CVPR 2019.
3. **CosFace**: Wang et al., *"CosFace: Large Margin Cosine Loss for Deep Face Recognition"*, CVPR 2018.
4. **Batch-Hard Triplet**: Hermans et al., *"In Defense of the Triplet Loss for Person Re-Identification"*, arXiv 2017.
5. **Open-Set Classification**: Scheirer et al., *"Toward Open Set Recognition"*, IEEE TPAMI 2013.
6. **Market-1501 Benchmark**: Zheng et al., *"Scalable Person Re-identification: A Benchmark"*, ICCV 2015.
