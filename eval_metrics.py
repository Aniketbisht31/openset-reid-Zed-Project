"""Comprehensive open-set person re-identification metrics.

All functions accept **numpy arrays** and return numpy arrays or scalars.

Sections
--------
Pair generation     generate_pairs
Verification        compute_roc · far_tar_curve · far_at_tar
Identification      compute_cmc_map
Open-set            openset_eval
Calibration         expected_calibration_error
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np


# ═════════════════════════════════════════════════════════════════════
# Pair generation
# ═════════════════════════════════════════════════════════════════════

def generate_pairs(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_pairs: int = 10_000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample genuine (same-ID) and impostor (diff-ID) cosine-similarity scores.

    Returns ``(genuine_scores, impostor_scores)`` each of length *n_pairs*.
    """
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)

    lbl2idx: Dict[int, list] = defaultdict(list)
    for i, l in enumerate(labels):
        if l >= 0:
            lbl2idx[int(l)].append(i)

    valid_lbls = [l for l, v in lbl2idx.items() if len(v) >= 2]
    all_lbls = list(lbl2idx.keys())
    assert len(valid_lbls) >= 1, "Need ≥ 1 ID with ≥ 2 images for genuine pairs"
    assert len(all_lbls) >= 2, "Need ≥ 2 IDs for impostor pairs"

    gen, imp = [], []
    for _ in range(n_pairs):
        l = rng.choice(valid_lbls)
        a, b = rng.choice(lbl2idx[l], 2, replace=False)
        gen.append(float(embeddings[a] @ embeddings[b]))
    for _ in range(n_pairs):
        l1, l2 = rng.choice(all_lbls, 2, replace=False)
        a, b = rng.choice(lbl2idx[l1]), rng.choice(lbl2idx[l2])
        imp.append(float(embeddings[a] @ embeddings[b]))

    return np.asarray(gen), np.asarray(imp)


# ═════════════════════════════════════════════════════════════════════
# Verification
# ═════════════════════════════════════════════════════════════════════

def compute_roc(
    gen_scores: np.ndarray,
    imp_scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """ROC curve → ``(fpr, tpr, thresholds, auroc)``."""
    from sklearn.metrics import roc_curve, roc_auc_score

    scores = np.concatenate([gen_scores, imp_scores])
    labels = np.concatenate([np.ones(len(gen_scores)),
                             np.zeros(len(imp_scores))])
    fpr, tpr, th = roc_curve(labels, scores)
    return fpr, tpr, th, float(roc_auc_score(labels, scores))


def far_tar_curve(
    gen_scores: np.ndarray,
    imp_scores: np.ndarray,
    n_th: int = 2000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FAR-vs-TAR curve.

    FAR = impostors accepted / total impostors.
    TAR = genuines accepted / total genuines.
    """
    lo = min(gen_scores.min(), imp_scores.min()) - 0.01
    hi = max(gen_scores.max(), imp_scores.max()) + 0.01
    th = np.linspace(lo, hi, n_th)
    far = np.array([(imp_scores >= t).mean() for t in th])
    tar = np.array([(gen_scores >= t).mean() for t in th])
    return far, tar, th


def far_at_tar(
    gen_scores: np.ndarray,
    imp_scores: np.ndarray,
    tar_targets: Tuple[float, ...] = (0.90, 0.95, 0.99),
) -> Dict[str, float]:
    """FAR at specific TAR levels (e.g. 90 %, 95 %, 99 %)."""
    far, tar, _ = far_tar_curve(gen_scores, imp_scores)
    out: Dict[str, float] = {}
    for t in tar_targets:
        mask = tar >= t
        out[f"FAR@TAR={t:.0%}"] = float(far[mask].min()) if mask.any() else float("nan")
    return out


# ═════════════════════════════════════════════════════════════════════
# Identification (closed-set)
# ═════════════════════════════════════════════════════════════════════

def compute_cmc_map(
    sim_mat: np.ndarray,
    q_labels: np.ndarray,
    g_labels: np.ndarray,
    q_cams: Optional[np.ndarray] = None,
    g_cams: Optional[np.ndarray] = None,
    max_rank: int = 50,
) -> Tuple[np.ndarray, float]:
    """CMC curve + mAP (standard re-id protocol).

    Args:
        sim_mat:  ``[Q, G]`` cosine similarity.
        q_labels: ``[Q]`` query identities.
        g_labels: ``[G]`` gallery identities.
        q_cams / g_cams: camera ids (same-PID same-cam pairs are removed).

    Returns:
        cmc ``[max_rank]`` — cmc[k] = rank-(k+1) accuracy.
        mAP — scalar mean average precision.
    """
    q_labels = np.asarray(q_labels)
    g_labels = np.asarray(g_labels)
    order = np.argsort(-sim_mat, axis=1)  # descending similarity

    cmc = np.zeros(max_rank, dtype=np.float64)
    aps: list[float] = []
    n_valid = 0

    for i in range(len(q_labels)):
        s_g = g_labels[order[i]]

        # validity mask — drop same-PID-same-cam and junk (label < 0)
        keep = np.ones(len(s_g), dtype=bool)
        if q_cams is not None and g_cams is not None:
            s_c = g_cams[order[i]]
            keep &= ~((s_g == q_labels[i]) & (s_c == q_cams[i]))
        keep &= s_g >= 0

        matches = s_g[keep] == q_labels[i]
        n_rel = int(matches.sum())
        if n_rel == 0:
            continue
        n_valid += 1

        # CMC — rank of first correct
        first = int(np.argmax(matches))
        if first < max_rank:
            cmc[first:] += 1

        # AP
        cum = matches.cumsum().astype(float)
        prec = cum / np.arange(1, len(matches) + 1)
        aps.append(float((prec * matches).sum() / n_rel))

    if n_valid > 0:
        cmc /= n_valid
    return cmc, float(np.mean(aps)) if aps else 0.0


# ═════════════════════════════════════════════════════════════════════
# Open-set
# ═════════════════════════════════════════════════════════════════════

def openset_eval(
    scores: np.ndarray,
    pred_labels: np.ndarray,
    true_labels: np.ndarray,
    is_known: np.ndarray,
    n_th: int = 2000,
) -> Dict:
    """Full open-set evaluation.

    Args:
        scores:      best cosine similarity per query ``[Q]``
        pred_labels: predicted identity label ``[Q]``
        true_labels: ground-truth label (−1 for unknown) ``[Q]``
        is_known:    ``[Q]`` boolean mask

    Returns:
        dict with arrays ``dirs``, ``fars``, ``thresholds``,
        ``oscr_far``, ``oscr_ccr``, scalar summaries, and
        ``DIR@FAR=…`` / ``FAR@DIR=…`` values.
    """
    scores = np.asarray(scores, dtype=np.float64)
    pred_labels = np.asarray(pred_labels)
    true_labels = np.asarray(true_labels)
    is_known = np.asarray(is_known, dtype=bool)

    n_kn = int(is_known.sum())
    n_un = int((~is_known).sum())
    correct = is_known & (pred_labels == true_labels)

    th = np.linspace(0, 1, n_th)
    dirs = np.empty(n_th)
    fars = np.empty(n_th)
    for i, t in enumerate(th):
        acc = scores >= t
        dirs[i] = (acc & correct).sum() / max(n_kn, 1)
        fars[i] = (acc & ~is_known).sum() / max(n_un, 1)

    res: Dict = {
        "thresholds": th, "dirs": dirs, "fars": fars,
        "n_known": n_kn, "n_unknown": n_un,
    }

    # DIR @ FAR
    for target_far in (0.001, 0.01, 0.05, 0.10):
        mask = fars <= target_far
        key = f"DIR@FAR={target_far:.1%}" if target_far < 0.01 else f"DIR@FAR={target_far:.0%}"
        res[key] = float(dirs[mask].max()) if mask.any() else 0.0

    # FAR @ DIR
    for target_dir in (0.50, 0.70, 0.80, 0.90, 0.95):
        mask = dirs >= target_dir
        res[f"FAR@DIR={target_dir:.0%}"] = float(fars[mask].min()) if mask.any() else float("nan")

    # OSCR
    order = np.argsort(-scores)
    cum_un = np.cumsum(~is_known[order])
    cum_cor = np.cumsum(correct[order])
    oscr_far = cum_un / max(n_un, 1)
    oscr_ccr = cum_cor / max(n_kn, 1)
    res["oscr_far"] = oscr_far
    res["oscr_ccr"] = oscr_ccr
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    res["OSCR_AUC"] = float(_trapz(oscr_ccr, oscr_far))

    return res


# ═════════════════════════════════════════════════════════════════════
# Calibration
# ═════════════════════════════════════════════════════════════════════

def expected_calibration_error(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 15,
) -> Tuple[float, Dict]:
    """Expected Calibration Error + per-bin data for reliability diagrams.

    Args:
        confidences: predicted probability ∈ [0, 1]  ``[N]``
        correct:     binary correctness                ``[N]``

    Returns:
        ece   — scalar ECE.
        bins  — ``{edges, accs, confs, counts}`` arrays.
    """
    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)

    edges = np.linspace(0, 1, n_bins + 1)
    accs = np.zeros(n_bins)
    confs = np.zeros(n_bins)
    counts = np.zeros(n_bins)

    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (confidences >= lo) & (confidences <= hi if b == n_bins - 1
                                      else confidences < hi)
        c = int(mask.sum())
        counts[b] = c
        if c > 0:
            accs[b] = correct[mask].mean()
            confs[b] = confidences[mask].mean()

    total = counts.sum()
    ece = float((counts * np.abs(accs - confs)).sum() / total) if total > 0 else 0.0

    return ece, {"edges": edges, "accs": accs, "confs": confs, "counts": counts}
