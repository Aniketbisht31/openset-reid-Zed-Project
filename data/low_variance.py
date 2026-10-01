"""Curated low-variance subset builder.

Groups images by clothing appearance (upper / lower body HSV colour
histograms) and constructs *same-clothing impostor pairs* — images of
**different** people who wear visually similar outfits.  These hard pairs
are the critical failure mode for a re-id system operating under low
inter-class variance.
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

# ── Colour histogram ────────────────────────────────────────────────


def extract_color_histogram(img_path: str, bins: int = 8) -> np.ndarray | None:
    """Compute a compact HSV histogram over upper- and lower-body crops.

    Returns a 1-D float32 vector of length ``2 × 3 × bins`` (upper + lower,
    H + S + V channels), or ``None`` if the image cannot be read.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Rough body-region crops (avoids head / feet edges)
    upper = hsv[int(h * 0.10) : int(h * 0.40), int(w * 0.15) : int(w * 0.85)]
    lower = hsv[int(h * 0.50) : int(h * 0.95), int(w * 0.10) : int(w * 0.90)]

    parts: list[float] = []
    for region in (upper, lower):
        if region.size == 0:
            parts.extend([0.0] * bins * 3)
            continue
        for ch in range(3):
            hi = cv2.calcHist(
                [region], [ch], None, [bins],
                [0, 180] if ch == 0 else [0, 256],
            )
            hi = hi.flatten().astype(np.float32)
            hi /= hi.sum() + 1e-8
            parts.extend(hi.tolist())

    return np.array(parts, dtype=np.float32)


# ── Subset builder ───────────────────────────────────────────────────

def _read_split_local(directory: str):
    """Lightweight reader that avoids importing the full data package."""
    import re
    _RE = re.compile(r"(-?\d+)_c(\d+)s\d+_\d+_\d+\.jpg")
    samples = []
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".jpg"):
            continue
        m = _RE.match(fname)
        if m is None:
            continue
        pid = int(m.group(1))
        camid = int(m.group(2)) - 1
        if pid < 0:
            continue
        samples.append((os.path.join(directory, fname), pid, camid))
    return samples


def _run_kmeans(X: np.ndarray, n_clusters: int, seed: int = 42) -> np.ndarray:
    """Run KMeans with sklearn if available, or pure-numpy Lloyd fallback."""
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        return km.fit_predict(X)
    except Exception:
        # Pure-numpy Lloyd's algorithm fallback
        rng = np.random.RandomState(seed)
        n_clusters = min(n_clusters, len(X))
        centers = X[rng.choice(len(X), n_clusters, replace=False)].copy()
        labels = np.zeros(len(X), dtype=int)
        for _ in range(50):
            # Compute squared euclidean distances [N, K]
            dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
            new_labels = np.argmin(dists, axis=-1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for k in range(n_clusters):
                mask = labels == k
                if mask.any():
                    centers[k] = X[mask].mean(axis=0)
        return labels


def build_low_variance_subset(
    root: str,
    known_pids: Optional[Set[int]] = None,
    n_clusters: int = 10,
    seed: int = 42,
) -> Tuple[Dict[int, list], List[Tuple[Tuple[str, int], Tuple[str, int]]]]:
    """Cluster training images by clothing and build same-clothing impostor pairs.

    Args:
        root:        Dataset root (must contain ``bounding_box_train/``).
        known_pids:  If given, only include these person IDs.
        n_clusters:  Number of KMeans clothing clusters.
        seed:        Random seed for reproducibility.

    Returns:
        clusters:       ``{cluster_id: [(path, pid, camid), ...]}``
        impostor_pairs: ``[((path_a, pid_a), (path_b, pid_b)), ...]``
                        where ``pid_a != pid_b`` but same clothing cluster.
    """
    train_dir = os.path.join(root, "bounding_box_train")
    samples = _read_split_local(train_dir)

    if known_pids is not None:
        samples = [(p, pid, c) for p, pid, c in samples if pid in known_pids]

    print(f"[low-var] Extracting colour histograms for {len(samples)} images ...")
    features, valid_samples = [], []
    for path, pid, camid in samples:
        hist = extract_color_histogram(path)
        if hist is not None:
            features.append(hist)
            valid_samples.append((path, pid, camid))

    X = np.stack(features)
    n_clusters = min(n_clusters, len(X))

    print(f"[low-var] KMeans clustering into {n_clusters} clothing groups ...")
    labels = _run_kmeans(X, n_clusters=n_clusters, seed=seed)

    clusters: Dict[int, list] = defaultdict(list)
    for i, (path, pid, camid) in enumerate(valid_samples):
        clusters[int(labels[i])].append((path, pid, camid))

    # Build same-clothing impostor pairs (different ID, same cluster)
    impostor_pairs: List[Tuple[Tuple[str, int], Tuple[str, int]]] = []
    for members in clusters.values():
        pid_groups: Dict[int, List[str]] = defaultdict(list)
        for path, pid, _ in members:
            pid_groups[pid].append(path)

        pids = list(pid_groups.keys())
        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                impostor_pairs.append((
                    (pid_groups[pids[i]][0], pids[i]),
                    (pid_groups[pids[j]][0], pids[j]),
                ))

    print(
        f"[low-var] {len(impostor_pairs)} same-clothing impostor pairs "
        f"across {n_clusters} clusters"
    )
    return dict(clusters), impostor_pairs


# ── CLI ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "./datasets/market1501"
    clusters, pairs = build_low_variance_subset(root)
    for cid, members in sorted(clusters.items()):
        pids = {pid for _, pid, _ in members}
        print(f"  Cluster {cid:>2d}: {len(members):>5d} images, "
              f"{len(pids):>3d} identities")
    print(f"\n  Total impostor pairs: {len(pairs)}")
