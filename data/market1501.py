"""Market-1501 dataset with open-set identity splits.

File-name convention::

    PPPP_cCsS_NNNNNN_DD.jpg
    │     │ │  │       └─ detection index
    │     │ │  └──────── sequence frame number
    │     │ └─────────── sequence index
    │     └───────────── camera id (1-based in filename)
    └─────────────────── person id  (-1 = junk, 0 = distractor)
"""
from __future__ import annotations

import os
import re
import random
from collections import defaultdict
from typing import Dict, Optional, Set, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ── Filename parsing ─────────────────────────────────────────────────

_FNAME_RE = re.compile(r"(-?\d+)_c(\d+)s\d+_\d+_\d+\.jpg")


def parse_market_filename(fname: str) -> Tuple[Optional[int], Optional[int]]:
    """Return ``(pid, camid)`` or ``(None, None)`` for unparseable names."""
    m = _FNAME_RE.match(fname)
    if m is None:
        return None, None
    return int(m.group(1)), int(m.group(2)) - 1          # camid 0-based


def read_split(directory: str):
    """Read every valid .jpg in *directory*.

    Returns:
        list of ``(abs_path, pid, camid)``
    """
    samples = []
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".jpg"):
            continue
        pid, camid = parse_market_filename(fname)
        if pid is None or pid < 0:                       # skip junk / bad names
            continue
        samples.append((os.path.join(directory, fname), pid, camid))
    return samples


# ── Open-set identity split ──────────────────────────────────────────

def split_known_unknown(
    root: str,
    unknown_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[Set[int], Set[int]]:
    """Partition *training* person-IDs into known and unknown sets.

    ``unknown_ratio`` of IDs are held out as impostors; they are never
    seen during metric-learning training.
    """
    train_dir = os.path.join(root, "bounding_box_train")
    samples = read_split(train_dir)
    all_pids = sorted({pid for _, pid, _ in samples})

    rng = random.Random(seed)
    rng.shuffle(all_pids)

    n_unknown = max(1, int(len(all_pids) * unknown_ratio))
    unknown_pids = set(all_pids[:n_unknown])
    known_pids   = set(all_pids[n_unknown:])
    return known_pids, unknown_pids


# ── Transforms ───────────────────────────────────────────────────────

def get_transforms(
    height: int = 256,
    width: int = 128,
    is_train: bool = True,
) -> transforms.Compose:
    """Standard re-id augmentation pipeline."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((height, width)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(10),
            transforms.RandomCrop((height, width)),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.15, saturation=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.4)),
        ])
    return transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ── Dataset ──────────────────────────────────────────────────────────

class Market1501(Dataset):
    """Market-1501 dataset that supports open-set train / eval splits.

    Args:
        root:        Path to dataset root (contains ``bounding_box_train/`` etc.)
        split:       ``'train'``, ``'query'``, or ``'gallery'``.
        known_pids:  Set of person IDs to *keep*.  If ``None`` all IDs pass.
        unknown_pids: Stored for later reference (evaluation).
        transform:   Torchvision transform pipeline.
        relabel:     Remap kept PIDs to contiguous ``[0, N)``.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        known_pids: Optional[Set[int]] = None,
        unknown_pids: Optional[Set[int]] = None,
        transform=None,
        relabel: bool = True,
    ):
        self.root = root
        self.split = split
        self.transform = transform
        self.relabel = relabel
        self.known_pids = known_pids
        self.unknown_pids = unknown_pids

        dir_map = {
            "train":   "bounding_box_train",
            "query":   "query",
            "gallery": "bounding_box_test",
        }
        if split not in dir_map:
            raise ValueError(f"Unknown split '{split}'")
        directory = os.path.join(root, dir_map[split])

        all_samples = read_split(directory)

        # Filter by identity set
        if known_pids is not None:
            all_samples = [
                (p, pid, c) for p, pid, c in all_samples if pid in known_pids
            ]

        # Build contiguous label map
        self.pid2label: Dict[int, int] = {}
        if relabel:
            unique_pids = sorted({pid for _, pid, _ in all_samples})
            self.pid2label = {pid: idx for idx, pid in enumerate(unique_pids)}

        self.samples = all_samples
        self.num_classes = len(self.pid2label) if self.pid2label else len(
            {pid for _, pid, _ in all_samples}
        )

    # ── sequence protocol ────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, pid, camid = self.samples[index]
        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        label = self.pid2label.get(pid, -1) if self.pid2label else pid
        return img, label, camid, path
