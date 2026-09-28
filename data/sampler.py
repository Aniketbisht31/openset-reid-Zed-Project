"""PK sampler for batch-hard triplet mining.

Each mini-batch contains *P* randomly chosen identities, each represented
by *K* randomly sampled images.  This guarantees that every anchor has at
least *K − 1* positive pairs within the batch.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterator, List

from torch.utils.data import Sampler


class PKSampler(Sampler[int]):
    """Yield indices so that consecutive ``P × K`` indices form one PK batch.

    Works with ``torch.utils.data.DataLoader(batch_size=P*K, sampler=…)``.

    Args:
        data_source: :class:`Market1501` dataset (needs ``.samples`` and
                     ``.pid2label``).
        p:           Number of identities per batch.
        k:           Number of images per identity per batch.
    """

    def __init__(self, data_source, p: int = 16, k: int = 4):
        super().__init__()
        self.p = p
        self.k = k

        # Group dataset indices by label
        self.pid_to_indices: dict[int, List[int]] = defaultdict(list)
        for idx, (_, pid, _) in enumerate(data_source.samples):
            label = data_source.pid2label.get(pid, pid)
            self.pid_to_indices[label].append(idx)

        self.pids = list(self.pid_to_indices.keys())
        assert len(self.pids) >= p, (
            f"Need at least P={p} identities but dataset has {len(self.pids)}"
        )

        self.batch_size = p * k
        # Number of complete batches per epoch
        self._n_batches = max(1, len(self.pids) // self.p)

    def __iter__(self) -> Iterator[int]:
        pid_pool = self.pids.copy()
        random.shuffle(pid_pool)
        ptr = 0

        for _ in range(self._n_batches):
            # Refill pool if exhausted
            if ptr + self.p > len(pid_pool):
                pid_pool = self.pids.copy()
                random.shuffle(pid_pool)
                ptr = 0

            selected_pids = pid_pool[ptr : ptr + self.p]
            ptr += self.p

            for pid in selected_pids:
                indices = self.pid_to_indices[pid]
                if len(indices) >= self.k:
                    chosen = random.sample(indices, self.k)
                else:
                    chosen = random.choices(indices, k=self.k)
                yield from chosen

    def __len__(self) -> int:
        return self._n_batches * self.batch_size
