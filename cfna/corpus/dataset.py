"""Turn the built corpus into training batches.

Key fairness rule from the build plan: *no single document dominates*. Instead of
sampling windows proportional to document length (which would let Moby-Dick or the
KJV Bible swamp the model), we sample a **document uniformly at random**, then a
window inside it. So every book contributes roughly equally regardless of length.

Train/val split is *by document* (whole held-out books), so validation bits/byte
measures generalization to unseen documents, not memorized windows.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from .build import load_manifest


class ByteCorpus:
    def __init__(self, corpus_dir: Path, split: str = "train"):
        self.dir = Path(corpus_dir)
        self.manifest = [m for m in load_manifest(self.dir) if m["split"] == split]
        if not self.manifest:
            raise ValueError(f"no documents with split={split!r} in {corpus_dir}")
        self.docs: List[np.ndarray] = []
        self.titles: List[str] = []
        for m in self.manifest:
            data = (self.dir / m["path"]).read_bytes()
            self.docs.append(np.frombuffer(data, dtype=np.uint8))
            self.titles.append(m["title"])
        self.total_bytes = int(sum(len(d) for d in self.docs))

    def sample_batch(self, seq_len: int, batch_size: int, rng: np.random.Generator):
        """Document-uniform sampling: pick a doc, then a window inside it."""
        rows = []
        eligible = [i for i, d in enumerate(self.docs) if len(d) > seq_len + 1]
        for _ in range(batch_size):
            di = eligible[int(rng.integers(0, len(eligible)))]
            d = self.docs[di]
            s = int(rng.integers(0, len(d) - seq_len - 1))
            rows.append(d[s:s + seq_len].astype(np.int64))
        return np.stack(rows)

    def iter_val_windows(self, seq_len: int, stride: Optional[int] = None):
        """Deterministic non-overlapping windows for validation bits/byte."""
        stride = stride or seq_len
        for d in self.docs:
            for s in range(0, len(d) - seq_len - 1, stride):
                yield d[s:s + seq_len].astype(np.int64)


def make_torch_batches(corpus: ByteCorpus, seq_len: int, batch_size: int,
                       n_batches: int, seed: int = 0, device=None):
    import torch

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_batches):
        arr = corpus.sample_batch(seq_len, batch_size, rng)
        t = torch.from_numpy(arr)
        out.append(t.to(device) if device else t)
    return out


def val_batches(corpus: ByteCorpus, seq_len: int, batch_size: int, max_batches: int = 16,
                device=None):
    import torch

    wins = list(corpus.iter_val_windows(seq_len))
    if not wins:
        return []
    rng = np.random.default_rng(123)
    rng.shuffle(wins)
    batches = []
    for i in range(0, min(len(wins), max_batches * batch_size), batch_size):
        chunk = wins[i:i + batch_size]
        if len(chunk) < batch_size:
            break
        t = torch.from_numpy(np.stack(chunk))
        batches.append(t.to(device) if device else t)
    return batches


__all__ = ["ByteCorpus", "make_torch_batches", "val_batches"]
