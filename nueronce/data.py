"""Tiny byte-level corpus + batching for the NUERONCE training demo.

The corpus is small and structured on purpose: the point of the demo is to show
the hand-built architecture *learns* (loss falls far below the uniform-byte
baseline and the model can continue the text), not to train a general model.
"""

from __future__ import annotations

from typing import List

import torch
from torch import Tensor

# A small, self-contained corpus drawn from the project's own design vocabulary.
CORPUS = (
    "NUERONCE separates understanding, thinking, remembering, and speaking. "
    "Perception forms dynamic information units from raw bytes. "
    "Typed memory keeps semantic, evidence, and authority channels apart. "
    "The hybrid core mixes a selective state space path with local attention. "
    "A planner decides what to say before the renderer decides how to word it. "
    "An independent verifier checks claims against evidence and revises. "
    "Retrieval is dense, sparse, and late interaction together. "
    "Provenance and authority are first class, not an afterthought. "
)


def corpus_bytes(repeat: int = 8) -> bytes:
    return (CORPUS * repeat).encode("utf-8")


# A larger, *non-repeating* corpus of distinct sentences (prose + code-ish text),
# used for the held-out generalization harness. Train and validation come from
# disjoint byte regions so val bits/byte measures generalization, not memorization.
LARGER_CORPUS = (
    "A foundation model learns reusable representations that transfer across tasks. "
    "Bytes are the smallest universal unit, so a byte model needs no fixed vocabulary. "
    "Dynamic patching spends more compute on hard spans and less on predictable ones. "
    "A selective state space layer carries information forward in linear time. "
    "Attention compares every position to every other and is exact but costly. "
    "Retrieval moves rare facts out of the weights and into an external store. "
    "Typed memory keeps goals, evidence, and authority in separate channels. "
    "A verifier checks each claim against the evidence before the answer is shown. "
    "Calibration means the stated confidence matches how often the model is right. "
    "Provenance records where every fact came from and whether it may be trusted. "
    "def add(a, b):\n    return a + b\n"
    "for i in range(10):\n    total += i\n"
    "class Cache:\n    def get(self, key):\n        return self.store.get(key)\n"
    "The quick brown fox jumps over the lazy dog near the river bank at dawn. "
    "Photosynthesis converts light, water, and carbon dioxide into sugar and oxygen. "
    "The treaty was signed in spring and the borders were redrawn that summer. "
    "A gradient points in the direction of steepest increase of a function. "
    "Entropy measures how surprising a distribution is, in bits per symbol. "
    "When the tests pass and the diff is small, the change is usually safe to ship. "
    "Long contexts stress the cache, so streaming inference keeps a bounded state. "
    "Sparse attention should compute sparsely, not compute densely and then mask. "
    "A held out set never seen during training reveals whether learning generalized. "
    "Repository repair turns an issue into a patch, then a test decides if it worked. "
    "Curiosity and care, applied patiently, compound into understanding over time. "
    "The river carried the small boat past the old stone bridge toward the sea. "
    "Every morning the baker opened the shop and the warm smell filled the street. "
    "Numbers that cannot be written as a fraction are called irrational numbers. "
    "The engine cooled slowly while the mechanic checked the oil and the belts. "
    "A good explanation makes a hard idea feel simple without leaving things out. "
    "Birds migrate south in autumn and return north when the warm weather comes. "
    "The library kept old maps that showed roads that no longer existed at all. "
    "Water expands when it freezes, which is why ice floats on the colder lake. "
    "She tuned the guitar by ear, then played a quiet song about the long road home. "
    "The committee read the report twice before they agreed on the final wording. "
    "Light from distant stars left them long before anyone here was alive to see it. "
    "He measured the board twice, marked the line, and cut it once with a steady hand. "
    "The garden needed rain, so the dry soil cracked along the edges of the beds. "
    "A promise kept in private is worth more than a loud promise made in public. "
    "The students compared their answers and found the same mistake in every one. "
    "Salt was once so valuable that roads and cities grew up around the salt trade. "
    "The clock in the tower stopped at noon and no one remembered how to fix it. "
    "When the wind dropped, the sailors rowed until the harbor lights came into view. "
    "A map is useful only when you also know where you are standing on it right now. "
    "The old teacher believed that a patient question taught more than a quick answer. "
    "Rust forms when iron meets water and air over many quiet and unhurried days. "
    "The market was loud with sellers calling prices for fish, bread, and bright cloth. "
    "A small leak, ignored for a season, can bring down a wall that stood for years. "
    "They followed the narrow path uphill until the whole valley opened below them. "
    "Honest measurement is the first duty of anyone who wants to learn the truth. "
    "The letter arrived late, but the news it carried was still warm and welcome. "
    "Bees tell each other where flowers are by dancing in tight and careful circles. "
    "The bridge swayed a little in the wind, yet it had carried traffic for a century. "
    "Practice turns a clumsy motion into a smooth one without you noticing the change. "
    "The court heard both sides, weighed the evidence, and ruled before the sun set. "
    "A seed holds a whole tree inside it, waiting for water, warmth, and a little time. "
    "The printer jammed on the last page, as printers seem to prefer to do. "
    "Mountains rise where plates collide and wear down again over unimaginable time. "
    "He saved a little each month, and after years the small sums had become enough. "
)


def larger_corpus_bytes() -> bytes:
    return LARGER_CORPUS.encode("utf-8")


def train_val_split(data: bytes, val_frac: float = 0.25) -> tuple:
    """Split a byte stream into disjoint train / validation regions."""
    cut = int(len(data) * (1.0 - val_frac))
    return data[:cut], data[cut:]


def make_batches(data: bytes, seq_len: int, batch_size: int, n_batches: int,
                 seed: int = 0, device=None) -> List[Tensor]:
    """Random contiguous windows of length ``seq_len`` as byte-id tensors."""
    g = torch.Generator().manual_seed(seed)
    buf = torch.tensor(list(data), dtype=torch.long)
    hi = len(buf) - seq_len - 1
    if hi <= 0:
        raise ValueError("corpus too short for seq_len; increase repeat or shorten seq_len")
    batches = []
    for _ in range(n_batches):
        starts = torch.randint(0, hi, (batch_size,), generator=g)
        rows = torch.stack([buf[s : s + seq_len] for s in starts])
        batches.append(rows.to(device) if device else rows)
    return batches


UNIFORM_BYTE_BPB = 8.0  # log2(256): the no-skill baseline in bits/byte


__all__ = ["CORPUS", "corpus_bytes", "make_batches", "UNIFORM_BYTE_BPB"]
