"""Retrieval-augmented training on engine — the port of
``nueronce.retrieval_train`` (needs PyTorch) for :class:`NueronceModel`.

Same task as the PyTorch version: a fresh, random ``key -> value`` fact per
example, so the value is recoverable *only* by retrieving the matching fact
document, not by memorizing weights. Gives the same with-vs-without ablation.
The retriever (``nueronce.impl.embed_text`` + ``nueronce.ops.cosine``) is frozen and
torch-free already, so it's reused directly rather than re-implemented.
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from ..impl import embed_text
from ..ops import cosine
from .optim import AdamW, clip_grad_norm_

_LETTERS = string.ascii_lowercase
_DIGITS = "0123456789"

_PREFIX = "lookup "
_MID = " = "
KEY_LEN = 3
SEQ_TEMPLATE_LEN = len(_PREFIX) + KEY_LEN + len(_MID) + 1 + 1
VALUE_POS = len(_PREFIX) + KEY_LEN + len(_MID)


def _rand_key(rng: np.random.Generator) -> str:
    return "".join(rng.choice(list(_LETTERS), size=KEY_LEN))


def _rand_value(rng: np.random.Generator) -> str:
    return _DIGITS[int(rng.integers(0, len(_DIGITS)))]


def fact_doc(key: str, value: str) -> str:
    return f"{_PREFIX}{key}{_MID}{value}"


def lookup_seq(key: str, value: str) -> str:
    return f"{_PREFIX}{key}{_MID}{value}\n"


def query_prefix(key: str) -> str:
    return f"{_PREFIX}{key}{_MID}"


@dataclass
class RetrievalStore:
    """In-memory store of fact docs with frozen embeddings + real top-k search."""

    dim: int = 256

    def __post_init__(self):
        self.texts: List[str] = []
        self.embs: List[np.ndarray] = []

    def add(self, text: str) -> int:
        self.texts.append(text)
        self.embs.append(embed_text(text, self.dim))
        return len(self.texts) - 1

    def retrieve(self, query: str, k: int) -> List[int]:
        qe = embed_text(query, self.dim)
        scores = [cosine(qe, e) for e in self.embs]
        return list(np.argsort(scores)[::-1][:k])


def make_retrieval_batch(rng: np.random.Generator, batch: int = 32, k: int = 3,
                         lc: int = 16, dim: int = 256) -> Dict:
    """Build a batch of fresh fact lookups + their retrieved neighbor docs."""
    keys, values = [], []
    store = RetrievalStore(dim=dim)
    for _ in range(batch):
        key, val = _rand_key(rng), _rand_value(rng)
        keys.append(key); values.append(val)
        store.add(fact_doc(key, val))

    T = SEQ_TEMPLATE_LEN
    seq = np.zeros((batch, T), dtype=np.int64)
    value_mask = np.zeros((batch, T), dtype=bool)
    neighbor_ids = np.zeros((batch, k, lc), dtype=np.int64)
    neighbor_mask = np.zeros((batch, k, lc), dtype=bool)
    recall_hits = 0

    for i in range(batch):
        s = lookup_seq(keys[i], values[i]).encode("utf-8")
        seq[i, : len(s)] = np.frombuffer(s, dtype=np.uint8)
        value_mask[i, VALUE_POS] = True

        topk = store.retrieve(query_prefix(keys[i]), k)
        if i in topk:
            recall_hits += 1
        order = [i] + [j for j in topk if j != i]
        order = order[:k]
        for slot, j in enumerate(order):
            db = store.texts[j].encode("utf-8")[:lc]
            neighbor_ids[i, slot, : len(db)] = np.frombuffer(db, dtype=np.uint8)
            neighbor_mask[i, slot, : len(db)] = True

    return {
        "seq_ids": seq, "value_mask": value_mask,
        "neighbor_ids": neighbor_ids, "neighbor_mask": neighbor_mask,
        "recall_at_k": recall_hits / batch,
    }


def _value_accuracy(logits, seq_ids: np.ndarray) -> float:
    pred = logits.data[:, VALUE_POS - 1].argmax(-1)   # logits at v-1 predict byte v
    tgt = seq_ids[:, VALUE_POS]
    return float((pred == tgt).mean())


def value_metrics(model, batch: Dict) -> Dict[str, float]:
    """Value-token loss and exact-match accuracy, with vs without retrieval."""
    from .tensor import no_grad
    with no_grad():
        lw_logits, _ = model.forward(batch["seq_ids"], batch["neighbor_ids"], batch["neighbor_mask"])
        lo_logits, _ = model.forward(batch["seq_ids"])
        return {
            "loss_with": model.masked_token_loss(lw_logits, batch["seq_ids"], batch["value_mask"]).item(),
            "loss_without": model.masked_token_loss(lo_logits, batch["seq_ids"], batch["value_mask"]).item(),
            "acc_with": _value_accuracy(lw_logits, batch["seq_ids"]),
            "acc_without": _value_accuracy(lo_logits, batch["seq_ids"]),
        }


def value_losses(model, batch: Dict) -> Tuple[float, float]:
    m = value_metrics(model, batch)
    return m["loss_with"], m["loss_without"]


def train_retrieval(model, steps: int = 600, batch: int = 32, k: int = 4, lr: float = 3e-3,
                    seed: int = 0, log_every: int = 50) -> List[Dict]:
    rng = np.random.default_rng(seed)
    opt = AdamW(list(model.parameters()), lr=lr, weight_decay=0.01)
    history = []
    for step in range(steps):
        b = make_retrieval_batch(rng, batch=batch, k=k)
        logits, _ = model.forward(b["seq_ids"], b["neighbor_ids"], b["neighbor_mask"])
        loss = model.masked_token_loss(logits, b["seq_ids"], b["value_mask"])
        model.zero_grad(); loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            mtr = value_metrics(model, b)
            history.append({"step": step, "train_value_loss": loss.item(),
                            "val_with_retrieval": mtr["loss_with"], "val_without_retrieval": mtr["loss_without"],
                            "acc_with": mtr["acc_with"], "acc_without": mtr["acc_without"],
                            "recall_at_k": b["recall_at_k"]})
    return history


__all__ = [
    "RetrievalStore", "make_retrieval_batch", "value_losses", "train_retrieval",
    "fact_doc", "lookup_seq", "query_prefix", "VALUE_POS", "SEQ_TEMPLATE_LEN",
]
