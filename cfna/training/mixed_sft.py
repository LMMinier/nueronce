"""Mixed conversational SFT dataset: rendering, register caps, and batching.

Torch-free on purpose (NumPy only), same split as the rest of ``cfna.training``:
the data/batch logic lives here where it can be unit-tested on any box, and the
thin PyTorch loop in ``scripts/train_conversation.py`` just wraps these arrays
in tensors on the machine that actually has the compute.

Record schema is ``cfna.training.dataset_prep``'s ``messages`` shape, with
three optional extra fields carried by prompt-aligned records::

    {"id": ..., "messages": [...], "source": ..., "category": ...,
     "system_message": str, "trusted_evidence": [str], "response_plan": [str]}

Records with evidence/plan/system extras render through
``prompting.format_training_example`` (single-turn, full canonical block
layout); plain records render through ``dialogue_data.encode_messages``
(multi-turn capable). Both paths produce the same canonical marker format and
a response-only target mask — see ``docs/FORMAT.md``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np

from ..prompting import format_training_example
from .dialogue_data import encode_messages


def render_record(rec: dict, *, system_default: str = "") -> Tuple[bytes, List[bool]]:
    """Render one record to ``(utf8_bytes, response_mask)`` in the canonical
    format. ``mask[i]`` is True iff byte ``i`` is an SFT loss target."""
    if rec.get("trusted_evidence") or rec.get("response_plan") or rec.get("system_message"):
        msgs = rec["messages"]
        if sum(1 for m in msgs if m["role"] == "user") != 1:
            raise ValueError(f"evidence/plan records must be single-turn: {rec.get('id')}")
        return format_training_example(
            system_message=rec.get("system_message", system_default),
            user_request=msgs[0]["content"],
            assistant_response=msgs[-1]["content"],
            trusted_evidence="\n".join(rec.get("trusted_evidence", [])),
            response_plan="\n".join(rec.get("response_plan", [])),
        )
    return encode_messages(rec["messages"], system=system_default)


def load_jsonl(paths: Sequence[str | Path]) -> List[dict]:
    records: List[dict] = []
    for p in paths:
        with Path(p).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def enforce_category_caps(records: Sequence[dict], *, cap_frac: float = 0.25,
                          seed: int = 0) -> List[dict]:
    """Downsample over-represented categories until no category exceeds
    ``cap_frac`` of the *final* total (the 77%-arithmetic poisoning lesson).

    Deterministic given ``seed``; drops are a seeded uniform sample of the
    offending category, never a prefix (generators enumerate templates in
    order, so a prefix would be a skewed slice)."""
    by_cat: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r)
    counts = {c: len(v) for c, v in by_cat.items()}
    while True:
        total = sum(counts.values())
        over = {c: n for c, n in counts.items() if n > cap_frac * total}
        if not over:
            break
        # Shrink the worst offender to exactly the cap given the others fixed:
        # n_c = cap * (T_other + n_c)  =>  n_c = cap * T_other / (1 - cap)
        c = max(over, key=lambda k: over[k])
        t_other = total - counts[c]
        counts[c] = int(np.floor(cap_frac * t_other / (1.0 - cap_frac)))
    rng = np.random.default_rng(seed)
    out: List[dict] = []
    for c in sorted(by_cat):
        rows = by_cat[c]
        if counts[c] < len(rows):
            keep = sorted(rng.permutation(len(rows))[: counts[c]])
            out.extend(rows[i] for i in keep)
        else:
            out.extend(rows)
    return out


def stride_sample(it: Iterable[dict], quota: int, total_hint: int) -> Iterator[dict]:
    """Take ~``quota`` records spread uniformly across an enumerated stream
    (not a prefix — see ``enforce_category_caps``)."""
    stride = max(1, total_hint // max(1, quota))
    taken = 0
    for i, rec in enumerate(it):
        if i % stride == 0 and taken < quota:
            taken += 1
            yield rec


def build_batches(records: Sequence[dict], *, batch_size: int = 8, max_len: int = 320,
                  seed: int = 0, loss: str = "response",
                  system_default: str = "") -> Iterator[Dict[str, np.ndarray]]:
    """Length-bucketed training batches: ``byte_ids`` [B,T] int64 and
    ``target_mask`` [B,T] bool. ``loss="response"`` masks to response bytes
    (the SFT contract); ``loss="full"`` targets every real byte (dialogue
    pretraining from scratch) — padding is never a target in either mode.
    Records rendering longer than ``max_len`` bytes are skipped, not truncated
    (a truncated response teaches the model to stop mid-sentence)."""
    if loss not in ("response", "full"):
        raise ValueError(f"loss must be 'response' or 'full', got {loss!r}")
    rendered: List[Tuple[bytes, List[bool]]] = []
    for rec in records:
        b, m = render_record(rec, system_default=system_default)
        if len(b) <= max_len:
            rendered.append((b, m))
    if not rendered:
        return
    order = np.argsort([len(b) for b, _ in rendered], kind="stable")
    groups = [order[i: i + batch_size] for i in range(0, len(order), batch_size)]
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    for idx in groups:
        rows = [rendered[i] for i in idx]
        t = max(len(b) for b, _ in rows)
        byte_ids = np.zeros((len(rows), t), dtype=np.int64)
        target_mask = np.zeros((len(rows), t), dtype=bool)
        for j, (b, m) in enumerate(rows):
            byte_ids[j, : len(b)] = np.frombuffer(b, dtype=np.uint8)
            target_mask[j, : len(m)] = m if loss == "response" else True
        yield {"byte_ids": byte_ids, "target_mask": target_mask}


def dataset_summary(records: Sequence[dict]) -> Dict[str, object]:
    cats: Dict[str, int] = defaultdict(int)
    for r in records:
        cats[r["category"]] += 1
    total = len(records)
    return {
        "total": total,
        "category_counts": dict(sorted(cats.items(), key=lambda kv: -kv[1])),
        "max_category_frac": (max(cats.values()) / total) if total else 0.0,
    }


__all__ = [
    "render_record", "load_jsonl", "enforce_category_caps", "stride_sample",
    "build_batches", "dataset_summary",
]
