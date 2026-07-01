"""Large-scale dialogue dataset preparation: validate, dedupe, split, shard.

Streaming by design — no stage holds the full corpus text in memory at once.
Pass 1 (:func:`build_clean_dataset`) reads records one at a time, validates
them, drops exact and near-duplicates, and streams survivors straight to a
single ``clean.jsonl`` file. Pass 2 (:func:`split_and_shard`) records only
byte offsets into that file (a few bytes per record, not the record itself),
deterministically shuffles those offsets, and streams each destination file
(validation/test/shard-N) by seeking + reading one line at a time.

Schema (see ``cfna.training.synthetic_dialogue``):
    {"id": ..., "messages": [{"role": "user"/"assistant", "content": ...}, ...],
     "source": ..., "category": ...}
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

VALID_ROLES = {"user", "assistant"}
_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize_text(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — used only for
    near-duplicate detection, never for the text actually written out."""
    s = s.lower()
    s = _PUNCT_RE.sub("", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def record_text_hash(record: dict) -> str:
    """Exact-duplicate key: the verbatim role+content sequence."""
    parts = [f"{m.get('role')}:{m.get('content')}" for m in record.get("messages", [])]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8", "replace")).hexdigest()


def record_normalized_hash(record: dict) -> str:
    """Near-duplicate key: normalized text, catching case/punctuation-only
    variants that ``record_text_hash`` would treat as distinct."""
    parts = [f"{m.get('role')}:{normalize_text(m.get('content', ''))}" for m in record.get("messages", [])]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def record_pair_hash(record: dict) -> str:
    """"Duplicate prompt-and-response pair" key: all user turns joined vs all
    assistant turns joined (for single-turn records this *is* the prompt and
    the response)."""
    msgs = record.get("messages", [])
    user_text = " ".join(m["content"] for m in msgs if m.get("role") == "user")
    asst_text = " ".join(m["content"] for m in msgs if m.get("role") == "assistant")
    return hashlib.sha256(f"{normalize_text(user_text)}\x1e{normalize_text(asst_text)}".encode("utf-8")).hexdigest()


def validate_record(record: object) -> Optional[str]:
    """Return None if ``record`` is a well-formed record, else a short reason
    string explaining why it was rejected."""
    if not isinstance(record, dict):
        return "not a JSON object"
    for key in ("id", "messages", "source", "category"):
        if key not in record:
            return f"missing field: {key}"
    if not isinstance(record["id"], str) or not record["id"]:
        return "empty or non-string id"
    msgs = record["messages"]
    if not isinstance(msgs, list) or not msgs:
        return "empty or non-list messages"
    has_user = has_assistant = False
    for m in msgs:
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            return "malformed message (missing role/content)"
        if m["role"] not in VALID_ROLES:
            return f"invalid role: {m['role']!r}"
        if not isinstance(m["content"], str) or not m["content"].strip():
            return "empty message content"
        has_user = has_user or m["role"] == "user"
        has_assistant = has_assistant or m["role"] == "assistant"
    if not (has_user and has_assistant):
        return "missing a user or assistant turn"
    if msgs[-1]["role"] != "assistant":
        return "conversation must end on an assistant turn (nothing to train on otherwise)"
    return None


@dataclass
class CleanStats:
    seen: int = 0
    accepted: int = 0
    rejected_invalid: int = 0
    rejected_exact_dup: int = 0
    rejected_near_dup: int = 0
    rejected_pair_dup: int = 0
    rejection_reasons: Counter = field(default_factory=Counter)
    category_counts: Counter = field(default_factory=Counter)


def build_clean_dataset(records: Iterable[dict], out_path: str) -> CleanStats:
    """Pass 1: validate + dedupe ``records``, streaming survivors to
    ``out_path`` as JSONL. Never holds more than one record's text in memory
    at a time; only the (small) hash sets accumulate across the whole pass."""
    stats = CleanStats()
    seen_exact, seen_norm, seen_pair = set(), set(), set()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for record in records:
            stats.seen += 1
            reason = validate_record(record)
            if reason is not None:
                stats.rejected_invalid += 1
                stats.rejection_reasons[reason] += 1
                continue
            h_exact = record_text_hash(record)
            if h_exact in seen_exact:
                stats.rejected_exact_dup += 1
                continue
            h_norm = record_normalized_hash(record)
            if h_norm in seen_norm:
                stats.rejected_near_dup += 1
                continue
            h_pair = record_pair_hash(record)
            if h_pair in seen_pair:
                stats.rejected_pair_dup += 1
                continue
            seen_exact.add(h_exact)
            seen_norm.add(h_norm)
            seen_pair.add(h_pair)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats.accepted += 1
            stats.category_counts[record["category"]] += 1
    return stats


@dataclass
class SplitResult:
    train_shard_paths: List[str]
    val_path: str
    test_path: str
    n_train: int
    n_val: int
    n_test: int
    train_category_counts: Dict[str, int]
    val_category_counts: Dict[str, int]
    test_category_counts: Dict[str, int]
    seed: int


def _index_lines(path: str) -> List[int]:
    """One streaming pass recording each line's byte offset (cheap: one int
    per record, not the record itself)."""
    offsets = []
    with open(path, "rb") as f:
        pos = f.tell()
        for line in f:
            if line.strip():
                offsets.append(pos)
            pos = f.tell()
    return offsets


def _read_at(f, offset: int) -> dict:
    f.seek(offset)
    return json.loads(f.readline())


def split_and_shard(clean_path: str, out_dir: str, *, num_shards: int = 10,
                    examples_per_shard: int = 10_000, val_size: int = 5_000,
                    test_size: int = 5_000, seed: int = 42) -> SplitResult:
    """Pass 2: deterministic seeded shuffle over line *offsets* (not text),
    then stream validation/test/shard files by seek+readline. Val and test are
    carved out first, so train shards are guaranteed disjoint from both."""
    offsets = _index_lines(clean_path)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(offsets))

    need = val_size + test_size + num_shards * examples_per_shard
    if len(offsets) < need:
        raise ValueError(
            f"clean dataset has {len(offsets)} records, need at least {need} "
            f"({val_size} val + {test_size} test + {num_shards}x{examples_per_shard} train)"
        )

    val_idx = order[:val_size]
    test_idx = order[val_size:val_size + test_size]
    train_idx = order[val_size + test_size: val_size + test_size + num_shards * examples_per_shard]

    out = Path(out_dir)
    (out / "train_shards").mkdir(parents=True, exist_ok=True)

    def _write(idxs, path) -> Dict[str, int]:
        cats: Counter = Counter()
        with open(clean_path, "r", encoding="utf-8") as src, open(path, "w", encoding="utf-8") as dst:
            for idx in idxs:
                rec = _read_at(src, offsets[idx])
                cats[rec["category"]] += 1
                dst.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return dict(cats)

    val_path = str(out / "validation.jsonl")
    test_path = str(out / "test.jsonl")
    val_cats = _write(val_idx, val_path)
    test_cats = _write(test_idx, test_path)

    shard_paths = []
    train_cats: Counter = Counter()
    for s in range(num_shards):
        shard_idx = train_idx[s * examples_per_shard:(s + 1) * examples_per_shard]
        shard_path = str(out / "train_shards" / f"shard_{s + 1:02d}.jsonl")
        cats = _write(shard_idx, shard_path)
        for k, v in cats.items():
            train_cats[k] += v
        shard_paths.append(shard_path)

    return SplitResult(
        train_shard_paths=shard_paths, val_path=val_path, test_path=test_path,
        n_train=len(train_idx), n_val=len(val_idx), n_test=len(test_idx),
        train_category_counts=dict(train_cats), val_category_counts=val_cats,
        test_category_counts=test_cats, seed=seed,
    )


def assert_no_leakage(train_shard_paths: List[str], val_path: str, test_path: str) -> None:
    """Post-hoc integrity check: normalized-hash sets of train/val/test must
    be pairwise disjoint. Raises AssertionError on any overlap."""
    def hashes(path: str) -> set:
        out = set()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.add(record_normalized_hash(json.loads(line)))
        return out

    val_h, test_h = hashes(val_path), hashes(test_path)
    assert val_h.isdisjoint(test_h), "validation and test sets overlap"
    for p in train_shard_paths:
        train_h = hashes(p)
        assert train_h.isdisjoint(val_h), f"{p} leaks into validation"
        assert train_h.isdisjoint(test_h), f"{p} leaks into test"


def write_manifest(out_dir: str, clean_stats: CleanStats, split: SplitResult, *,
                    source_description: str, license_description: str) -> dict:
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": {
            "description": source_description,
            "license": license_description,
        },
        "cleaning": {
            "records_seen": clean_stats.seen,
            "accepted": clean_stats.accepted,
            "rejected_invalid": clean_stats.rejected_invalid,
            "rejected_exact_duplicate": clean_stats.rejected_exact_dup,
            "rejected_near_duplicate": clean_stats.rejected_near_dup,
            "rejected_duplicate_pair": clean_stats.rejected_pair_dup,
            "rejection_reasons": dict(clean_stats.rejection_reasons),
            "category_counts_after_cleaning": dict(clean_stats.category_counts),
        },
        "split": {
            "seed": split.seed,
            "n_train": split.n_train,
            "n_validation": split.n_val,
            "n_test": split.n_test,
            "num_shards": len(split.train_shard_paths),
            "examples_per_shard": split.n_train // len(split.train_shard_paths) if split.train_shard_paths else 0,
            "train_category_counts": split.train_category_counts,
            "validation_category_counts": split.val_category_counts,
            "test_category_counts": split.test_category_counts,
        },
    }
    path = Path(out_dir) / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return manifest


__all__ = [
    "VALID_ROLES", "normalize_text", "record_text_hash", "record_normalized_hash",
    "record_pair_hash", "validate_record", "CleanStats", "build_clean_dataset",
    "SplitResult", "split_and_shard", "assert_no_leakage", "write_manifest",
]
