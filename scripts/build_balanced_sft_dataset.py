#!/usr/bin/env python3
"""Build a balanced synthetic SFT curriculum.

This is meant as the next fine-tuning dataset after the 100K run revealed a
category imbalance. It reuses only self-authored templates and writes standard
train shard / validation / test files compatible with ``scripts/train_sft.py``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

# Allow direct execution from a clean checkout without requiring installation
# or an externally configured PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cfna.training.dataset_prep import assert_no_leakage, record_normalized_hash, validate_record
from cfna.training.synthetic_dialogue import GENERATORS


def _write_records(records, path: Path):
    counts = {}
    total = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            reason = validate_record(rec)
            if reason is not None:
                raise ValueError(f"invalid generated record {rec.get('id')}: {reason}")
            counts[rec["category"]] = counts.get(rec["category"], 0) + 1
            total += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return total, counts


def _with_id(rec: dict, prefix: str, i: int) -> dict:
    out = dict(rec)
    out["id"] = f"{prefix}-{i:05d}-{rec['id']}"
    if prefix.startswith("train-repeat"):
        out["source"] = f"{rec.get('source', 'unknown')}|balanced-train-weight"
    return out


def _unique_by_normalized_hash(records):
    seen = set()
    out = []
    for rec in records:
        h = record_normalized_hash(rec)
        if h in seen:
            continue
        seen.add(h)
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/sft_balanced")
    ap.add_argument("--train-examples-per-category", type=int, default=500)
    ap.add_argument("--val-per-category", type=int, default=4)
    ap.add_argument("--test-per-category", type=int, default=4)
    ap.add_argument("--num-shards", type=int, default=5)
    ap.add_argument("--examples-per-shard", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=43)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clean_path = out / "clean.jsonl"
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    train, val, test = [], [], []
    unique_counts = {}
    train_unique_counts = {}
    train_pools = {}
    heldout_hashes = set()
    for name, gen_fn in GENERATORS:
        records = _unique_by_normalized_hash(list(gen_fn()))
        unique_counts[name] = len(records)
        order = rng.permutation(len(records))
        need_holdout = args.val_per_category + args.test_per_category
        if len(records) <= need_holdout:
            raise ValueError(f"{name} has {len(records)} unique records, need > {need_holdout}")
        val_idx = order[:args.val_per_category]
        test_idx = order[args.val_per_category:need_holdout]
        train_idx = order[need_holdout:]
        val_records = [records[int(i)] for i in val_idx]
        test_records = [records[int(i)] for i in test_idx]
        heldout_hashes.update(record_normalized_hash(r) for r in val_records)
        heldout_hashes.update(record_normalized_hash(r) for r in test_records)
        val.extend(_with_id(r, f"val-{name}", j) for j, r in enumerate(val_records))
        test.extend(_with_id(r, f"test-{name}", j) for j, r in enumerate(test_records))
        train_pools[name] = [records[int(i)] for i in train_idx]

    for name, _ in GENERATORS:
        train_pool = [r for r in train_pools[name]
                      if record_normalized_hash(r) not in heldout_hashes]
        if not train_pool:
            raise ValueError(f"{name} has no train records after global holdout filtering")
        train_unique_counts[name] = len(train_pool)
        weighted = itertools.islice(itertools.cycle(train_pool), args.train_examples_per_category)
        train.extend(_with_id(rec, f"train-repeat-{name}", j) for j, rec in enumerate(weighted))

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    clean_total, clean_counts = _write_records([*train, *val, *test], clean_path)
    train_total = args.num_shards * args.examples_per_shard
    if len(train) < train_total:
        raise ValueError(f"train set has {len(train)} weighted records, need {train_total}")
    train = train[:train_total]

    (out / "train_shards").mkdir(parents=True, exist_ok=True)
    shard_paths = []
    for s in range(args.num_shards):
        path = out / "train_shards" / f"shard_{s + 1:02d}.jsonl"
        _write_records(train[s * args.examples_per_shard:(s + 1) * args.examples_per_shard], path)
        shard_paths.append(str(path))
    val_path = out / "validation.jsonl"
    test_path = out / "test.jsonl"
    _, val_counts = _write_records(val, val_path)
    _, test_counts = _write_records(test, test_path)
    assert_no_leakage(shard_paths, str(val_path), str(test_path))

    train_counts = Counter(r["category"] for r in train)
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": {
            "description": (
                "balanced synthetic SFT curriculum from cfna.training.synthetic_dialogue; "
                "train may repeat small-category records as explicit weighting, while "
                "validation/test are unique and held out before repetition"
            ),
            "license": "Original self-authored repository templates; no external corpus.",
        },
        "split": {
            "seed": args.seed,
            "n_train": len(train),
            "n_validation": len(val),
            "n_test": len(test),
            "num_shards": args.num_shards,
            "examples_per_shard": args.examples_per_shard,
            "train_category_counts": dict(train_counts),
            "validation_category_counts": val_counts,
            "test_category_counts": test_counts,
        },
    }
    manifest["balanced_dataset"] = {
        "train_examples_per_category_requested": args.train_examples_per_category,
        "val_per_category": args.val_per_category,
        "test_per_category": args.test_per_category,
        "categories": [name for name, _ in GENERATORS],
        "unique_available_by_category": unique_counts,
        "train_unique_available_after_holdout": train_unique_counts,
        "clean_category_counts": clean_counts,
        "train_repetition_is_weighting_not_new_data": True,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"done in {time.time() - t0:.1f}s. manifest -> {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
