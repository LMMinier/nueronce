#!/usr/bin/env python3
"""Build the large-scale synthetic dialogue SFT dataset: generate -> validate
-> dedupe -> deterministic split -> shard -> manifest.

Usage:
    python scripts/build_sft_dataset.py --out-dir data/sft_100k \
        --num-shards 10 --examples-per-shard 10000 \
        --val-size 5000 --test-size 5000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from cfna.training.dataset_prep import assert_no_leakage, build_clean_dataset, split_and_shard, write_manifest
from cfna.training.synthetic_dialogue import GENERATORS, generate_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default="data/sft_100k")
    ap.add_argument("--num-shards", type=int, default=10)
    ap.add_argument("--examples-per-shard", type=int, default=10_000)
    ap.add_argument("--val-size", type=int, default=5_000)
    ap.add_argument("--test-size", type=int, default=5_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_path = out_dir / "clean.jsonl"

    print("generating raw synthetic records (streaming)...")
    raw_counts = {name: 0 for name, _ in GENERATORS}
    t0 = time.time()

    def _counted():
        for name, fn in GENERATORS:
            for rec in fn():
                raw_counts[rec["category"]] += 1
                yield rec

    stats = build_clean_dataset(_counted(), str(clean_path))
    print(f"raw generated: {sum(raw_counts.values()):,} | "
          f"clean accepted: {stats.accepted:,} | invalid: {stats.rejected_invalid:,} | "
          f"exact dup: {stats.rejected_exact_dup:,} | near dup: {stats.rejected_near_dup:,} | "
          f"pair dup: {stats.rejected_pair_dup:,}")

    print("splitting + sharding (streaming, offset-based deterministic shuffle)...")
    split = split_and_shard(
        str(clean_path), str(out_dir), num_shards=args.num_shards,
        examples_per_shard=args.examples_per_shard, val_size=args.val_size,
        test_size=args.test_size, seed=args.seed,
    )
    print(f"train: {split.n_train:,} ({len(split.train_shard_paths)} shards x "
          f"{args.examples_per_shard:,}) | val: {split.n_val:,} | test: {split.n_test:,}")

    print("verifying no train/val/test leakage...")
    assert_no_leakage(split.train_shard_paths, split.val_path, split.test_path)
    print("no leakage confirmed.")

    manifest = write_manifest(
        str(out_dir), stats, split,
        source_description=(
            "cfna-synthetic-template-v1: programmatically generated (self-authored templates x "
            "randomized/enumerated parameters, no third-party corpus). See "
            "cfna/training/synthetic_dialogue.py for every template and entity table."
        ),
        license_description="Original content authored for this repository; no external license to track.",
    )
    manifest["raw_generated_category_counts"] = raw_counts
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\ndone in {time.time() - t0:.1f}s. manifest -> {out_dir / 'manifest.json'}")
    print(f"clean intermediate file kept at {clean_path} (delete if disk space matters).")


if __name__ == "__main__":
    main()
