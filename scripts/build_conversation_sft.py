#!/usr/bin/env python3
"""Build the mixed conversational SFT dataset (PLAN.md Phase 2).

Mix, all rendered in the canonical prompt format (docs/FORMAT.md):

- every synthetic dialogue register from ``cfna.training.synthetic_dialogue``
  (arithmetic/classification stride-sampled, not prefix-sliced),
- prompt-aligned direct/grounded/edge records (evidence + plan block skills)
  from ``scripts/build_prompt_aligned_sft.py``'s generators,
- the hand-written seed set from ``cfna.training.dialogue_data``.

Pipeline: generate -> validate+dedupe (``dataset_prep``) -> enforce the <=25%
per-register cap (the 77%-arithmetic poisoning lesson) -> deterministic
split/shard -> leakage check -> manifest. Deterministic given --seed, so the
dataset is reproduced on any machine with one command instead of being
committed to git (data/ is gitignored).

Usage:
    python scripts/build_conversation_sft.py --out-dir data/conversation_sft
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

from cfna.training.dataset_prep import (
    assert_no_leakage,
    build_clean_dataset,
    split_and_shard,
)
from cfna.training.dialogue_data import SFT_DATASET
from cfna.training.mixed_sft import dataset_summary, enforce_category_caps, load_jsonl, stride_sample
from cfna.training.synthetic_dialogue import GENERATORS

CAPPED = {"arithmetic": 15_000, "classification": 15_000}  # generous pre-cap quotas


def _load_prompt_aligned_module():
    path = Path(__file__).parent / "build_prompt_aligned_sft.py"
    spec = importlib.util.spec_from_file_location("build_prompt_aligned_sft", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def generate_records(n_direct: int, n_grounded: int, n_edge: int):
    """Yield every candidate record in the unified messages schema."""
    totals = {"arithmetic": 98_450, "classification": 21_171}
    for name, fn in GENERATORS:
        it = fn()
        first = next(iter(it), None)
        if first is None:
            continue
        cat = first["category"]
        def _chain(first=first, it=it):
            yield first
            yield from it
        if cat in CAPPED:
            yield from stride_sample(_chain(), CAPPED[cat], totals.get(cat, CAPPED[cat]))
        else:
            yield from _chain()

    pa = _load_prompt_aligned_module()
    for r in pa.build_records(n_direct, n_grounded, n_edge):
        yield {
            "id": r["id"],
            "source": "cfna-prompt-aligned-v1",
            "category": f"pa_{r['category']}",
            "messages": [
                {"role": "user", "content": r["user_request"]},
                {"role": "assistant", "content": r["assistant_response"]},
            ],
            "system_message": r["system_message"],
            "trusted_evidence": r["trusted_evidence"],
            "response_plan": r["response_plan"],
        }

    for i, (prompt, response) in enumerate(SFT_DATASET):
        yield {
            "id": f"hw-{i:04d}",
            "source": "cfna-handwritten-v1",
            "category": "handwritten",
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/conversation_sft")
    ap.add_argument("--direct", type=int, default=8000)
    ap.add_argument("--grounded", type=int, default=8000)
    ap.add_argument("--edge", type=int, default=4000)
    ap.add_argument("--cap-frac", type=float, default=0.25)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--val-size", type=int, default=2500)
    ap.add_argument("--test-size", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--manifest-copy", default="metrics/conversation_sft_manifest.json",
                    help="Tracked copy of the manifest ('' to skip)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("pass 1: generate -> validate -> dedupe ...")
    clean_path = out_dir / "clean.jsonl"
    stats = build_clean_dataset(generate_records(args.direct, args.grounded, args.edge), str(clean_path))
    print(f"  seen {stats.seen:,} | accepted {stats.accepted:,} | invalid {stats.rejected_invalid:,} | "
          f"dups {stats.rejected_exact_dup + stats.rejected_near_dup + stats.rejected_pair_dup:,}")

    print(f"pass 2: enforce <= {args.cap_frac:.0%} per-register cap ...")
    records = enforce_category_caps(load_jsonl([clean_path]), cap_frac=args.cap_frac, seed=args.seed)
    summary = dataset_summary(records)
    capped_path = out_dir / "capped.jsonl"
    with capped_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  kept {summary['total']:,} records | max register share "
          f"{summary['max_category_frac']:.1%}")
    assert summary["max_category_frac"] <= args.cap_frac + 0.01, "register cap violated"

    print("pass 3: split + shard ...")
    n_train = summary["total"] - args.val_size - args.test_size
    per_shard = max(1, n_train // args.num_shards)
    split = split_and_shard(
        str(capped_path), str(out_dir), num_shards=args.num_shards,
        examples_per_shard=per_shard, val_size=args.val_size,
        test_size=args.test_size, seed=args.seed,
    )
    print(f"  train {split.n_train:,} ({len(split.train_shard_paths)} shards) | "
          f"val {split.n_val:,} | test {split.n_test:,}")
    assert_no_leakage(split.train_shard_paths, split.val_path, split.test_path)
    print("  no train/val/test leakage.")

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prompt_format": "canonical",
        "seed": args.seed,
        "cap_frac": args.cap_frac,
        "clean_stats": {
            "seen": stats.seen, "accepted": stats.accepted,
            "rejected_invalid": stats.rejected_invalid,
            "rejected_exact_dup": stats.rejected_exact_dup,
            "rejected_near_dup": stats.rejected_near_dup,
            "rejected_pair_dup": stats.rejected_pair_dup,
        },
        "after_caps": summary,
        "splits": {
            "train": split.n_train, "val": split.n_val, "test": split.n_test,
            "train_shards": split.train_shard_paths,
            "train_category_counts": split.train_category_counts,
        },
        "rebuild_command": (
            f"python scripts/build_conversation_sft.py --out-dir {args.out_dir} "
            f"--direct {args.direct} --grounded {args.grounded} --edge {args.edge} "
            f"--cap-frac {args.cap_frac} --num-shards {args.num_shards} "
            f"--val-size {args.val_size} --test-size {args.test_size} --seed {args.seed}"
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.manifest_copy:
        Path(args.manifest_copy).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest_copy).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"done in {time.time() - t0:.1f}s -> {out_dir}/manifest.json")


if __name__ == "__main__":
    main()
