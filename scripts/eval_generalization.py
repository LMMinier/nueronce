#!/usr/bin/env python3
"""Run the >=200-prompt generalization suite against a trained NueronceModel
checkpoint: memorization probes (sampled verbatim from training) vs novel
prompts (new operands/entities/phrasings), scored and compared side by side.

Usage:
    python scripts/eval_generalization.py \
        --ckpt checkpoints/micro_nueronce_sft_100k/best.pt \
        --train-dir data/sft_100k/train_shards \
        --out metrics/generalization_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nueronce.engine.nueronce_model import NueronceModel, NueronceConfig
from nueronce.training.generalization_eval import (
    build_novel_prompts, build_seen_user_text_index, run_generalization_eval,
    sample_memorized_probes,
)
from nueronce.training.sharded_sft import load_checkpoint, load_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--train-dir", type=str, required=True, help="directory of shard_NN.jsonl files")
    ap.add_argument("--num-memorized-probes", type=int, default=100)
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="metrics/generalization_results.json")
    args = ap.parse_args()

    payload = load_checkpoint(args.ckpt)
    model = NueronceModel(NueronceConfig(**payload["config"]))
    for p, arr in zip(model.parameters(), payload["params"]):
        p.data = arr.copy()
    print(f"loaded {args.ckpt} ({model.num_params():,} params)")

    shard_paths = sorted(str(p) for p in Path(args.train_dir).glob("shard_*.jsonl"))
    print(f"indexing {len(shard_paths)} shard files for seen/novel classification...")
    seen_user_texts = build_seen_user_text_index(shard_paths)

    all_train_records = []
    for sp in shard_paths:
        all_train_records.extend(load_jsonl(sp))
    memorized = sample_memorized_probes(all_train_records, args.num_memorized_probes, seed=args.seed)
    novel = build_novel_prompts()
    prompts = memorized + novel
    print(f"evaluating {len(prompts)} prompts ({len(memorized)} memorized probes + {len(novel)} novel)...")

    results = run_generalization_eval(model, prompts, seen_user_texts, max_new=args.max_new)

    print(f"\noverall: {results['overall']}")
    print(f"memorized (n={results['memorized']['n']}): {results['memorized']}")
    print(f"novel (n={results['novel']['n']}): {results['novel']}")
    print("\nby category:")
    for cat, stats in sorted(results["by_category"].items()):
        print(f"  {cat:24s} n={stats['n']:4d} coherent={stats['coherent_rate']} "
              f"check_pass={stats['check_pass_rate']}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nfull results -> {out_path}")


if __name__ == "__main__":
    main()
