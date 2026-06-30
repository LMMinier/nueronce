#!/usr/bin/env python3
"""Retrieval-augmented training demo + ablation.

Trains CFNAModel on the synthetic ``lookup <key> = <value>`` task where each fact
is fresh-random and used once, so the value is recoverable *only* by retrieving
the matching document. Reports, at each checkpoint, the value-token loss and
exact-match accuracy WITH vs WITHOUT retrieval — the ablation that shows the model
genuinely uses the retrieval path.

Usage:  python scripts/train_retrieval.py [--steps 800] [--k 2]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cfna.model import CFNAModel, ModelConfig
from cfna.retrieval_train import train_retrieval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--k", type=int, default=2, help="neighbors retrieved (1 correct + distractors)")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(Path(__file__).parent / "retrieval_metrics.json"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model = CFNAModel(ModelConfig())
    print(f"model: {model.num_params():,} params | task: lookup <key> = <value> (10 possible values)\n")

    hist = train_retrieval(model, steps=args.steps, batch=args.batch, k=args.k,
                           lr=args.lr, seed=args.seed, log_every=max(1, args.steps // 10))

    print(f"{'step':>5} | {'train L':>7} | {'WITH L':>6} | {'W/O L':>6} | {'WITH acc':>8} | {'W/O acc':>7} | recall@k")
    for h in hist:
        print(f"{h['step']:5d} | {h['train_value_loss']:7.3f} | {h['val_with_retrieval']:6.3f} | "
              f"{h['val_without_retrieval']:6.3f} | {h['acc_with']:8.2f} | {h['acc_without']:7.2f} | {h['recall_at_k']:.2f}")

    final = hist[-1]
    print("\n=== ablation (final) ===")
    print(f"value accuracy  WITH retrieval: {final['acc_with']:.2f}   "
          f"WITHOUT retrieval: {final['acc_without']:.2f}  (chance = 0.10)")
    print(f"value loss      WITH retrieval: {final['val_with_retrieval']:.2f}   "
          f"WITHOUT retrieval: {final['val_without_retrieval']:.2f}")
    print("The value is fresh-random per example, so it cannot be memorized in the\n"
          "weights — recovering it requires the retrieval path. The gap is the proof.")

    Path(args.out).write_text(json.dumps({"config": "CFNA-default", "history": hist}, indent=2))
    print(f"\nmetrics written to {args.out}")


if __name__ == "__main__":
    main()
