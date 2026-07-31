#!/usr/bin/env python3
"""Matched-parameter comparison on a held-out split.

Trains NUERONCE and from-scratch baselines (byte Transformer, pure SSM) on the train
region of a non-repeating corpus, then reports bits/byte on a disjoint held-out
region. This is the honest signal: does the architecture *generalize*, not just
memorize the training bytes?

Usage:  python scripts/compare_baselines.py [--steps 300] [--seq 64]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from nueronce.baselines import BaselineConfig, ByteSSMLM, ByteTransformerLM
from nueronce.data import larger_corpus_bytes, make_batches, train_val_split
from nueronce.eval import compare
from nueronce.model import NUERONCEModel, ModelConfig


def nueronce_factory():
    return NUERONCEModel(ModelConfig(
        byte_embed_dim=32, d_local=64, d_model=96, p_max=24, physical_blocks=2,
        logical_depth=3, n_heads=4, unit_window=16, decoder_window=24,
        decoder_layers=2, d_state=12, channel_dim=16, min_patch=3, max_patch=20,
    ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parameter-tolerance", type=float, default=0.05)
    ap.add_argument("--out", type=str, default=str(Path(__file__).parent / "baseline_metrics.json"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    data = larger_corpus_bytes()
    train_bytes, val_bytes = train_val_split(data, val_frac=0.25)
    print(f"corpus {len(data)} bytes -> train {len(train_bytes)} / held-out {len(val_bytes)} (disjoint)")

    train_batches = make_batches(train_bytes, args.seq, args.batch, args.steps, seed=args.seed)
    val_batches = make_batches(val_bytes, args.seq, args.batch, 8, seed=10_000 + args.seed)

    factories = {
        "NUERONCE": nueronce_factory,
        "ByteTransformer": lambda: ByteTransformerLM(BaselineConfig(d_model=132, n_layers=5, n_heads=4, max_len=args.seq)),
        "ByteSSM": lambda: ByteSSMLM(BaselineConfig(d_model=136, n_layers=4, d_state=12, max_len=args.seq)),
    }
    results = compare(factories, train_batches, val_batches, lr=args.lr, seed=args.seed)
    counts = [row["params"] for row in results.values()]
    parameter_spread = (max(counts) - min(counts)) / min(counts)
    parameter_gate = parameter_spread <= args.parameter_tolerance

    print(f"\n{'model':<16} {'params':>10} {'train bpb':>10} {'heldout bpb':>12}")
    for name, r in results.items():
        print(f"{name:<16} {r['params']:>10,} {r['final_train_bpb']:>10.3f} {r['heldout_bpb']:>12.3f}")

    print("\nHeld-out bits/byte is the generalization signal (lower is better).")
    print("Parameter gate + same batches/steps/optimizer; uniform-byte baseline = 8.0 bpb.")
    report = {
        "protocol": "matched_byte_lm_v2",
        "seed": args.seed,
        "steps": args.steps,
        "sequence_bytes": args.seq,
        "batch_size": args.batch,
        "learning_rate": args.lr,
        "optimizer": "torch.optim.AdamW_for_all_models",
        "corpus_sha256": hashlib.sha256(data).hexdigest(),
        "parameter_tolerance": args.parameter_tolerance,
        "parameter_spread": parameter_spread,
        "parameter_gate_passed": parameter_gate,
        "claim_eligible": parameter_gate,
        "results": results,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    if not parameter_gate:
        print(f"WARNING: parameter gate failed: spread={parameter_spread:.3%} "
              f"> tolerance={args.parameter_tolerance:.3%}; no superiority claim is eligible.")
    print(f"metrics written to {args.out}")


if __name__ == "__main__":
    main()
