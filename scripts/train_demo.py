#!/usr/bin/env python3
"""Train the from-scratch CFNA model on the tiny demo corpus.

This is the "give the architecture a run" entry point. It trains the hand-built
two-level byte model for a few hundred steps on CPU and reports:

  - the loss / bits-per-byte curve (should fall far below the 8.0 bpb baseline),
  - a sample continuation of a prompt,
  - dynamic-patching statistics (how many units the learned patcher forms),
  - a metrics JSON written next to the script.

Usage:
    python scripts/train_demo.py [--steps 400] [--seq 96] [--batch 16]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from cfna.data import UNIFORM_BYTE_BPB, corpus_bytes, make_batches
from cfna.model import CFNAModel, ModelConfig
from cfna.segment import segment_ids_from_boundaries


def patch_stats(model: CFNAModel, batch: torch.Tensor) -> float:
    c = model.cfg
    with torch.no_grad():
        _, blogits = model.perception(batch)
        prob = torch.sigmoid(blogits)
        _, n_units = segment_ids_from_boundaries(
            prob, tau=c.tau, min_patch=c.min_patch, max_patch=c.max_patch, p_max=c.p_max
        )
    return float(n_units.float().mean())


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seq", type=int, default=96)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(Path(__file__).parent / "train_metrics.json"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    data = corpus_bytes(repeat=10)
    model = CFNAModel(ModelConfig())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    n_params = model.num_params()
    print(f"CFNA model: {n_params:,} parameters | corpus: {len(data)} bytes")

    batches = make_batches(data, args.seq, args.batch, args.steps, seed=args.seed)
    eval_batch = make_batches(data, args.seq, args.batch, 1, seed=args.seed + 999)[0]

    history = []
    first_bpb = None
    t0 = time.time()
    model.train()
    for step, batch in enumerate(batches):
        loss, parts = model.loss(batch)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if first_bpb is None:
            first_bpb = parts["bpb"]
        if step % 25 == 0 or step == len(batches) - 1:
            units = patch_stats(model, eval_batch)
            history.append({"step": step, **parts, "avg_units": units})
            print(f"step {step:4d} | loss {parts['loss']:.3f} | lm {parts['lm']:.3f} "
                  f"| bpb {parts['bpb']:.3f} | boundary {parts['boundary']:.3f} "
                  f"| ~units/seq {units:.1f}")

    train_secs = time.time() - t0

    # final eval
    model.eval()
    with torch.no_grad():
        final_loss, final = model.loss(eval_batch)
    sample = model.generate(b"CFNA separates ", max_new=80, greedy=True)
    try:
        sample_text = sample.decode("utf-8", errors="replace")
    except Exception:
        sample_text = repr(sample)

    print("\n=== result ===")
    print(f"first bpb: {first_bpb:.3f}  ->  final bpb: {final['bpb']:.3f} "
          f"(uniform baseline {UNIFORM_BYTE_BPB:.1f})")
    print(f"train time: {train_secs:.1f}s")
    print("sample continuation (greedy):")
    print("  " + sample_text.replace("\n", "\n  "))

    metrics = {
        "n_params": n_params,
        "steps": args.steps,
        "first_bpb": first_bpb,
        "final_bpb": final["bpb"],
        "final_lm": final["lm"],
        "final_boundary": final["boundary"],
        "uniform_baseline_bpb": UNIFORM_BYTE_BPB,
        "train_seconds": train_secs,
        "avg_units_per_seq": history[-1]["avg_units"],
        "sample": sample_text,
        "history": history,
    }
    Path(args.out).write_text(json.dumps(metrics, indent=2))
    print(f"\nmetrics written to {args.out}")
    return metrics


if __name__ == "__main__":
    main()
