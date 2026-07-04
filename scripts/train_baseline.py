#!/usr/bin/env python3
"""Train a matched from-scratch baseline (ByteTransformerLM / ByteSSMLM) on
the SAME corpus, budget, and save/resume discipline as scripts/
train_checkpoint.py — the equal-compute face-off arm.

Fairness contract (the whole point):
- same ByteCorpus, same seq/batch/AMP, same wall-clock budget, same
  atomic/resumable checkpointing and held-out bpb metric;
- parameter count printed at start — size the baseline to match the CFNA
  rung under comparison (e.g. ~34M vs base_35m) and record both counts;
- the baseline is built from the same hand-rolled primitives (cfna/nn.py),
  so the comparison reflects architecture, not library kernels.

Usage (Colab/desktop, matched vs base_35m):
    python scripts/train_baseline.py --kind transformer --corpus corpus_large \
        --d-model 512 --n-layers 10 --n-heads 8 --max-len 512 \
        --minutes 170 --seq 192 --batch 16 --lr 3e-4 --amp --resume \
        --out checkpoints/baseline_tf_34m.pt
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from cfna.baselines import BaselineConfig, ByteSSMLM, ByteTransformerLM
from cfna.corpus.dataset import ByteCorpus, val_batches

LN2 = math.log(2.0)


@torch.no_grad()
def heldout_bpb(model, batches) -> float:
    if not batches:
        return float("nan")
    model.eval()
    return float(np.mean([model.lm_loss(b).item() for b in batches]) / LN2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["transformer", "ssm"], default="transformer")
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=10)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--seq", type=int, default=192)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints/baseline.pt")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    amp = bool(args.amp and device == "cuda")
    torch.manual_seed(args.seed)

    cfg = BaselineConfig(d_model=args.d_model, n_layers=args.n_layers,
                         n_heads=args.n_heads, max_len=args.max_len)
    model = (ByteTransformerLM if args.kind == "transformer" else ByteSSMLM)(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    train = ByteCorpus(args.corpus, "train")
    val = ByteCorpus(args.corpus, "val")
    valb = val_batches(val, args.seq, args.batch, max_batches=8)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    history, step = [], 0
    if args.resume and out.exists():
        ck = torch.load(out, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"])
        if ck.get("optimizer") is not None:
            opt.load_state_dict(ck["optimizer"])
        for g in opt.param_groups:
            g["lr"] = args.lr
        step, history = int(ck.get("step", 0)), ck.get("history", [])
        print(f"resumed from step {step}")
    model.to(device)
    for st in opt.state.values():
        for k, v in list(st.items()):
            if torch.is_tensor(v):
                st[k] = v.to(device)
    valb = [b.to(device) for b in valb]
    print(f"baseline={args.kind} | {model.num_params():,} params | device {device}"
          f"{' +amp' if amp else ''} | seq {args.seq} batch {args.batch} | "
          f"budget {args.minutes} min")

    rng = np.random.default_rng(args.seed + step)
    t0 = time.time()
    best = min((h["heldout_bpb"] for h in history), default=float("inf"))

    def save():
        tmp = Path(str(out) + ".tmp")
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "optimizer": opt.state_dict(), "kind": args.kind,
                    "config": vars(cfg), "step": step, "history": history}, tmp)
        tmp.replace(out)

    while (time.time() - t0) < args.minutes * 60:
        batch = torch.from_numpy(train.sample_batch(args.seq, args.batch, rng)).to(device)
        model.train()
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss = model.lm_loss(batch)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1
        if step % 50 == 0:
            vb = heldout_bpb(model, valb)
            mins = (time.time() - t0) / 60
            history.append({"step": step, "train_bpb": float(loss.item()) / LN2,
                            "heldout_bpb": vb, "minutes": mins})
            best = min(best, vb)
            print(f"step {step:6d} | {mins:5.1f}m | train bpb {loss.item()/LN2:.3f} | "
                  f"held-out bpb {vb:.3f}", flush=True)
            save()
    save()
    Path(str(out) + ".json").write_text(json.dumps(
        {"kind": args.kind, "config": vars(cfg), "n_params": model.num_params(),
         "history": history}, indent=2))
    print(f"\nsaved -> {out} ({step} steps, best held-out bpb {best:.3f})")


if __name__ == "__main__":
    main()
