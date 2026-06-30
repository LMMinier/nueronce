#!/usr/bin/env python3
"""Convert the public-domain corpus into CFNA weights (a real checkpoint).

Trains on the train documents, evaluates bits/byte on the held-out documents, and
saves a checkpoint (weights + config + metrics). Time-budgeted so it fits a
bounded window; resumable-friendly (saves periodically).

Usage:
    python scripts/train_checkpoint.py --minutes 20 --out checkpoints/cfna_chat.pt
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from cfna.corpus.dataset import ByteCorpus, val_batches
from cfna.model import CFNAModel, ModelConfig

LN2 = math.log(2.0)


def chat_config() -> ModelConfig:
    """A modest (~few-M param) config sized to actually train on CPU in-window."""
    return ModelConfig(
        byte_embed_dim=64, d_local=128, d_model=256, p_max=48, physical_blocks=3,
        logical_depth=4, n_heads=8, unit_window=48, decoder_window=64,
        decoder_layers=3, d_state=16, channel_dim=24, ret_byte_dim=32,
        min_patch=3, max_patch=24, boundary_loss_weight=0.2,
    )


@torch.no_grad()
def heldout_bpb(model, batches) -> float:
    if not batches:
        return float("nan")
    model.eval()
    return float(np.mean([model.lm_loss(b).item() for b in batches]) / LN2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=str, default="corpus")
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--seq", type=int, default=192)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", type=str, default="checkpoints/cfna_chat.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    train = ByteCorpus(args.corpus, "train")
    val = ByteCorpus(args.corpus, "val")
    valb = val_batches(val, args.seq, args.batch, max_batches=8)
    print(f"train {train.total_bytes/1e6:.2f} MB / {len(train.docs)} docs | "
          f"held-out {val.total_bytes/1e6:.2f} MB / {len(val.docs)} docs ({', '.join(val.titles)})")

    cfg = chat_config()
    model = CFNAModel(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    print(f"model: {model.num_params():,} params | seq {args.seq} batch {args.batch} | budget {args.minutes} min")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    history = []
    t0 = time.time()
    step = 0
    best_val = float("inf")

    def save(tag="last"):
        torch.save({"state_dict": model.state_dict(), "config": vars(cfg),
                    "step": step, "history": history}, out)

    while (time.time() - t0) < args.minutes * 60:
        batch = torch.from_numpy(train.sample_batch(args.seq, args.batch, rng))
        model.train()
        loss, parts = model.loss(batch)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1

        if step % 50 == 0:
            vb = heldout_bpb(model, valb)
            mins = (time.time() - t0) / 60
            history.append({"step": step, "train_bpb": parts["bpb"], "heldout_bpb": vb, "minutes": mins})
            print(f"step {step:5d} | {mins:5.1f}m | train bpb {parts['bpb']:.3f} | held-out bpb {vb:.3f}")
            if vb < best_val:
                best_val = vb
                save("best")

    save("final")
    Path(str(out) + ".json").write_text(json.dumps({"config": vars(cfg), "history": history}, indent=2))
    print(f"\nsaved checkpoint -> {out}  ({step} steps, best held-out bpb {best_val:.3f})")


if __name__ == "__main__":
    main()
