#!/usr/bin/env python3
"""Convert a built corpus into CFNA weights (a real checkpoint).

Trains on train documents, evaluates bits/byte on held-out documents, and saves
weights + optimizer + config + history. Runs for a bounded time, saves every 50
steps, resumes cleanly across CPU/CUDA, and uses atomic checkpoint replacement so
an interrupted Google Drive write does not destroy the last valid checkpoint.

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
from cfna.model import CFNAModel, CONFIG_PRESETS, ModelConfig

LN2 = math.log(2.0)


def chat_config() -> ModelConfig:
    """A modest config sized to train in a bounded local window."""
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


def optimizer_to(opt: torch.optim.Optimizer, device: torch.device | str) -> None:
    """Move resumed optimizer tensors to the active model device."""
    for state in opt.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def atomic_torch_save(payload: dict, out: Path) -> None:
    """Write beside the destination and replace only after torch.save succeeds."""
    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(out)


def atomic_json_save(payload: dict, out: Path) -> None:
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=str, default="corpus")
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--seq", type=int, default=192)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", type=str, default="checkpoints/cfna_chat.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue from an existing checkpoint")
    ap.add_argument("--preset", default="", choices=[""] + sorted(CONFIG_PRESETS),
                    help="cfna.model.CONFIG_PRESETS rung (default: local chat_config)")
    ap.add_argument("--device", default="auto", help="auto|cuda|cpu")
    ap.add_argument("--amp", action="store_true", help="fp16 autocast (CUDA only)")
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="accumulate N micro-batches per optimizer step "
                         "(effective batch = batch*N; fewer, larger steps)")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the model (one-time warmup, then faster)")
    ap.add_argument("--save-every-min", type=float, default=0.0,
                    help="also save at least every N minutes (0 = only on log interval)")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    amp = bool(args.amp and device == "cuda")
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    train = ByteCorpus(args.corpus, "train")
    val = ByteCorpus(args.corpus, "val")
    valb = val_batches(val, args.seq, args.batch, max_batches=8)
    print(f"train {train.total_bytes/1e6:.2f} MB / {len(train.docs)} docs | "
          f"held-out {val.total_bytes/1e6:.2f} MB / {len(val.docs)} docs "
          f"({', '.join(val.titles[:8])}{' ...' if len(val.titles) > 8 else ''})")

    cfg = CONFIG_PRESETS[args.preset]() if args.preset else chat_config()
    model = CFNAModel(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    history = []
    step = 0

    if args.resume and out.exists():
        ck = torch.load(out, map_location="cpu", weights_only=False)
        if ck.get("config") != vars(cfg):
            raise SystemExit("resume config mismatch — checkpoint was created with another preset")
        model.load_state_dict(ck["state_dict"])
        if ck.get("optimizer") is not None:
            opt.load_state_dict(ck["optimizer"])
        for group in opt.param_groups:
            group["lr"] = args.lr  # CLI lr wins on resume (convergence ladder)
        step = int(ck.get("step", 0))
        history = ck.get("history", [])
        print(f"resumed from step {step}")

    model.to(device)
    optimizer_to(opt, device)
    valb = [b.to(device) for b in valb]
    train_model = model
    if args.compile:
        try:
            train_model = torch.compile(model)
            print("torch.compile enabled (first steps slower while it warms up)")
        except Exception as e:
            print("torch.compile unavailable, continuing eager:", e)
    eff_batch = args.batch * max(1, args.grad_accum)
    print(f"model: {model.num_params():,} params | device {device}{' +amp' if amp else ''} | "
          f"seq {args.seq} batch {args.batch}x{args.grad_accum}accum (eff {eff_batch}) | "
          f"budget {args.minutes} min")

    rng = np.random.default_rng(args.seed + step)
    t0 = time.time()
    best_val = min((h["heldout_bpb"] for h in history), default=float("inf"))

    def save():
        payload = {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer": opt.state_dict(),
            "config": vars(cfg),
            "step": step,
            "history": history,
            "corpus": str(Path(args.corpus).resolve()),
        }
        atomic_torch_save(payload, out)

    last_timed_save = time.time()
    while (time.time() - t0) < args.minutes * 60:
        model.train()
        opt.zero_grad(set_to_none=True)
        parts = None
        # Gradient accumulation: sum grads over grad_accum micro-batches, scaling
        # each loss by 1/accum so the effective batch behaves like one big batch.
        for micro in range(max(1, args.grad_accum)):
            batch = torch.from_numpy(train.sample_batch(args.seq, args.batch, rng)).to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss, parts = train_model.loss(batch)
                loss = loss / max(1, args.grad_accum)
            scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1

        # Timed safety save (Colab-runtime-death insurance, independent of the
        # log interval) so an idle timeout loses at most save-every-min minutes.
        if args.save_every_min and (time.time() - last_timed_save) >= args.save_every_min * 60:
            save()
            last_timed_save = time.time()

        if step % 50 == 0:
            vb = heldout_bpb(model, valb)
            mins = (time.time() - t0) / 60
            history.append({
                "step": step,
                "train_bpb": parts["bpb"],
                "heldout_bpb": vb,
                "minutes": mins,
            })
            print(f"step {step:5d} | {mins:5.1f}m | train bpb {parts['bpb']:.3f} | "
                  f"held-out bpb {vb:.3f}", flush=True)
            best_val = min(best_val, vb)
            save()

    save()
    atomic_json_save({"config": vars(cfg), "history": history}, Path(str(out) + ".json"))
    print(f"\nsaved checkpoint -> {out}  ({step} steps, best held-out bpb {best_val:.3f})")


if __name__ == "__main__":
    main()
