#!/usr/bin/env python3
"""Resumable conversational training for NUERONCEModel (PLAN.md Phases 3-5).

Runs on whatever compute exists (CUDA GPU, Colab T4, or CPU): the data/batch
logic is the torch-free ``nueronce.training.mixed_sft`` (unit-tested without a
GPU); this script is only the thin torch loop around it.

Two loss modes:
- ``--loss full``     next-byte CE on every real byte (dialogue pretraining /
                      from-scratch runs; padding is never a target),
- ``--loss response`` the SFT contract — CE on assistant-response bytes only.

Checkpoints are ``nueronce.chat.load_checkpoint``-compatible and stamp
``meta.prompt_format`` per docs/FORMAT.md. ``best.pt`` is best-by-val (never
last-step); ``latest.pt`` is for resume.

Usage (desktop GPU / Colab):
    python scripts/build_conversation_sft.py --out-dir data/conversation_sft
    python scripts/train_conversation.py --data data/conversation_sft \
        --preset chat_11m --out-dir checkpoints/conv_11m --minutes 120 --amp
    # resume later:
    python scripts/train_conversation.py --data data/conversation_sft \
        --preset chat_11m --out-dir checkpoints/conv_11m --minutes 120 --amp --resume
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from nueronce.model import NUERONCEModel, CONFIG_PRESETS, ModelConfig
from nueronce.training.dialogue_data import PROMPT_FORMAT
from nueronce.training.mixed_sft import build_batches, load_jsonl

LN2 = math.log(2.0)


def to_torch(batch, device):
    return (torch.from_numpy(batch["byte_ids"]).to(device),
            torch.from_numpy(batch["target_mask"]).to(device))


@torch.no_grad()
def evaluate(model, val_batches, device):
    """Masked val loss (bits/byte over target bytes) + teacher-forced byte
    accuracy on target bytes — the checkpoint-health metric from FORMAT.md."""
    model.eval()
    losses, correct, total = [], 0, 0
    for batch in val_batches:
        byte_ids, mask = to_torch(batch, device)
        logits, _ = model(byte_ids)
        losses.append(model.masked_token_loss(logits, byte_ids, mask).item())
        pred = logits[:, :-1].argmax(dim=-1)
        tgt = byte_ids[:, 1:]
        sel = mask[:, 1:]
        correct += int((pred.eq(tgt) & sel).sum().item())
        total += int(sel.sum().item())
    return float(np.mean(losses)), (correct / total if total else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/conversation_sft")
    ap.add_argument("--out-dir", default="checkpoints/conversation")
    ap.add_argument("--preset", default="chat_11m", choices=sorted(CONFIG_PRESETS))
    ap.add_argument("--init-from", default="", help="warm-start weights from this checkpoint")
    ap.add_argument("--resume", action="store_true", help="continue from <out-dir>/latest.pt")
    ap.add_argument("--loss", default="response", choices=["response", "full"])
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = time budget only")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--val-batches", type=int, default=16)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action="store_true", help="fp16 autocast (CUDA only)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--metrics-out", default="metrics/conversation_train.jsonl")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    amp = bool(args.amp and device == "cuda")
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    shards = sorted(glob.glob(str(Path(args.data) / "train" / "shard_*.jsonl")))
    if not shards:
        raise SystemExit(f"no train shards under {args.data}/train — run scripts/build_conversation_sft.py first")
    train_records = load_jsonl(shards)
    val_records = load_jsonl([Path(args.data) / "val.jsonl"]) if (Path(args.data) / "val.jsonl").exists() \
        else load_jsonl([Path(args.data) / "validation.jsonl"])
    val_batches = list(build_batches(val_records, batch_size=args.batch, max_len=args.max_len,
                                     seed=1, loss="response"))[: args.val_batches]

    cfg = CONFIG_PRESETS[args.preset]()
    model = NUERONCEModel(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    step, epoch, history = 0, 0, []
    best_val = float("inf")

    if args.init_from and not args.resume:
        ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"])
        print(f"warm-started weights from {args.init_from} (step {ck.get('step', '?')})")
    if args.resume and (out_dir / "latest.pt").exists():
        ck = torch.load(out_dir / "latest.pt", map_location="cpu", weights_only=False)
        if ck.get("config") != vars(cfg):
            raise SystemExit("resume config mismatch — wrong --preset for this out-dir")
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["optimizer"])
        for group in opt.param_groups:
            group["lr"] = args.lr  # CLI lr wins on resume (convergence ladder)
        step, epoch = int(ck.get("step", 0)), int(ck.get("epoch", 0))
        history = ck.get("history", [])
        best_val = min((h["val_loss"] for h in history if "val_loss" in h), default=float("inf"))
        print(f"resumed from step {step} (epoch {epoch}, best val {best_val:.4f})")

    model.to(device)
    n_params = model.num_params()
    print(f"{args.preset}: {n_params:,} params | device {device}{' +amp' if amp else ''} | "
          f"loss={args.loss} | {len(train_records):,} train records | "
          f"{len(val_batches)} val batches | budget {args.minutes} min")

    def save(path, extra=None):
        payload = {
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "optimizer": opt.state_dict(),
            "config": vars(cfg), "step": step, "epoch": epoch, "history": history,
            "meta": {
                "prompt_format": PROMPT_FORMAT, "preset": args.preset,
                "loss_mode": args.loss, "n_params": n_params,
                "data": str(args.data), **(extra or {}),
            },
        }
        tmp = Path(str(path) + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)

    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    done = False
    while not done:
        epoch += 1
        for batch in build_batches(train_records, batch_size=args.batch, max_len=args.max_len,
                                   seed=args.seed + epoch, loss=args.loss):
            byte_ids, mask = to_torch(batch, device)
            model.train()
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits, _ = model(byte_ids)
                loss = model.masked_token_loss(logits, byte_ids, mask)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            step += 1

            if step % args.val_every == 0:
                val_loss, val_acc = evaluate(model, val_batches, device)
                mins = (time.time() - t0) / 60
                rec = {"step": step, "epoch": epoch, "minutes": round(mins, 2),
                       "train_loss": float(loss.item()), "val_loss": val_loss,
                       "val_bpb": val_loss / LN2, "val_byte_acc": val_acc}
                history.append(rec)
                with metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"step {step:6d} | ep {epoch} | {mins:6.1f}m | train {rec['train_loss']:.4f} | "
                      f"val {val_loss:.4f} ({rec['val_bpb']:.3f} bpb) | acc {val_acc:.3f}", flush=True)
                save(out_dir / "latest.pt")
                if val_loss < best_val:
                    best_val = val_loss
                    save(out_dir / "best.pt", extra={"best_val_loss": best_val, "best_val_acc": val_acc})

            if (time.time() - t0) >= args.minutes * 60 or (args.max_steps and step >= args.max_steps):
                done = True
                break

    save(out_dir / "latest.pt")
    val_loss, val_acc = evaluate(model, val_batches, device)
    print(f"\nfinished: {step} steps / {epoch} epochs | final val {val_loss:.4f} "
          f"(acc {val_acc:.3f}) | best val {best_val:.4f} -> {out_dir}/best.pt")


if __name__ == "__main__":
    main()
