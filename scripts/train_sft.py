#!/usr/bin/env python3
"""Supervised instruction tuning (VGRFT stage 1) on top of a byte-LM checkpoint.

Every checkpoint produced by ``train_checkpoint.py`` is trained with pure
next-byte cross-entropy on monologic prose — it learns what English looks like,
not what to say back when spoken to (see ``cfna/chat.py``'s honest framing).
This script closes that gap for real: it loads a pretrained checkpoint (or
starts fresh if none exists), then runs an actual SFT pass over the small
(prompt, response) dialogue set in ``cfna.training.sft`` through
``VGRFTTrainer.supervised_instruction_tune`` — masking the loss to response
bytes only — and saves the tuned weights.

Usage:
    python scripts/train_sft.py --ckpt checkpoints/cfna_chat.pt \
        --out checkpoints/cfna_chat_sft.pt --steps 400
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cfna.chat import Conversation, load_checkpoint
from cfna.model import CFNAModel, ModelConfig
from cfna.training.sft import SFT_DATASET, held_out_split, sft_eval, make_sft_batch, TorchSFTBackend
from cfna.training.vgrft import VGRFTTrainer

DEMO_TURNS = ["Hello, who are you?", "What is two plus two?", "Thank you, goodbye."]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/cfna_chat.pt",
                     help="pretrained byte-LM checkpoint to fine-tune (falls back to a fresh model if missing)")
    ap.add_argument("--out", type=str, default="checkpoints/cfna_chat_sft.pt")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    ckpt_path = Path(args.ckpt)
    if ckpt_path.exists():
        model, ckpt = load_checkpoint(str(ckpt_path))
        print(f"loaded pretrained checkpoint {ckpt_path} ({model.num_params():,} params)")
    else:
        model = CFNAModel(ModelConfig())
        ckpt = {"config": vars(model.cfg)}
        print(f"no checkpoint at {ckpt_path}; starting SFT from a freshly-initialized "
              f"({model.num_params():,}-param) model — pretrain first for a real result")

    train_ex, val_ex = held_out_split(SFT_DATASET, val_frac=args.val_frac, seed=args.seed)
    print(f"SFT set: {len(train_ex)} train turns / {len(val_ex)} held-out turns\n")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    trainer = VGRFTTrainer(TorchSFTBackend(model, opt))
    history = trainer.supervised_instruction_tune(
        train_ex, steps=args.steps, batch_size=args.batch, val_examples=val_ex,
        seed=args.seed, log_every=max(1, args.steps // 10),
    )

    for rec in history:
        line = f"step {rec['step']:5d} | train loss {rec['train_loss']:.3f}"
        if "val_loss" in rec:
            line += f" | held-out loss {rec['val_loss']:.3f}"
        print(line)

    # Held-out response-byte loss, computed fresh so it's independent of the
    # training-loop bookkeeping above.
    val_batch = make_sft_batch(val_ex)
    final_val = sft_eval(model, val_batch)
    print(f"\nfinal held-out SFT loss (response bytes only): {final_val:.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": vars(model.cfg),
                "step": ckpt.get("step", 0), "history": ckpt.get("history", []),
                "sft_history": history}, out)
    print(f"saved SFT-tuned checkpoint -> {out}")

    print("\n=== sample turns after SFT ===")
    convo = Conversation(model, temperature=0.0, max_new=60)
    for turn in DEMO_TURNS:
        reply = convo.say(turn)
        print(f"User:      {turn}")
        print(f"Assistant: {reply}\n")


if __name__ == "__main__":
    main()
