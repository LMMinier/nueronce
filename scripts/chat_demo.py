#!/usr/bin/env python3
"""Hold a short conversation with the trained CFNA checkpoint.

Honest expectation: a ~11M-param byte model trained for minutes on ~14 MB of
public-domain books and speeches produces *English-shaped continuations* in the
register it learned — not a genuine instruct-tuned assistant. This script shows
the model responding to prompts so you can see what it actually learned.

Usage:  python scripts/chat_demo.py [--ckpt checkpoints/cfna_chat.pt] [--temp 0.7]
"""

from __future__ import annotations

import argparse

from cfna.chat import Conversation, load_checkpoint

DEFAULT_TURNS = [
    "Good evening. Who are you?",
    "Tell me about the sea.",
    "What is your opinion of liberty?",
    "Describe the morning.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/cfna_chat.pt")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    torch.manual_seed(args.seed)

    model, ckpt = load_checkpoint(args.ckpt)
    hist = ckpt.get("history", [])
    bpb = hist[-1]["heldout_bpb"] if hist else float("nan")
    print(f"loaded {model.num_params():,}-param checkpoint "
          f"(step {ckpt.get('step','?')}, held-out bpb {bpb:.3f})\n")

    convo = Conversation(model, system="A conversation.", temperature=args.temp, max_new=args.max_new)
    for turn in DEFAULT_TURNS:
        reply = convo.say(turn)
        print(f"User:      {turn}")
        print(f"Assistant: {reply}\n")


if __name__ == "__main__":
    main()
