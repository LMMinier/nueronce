#!/usr/bin/env python3
"""Chat/probe a CFNA checkpoint through the coherent inference wrapper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cfna.coherent_inference import respond, run_probes


def _torch_model_fn(ckpt: str, temp: float, max_new: int):
    from cfna.chat import Conversation, load_checkpoint

    model, _ = load_checkpoint(ckpt)
    convo = Conversation(
        model,
        system="You are CFNA, a small byte-level research assistant. Answer briefly.",
        temperature=temp,
        max_new=max_new,
    )
    return convo.say


def _micro_model_fn(ckpt: str, temp: float, max_new: int):
    from cfna.microtorch.chat import MicroConversation, load_checkpoint

    model, _ = load_checkpoint(ckpt)
    convo = MicroConversation(
        model,
        system="You are CFNA, a small byte-level research assistant. Answer briefly.",
        temperature=temp,
        max_new=max_new,
    )
    return convo.say


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["torch", "microtorch"], default="torch")
    ap.add_argument("--ckpt", default="checkpoints/cfna_chat.pt")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--no-assist-tools", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--json", default="")
    ap.add_argument("prompt", nargs="*", help="prompt text; omit with --probe")
    args = ap.parse_args()

    model_fn = (_torch_model_fn if args.backend == "torch" else _micro_model_fn)(
        args.ckpt, args.temp, args.max_new
    )
    assist = not args.no_assist_tools
    if args.probe:
        res = run_probes(model_fn, assist_tools=assist)
        print(json.dumps(res, indent=2))
        if args.json:
            p = Path(args.json)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(res, indent=2), encoding="utf-8")
        return
    prompt = " ".join(args.prompt).strip()
    if not prompt:
        raise SystemExit("provide a prompt or use --probe")
    res = respond(prompt, model_fn, assist_tools=assist)
    print(res.text)
    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res.__dict__, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
