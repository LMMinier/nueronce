#!/usr/bin/env python3
"""Run a fixed conversational probe against a Nueronce Engine checkpoint."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_PROMPTS = [
    "Hello. Introduce yourself in one sentence.",
    "What is two plus three?",
    "Explain why evidence matters when answering a question.",
    "Write one sentence about the ocean.",
    "I feel stuck while learning. Give me one practical next step.",
    "What should an assistant do when it is unsure?",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="metrics/nueronce_355m_conversation/chat_probe.json")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--max-ctx", type=int, default=288)
    ap.add_argument("--prompt", action="append", default=[])
    args = ap.parse_args()

    checkpoint = Path(args.checkpoint)
    with checkpoint.open("rb") as f:
        payload = pickle.load(f)
    position_mode = (payload.get("meta") or {}).get("position_mode", "baseline")
    if position_mode == "phi_rope":
        from nueronce.engine.rft_attention import install_phi_rotary_attention
        install_phi_rotary_attention()

    from nueronce.engine.chat import MicroConversation
    from nueronce.engine.nueronce_model import NueronceConfig, NueronceModel

    model = NueronceModel(NueronceConfig(**payload["config"]))
    params = list(model.parameters())
    arrays = payload["params"]
    if len(params) != len(arrays):
        raise ValueError(f"parameter-list mismatch: checkpoint={len(arrays)} model={len(params)}")
    for p, arr in zip(params, arrays):
        p.data = arr.copy()

    prompt_format = MicroConversation.resolve_format(payload)
    prompts = args.prompt or DEFAULT_PROMPTS
    results = []
    for prompt in prompts:
        chat = MicroConversation(
            model=model,
            temperature=args.temperature,
            max_new=args.max_new,
            max_ctx=args.max_ctx,
            prompt_format=prompt_format,
            use_incremental=False,
        )
        reply = chat.say(prompt)
        printable = sum(ch.isprintable() or ch.isspace() for ch in reply) / max(1, len(reply))
        results.append({
            "prompt": prompt,
            "reply": reply,
            "nonempty": bool(reply.strip()),
            "printable_fraction": printable,
        })

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": str(checkpoint),
        "position_mode": position_mode,
        "prompt_format": prompt_format,
        "temperature": args.temperature,
        "pass_nonempty": sum(int(r["nonempty"]) for r in results),
        "pass_printable": sum(int(r["printable_fraction"] >= 0.95) for r in results),
        "total": len(results),
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
