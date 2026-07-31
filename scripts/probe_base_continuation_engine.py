#!/usr/bin/env python3
"""Probe raw base-model continuations without applying chat serialization."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import pickle
import time
from pathlib import Path

from nueronce.engine.nueronce_model import NueronceConfig, NueronceModel


DEFAULT_PROMPTS = [
    "Once upon a time",
    "The evidence shows that",
    "Two plus three is",
]


def repetition_stats(data: bytes, n: int = 4) -> dict:
    grams = [data[i:i + n] for i in range(max(0, len(data) - n + 1))]
    counts = Counter(grams)
    repeated = sum(count - 1 for count in counts.values())
    return {
        "bytes": len(data),
        "ngram": n,
        "repeated_ngram_fraction": repeated / max(1, len(grams)),
        "dominant_ngram_fraction": max(counts.values(), default=0) / max(1, len(grams)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-new", type=int, default=24)
    parser.add_argument("--max-ctx", type=int, default=1024)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--dense", action="store_true")
    args = parser.parse_args()

    with args.checkpoint.open("rb") as handle:
        payload = pickle.load(handle)
    model = NueronceModel(NueronceConfig(**payload["config"]))
    params = list(model.parameters())
    if len(params) != len(payload["params"]):
        raise ValueError("parameter-list mismatch")
    for parameter, stored in zip(params, payload["params"]):
        parameter.data = stored.copy()

    results = []
    for prompt in args.prompt or DEFAULT_PROMPTS:
        prompt_bytes = prompt.encode("utf-8")
        generator = model
        if not args.dense:
            from nueronce.engine.incremental import IncrementalGenerator
            generator = IncrementalGenerator(model)
        generated = generator.generate(
            prompt_bytes,
            max_new=args.max_new,
            max_ctx=args.max_ctx,
            temperature=args.temperature,
            greedy=args.temperature <= 0.0,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            stop_bytes=None,
            min_new=0,
        )
        continuation = generated[len(prompt_bytes):]
        results.append({
            "prompt": prompt,
            "continuation_bytes_hex": continuation.hex(),
            "continuation": continuation.decode("utf-8", errors="replace"),
            "repetition": repetition_stats(continuation),
        })

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": str(args.checkpoint),
        "step": int((payload.get("meta") or {}).get("step", 0)),
        "mode": "raw_base_continuation_dense" if args.dense else "raw_base_continuation_incremental",
        "decoding": {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
        },
        "max_repeated_4gram_fraction": max(
            (r["repetition"]["repeated_ngram_fraction"] for r in results), default=0.0
        ),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
