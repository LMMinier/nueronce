#!/usr/bin/env python3
"""Measure peak GPU memory and forward time vs context length: CFNA vs a
dense-attention transformer at (approximately) matched parameters.

This produces the memory-flatness figure the architecture predicts: CFNA's
core runs on patch-compressed units with windowed attention and a fixed-size
SSM state, so its footprint should grow far slower with context than a dense
transformer whose attention is O(T^2). Random weights are fine — memory and
FLOP shape do not depend on weight values — so this needs no training and
runs in minutes.

Honest scope: forward-pass peak memory + wall time, batch 1, fp16 autocast on
CUDA (fp32 on CPU, where only time is reported). Generation-time KV growth is
a separate (also real) effect not measured here. Writes JSON to metrics/.

Usage (Colab GPU):
    python scripts/bench_context_scaling.py --preset base_35m \
        --baseline-d-model 512 --baseline-layers 10 \
        --contexts 512,1024,2048,4096,8192,16384
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from cfna.baselines import BaselineConfig, ByteTransformerLM
from cfna.model import CFNAModel, CONFIG_PRESETS


def measure(model, ctx: int, device: str, amp: bool):
    model.eval()
    x = torch.randint(32, 126, (1, ctx), device=device)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        try:
            out = model(x)
            _ = out[0] if isinstance(out, tuple) else out
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return {"ctx": ctx, "oom": True}
    if device == "cuda":
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1e6
    else:
        peak = None
    return {"ctx": ctx, "oom": False, "seconds": round(time.time() - t0, 4),
            "peak_mb": round(peak, 1) if peak is not None else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="base_35m", choices=sorted(CONFIG_PRESETS))
    ap.add_argument("--baseline-d-model", type=int, default=512)
    ap.add_argument("--baseline-layers", type=int, default=10)
    ap.add_argument("--baseline-heads", type=int, default=8)
    ap.add_argument("--contexts", default="512,1024,2048,4096,8192,16384")
    ap.add_argument("--out", default="metrics/context_scaling.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    ctxs = [int(c) for c in args.contexts.split(",")]

    cfna = CFNAModel(CONFIG_PRESETS[args.preset]()).to(device)
    results = {"device": device,
               "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
               "cfna": {"preset": args.preset, "n_params": cfna.num_params(), "rows": []},
               "transformer": {"rows": []}}
    print(f"CFNA {args.preset}: {cfna.num_params():,} params")
    for ctx in ctxs:
        row = measure(cfna, ctx, device, amp)
        results["cfna"]["rows"].append(row)
        print(f"  cfna ctx {ctx:6d}: {row}")
    del cfna
    if device == "cuda":
        torch.cuda.empty_cache()

    for ctx in ctxs:
        # dense attention needs max_len >= ctx; rebuild per context
        cfg = BaselineConfig(d_model=args.baseline_d_model, n_layers=args.baseline_layers,
                             n_heads=args.baseline_heads, max_len=ctx)
        tf = ByteTransformerLM(cfg).to(device)
        if not results["transformer"].get("n_params"):
            results["transformer"]["n_params"] = tf.num_params()
            print(f"transformer baseline: {tf.num_params():,} params "
                  f"(match this to the preset; adjust --baseline-* if off)")
        row = measure(tf, ctx, device, amp)
        results["transformer"]["rows"].append(row)
        print(f"  tf   ctx {ctx:6d}: {row}")
        del tf
        if device == "cuda":
            torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
