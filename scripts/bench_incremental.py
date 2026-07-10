#!/usr/bin/env python3
"""Benchmark dense vs incremental generation on the NumPy backend.

This is the 'deem itself sufficient' gate for the incremental engine: it
reports wall-clock bytes/sec for both paths on identical greedy generations
(so outputs are verifiably equal) at a chat-shaped context length. Run on
any CPU-only box; commit the printed block to the PR/report.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from nueronce.engine.nueronce_model import NueronceModel, NueronceConfig
from nueronce.engine.incremental import IncrementalGenerator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--max-ctx", type=int, default=320)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    model = NueronceModel(NueronceConfig())  # default mid config
    rng = np.random.default_rng(args.seed)
    prompt = bytes(int(b) for b in rng.integers(97, 122, size=args.prompt_len))

    t0 = time.time()
    dense = model.generate(prompt, max_new=args.max_new, greedy=True, max_ctx=args.max_ctx)
    t_dense = time.time() - t0

    inc = IncrementalGenerator(model)
    t0 = time.time()
    fast = inc.generate(prompt, max_new=args.max_new, greedy=True, max_ctx=args.max_ctx)
    t_inc = time.time() - t0

    assert fast == dense, "outputs diverged — do NOT ship"
    n = args.max_new
    print(f"config: default NueronceConfig ({model.num_params():,} params) | "
          f"prompt {args.prompt_len}B | {n} new bytes | max_ctx {args.max_ctx}")
    print(f"dense      : {t_dense:7.2f}s  ({n / t_dense:6.2f} bytes/s)")
    print(f"incremental: {t_inc:7.2f}s  ({n / t_inc:6.2f} bytes/s)")
    print(f"speedup    : {t_dense / t_inc:5.2f}x  | outputs byte-identical: True")


if __name__ == "__main__":
    main()
