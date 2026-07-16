#!/usr/bin/env python3
"""Start NUERONCE 355M from random initialization with a one-step safety gate."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--workspace", default="runs/nueronce_355m_frontier_from_scratch")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--position-mode", choices=["baseline", "phi_rope"], default="phi_rope")
    ap.add_argument("--smoke-only", action="store_true")
    args = ap.parse_args()

    workspace = Path(args.workspace)
    save_dir = workspace / "base" / "checkpoints"
    metrics_dir = workspace / "base" / "metrics"

    smoke = [
        sys.executable,
        "scripts/train_nueronce_engine_355m_base_rft.py",
        "--corpus", args.corpus,
        "--save-dir", str(save_dir),
        "--metrics-dir", str(metrics_dir),
        "--position-mode", args.position_mode,
        "--seq", "16",
        "--batch", "1",
        "--lr", "1e-5",
        "--additional-steps", "1",
        "--seed", str(args.seed),
        "--eval-every", "1",
        "--checkpoint-every", "1",
        "--val-batches", "1",
    ]
    run(smoke)

    if args.smoke_only:
        return

    warmup = [
        sys.executable,
        "scripts/train_nueronce_engine_355m_base_rft.py",
        "--corpus", args.corpus,
        "--save-dir", str(save_dir),
        "--metrics-dir", str(metrics_dir),
        "--position-mode", args.position_mode,
        "--seq", "16",
        "--batch", "1",
        "--lr", "1e-5",
        "--additional-steps", "999",
        "--seed", str(args.seed),
        "--eval-every", "100",
        "--checkpoint-every", "100",
        "--val-batches", "4",
    ]
    run(warmup)


if __name__ == "__main__":
    main()
