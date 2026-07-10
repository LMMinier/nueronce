#!/usr/bin/env python3
"""Run NUERONCE MicroTorch 35M training with CPU-safe BLAS defaults.

MicroTorch performs many small matrix multiplications. On machines exposing large
OpenBLAS/OMP thread pools, thread-launch overhead can make backward hundreds of
times slower. This wrapper pins those libraries to one worker unless the caller
already selected a value, then launches the requested training program in a
fresh child process so the settings take effect before NumPy is imported.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROGRAMS = {
    "pretrain": ROOT / "scripts/train_microtorch_35m_split_pretrain.py",
    "sft": ROOT / "scripts/train_microtorch_35m_split_step.py",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("program", choices=tuple(PROGRAMS))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    ns = parser.parse_args()

    env = os.environ.copy()
    env.setdefault("MICROTORCH_DTYPE", "float32")
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env.setdefault(name, "1")

    command = [sys.executable, str(PROGRAMS[ns.program]), *ns.args]
    return subprocess.run(command, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
