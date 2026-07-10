#!/usr/bin/env python3
"""Non-destructive load benchmark for the Nueronce Engine 35M trainer.

Each sequence length receives its own temporary checkpoint copy and fresh
prepare/backward subprocesses. This prevents one slow case from consuming a
shared command timeout and guarantees the source checkpoint is never modified.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "scripts" / "train_nueronce_engine_35m_split_pretrain.py"


def _last_json(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError(f"no JSON object found in subprocess output:\n{stdout[-2000:]}")


def _run(command: list[str], timeout_s: float, env: dict[str, str]) -> tuple[dict, float]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=False,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout[-4000:]}\n"
            f"stderr:\n{completed.stderr[-4000:]}"
        )
    return _last_json(completed.stdout), elapsed


def run_case(
    source_checkpoint: Path,
    document: Path,
    seq_len: int,
    timeout_s: float,
    seed: int,
) -> dict:
    env = os.environ.copy()
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[name] = "1"

    with tempfile.TemporaryDirectory(prefix=f"nueronce-load-{seq_len}-") as raw_tmp:
        tmp = Path(raw_tmp)
        checkpoint = tmp / source_checkpoint.name
        plan = tmp / "step.plan.pkl"
        shutil.copy2(source_checkpoint, checkpoint)

        prepare_command = [
            sys.executable,
            str(TRAINER),
            "prepare",
            "--checkpoint",
            str(checkpoint),
            "--plan",
            str(plan),
            "--document",
            str(document),
            "--seq-len",
            str(seq_len),
            "--seed",
            str(seed),
        ]
        backward_command = [
            sys.executable,
            str(TRAINER),
            "backward",
            "--plan",
            str(plan),
        ]

        prepared, prepare_wall_s = _run(prepare_command, timeout_s, env)
        completed, backward_wall_s = _run(backward_command, timeout_s, env)

        targets = int(completed.get("supervised_target_bytes", seq_len - 1))
        measured_compute_s = sum(
            float(completed.get(key, 0.0))
            for key in ("forward_s", "backward_s", "update_s")
        )
        return {
            "seq_len": seq_len,
            "status": "completed",
            "loss": completed.get("loss"),
            "grad_norm": completed.get("grad_norm"),
            "grad_tensors": completed.get("grad_tensors"),
            "clip_scale": completed.get("clip_scale"),
            "prepare_forward_s": prepared.get("forward_s"),
            "train_forward_s": completed.get("forward_s"),
            "backward_s": completed.get("backward_s"),
            "update_s": completed.get("update_s"),
            "prepare_wall_s": prepare_wall_s,
            "backward_process_wall_s": backward_wall_s,
            "peak_rss_kib_prepare": prepared.get("peak_rss_kib"),
            "peak_rss_kib_backward": completed.get("peak_rss_kib"),
            "supervised_target_bytes": targets,
            "targets_per_compute_second": (
                targets / measured_compute_s if measured_compute_s > 0 else None
            ),
            "source_checkpoint_unchanged": True,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--document", type=Path, required=True)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--output", type=Path, default=Path("metrics/nueronce_engine_load.json"))
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    document = args.document.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not document.is_file():
        raise FileNotFoundError(document)

    source_stat = checkpoint.stat()
    results: list[dict] = []
    for seq_len in args.seq_lens:
        started = time.perf_counter()
        try:
            result = run_case(checkpoint, document, seq_len, args.timeout_s, args.seed)
        except subprocess.TimeoutExpired as exc:
            result = {
                "seq_len": seq_len,
                "status": "timeout",
                "timeout_s": args.timeout_s,
                "command": exc.cmd,
            }
        except Exception as exc:  # keep later sequence cases running
            result = {
                "seq_len": seq_len,
                "status": "error",
                "error": repr(exc),
            }
        result["case_wall_s"] = time.perf_counter() - started
        results.append(result)
        print(json.dumps(result), flush=True)

    source_after = checkpoint.stat()
    report = {
        "format": "nueronce-engine-load-v1",
        "checkpoint": str(checkpoint),
        "document": str(document),
        "timeout_s_per_process": args.timeout_s,
        "blas_threads": 1,
        "source_checkpoint_unchanged": (
            source_stat.st_size == source_after.st_size
            and source_stat.st_mtime_ns == source_after.st_mtime_ns
        ),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "logged", "output": str(args.output), "cases": len(results)}))


if __name__ == "__main__":
    main()
