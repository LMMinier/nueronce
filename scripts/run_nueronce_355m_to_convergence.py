#!/usr/bin/env python3
"""Orchestrate 355M base pretraining -> conversational SFT -> chat probe.

The controller is restart-safe because each stage owns atomic checkpoints. It
runs base pretraining in bounded chunks, requires the held-out BPB gate before
SFT, then runs response-only conversational training until validation plateaus
or a hard safety limit is reached. It is intended to be launched detached by
``scripts/nueronce_355m_job.py`` on the user's machine or Colab runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_step(path: Path) -> int:
    """Compatibility helper for tests and small checkpoints.

    The live controller intentionally does not call this for 355M checkpoints;
    progress is read from lightweight JSONL metrics to avoid loading gigabytes
    of parameters merely to inspect metadata.
    """
    if not path.exists():
        return 0
    with path.open("rb") as f:
        payload = pickle.load(f)
    meta = payload.get("meta") or {}
    return int(meta.get("global_step", meta.get("optimizer_step", payload.get("step", 0))))


def jsonl_records(path: Path, event: str | None = None) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event is None or rec.get("event") == event:
                out.append(rec)
    return out


def latest_metric_step(path: Path) -> int:
    records = jsonl_records(path, "train")
    return int(records[-1].get("step", 0)) if records else 0


def run(cmd: Iterable[str], *, cwd: Path) -> None:
    command = [str(x) for x in cmd]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--source-checkpoint", required=True)
    ap.add_argument("--workspace", default="runs/nueronce_355m_convergence")
    ap.add_argument("--position-mode", choices=["baseline", "phi_rope"], default="phi_rope")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--base-target-bpb", type=float, default=1.8)
    ap.add_argument("--base-chunk-steps", type=int, default=100)
    ap.add_argument("--base-max-steps", type=int, default=50_000)
    ap.add_argument("--base-patience", type=int, default=8)
    ap.add_argument("--base-min-delta", type=float, default=2e-3)
    ap.add_argument("--base-seq", type=int, default=16)
    ap.add_argument("--base-batch", type=int, default=1)
    ap.add_argument("--base-lr", type=float, default=1e-5)
    ap.add_argument("--base-eval-every", type=int, default=100)
    ap.add_argument("--base-val-batches", type=int, default=4)

    ap.add_argument("--sft-data", default="data/conversation_sft")
    ap.add_argument("--build-sft-if-missing", action="store_true")
    ap.add_argument("--sft-batch", type=int, default=1)
    ap.add_argument("--sft-max-len", type=int, default=128)
    ap.add_argument("--sft-lr", type=float, default=1e-5)
    ap.add_argument("--sft-eval-every", type=int, default=100)
    ap.add_argument("--sft-checkpoint-every", type=int, default=100)
    ap.add_argument("--sft-eval-batch", type=int, default=1)
    ap.add_argument("--sft-val-examples", type=int, default=128)
    ap.add_argument("--sft-test-examples", type=int, default=128)
    ap.add_argument("--sft-min-steps", type=int, default=1_000)
    ap.add_argument("--sft-max-steps", type=int, default=50_000)
    ap.add_argument("--sft-patience", type=int, default=8)
    ap.add_argument("--sft-min-delta", type=float, default=1e-3)
    ap.add_argument("--skip-chat-probe", action="store_true")
    args = ap.parse_args()

    source = Path(args.source_checkpoint).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    corpus = Path(args.corpus).resolve()
    if not corpus.exists():
        raise FileNotFoundError(corpus)
    source_hash = sha256_file(source)

    workspace = Path(args.workspace).resolve()
    base_save = workspace / "base" / "checkpoints"
    base_metrics = workspace / "base" / "metrics"
    sft_save = workspace / "conversation" / "checkpoints"
    sft_metrics = workspace / "conversation" / "metrics"
    state_path = workspace / "state.json"
    workspace.mkdir(parents=True, exist_ok=True)

    previous = {}
    if state_path.exists():
        try:
            previous = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            previous = {}
    if previous.get("source_checkpoint_sha256") not in (None, source_hash):
        raise SystemExit("workspace belongs to a different source checkpoint")
    if previous.get("position_mode") not in (None, args.position_mode):
        raise SystemExit("workspace belongs to a different position mode")
    if previous.get("status") == "completed":
        print(json.dumps(previous, indent=2), flush=True)
        return

    state = {
        "status": "resuming" if previous else "starting",
        "phase": previous.get("phase", "base"),
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": source_hash,
        "position_mode": args.position_mode,
        "started_at": previous.get(
            "started_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        ),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **previous,
    }
    state["status"] = "resuming" if previous else "starting"
    atomic_json(state, state_path)

    latest_base = base_save / "latest.pkl"
    base_metrics_file = base_metrics / "base_metrics.jsonl"
    best_bpb = float(state.get("best_base_bpb", "inf"))
    bad_checks = int(state.get("base_bad_checks", 0))
    current_step = max(int(state.get("base_step", 0)), latest_metric_step(base_metrics_file))

    while state.get("base_gate") != "passed" and current_step < args.base_max_steps:
        resume_args: list[str] = []
        if not latest_base.exists():
            resume_args = ["--resume-from", str(source)]
        remaining = args.base_max_steps - current_step
        chunk = min(args.base_chunk_steps, remaining)
        run([
            sys.executable,
            "scripts/train_nueronce_engine_355m_base_rft.py",
            "--corpus", str(corpus),
            "--save-dir", str(base_save),
            "--metrics-dir", str(base_metrics),
            "--position-mode", args.position_mode,
            "--seq", str(args.base_seq),
            "--batch", str(args.base_batch),
            "--lr", str(args.base_lr),
            "--additional-steps", str(chunk),
            "--seed", str(args.seed),
            "--eval-every", str(args.base_eval_every),
            "--checkpoint-every", str(max(1, min(chunk, args.base_eval_every))),
            "--val-batches", str(args.base_val_batches),
            *resume_args,
        ], cwd=REPO_ROOT)

        current_step = latest_metric_step(base_metrics_file)
        vals = jsonl_records(base_metrics_file, "validation")
        if not vals:
            raise RuntimeError("base trainer produced no validation record")
        latest_val = vals[-1]
        bpb = float(latest_val["heldout_bpb"])
        if bpb < best_bpb - args.base_min_delta:
            best_bpb = bpb
            bad_checks = 0
        else:
            bad_checks += 1
        state.update({
            "status": "running",
            "phase": "base",
            "base_step": current_step,
            "latest_base_validation_step": int(latest_val["step"]),
            "latest_base_bpb": bpb,
            "best_base_bpb": best_bpb,
            "base_bad_checks": bad_checks,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        if bpb <= args.base_target_bpb:
            state["base_gate"] = "passed"
        elif bad_checks >= args.base_patience:
            state.update({
                "status": "stopped",
                "base_gate": "failed_plateau",
                "reason": (
                    f"held-out BPB plateaued above the {args.base_target_bpb} SFT gate; "
                    "conversational training was not started"
                ),
            })
        atomic_json(state, state_path)
        if state.get("base_gate") == "passed":
            break
        if state.get("base_gate") == "failed_plateau":
            raise SystemExit(2)

    if state.get("base_gate") != "passed":
        vals = jsonl_records(base_metrics_file, "validation")
        last_bpb = float(vals[-1]["heldout_bpb"]) if vals else float("inf")
        if last_bpb <= args.base_target_bpb:
            state["base_gate"] = "passed"
        else:
            state.update({
                "status": "stopped",
                "base_gate": "failed_max_steps",
                "reason": f"base max steps reached with held-out BPB {last_bpb:.4f}",
            })
            atomic_json(state, state_path)
            raise SystemExit(3)

    if not latest_base.exists():
        raise FileNotFoundError("base gate passed but latest base checkpoint is missing")

    sft_data = Path(args.sft_data).resolve()
    train_dir = sft_data / "train_shards"
    validation = sft_data / "validation.jsonl"
    test = sft_data / "test.jsonl"
    if not (train_dir.exists() and validation.exists() and test.exists()):
        if not args.build_sft_if_missing:
            raise FileNotFoundError(
                f"SFT data missing under {sft_data}; rerun with --build-sft-if-missing"
            )
        run([
            sys.executable,
            "scripts/build_conversation_sft.py",
            "--out-dir", str(sft_data),
            "--seed", str(args.seed),
        ], cwd=REPO_ROOT)

    state.update({
        "status": "running",
        "phase": "conversation_sft",
        "base_checkpoint": str(latest_base),
        "base_checkpoint_sha256": sha256_file(latest_base),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    atomic_json(state, state_path)

    summary_path = sft_metrics / "conversation_summary.json"
    if not summary_path.exists():
        run([
            sys.executable,
            "scripts/train_nueronce_engine_355m_sft_converge.py",
            "--train-dir", str(train_dir),
            "--validation", str(validation),
            "--test", str(test),
            "--init-checkpoint", str(latest_base),
            "--save-dir", str(sft_save),
            "--metrics-dir", str(sft_metrics),
            "--position-mode", args.position_mode,
            "--batch", str(args.sft_batch),
            "--max-len", str(args.sft_max_len),
            "--lr", str(args.sft_lr),
            "--seed", str(args.seed),
            "--eval-every", str(args.sft_eval_every),
            "--checkpoint-every", str(args.sft_checkpoint_every),
            "--eval-batch", str(args.sft_eval_batch),
            "--val-examples", str(args.sft_val_examples),
            "--test-examples", str(args.sft_test_examples),
            "--min-steps", str(args.sft_min_steps),
            "--max-steps", str(args.sft_max_steps),
            "--patience", str(args.sft_patience),
            "--min-delta", str(args.sft_min_delta),
        ], cwd=REPO_ROOT)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    best_checkpoint = Path(summary["best_checkpoint"])
    state.update({
        "status": "running",
        "phase": "chat_probe",
        "conversation_summary": summary,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    atomic_json(state, state_path)

    if not args.skip_chat_probe:
        probe_out = sft_metrics / "chat_probe.json"
        if not probe_out.exists():
            run([
                sys.executable,
                "scripts/probe_nueronce_engine_chat.py",
                "--checkpoint", str(best_checkpoint),
                "--out", str(probe_out),
                "--temperature", "0",
            ], cwd=REPO_ROOT)
        state["chat_probe"] = json.loads(probe_out.read_text(encoding="utf-8"))

    state.update({
        "status": "completed",
        "phase": "done",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    atomic_json(state, state_path)
    print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    main()
