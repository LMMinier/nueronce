#!/usr/bin/env python3
"""Two-process, crash-safe response-only SFT for MicroTorch CFNA base-35M.

MicroTorch graphs contain Python backward closures and cannot be safely
serialized. This runner therefore splits one optimization step into two
reproducible processes:

1. ``prepare`` selects the phase record and exact byte window, verifies the
   checkpoint, evaluates the forward loss without retaining a graph, and writes
   a compact immutable step plan.
2. ``backward`` verifies the checkpoint hash, reconstructs the exact forward
   graph from the saved bytes, performs backward/update, atomically advances the
   model checkpoint, and marks the plan consumed.

The backward process gets its own complete runtime and does not spend time
searching data or performing a separate planning pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cfna.microtorch import functional as F
from cfna.microtorch.cfna_model import MicroCFNAModel, preset_configs
from cfna.microtorch.optim import clip_grad_norm_, validate_finite_gradients
from cfna.microtorch.tensor import no_grad
from scripts.train_microtorch_35m_phased_sft import (
    PHASES,
    PagedFactorState,
    choose_window,
    load_records,
    render_record,
)

PLAN_VERSION = "microtorch-base35m-split-step-v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_pickle(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_model_checkpoint(checkpoint_path: Path):
    cfg = preset_configs()["base_35m"]
    model = MicroCFNAModel(cfg)
    with checkpoint_path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    params = list(model.parameters())
    if len(checkpoint["params"]) != len(params):
        raise RuntimeError("checkpoint parameter count does not match base_35m")
    for parameter, stored in zip(params, checkpoint["params"]):
        parameter.data[...] = stored
    return cfg, model, checkpoint


def response_loss(model, batch: np.ndarray, first_target: int):
    logits, _ = model.forward(batch)
    response_logits = logits[:, first_target - 1:-1].reshape(-1, 256)
    response_targets = batch[:, first_target:].reshape(-1)
    return F.cross_entropy(response_logits, response_targets)


def prepare(args) -> None:
    started = time.time()
    checkpoint_hash = sha256(args.checkpoint)
    cfg, model, checkpoint = load_model_checkpoint(args.checkpoint)
    next_step = int(checkpoint["meta"].get("step", 0)) + 1
    history = checkpoint["meta"].get("history", [])
    phase_seen = sum(1 for item in history if item.get("phase") == args.phase)
    records = load_records(ROOT, args.phase)
    record = records[phase_seen % len(records)]
    context, response = render_record(record)
    sequence, target_mask, response_offset = choose_window(
        context, response, args.seq_len, phase_seen
    )
    batch = np.frombuffer(sequence, dtype=np.uint8).astype(np.int64)[None, :]
    first_target = int(np.nonzero(target_mask)[0][0])

    t0 = time.time()
    with no_grad():
        loss = response_loss(model, batch, first_target)
    forward_s = time.time() - t0
    if not np.isfinite(loss.item()):
        raise FloatingPointError("prepare forward produced nonfinite loss")

    plan = {
        "version": PLAN_VERSION,
        "status": "prepared",
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_step": next_step - 1,
        "next_step": next_step,
        "phase": args.phase,
        "record_id": record.get("id"),
        "category": record.get("category"),
        "objective": "assistant_response_masked_sft",
        "sequence": sequence,
        "target_mask": target_mask,
        "first_target": first_target,
        "response_offset": response_offset,
        "sequence_bytes": len(sequence),
        "supervised_target_bytes": int(target_mask.sum()),
        "expected_forward_loss": float(loss.item()),
        "prepare_forward_s": forward_s,
        "lr": args.lr,
        "max_grad_norm": args.max_grad_norm,
        "config": cfg.__dict__,
        "created_unix": time.time(),
    }
    atomic_pickle(args.plan, plan)
    print(json.dumps({
        "status": "prepared",
        "plan": str(args.plan),
        "next_step": next_step,
        "phase": args.phase,
        "record_id": record.get("id"),
        "loss": float(loss.item()),
        "forward_s": forward_s,
        "elapsed_s": time.time() - started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "checkpoint_sha256": checkpoint_hash,
    }), flush=True)


def backward(args) -> None:
    started = time.time()
    with args.plan.open("rb") as handle:
        plan = pickle.load(handle)
    if plan.get("version") != PLAN_VERSION:
        raise RuntimeError("unsupported or invalid split-step plan")
    if plan.get("status") != "prepared":
        raise RuntimeError(f"step plan is not runnable: status={plan.get('status')}")

    checkpoint_path = Path(plan["checkpoint_path"])
    current_hash = sha256(checkpoint_path)
    if current_hash != plan["checkpoint_sha256"]:
        raise RuntimeError(
            "checkpoint changed after prepare; refusing stale backward run"
        )

    cfg, model, checkpoint = load_model_checkpoint(checkpoint_path)
    if int(checkpoint["meta"].get("step", 0)) != int(plan["checkpoint_step"]):
        raise RuntimeError("checkpoint step changed after prepare")

    old_opt = checkpoint["optimizer"]
    optimizer = PagedFactorState(
        model.parameters(),
        lr=plan["lr"] if plan["lr"] is not None else old_opt.get("lr", 1e-5),
        step=old_opt.get("t", checkpoint["meta"].get("step", 0)),
        state=old_opt["v"],
    )
    batch = np.frombuffer(plan["sequence"], dtype=np.uint8).astype(np.int64)[None, :]
    optimizer.zero_grad()

    t0 = time.time()
    loss = response_loss(model, batch, int(plan["first_target"]))
    graph_forward_s = time.time() - t0
    expected = float(plan["expected_forward_loss"])
    if not np.isclose(loss.item(), expected, rtol=1e-6, atol=1e-7):
        raise RuntimeError(
            f"reconstructed forward loss {loss.item()} != prepared loss {expected}"
        )

    t0 = time.time()
    loss.backward()
    backward_s = time.time() - t0
    grad_tensors = validate_finite_gradients(model.parameters())
    grad_norm = clip_grad_norm_(model.parameters(), float(plan["max_grad_norm"]))

    t0 = time.time()
    optimizer.update()
    update_s = time.time() - t0

    record_log = {
        "step": int(plan["next_step"]),
        "phase": plan["phase"],
        "objective": plan["objective"],
        "record_id": plan["record_id"],
        "category": plan["category"],
        "response_offset": int(plan["response_offset"]),
        "sequence_bytes": int(plan["sequence_bytes"]),
        "supervised_target_bytes": int(plan["supervised_target_bytes"]),
        "loss": float(loss.item()),
        "grad_norm": float(grad_norm),
        "grad_tensors": int(grad_tensors),
        "prepare_forward_s": float(plan["prepare_forward_s"]),
        "graph_forward_s": graph_forward_s,
        "backward_s": backward_s,
        "update_s": update_s,
        "execution": "split_prepare_backward",
    }
    history = list(checkpoint["meta"].get("history", []))
    history.append(record_log)
    payload = {
        "format": "microtorch-base35m-phased-sft-v3-split",
        "config": cfg.__dict__,
        "params": [p.data for p in model.parameters()],
        "optimizer": {
            "name": "streamfactor",
            "lr": optimizer.lr,
            "t": optimizer.step_index,
            "v": optimizer.state,
        },
        "meta": {
            **checkpoint["meta"],
            "step": int(plan["next_step"]),
            "data_cursor": int(plan["next_step"]),
            "training_objective": "response-only masked SFT",
            "phase": plan["phase"],
            "history": history,
        },
    }
    t0 = time.time()
    atomic_pickle(checkpoint_path, payload)
    save_s = time.time() - t0

    completed = dict(plan)
    completed.update({
        "status": "completed",
        "completed_unix": time.time(),
        "result": record_log,
        "output_checkpoint_sha256": sha256(checkpoint_path),
    })
    atomic_pickle(args.plan, completed)

    print(json.dumps({
        **record_log,
        "status": "completed",
        "plan": str(args.plan),
        "checkpoint": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": completed["output_checkpoint_sha256"],
        "save_s": save_s,
        "elapsed_s": time.time() - started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--phase", choices=tuple(PHASES), required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--seq-len", type=int, default=9)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.set_defaults(func=prepare)

    b = sub.add_parser("backward")
    b.add_argument("--plan", type=Path, required=True)
    b.set_defaults(func=backward)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
