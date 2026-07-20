#!/usr/bin/env python3
"""Train the 355M NUERONCE checkpoint on conversational SFT until a measured stop.

This is a restart-safe response-only SFT loop for the Nueronce Engine. It can
initialize from a base-pretraining checkpoint, resumes from ``latest.pt`` after
interruptions, keeps ``best.pt`` by validation loss, and stops when validation
has failed to improve by ``min_delta`` for ``patience`` evaluations after
``min_steps``. A hard ``max_steps`` safety bound is always required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LN2 = 0.6931471805599453


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pickle(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def atomic_pickle(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def load_jsonl(path: Path) -> List[dict]:
    out: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def list_shards(train_dir: Path) -> List[Path]:
    shards = sorted(train_dir.glob("shard_*.jsonl"))
    if not shards:
        raise FileNotFoundError(f"no shard_*.jsonl files found in {train_dir}")
    return shards


def copy_params(payload: dict, model) -> None:
    arrays = payload.get("params")
    if arrays is None:
        raise ValueError("checkpoint has no 'params' list")
    params = list(model.parameters())
    if len(arrays) != len(params):
        raise ValueError(f"parameter-list mismatch: checkpoint={len(arrays)} model={len(params)}")
    for i, (p, arr) in enumerate(zip(params, arrays)):
        if tuple(p.shape) != tuple(arr.shape):
            raise ValueError(f"parameter {i} shape mismatch: checkpoint={arr.shape} model={p.shape}")
        p.data = arr.copy()


def make_batch(records: List[dict], max_len: int) -> dict:
    from nueronce.training.dialogue_data import make_conversation_batch

    return make_conversation_batch([r["messages"] for r in records], max_len=max_len)


def evaluate(model, records: List[dict], *, max_len: int, batch_size: int,
             max_examples: int | None = None) -> dict:
    from nueronce.engine.tensor import no_grad

    if max_examples is not None:
        records = records[:max_examples]
    total_loss = 0.0
    total_correct = 0
    total_targets = 0
    with no_grad():
        for i in range(0, len(records), batch_size):
            chunk = records[i:i + batch_size]
            if not chunk:
                continue
            batch = make_batch(chunk, max_len)
            logits, _ = model.forward(batch["byte_ids"])
            loss = model.masked_token_loss(logits, batch["byte_ids"], batch["target_mask"])
            total_loss += loss.item() * len(chunk)
            pred = logits.data[:, :-1].argmax(-1)
            tgt = batch["byte_ids"][:, 1:]
            sel = batch["target_mask"][:, 1:]
            total_correct += int((pred[sel] == tgt[sel]).sum())
            total_targets += int(sel.sum())
    n = max(1, len(records))
    mean = total_loss / n
    return {
        "loss": mean,
        "bits_per_byte": mean / LN2,
        "byte_accuracy": total_correct / max(1, total_targets),
        "n_examples": len(records),
        "n_targets": total_targets,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--validation", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--init-checkpoint", default="")
    ap.add_argument("--save-dir", default="checkpoints/nueronce_355m_conversation")
    ap.add_argument("--metrics-dir", default="metrics/nueronce_355m_conversation")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--grad-accum-steps", type=int, default=1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--tile-rows", type=int, default=128)
    ap.add_argument("--no-momentum", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps-per-shard", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--checkpoint-every", type=int, default=100)
    ap.add_argument("--eval-batch", type=int, default=1)
    ap.add_argument("--val-examples", type=int, default=128)
    ap.add_argument("--test-examples", type=int, default=128)
    ap.add_argument("--min-steps", type=int, default=1_000)
    ap.add_argument("--max-steps", type=int, default=50_000)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--min-delta", type=float, default=1e-3)
    args = ap.parse_args()

    if args.max_steps <= 0:
        raise SystemExit("--max-steps must be positive")
    if args.eval_every <= 0:
        raise SystemExit("--eval-every must be positive")
    if args.steps_per_shard <= 0:
        raise SystemExit("--steps-per-shard must be positive")

    from nueronce.engine.optim import StreamFactor, clip_grad_norm_
    from nueronce.engine.scaling import base_355m_config, enable_training_dtype

    enable_training_dtype("float32")

    from nueronce.engine.nueronce_model import NueronceModel
    from nueronce.training.dialogue_data import PROMPT_FORMAT

    shards = list_shards(Path(args.train_dir))
    val_records = load_jsonl(Path(args.validation))
    test_records = load_jsonl(Path(args.test))
    save_dir = Path(args.save_dir)
    metrics_dir = Path(args.metrics_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    latest = save_dir / "latest.pt"
    best = save_dir / "best.pt"
    metrics_path = metrics_dir / "conversation_metrics.jsonl"

    np.random.seed(args.seed)
    cfg = base_355m_config()
    model = NueronceModel(cfg)
    params = list(model.parameters())
    opt = StreamFactor(
        params,
        lr=args.lr,
        weight_decay=0.01,
        tile_rows=args.tile_rows,
        momentum=not args.no_momentum,
    )

    meta = {
        "stage": "response_only_conversation_sft",
        "global_step": 0,
        "examples_seen": 0,
        "best_val_loss": float("inf"),
        "best_val_accuracy": 0.0,
        "bad_evals": 0,
        "prompt_format": PROMPT_FORMAT,
        "source_checkpoint_sha256": None,
        "seed": args.seed,
        "max_len": args.max_len,
        "steps_per_shard": args.steps_per_shard,
    }

    if latest.exists():
        payload = load_pickle(latest)
        copy_params(payload, model)
        state = payload.get("optimizer")
        if state is not None:
            opt.load_state_dict(state)
        meta.update(payload.get("meta", {}))
        opt.lr = args.lr
        print(f"resumed SFT from {latest} at step {meta['global_step']}")
    elif args.init_checkpoint:
        init = Path(args.init_checkpoint)
        if not init.exists():
            raise FileNotFoundError(init)
        payload = load_pickle(init)
        copy_params(payload, model)
        meta["source_checkpoint_sha256"] = sha256_file(init)
        print(f"initialized SFT from base checkpoint {init}")
    else:
        raise SystemExit("no latest SFT checkpoint and no --init-checkpoint was provided")

    if model.num_params() != 352_993_825:
        raise SystemExit(f"unexpected parameter count: {model.num_params():,}")

    def log(rec: dict) -> None:
        rec = {**rec, "wall_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)

    def save(path: Path) -> None:
        payload = {
            "config": vars(cfg),
            "params": [p.data.copy() for p in params],
            "optimizer": opt.state_dict(),
            "optimizer_name": "streamfactor",
            "opt_lr": opt.lr,
            "meta": dict(meta),
        }
        atomic_pickle(payload, path)

    if int(meta["global_step"]) == 0:
        pre = evaluate(
            model,
            val_records,
            max_len=args.max_len,
            batch_size=args.eval_batch,
            max_examples=args.val_examples,
        )
        meta["best_val_loss"] = pre["loss"]
        meta["best_val_accuracy"] = pre["byte_accuracy"]
        save(best)
        log({"event": "pre_sft", **pre, "step": 0, "lr": opt.lr})

    start_time = time.time()
    stop_reason = "max_steps"
    cached_shard_index = -1
    cached_records: List[dict] = []

    while int(meta["global_step"]) < args.max_steps:
        step = int(meta["global_step"])
        shard_index = (step // args.steps_per_shard) % len(shards)
        if shard_index != cached_shard_index:
            cached_records = load_jsonl(shards[shard_index])
            if not cached_records:
                raise RuntimeError(f"empty shard: {shards[shard_index]}")
            cached_shard_index = shard_index

        rng = np.random.default_rng(args.seed + 1_000_003 * step)
        replace = len(cached_records) < args.batch
        idx = rng.choice(len(cached_records), size=args.batch, replace=replace)
        chunk = [cached_records[int(i)] for i in np.atleast_1d(idx)]
        batch = make_batch(chunk, args.max_len)
        model.zero_grad()
        micro = max(1, int(np.ceil(len(chunk) / max(1, args.grad_accum_steps))))
        total = 0.0
        seen = 0
        for start in range(0, len(chunk), micro):
            stop = min(start + micro, len(chunk))
            ids = batch["byte_ids"][start:stop]
            mask = batch["target_mask"][start:stop]
            if ids.shape[0] == 0:
                continue
            logits, _ = model.forward(ids)
            loss = model.masked_token_loss(logits, ids, mask)
            weight = ids.shape[0] / max(1, batch["byte_ids"].shape[0])
            (loss * weight).backward()
            total += loss.item() * ids.shape[0]
            seen += ids.shape[0]
        grad_norm = clip_grad_norm_(params, args.grad_clip)
        opt.step()
        meta["global_step"] = step + 1
        meta["examples_seen"] = int(meta["examples_seen"]) + len(chunk)

        current = int(meta["global_step"])
        log({
            "event": "train",
            "step": current,
            "shard": shard_index + 1,
            "train_loss": total / max(1, seen),
            "grad_norm": grad_norm,
            "lr": opt.lr,
            "examples_seen": meta["examples_seen"],
            "elapsed_seconds": time.time() - start_time,
        })

        if args.checkpoint_every > 0 and current % args.checkpoint_every == 0:
            save(latest)

        if current % args.eval_every == 0:
            val = evaluate(
                model,
                val_records,
                max_len=args.max_len,
                batch_size=args.eval_batch,
                max_examples=args.val_examples,
            )
            improved = val["loss"] < float(meta["best_val_loss"]) - args.min_delta
            if improved:
                meta["best_val_loss"] = val["loss"]
                meta["best_val_accuracy"] = val["byte_accuracy"]
                meta["bad_evals"] = 0
                save(best)
            else:
                meta["bad_evals"] = int(meta.get("bad_evals", 0)) + 1
            save(latest)
            log({
                "event": "validation",
                **val,
                "step": current,
                "improved": improved,
                "bad_evals": meta["bad_evals"],
                "best_val_loss": meta["best_val_loss"],
                "best_val_accuracy": meta["best_val_accuracy"],
                "lr": opt.lr,
            })
            if current >= args.min_steps and int(meta["bad_evals"]) >= args.patience:
                stop_reason = "validation_plateau"
                break

    save(latest)
    chosen = best if best.exists() else latest
    payload = load_pickle(chosen)
    copy_params(payload, model)
    test = evaluate(
        model,
        test_records,
        max_len=args.max_len,
        batch_size=args.eval_batch,
        max_examples=args.test_examples,
    )
    summary = {
        "event": "final",
        "stop_reason": stop_reason,
        "step": int(meta["global_step"]),
        "best_val_loss": float(meta["best_val_loss"]),
        "best_val_accuracy": float(meta["best_val_accuracy"]),
        "test": test,
        "best_checkpoint": str(chosen.resolve()),
        "best_checkpoint_sha256": sha256_file(chosen),
        "source_checkpoint_sha256": meta.get("source_checkpoint_sha256"),
    }
    (metrics_dir / "conversation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    log(summary)


if __name__ == "__main__":
    main()
