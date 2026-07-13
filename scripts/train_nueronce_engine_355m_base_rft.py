#!/usr/bin/env python3
"""Resumable 355M NUERONCE base pretraining with an optional Phi-RoPE ablation.

Unlike ``train_nueronce_engine_355m.py`` (response-only conversation SFT), this
launcher performs ordinary next-byte base pretraining on the built corpus.
The ``--position-mode phi_rope`` path changes only self-attention Q/K geometry;
it adds no parameters and can load the same baseline checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LN2 = 0.6931471805599453


def atomic_pickle(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def deterministic_val_batches(corpus, seq_len: int, batch_size: int,
                              max_batches: int) -> list[np.ndarray]:
    windows = list(corpus.iter_val_windows(seq_len))
    if not windows:
        return []
    rng = np.random.default_rng(123)
    rng.shuffle(windows)
    out = []
    limit = min(len(windows), batch_size * max_batches)
    for i in range(0, limit, batch_size):
        chunk = windows[i:i + batch_size]
        if len(chunk) != batch_size:
            break
        out.append(np.stack(chunk))
    return out


def evaluate(model, batches: Iterable[np.ndarray]) -> dict:
    from nueronce.engine.tensor import no_grad

    losses = []
    with no_grad():
        for batch in batches:
            loss = model.lm_loss(batch)
            losses.append(loss.item())
    if not losses:
        return {"loss": float("nan"), "bpb": float("nan")}
    mean = float(np.mean(losses))
    return {"loss": mean, "bpb": mean / LN2}


def load_into(payload: dict, model, opt) -> int:
    arrays = payload.get("params")
    if arrays is None:
        raise ValueError("checkpoint has no 'params' list")
    params = list(model.parameters())
    if len(arrays) != len(params):
        raise ValueError(f"parameter-list mismatch: checkpoint={len(arrays)} model={len(params)}")
    for index, (p, arr) in enumerate(zip(params, arrays)):
        if tuple(p.shape) != tuple(arr.shape):
            raise ValueError(
                f"parameter {index} shape mismatch: checkpoint={arr.shape} model={p.shape}"
            )
        p.data = arr.copy()

    state = payload.get("optimizer")
    if state is not None:
        opt.load_state_dict(state)
    else:
        legacy = payload.get("optimizer_state")
        if legacy is not None:
            opt.load_state_dict(legacy)
    meta = payload.get("meta", {})
    return int(meta.get("global_step", meta.get("optimizer_step", payload.get("step", 0))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--save-dir", default="checkpoints/nueronce_engine_355m_base")
    ap.add_argument("--metrics-dir", default="metrics/nueronce_engine_355m_base")
    ap.add_argument("--resume-from", default="")
    ap.add_argument("--position-mode", choices=["baseline", "phi_rope"], default="baseline")
    ap.add_argument("--seq", type=int, default=16)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--additional-steps", type=int, default=10)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--tile-rows", type=int, default=128)
    ap.add_argument("--no-momentum", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--val-batches", type=int, default=4)
    args = ap.parse_args()

    from nueronce.corpus.dataset import ByteCorpus
    from nueronce.engine.optim import StreamFactor, clip_grad_norm_
    from nueronce.engine.scaling import base_355m_config, enable_training_dtype

    enable_training_dtype("float32")
    if args.position_mode == "phi_rope":
        from nueronce.engine.rft_attention import install_phi_rotary_attention
        install_phi_rotary_attention()

    from nueronce.engine.nueronce_model import NueronceModel

    train = ByteCorpus(Path(args.corpus), "train")
    val = ByteCorpus(Path(args.corpus), "val")
    val_batches = deterministic_val_batches(val, args.seq, args.batch, args.val_batches)

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

    save_dir = Path(args.save_dir)
    metrics_dir = Path(args.metrics_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    latest = save_dir / "latest.pkl"
    metrics_path = metrics_dir / "base_metrics.jsonl"

    resume = Path(args.resume_from) if args.resume_from else latest
    global_step = 0
    source_sha256 = None
    if resume.exists():
        with resume.open("rb") as f:
            payload = pickle.load(f)
        global_step = load_into(payload, model, opt)
        opt.lr = args.lr
        source_sha256 = sha256_file(resume)
        print(f"resumed {resume} at step {global_step}; sha256={source_sha256}")
    else:
        print("starting fresh; no resume checkpoint found")

    print(
        f"model={model.num_params():,} params position_mode={args.position_mode} "
        f"seq={args.seq} batch={args.batch} lr={args.lr:g}"
    )
    if model.num_params() != 352_993_825:
        raise SystemExit(f"unexpected parameter count: {model.num_params():,}")

    stop_step = global_step + args.additional_steps
    t0 = time.time()

    def log(record: dict) -> None:
        record = {**record, "wall_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    def save() -> None:
        payload = {
            "config": vars(cfg),
            "params": [p.data.copy() for p in params],
            "optimizer": opt.state_dict(),
            "optimizer_name": "streamfactor",
            "opt_lr": opt.lr,
            "meta": {
                "stage": "byte_base_pretraining",
                "global_step": global_step,
                "optimizer_step": opt.t,
                "position_mode": args.position_mode,
                "source_checkpoint_sha256": source_sha256,
                "dtype": "float32",
                "preset": "base_355m",
                "seq": args.seq,
                "batch": args.batch,
                "seed": args.seed,
            },
        }
        atomic_pickle(payload, latest)
        print(f"checkpoint={latest} sha256={sha256_file(latest)}", flush=True)

    while global_step < stop_step:
        step_rng = np.random.default_rng(args.seed + 1_000_003 * global_step)
        batch = train.sample_batch(args.seq, args.batch, step_rng)
        model.zero_grad()
        loss, stats = model.loss(batch)
        loss.backward()
        grad_norm = clip_grad_norm_(params, args.grad_clip)
        opt.step()
        global_step += 1

        log({
            "event": "train",
            "step": global_step,
            "loss": stats["loss"],
            "lm_loss": stats["lm"],
            "train_bpb": stats["bpb"],
            "boundary_loss": stats["boundary"],
            "grad_norm": grad_norm,
            "lr": opt.lr,
            "position_mode": args.position_mode,
            "elapsed_seconds": time.time() - t0,
        })

        if args.eval_every > 0 and global_step % args.eval_every == 0:
            held = evaluate(model, val_batches)
            log({
                "event": "validation",
                "step": global_step,
                "heldout_loss": held["loss"],
                "heldout_bpb": held["bpb"],
                "position_mode": args.position_mode,
                "elapsed_seconds": time.time() - t0,
            })

        if args.checkpoint_every > 0 and global_step % args.checkpoint_every == 0:
            save()

    save()


if __name__ == "__main__":
    main()
