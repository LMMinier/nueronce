#!/usr/bin/env python3
"""ForgeLoop response-only SFT on the NumPy-only Nueronce Engine."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

from nueronce.engine.nueronce_model import NueronceConfig, NueronceModel
from nueronce.engine.optim import StreamFactor, clip_grad_norm_
from nueronce.engine.scaling import enable_training_dtype
from nueronce.engine.tensor import no_grad
from nueronce.training.dialogue_data import encode_example


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def target_window(row: dict, system: str, max_len: int,
                  min_context: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Keep response targets when a formatted example exceeds ``max_len``."""
    encoded, mask = encode_example(row["prompt"], row["response"], system=system)
    data = np.frombuffer(encoded, dtype=np.uint8).astype(np.int64)
    mask_array = np.asarray(mask, dtype=bool)
    targets = np.flatnonzero(mask_array)
    if not len(targets):
        raise ValueError("example has no response targets")
    if len(data) > max_len:
        end = min(len(data), int(targets[-1]) + 1)
        start = max(0, end - max_len)
        data, mask_array = data[start:end], mask_array[start:end]
        # If the crop begins inside a long response, use its leading bytes as
        # causal context instead of supervising an all-target window with no
        # prompt/previous-response conditioning.
        if start > 0:
            mask_array[:min(min_context, len(mask_array) - 1)] = False
    if not bool(mask_array[1:].any()):
        raise ValueError("target-preserving crop produced no shifted targets")
    return data[None, :], mask_array[None, :]


def load_engine_checkpoint(path: Path, execution_depth: int | None = None):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    config = dict(payload["config"])
    previous_execution_depth = int(config.get("execution_depth", 0))
    if execution_depth is not None:
        config["execution_depth"] = execution_depth
    model = NueronceModel(NueronceConfig(**config))
    params = list(model.parameters())
    stored_params = payload["params"]
    adding_executor = previous_execution_depth == 0 and int(config.get("execution_depth", 0)) > 0
    if len(params) != len(stored_params) and not (adding_executor and len(params) > len(stored_params)):
        raise ValueError("checkpoint parameter count does not match engine model")
    for parameter, stored in zip(params, stored_params):
        parameter.data[...] = stored
    optimizer = StreamFactor(params, lr=5e-5, weight_decay=0.01, momentum=False)
    if payload.get("optimizer") and not adding_executor:
        optimizer.load_state_dict(payload["optimizer"])
    return model, optimizer, payload


def atomic_save(path: Path, model, optimizer, meta: dict) -> None:
    payload = {
        "format": "nueronce-engine-forgeloop-sft-v1",
        "config": vars(model.cfg),
        "params": [parameter.data.copy() for parameter in model.parameters()],
        "optimizer": optimizer.state_dict(),
        "meta": meta,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def evaluate(model, rows, system: str, max_len: int, count: int) -> float:
    losses = []
    with no_grad():
        for row in rows[:count]:
            ids, mask = target_window(row, system, max_len)
            losses.append(model.masked_loss(ids, mask).item())
    return float(np.mean(losses))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--system-file", type=Path, required=True)
    parser.add_argument("--max-len", type=int, default=384)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-examples", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--reasoning-mode", choices=("fixed", "equilibrium"), default="fixed")
    parser.add_argument("--min-depth", type=int, default=None,
                        help="minimum sampled logical depth; defaults to checkpoint depth")
    parser.add_argument("--max-depth", type=int, default=None,
                        help="maximum sampled logical depth; defaults to checkpoint depth")
    parser.add_argument("--reasoning-damping", type=float, default=1.0)
    parser.add_argument("--execution-depth", type=int, default=None,
                        help="enable the addressable execution register at this depth")
    args = parser.parse_args()

    enable_training_dtype(args.dtype)
    model, optimizer, payload = load_engine_checkpoint(args.checkpoint, args.execution_depth)
    min_depth = args.min_depth if args.min_depth is not None else model.cfg.logical_depth
    max_depth = args.max_depth if args.max_depth is not None else model.cfg.logical_depth
    if not 1 <= min_depth <= max_depth:
        raise ValueError("reasoning depth range must satisfy 1 <= min_depth <= max_depth")
    model.cfg.reasoning_mode = args.reasoning_mode
    model.cfg.reasoning_min_depth = min_depth
    model.cfg.reasoning_halt_epsilon = 0.0  # training always unrolls the sampled depth
    model.cfg.reasoning_damping = args.reasoning_damping
    optimizer.lr = args.lr
    train_rows, val_rows = read_jsonl(args.train), read_jsonl(args.val)
    system = args.system_file.read_text(encoding="utf-8").strip()
    meta = dict(payload.get("meta", {}))
    meta["prompt_format"] = "canonical"
    step = int(meta.get("engine_sft_step", 0))
    rng = np.random.default_rng(args.seed + step)
    print(json.dumps({"event": "start", "backend": "nueronce-engine-numpy",
                      "torch_used": False, "params": model.num_params(), "step": step,
                      "dtype": args.dtype,
                      "reasoning_mode": args.reasoning_mode,
                      "reasoning_depth_range": [min_depth, max_depth],
                      "execution_depth": getattr(model.cfg, "execution_depth", 0),
                      "max_len": args.max_len, "activation_checkpointing": model.cfg.activation_checkpointing}),
          flush=True)

    for _ in range(args.steps):
        row = train_rows[int(rng.integers(0, len(train_rows)))]
        model.cfg.logical_depth = int(rng.integers(min_depth, max_depth + 1))
        ids, mask = target_window(row, system, args.max_len)
        started = time.time()
        loss = model.masked_loss(ids, mask)
        model.zero_grad()
        loss.backward()
        grad_norm = clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1
        record = {"event": "train", "engine_sft_step": step, "loss": loss.item(),
                  "grad_norm": grad_norm, "sequence_bytes": int(ids.shape[1]),
                  "target_bytes": int(mask.sum()), "seconds": time.time() - started}
        record["logical_depth"] = model.cfg.logical_depth
        print(json.dumps(record), flush=True)
        meta.update(record)
        if step % args.eval_every == 0:
            meta["val_loss"] = evaluate(model, val_rows, system, args.max_len, args.eval_examples)
            print(json.dumps({"event": "validation", "engine_sft_step": step,
                              "val_loss": meta["val_loss"]}), flush=True)
        if step % args.save_every == 0:
            atomic_save(args.checkpoint, model, optimizer, meta)

    atomic_save(args.checkpoint, model, optimizer, meta)


if __name__ == "__main__":
    main()
