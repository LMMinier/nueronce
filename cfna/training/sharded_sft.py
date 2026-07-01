"""Continuous, resumable, sharded VGRFT stage-1 SFT training for the *full*
``MicroCFNAModel`` at real scale (10 shards x 10,000 conversations = 100,000),
entirely on the from-scratch microtorch engine (no PyTorch anywhere).

Design notes:

- **Continuous**: one model + one optimizer instance lives across all shards;
  nothing is reset between them (see ``run_sharded_sft``'s single ``model``/
  ``opt`` pair threaded through the whole loop).
- **Resumable without serializing RNG state**: each shard's example order is
  ``np.random.default_rng(seed + 1000 * shard_index).permutation(N)`` — a pure
  function of (seed, shard_index), so resuming mid-shard only needs the saved
  ``step_within_shard`` to know how many examples of that deterministic order
  to skip, not a serialized generator.
- **Two validation cadences**: a cheap periodic check on a fixed *subset* of
  validation during training (signal, not the metric of record), and a full
  pass over the entire fixed validation set before training, at the end of
  every shard, and after all shards (the metric of record, used for
  best-checkpoint selection and LR decay).
- **Policy for "validation repeatedly worsens"**: decay the learning rate
  (bounded by ``min_lr``) and keep training through all ten shards regardless
  (per the run's explicit "continuous training across all ten shards"
  requirement) — the safeguard against a late bad shard is best-checkpoint
  selection, not early termination.
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..microtorch.cfna_model import MicroCFNAModel, MicroModelConfig
from ..microtorch.optim import AdamW, clip_grad_norm_
from ..microtorch.tensor import no_grad
from .dialogue_data import make_conversation_batch

LN2 = 0.6931471805599453


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_jsonl(path: str) -> List[dict]:
    """Load one JSONL file fully into memory. Used per-shard (~10k short
    records, a few MB) and for the fixed val/test sets — never for the whole
    multi-shard corpus at once, which is the actual streaming requirement."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _batch_from_records(records: List[dict], max_len: int) -> Dict[str, np.ndarray]:
    return make_conversation_batch([r["messages"] for r in records], max_len=max_len)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def evaluate(model: MicroCFNAModel, records: List[dict], *, batch_size: int = 64,
             max_len: int = 288, max_examples: Optional[int] = None) -> Dict[str, float]:
    """Full (or subset) masked-loss / byte-accuracy pass over ``records``.
    No gradient graph is built (``no_grad``), so this is cheap relative to a
    training step."""
    if max_examples is not None:
        records = records[:max_examples]
    total_loss, total_correct, total_count = 0.0, 0, 0
    with no_grad():
        for i in range(0, len(records), batch_size):
            chunk = records[i:i + batch_size]
            batch = _batch_from_records(chunk, max_len)
            logits, _ = model.forward(batch["byte_ids"])
            loss = model.masked_token_loss(logits, batch["byte_ids"], batch["target_mask"])
            total_loss += loss.item() * len(chunk)
            pred = logits.data[:, :-1].argmax(-1)
            tgt = batch["byte_ids"][:, 1:]
            sel = batch["target_mask"][:, 1:]
            if sel.sum() > 0:
                total_correct += int((pred[sel] == tgt[sel]).sum())
                total_count += int(sel.sum())
    n = max(1, len(records))
    avg_loss = total_loss / n
    return {
        "loss": avg_loss,
        "bits_per_byte": avg_loss / LN2,
        "byte_accuracy": total_correct / max(1, total_count),
        "n_examples": len(records),
    }


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #

def save_checkpoint(path: str, model: MicroCFNAModel, opt: AdamW, meta: dict) -> None:
    payload = {
        "config": vars(model.cfg),
        "params": [p.data.copy() for p in model.parameters()],
        "opt_m": [m.copy() for m in opt.m],
        "opt_v": [v.copy() for v in opt.v],
        "opt_t": opt.t,
        "opt_lr": opt.lr,
        "meta": meta,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f)
    tmp.replace(p)  # atomic on POSIX: a crash mid-write never corrupts the real file


def load_checkpoint(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def new_model_and_optimizer(cfg: MicroModelConfig, lr: float, seed: int):
    np.random.seed(seed)
    model = MicroCFNAModel(cfg)
    opt = AdamW(list(model.parameters()), lr=lr, weight_decay=0.01)
    return model, opt


def apply_checkpoint(payload: dict, model: MicroCFNAModel, opt: AdamW) -> None:
    for p, arr in zip(model.parameters(), payload["params"]):
        p.data = arr.copy()
    opt.m = [m.copy() for m in payload["opt_m"]]
    opt.v = [v.copy() for v in payload["opt_v"]]
    opt.t = payload["opt_t"]
    opt.lr = payload["opt_lr"]


# --------------------------------------------------------------------------- #
# Training driver
# --------------------------------------------------------------------------- #

@dataclass
class ShardedSFTConfig:
    train_dir: str
    val_path: str
    test_path: str
    save_dir: str
    metrics_dir: str = "metrics"
    num_shards: int = 10
    examples_per_shard: int = 10_000
    batch_size: int = 32
    max_len: int = 288
    lr: float = 2e-3
    lr_decay_factor: float = 0.7
    min_lr: float = 1e-4
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    periodic_val_every: int = 200
    periodic_val_examples: int = 256
    full_val_examples: Optional[int] = None  # None = the entire fixed validation set
    checkpoint_every_steps: int = 500
    log_every: int = 50
    seed: int = 42
    resume: bool = True


def _shard_path(cfg: ShardedSFTConfig, shard_idx: int) -> str:
    return str(Path(cfg.train_dir) / f"shard_{shard_idx + 1:02d}.jsonl")


def run_sharded_sft(model_cfg: MicroModelConfig, cfg: ShardedSFTConfig,
                    log_fn=print) -> dict:
    save_dir = Path(cfg.save_dir)
    metrics_dir = Path(cfg.metrics_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "shard_metrics.jsonl"
    latest_path = save_dir / "latest.pt"
    best_path = save_dir / "best.pt"

    val_records = load_jsonl(cfg.val_path)
    val_subset = val_records[: cfg.periodic_val_examples]

    meta = {
        "next_shard_index": 0, "step_within_shard": 0, "global_step": 0,
        "examples_seen": 0, "best_val_loss": float("inf"), "best_shard": None,
        "lr": cfg.lr, "consecutive_no_improve": 0, "elapsed_seconds": 0.0,
    }

    if cfg.resume and latest_path.exists():
        payload = load_checkpoint(str(latest_path))
        model = MicroCFNAModel(MicroModelConfig(**payload["config"]))
        opt = AdamW(list(model.parameters()), lr=payload["opt_lr"], weight_decay=0.01)
        apply_checkpoint(payload, model, opt)
        meta = payload["meta"]
        log_fn(f"resumed from {latest_path}: shard {meta['next_shard_index']}, "
               f"step_within_shard {meta['step_within_shard']}, examples_seen {meta['examples_seen']:,}")
    else:
        model, opt = new_model_and_optimizer(model_cfg, cfg.lr, cfg.seed)
        log_fn(f"fresh model: {model.num_params():,} params")

    def log_metric(rec: dict) -> None:
        rec["wall_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        log_fn(json.dumps(rec))

    t0 = time.time() - meta["elapsed_seconds"]

    def full_val() -> Dict[str, float]:
        return evaluate(model, val_records, batch_size=max(64, cfg.batch_size * 2),
                        max_len=cfg.max_len, max_examples=cfg.full_val_examples)

    if meta["global_step"] == 0:
        pre = full_val()
        log_metric({"event": "pre_training", "shard": 0, "step": 0, "examples_seen": 0,
                   "val_loss": pre["loss"], "val_bits_per_byte": pre["bits_per_byte"],
                   "val_byte_accuracy": pre["byte_accuracy"], "lr": opt.lr,
                   "elapsed_seconds": time.time() - t0})
        meta["best_val_loss"] = pre["loss"]

    shard_summaries: List[dict] = []

    for shard_idx in range(meta["next_shard_index"], cfg.num_shards):
        shard_path = _shard_path(cfg, shard_idx)
        records = load_jsonl(shard_path)
        assert len(records) == cfg.examples_per_shard, (
            f"{shard_path} has {len(records)} records, expected {cfg.examples_per_shard}")

        order = np.random.default_rng(cfg.seed + 1000 * shard_idx).permutation(len(records))
        start_step = meta["step_within_shard"] if shard_idx == meta["next_shard_index"] else 0
        n_steps = len(records) // cfg.batch_size
        train_loss_ema = None

        for step in range(start_step, n_steps):
            idx = order[step * cfg.batch_size:(step + 1) * cfg.batch_size]
            chunk = [records[i] for i in idx]
            batch = _batch_from_records(chunk, cfg.max_len)

            model.zero_grad()
            accum_loss = 0.0
            micro_bs = max(1, cfg.batch_size // cfg.grad_accum_steps)
            for a in range(cfg.grad_accum_steps):
                sub = slice(a * micro_bs, (a + 1) * micro_bs)
                sub_ids, sub_mask = batch["byte_ids"][sub], batch["target_mask"][sub]
                if sub_ids.shape[0] == 0:
                    continue
                logits, _ = model.forward(sub_ids)
                loss = model.masked_token_loss(logits, sub_ids, sub_mask)
                (loss * (sub_ids.shape[0] / batch["byte_ids"].shape[0])).backward()
                accum_loss += loss.item() * sub_ids.shape[0]
            clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            step_loss = accum_loss / batch["byte_ids"].shape[0]
            train_loss_ema = step_loss if train_loss_ema is None else 0.95 * train_loss_ema + 0.05 * step_loss
            meta["global_step"] += 1
            meta["examples_seen"] += batch["byte_ids"].shape[0]
            meta["step_within_shard"] = step + 1
            meta["elapsed_seconds"] = time.time() - t0

            if meta["global_step"] % cfg.log_every == 0:
                log_fn(f"shard {shard_idx + 1}/{cfg.num_shards} step {step + 1}/{n_steps} "
                       f"| examples_seen {meta['examples_seen']:,} | train_loss(ema) {train_loss_ema:.4f} "
                       f"| lr {opt.lr:.2e} | elapsed {meta['elapsed_seconds']:.0f}s")

            if meta["global_step"] % cfg.periodic_val_every == 0:
                pv = evaluate(model, val_subset, batch_size=max(64, cfg.batch_size * 2), max_len=cfg.max_len)
                log_metric({"event": "periodic", "shard": shard_idx + 1, "step": meta["global_step"],
                           "examples_seen": meta["examples_seen"], "train_loss": train_loss_ema,
                           "val_loss_subset": pv["loss"], "val_byte_accuracy_subset": pv["byte_accuracy"],
                           "lr": opt.lr, "elapsed_seconds": meta["elapsed_seconds"]})

            if meta["global_step"] % cfg.checkpoint_every_steps == 0:
                save_checkpoint(str(latest_path), model, opt, meta)

        # end-of-shard: full validation, checkpoint, LR policy
        meta["next_shard_index"] = shard_idx + 1
        meta["step_within_shard"] = 0
        val = full_val()
        is_best = val["loss"] < meta["best_val_loss"]
        if is_best:
            meta["best_val_loss"] = val["loss"]
            meta["best_shard"] = shard_idx + 1
            meta["consecutive_no_improve"] = 0
            save_checkpoint(str(best_path), model, opt, dict(meta))
        else:
            meta["consecutive_no_improve"] += 1
            opt.lr = max(cfg.min_lr, opt.lr * cfg.lr_decay_factor)
            meta["lr"] = opt.lr

        save_checkpoint(str(latest_path), model, opt, meta)

        rec = {
            "event": "end_of_shard", "shard": shard_idx + 1, "step": meta["global_step"],
            "examples_seen": meta["examples_seen"], "train_loss": train_loss_ema,
            "val_loss": val["loss"], "val_bits_per_byte": val["bits_per_byte"],
            "val_byte_accuracy": val["byte_accuracy"], "lr": opt.lr, "is_best": is_best,
            "elapsed_seconds": meta["elapsed_seconds"],
        }
        log_metric(rec)
        shard_summaries.append(rec)

    test_records = load_jsonl(cfg.test_path)
    # Final model selection is the best-validation checkpoint, not shard 10's.
    best_payload = load_checkpoint(str(best_path)) if best_path.exists() else load_checkpoint(str(latest_path))
    best_model = MicroCFNAModel(MicroModelConfig(**best_payload["config"]))
    for p, arr in zip(best_model.parameters(), best_payload["params"]):
        p.data = arr.copy()
    test_metrics = evaluate(best_model, test_records, batch_size=max(64, cfg.batch_size * 2), max_len=cfg.max_len)
    final_val = evaluate(model, val_records, batch_size=max(64, cfg.batch_size * 2), max_len=cfg.max_len)

    log_metric({"event": "final", "shard": cfg.num_shards, "step": meta["global_step"],
               "examples_seen": meta["examples_seen"], "final_val_loss": final_val["loss"],
               "test_loss": test_metrics["loss"], "test_bits_per_byte": test_metrics["bits_per_byte"],
               "test_byte_accuracy": test_metrics["byte_accuracy"], "best_shard": meta["best_shard"],
               "best_val_loss": meta["best_val_loss"], "elapsed_seconds": meta["elapsed_seconds"]})

    summary = {
        "model_config": vars(model_cfg), "num_params": model.num_params(),
        "training_config": asdict(cfg), "shard_summaries": shard_summaries,
        "best_shard": meta["best_shard"], "best_val_loss": meta["best_val_loss"],
        "final_val_metrics": final_val, "test_metrics": test_metrics,
        "total_examples_seen": meta["examples_seen"], "total_steps": meta["global_step"],
        "elapsed_seconds": meta["elapsed_seconds"],
    }
    (metrics_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


__all__ = [
    "ShardedSFTConfig", "run_sharded_sft", "evaluate", "load_jsonl",
    "save_checkpoint", "load_checkpoint", "apply_checkpoint", "new_model_and_optimizer",
]
