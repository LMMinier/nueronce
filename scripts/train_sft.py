#!/usr/bin/env python3
"""Supervised instruction tuning (VGRFT stage 1) on top of a byte-LM checkpoint.

Every checkpoint produced by ``train_checkpoint.py`` is trained with pure
next-byte cross-entropy on monologic prose — it learns what English looks like,
not what to say back when spoken to (see ``nueronce/chat.py``'s honest framing).
This script closes that gap for real, with three backends:

- ``--backend torch --model small`` (default, backward-compatible): fine-tunes
  the real PyTorch ``NUERONCEModel`` on the small 61-turn hand-written dialogue set.
- ``--backend engine --model small``: the same small dialogue set, but on
  the from-scratch Nueronce Engine (``MicroByteLM``), no PyTorch needed.
- ``--backend engine --model full-nueronce``: the real, large-scale run —
  continuous sharded SFT of the full ``NueronceModel`` over a JSONL dialogue
  corpus (e.g. 10 shards x 10,000 conversations), with resumable
  shard/step/optimizer state, checkpointing, LR decay, and validation/test
  evaluation. See ``nueronce.training.sharded_sft``.

Usage (small, backward-compatible):
    python scripts/train_sft.py --ckpt checkpoints/nueronce_chat.pt \\
        --out checkpoints/nueronce_chat_sft.pt --steps 400

Usage (large-scale, engine, full architecture):
    python scripts/train_sft.py \\
        --backend engine --model full-nueronce \\
        --train-dir data/sft_100k/train_shards \\
        --validation data/sft_100k/validation.jsonl \\
        --test data/sft_100k/test.jsonl \\
        --num-shards 10 --examples-per-shard 10000 \\
        --save-dir checkpoints/micro_nueronce_sft_100k \\
        --resume --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

DEMO_TURNS = ["Hello, who are you?", "What is two plus two?", "Thank you, goodbye."]


def run_torch_small(args):
    import torch

    from nueronce.chat import Conversation, load_checkpoint
    from nueronce.model import NUERONCEModel, ModelConfig
    from nueronce.training.sft import SFT_DATASET, held_out_split, sft_eval, make_sft_batch, TorchSFTBackend
    from nueronce.training.vgrft import VGRFTTrainer

    torch.manual_seed(args.seed)

    ckpt_path = Path(args.ckpt)
    if ckpt_path.exists():
        model, ckpt = load_checkpoint(str(ckpt_path))
        print(f"loaded pretrained checkpoint {ckpt_path} ({model.num_params():,} params)")
    else:
        model = NUERONCEModel(ModelConfig())
        ckpt = {"config": vars(model.cfg)}
        print(f"no checkpoint at {ckpt_path}; starting SFT from a freshly-initialized "
              f"({model.num_params():,}-param) model — pretrain first for a real result")

    train_ex, val_ex = held_out_split(SFT_DATASET, val_frac=args.val_frac, seed=args.seed)
    print(f"SFT set: {len(train_ex)} train turns / {len(val_ex)} held-out turns\n")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    trainer = VGRFTTrainer(TorchSFTBackend(model, opt))
    history = trainer.supervised_instruction_tune(
        train_ex, steps=args.steps, batch_size=args.batch, val_examples=val_ex,
        seed=args.seed, log_every=max(1, args.steps // 10),
    )

    for rec in history:
        line = f"step {rec['step']:5d} | train loss {rec['train_loss']:.3f}"
        if "val_loss" in rec:
            line += f" | held-out loss {rec['val_loss']:.3f}"
        print(line)

    val_batch = make_sft_batch(val_ex)
    final_val = sft_eval(model, val_batch)
    print(f"\nfinal held-out SFT loss (response bytes only): {final_val:.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": vars(model.cfg),
                "step": ckpt.get("step", 0), "history": ckpt.get("history", []),
                "sft_history": history}, out)
    print(f"saved SFT-tuned checkpoint -> {out}")

    print("\n=== sample turns after SFT ===")
    convo = Conversation(model, temperature=0.0, max_new=60)
    for turn in DEMO_TURNS:
        reply = convo.say(turn)
        print(f"User:      {turn}")
        print(f"Assistant: {reply}\n")


def run_engine_small(args):
    import numpy as np

    from nueronce.engine.models import MicroByteLM, MicroSFTBackend
    from nueronce.training.dialogue_data import SFT_DATASET, held_out_split
    from nueronce.training.vgrft import VGRFTTrainer

    np.random.seed(args.seed)
    model = MicroByteLM(d_model=32, n_heads=4, window=16, d_state=8)
    print(f"engine MicroByteLM: {sum(p.data.size for p in model.parameters()):,} params")

    train_ex, val_ex = held_out_split(SFT_DATASET, val_frac=args.val_frac, seed=args.seed)
    print(f"SFT set: {len(train_ex)} train turns / {len(val_ex)} held-out turns\n")

    trainer = VGRFTTrainer(MicroSFTBackend(model, lr=args.lr))
    history = trainer.supervised_instruction_tune(
        train_ex, steps=args.steps, batch_size=args.batch, val_examples=val_ex,
        seed=args.seed, log_every=max(1, args.steps // 10),
    )
    for rec in history:
        line = f"step {rec['step']:5d} | train loss {rec['train_loss']:.3f}"
        if "val_loss" in rec:
            line += f" | held-out loss {rec['val_loss']:.3f}"
        print(line)


def run_engine_full_nueronce(args):
    from nueronce.engine.nueronce_model import NueronceConfig
    from nueronce.training.sharded_sft import ShardedSFTConfig, run_sharded_sft

    if not args.train_dir or not args.validation or not args.test:
        raise SystemExit("--backend engine --model full-nueronce requires --train-dir, "
                          "--validation, and --test")

    # Deliberately unchanged from the small demo config (~112K params) — this
    # run tests whether more data helps *this* architecture, not a bigger one.
    model_cfg = NueronceConfig(
        byte_embed_dim=16, d_local=24, d_model=32, p_max=16, physical_blocks=1,
        logical_depth=2, n_heads=4, unit_window=12, decoder_window=16,
        decoder_layers=1, d_state=8, channel_dim=8, ret_byte_dim=8,
        min_patch=2, max_patch=14,
    )
    cfg = ShardedSFTConfig(
        train_dir=args.train_dir, val_path=args.validation, test_path=args.test,
        save_dir=args.save_dir, metrics_dir=args.metrics_dir,
        num_shards=args.num_shards, examples_per_shard=args.examples_per_shard,
        batch_size=args.batch, max_len=args.max_len, lr=args.lr,
        lr_decay_factor=args.lr_decay_factor, min_lr=args.min_lr,
        grad_clip=args.grad_clip, grad_accum_steps=args.grad_accum_steps,
        periodic_val_every=args.periodic_val_every, periodic_val_examples=args.periodic_val_examples,
        full_val_examples=args.full_val_examples, checkpoint_every_steps=args.checkpoint_every_steps,
        log_every=args.log_every, epochs=args.epochs, max_steps=args.max_steps,
        additional_steps=args.additional_steps, seed=args.seed, resume=args.resume,
    )
    summary = run_sharded_sft(model_cfg, cfg)
    print(f"\ndone: best shard {summary['best_shard']} (val loss {summary['best_val_loss']:.4f}) | "
          f"test loss {summary['test_metrics']['loss']:.4f} "
          f"(bits/byte {summary['test_metrics']['bits_per_byte']:.4f}, "
          f"byte acc {summary['test_metrics']['byte_accuracy']:.4f})")


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_hash(train_dir: str) -> str:
    root = Path(train_dir).parent
    manifest = root / "manifest.json"
    return _sha256_path(manifest) if manifest.exists() else ""


def _load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _record_to_training_bytes(rec: dict):
    from nueronce.prompting import format_training_example
    from nueronce.training.dialogue_data import encode_messages

    if "messages" in rec:
        return encode_messages(rec["messages"], system=rec.get("system_message", ""))

    return format_training_example(
        system_message=rec.get("system_message", ""),
        user_request=rec["user_request"],
        trusted_evidence="\n".join(rec.get("trusted_evidence", [])),
        response_plan="\n".join(rec.get("response_plan", [])),
        assistant_response=rec["assistant_response"],
    )


def _torch_batch_from_records(records, *, max_len: int, device, max_neighbors: int = 4, neighbor_len: int = 128):
    import torch

    encoded = [_record_to_training_bytes(r) for r in records]
    width = min(max(len(b) for b, _ in encoded), max_len)
    byte_ids = torch.zeros((len(records), width), dtype=torch.long, device=device)
    target_mask = torch.zeros((len(records), width), dtype=torch.bool, device=device)
    neighbor_ids = torch.zeros((len(records), max_neighbors, neighbor_len), dtype=torch.long, device=device)
    neighbor_mask = torch.zeros((len(records), max_neighbors, neighbor_len), dtype=torch.bool, device=device)
    has_neighbor = False
    for i, ((b, m), rec) in enumerate(zip(encoded, records)):
        if len(b) > width:
            target_positions = [j for j, flag in enumerate(m) if flag]
            if target_positions:
                first_t, last_t = target_positions[0], target_positions[-1]
                response_len = last_t - first_t + 1
                prefix_keep = max(1, min(width // 3, width - min(width, response_len)))
                start = max(0, first_t - prefix_keep)
                if last_t >= start + width:
                    start = max(0, last_t - width + 1)
                start = min(start, max(0, len(b) - width))
            else:
                start = max(0, len(b) - width)
            b = b[start:start + width]
            m = m[start:start + width]
        else:
            b = b[:width]
            m = m[:width]
        byte_ids[i, :len(b)] = torch.tensor(list(b), dtype=torch.long, device=device)
        target_mask[i, :len(m)] = torch.tensor(m, dtype=torch.bool, device=device)
        for j, ev in enumerate(rec.get("trusted_evidence", [])[:max_neighbors]):
            evb = str(ev).encode("utf-8")[:neighbor_len]
            if evb:
                neighbor_ids[i, j, :len(evb)] = torch.tensor(list(evb), dtype=torch.long, device=device)
                neighbor_mask[i, j, :len(evb)] = True
                has_neighbor = True
    if not bool(target_mask.any().item()):
        raise ValueError(
            f"max_len={max_len} truncated away all assistant response targets; "
            "increase max_len or shorten the prompt-aligned examples"
        )
    return {
        "byte_ids": byte_ids,
        "target_mask": target_mask,
        "neighbor_ids": neighbor_ids if has_neighbor else None,
        "neighbor_mask": neighbor_mask if has_neighbor else None,
    }


def _torch_eval(model, records, *, batch_size: int, max_len: int, device, max_examples: int | None = None) -> dict:
    import torch

    model.eval()
    if max_examples is not None:
        records = records[:max_examples]
    total_loss = 0.0
    total_examples = 0
    correct = 0
    count = 0
    with torch.no_grad():
        for i in range(0, len(records), batch_size):
            chunk = records[i:i + batch_size]
            batch = _torch_batch_from_records(chunk, max_len=max_len, device=device)
            logits, _ = model(batch["byte_ids"], batch["neighbor_ids"], batch["neighbor_mask"])
            loss = model.masked_token_loss(logits, batch["byte_ids"], batch["target_mask"])
            total_loss += float(loss.item()) * len(chunk)
            pred = logits[:, :-1].argmax(-1)
            tgt = batch["byte_ids"][:, 1:]
            sel = batch["target_mask"][:, 1:]
            if int(sel.sum().item()) > 0:
                correct += int((pred[sel] == tgt[sel]).sum().item())
                count += int(sel.sum().item())
            total_examples += len(chunk)
    loss = total_loss / max(1, total_examples)
    return {
        "loss": loss,
        "bits_per_byte": loss / 0.6931471805599453,
        "byte_accuracy": correct / max(1, count),
        "n_examples": total_examples,
    }


def _save_torch_checkpoint(path: Path, model, opt, scheduler, meta: dict):
    import numpy as np
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "config": vars(model.cfg),
        "optimizer": opt.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "meta": meta,
        "step": meta.get("global_step", 0),
        "prompt_format_version": "nueronce.prompting.v1",
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _load_torch_training_checkpoint(path: Path, model, opt, scheduler):
    import numpy as np
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"])
    if "optimizer" in payload and payload["optimizer"] is not None:
        opt.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if payload.get("torch_rng_state") is not None:
        torch.set_rng_state(payload["torch_rng_state"])
    if torch.cuda.is_available() and payload.get("cuda_rng_state"):
        torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
    if payload.get("python_random_state") is not None:
        random.setstate(payload["python_random_state"])
    if payload.get("numpy_random_state") is not None:
        np.random.set_state(payload["numpy_random_state"])
    return payload.get("meta", {})


def run_torch_full_nueronce(args):
    import numpy as np
    import torch

    from nueronce.chat import load_checkpoint

    if not args.train_dir or not args.validation or not args.test:
        raise SystemExit("--backend torch --model full-nueronce requires --train-dir, --validation, and --test")
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise SystemExit(f"requested checkpoint does not exist: {ckpt_path}")
    model, source_ckpt = load_checkpoint(str(ckpt_path))
    print(f"loaded PyTorch NUERONCEModel checkpoint {ckpt_path} ({model.num_params():,} params)")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    train_dir = Path(args.train_dir)
    shards = sorted(train_dir.glob("shard_*.jsonl"))
    if not shards:
        raise SystemExit(f"no shard_*.jsonl files found in {train_dir}")
    val_records = _load_jsonl(Path(args.validation))
    test_records = _load_jsonl(Path(args.test))
    save_dir = Path(args.save_dir)
    metrics_dir = Path(args.metrics_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "torch_sharded_metrics.jsonl"
    latest_path = save_dir / "latest.pt"
    best_path = save_dir / "best.pt"

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_planned = args.max_steps or sum(len(_load_jsonl(p)) // args.batch for p in shards)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_planned), eta_min=args.min_lr)
    meta = {
        "global_step": 0,
        "next_shard_index": 0,
        "step_within_shard": 0,
        "best_val_loss": float("inf"),
        "patience_bad_checks": 0,
        "examples_seen": 0,
        "starting_checkpoint": str(ckpt_path),
        "starting_checkpoint_sha256": _sha256_path(ckpt_path),
        "dataset_manifest_hash": _manifest_hash(args.train_dir),
        "prompt_format_version": "nueronce.prompting.v1",
        "source_checkpoint_step": source_ckpt.get("step"),
    }
    if args.resume and latest_path.exists():
        meta = _load_torch_training_checkpoint(latest_path, model, opt, scheduler)
        if args.reset_scheduler_on_resume:
            for group in opt.param_groups:
                group["lr"] = args.lr
            remaining = (args.max_steps - int(meta.get("global_step", 0))) if args.max_steps else total_planned
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, remaining), eta_min=args.min_lr
            )
            print(f"reset scheduler on resume: lr {args.lr:g}, remaining T_max {max(1, remaining)}")
        model.to(device)
        print(f"resumed {latest_path} at global_step {meta.get('global_step')}")

    def log(rec):
        rec["time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec))

    def validate(event):
        m = _torch_eval(model, val_records, batch_size=max(1, args.batch), max_len=args.max_len,
                        device=device, max_examples=args.full_val_examples)
        rec = {"event": event, "step": meta["global_step"], **{f"val_{k}": v for k, v in m.items()}}
        previous_best = meta["best_val_loss"]
        is_best = m["loss"] < previous_best
        is_significant = m["loss"] < (previous_best - args.min_delta)
        rec["is_best"] = is_best
        rec["is_significant_improvement"] = is_significant
        rec["patience_bad_checks"] = meta["patience_bad_checks"]
        if is_best:
            meta["best_val_loss"] = m["loss"]
            _save_torch_checkpoint(best_path, model, opt, scheduler, dict(meta))
        if is_significant:
            meta["patience_bad_checks"] = 0
        else:
            meta["patience_bad_checks"] += 1
        rec["patience_bad_checks_after"] = meta["patience_bad_checks"]
        log(rec)
        return m

    if meta["global_step"] == 0:
        validate("pre_training")

    t0 = time.time()
    stop = False
    for shard_idx in range(meta["next_shard_index"], len(shards)):
        records = _load_jsonl(shards[shard_idx])
        order = np.random.default_rng(args.seed + 1000 * shard_idx).permutation(len(records))
        n_steps = len(records) // args.batch
        start_step = meta["step_within_shard"] if shard_idx == meta["next_shard_index"] else 0
        for local_step in range(start_step, n_steps):
            if args.max_steps and meta["global_step"] >= args.max_steps:
                stop = True
                break
            idx = order[local_step * args.batch:(local_step + 1) * args.batch]
            chunk = [records[int(i)] for i in idx]
            batch = _torch_batch_from_records(chunk, max_len=args.max_len, device=device)
            model.train()
            opt.zero_grad(set_to_none=True)
            accum = 0.0
            grad_norm = 0.0
            micro = max(1, math.ceil(args.batch / max(1, args.grad_accum_steps)))
            for start in range(0, args.batch, micro):
                sub_ids = batch["byte_ids"][start:start + micro]
                sub_mask = batch["target_mask"][start:start + micro]
                if sub_ids.numel() == 0:
                    continue
                sub_neighbor_ids = batch["neighbor_ids"][start:start + micro] if batch["neighbor_ids"] is not None else None
                sub_neighbor_mask = batch["neighbor_mask"][start:start + micro] if batch["neighbor_mask"] is not None else None
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits, _ = model(sub_ids, sub_neighbor_ids, sub_neighbor_mask)
                    loss = model.masked_token_loss(logits, sub_ids, sub_mask) / max(1, args.grad_accum_steps)
                scaler.scale(loss).backward()
                accum += float(loss.detach().item())
            scaler.unscale_(opt)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item())
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            meta["global_step"] += 1
            meta["examples_seen"] += len(chunk)
            meta["next_shard_index"] = shard_idx
            meta["step_within_shard"] = local_step + 1
            if meta["global_step"] % args.log_every == 0:
                progress = None
                if args.max_steps:
                    progress = min(1.0, meta["global_step"] / max(1, args.max_steps))
                log({"event": "train", "step": meta["global_step"], "shard": shard_idx + 1,
                     "train_loss": accum, "grad_norm": grad_norm, "lr": opt.param_groups[0]["lr"],
                     "examples_seen": meta["examples_seen"], "progress": progress,
                     "progress_bar": ("[" + "#" * int((progress or 0) * 20) +
                                      "-" * (20 - int((progress or 0) * 20)) + "]") if progress is not None else None,
                     "elapsed_seconds": time.time() - t0})
            if meta["global_step"] % args.periodic_val_every == 0:
                validate("periodic_val")
                if meta["patience_bad_checks"] >= args.patience:
                    stop = True
                    break
            if meta["global_step"] % args.checkpoint_every_steps == 0:
                _save_torch_checkpoint(latest_path, model, opt, scheduler, dict(meta))
        if stop:
            break
        meta["next_shard_index"] = shard_idx + 1
        meta["step_within_shard"] = 0
        _save_torch_checkpoint(latest_path, model, opt, scheduler, dict(meta))

    validate("final_val")
    _save_torch_checkpoint(latest_path, model, opt, scheduler, dict(meta))
    best_payload = torch.load(best_path if best_path.exists() else latest_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_payload["state_dict"])
    model.to(device)
    test_metrics = _torch_eval(model, test_records, batch_size=max(1, args.batch), max_len=args.max_len,
                               device=device, max_examples=args.full_val_examples)
    summary = {
        "num_params": model.num_params(),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "training_meta": meta,
        "test_metrics": test_metrics,
        "best_checkpoint": str(best_path if best_path.exists() else latest_path),
        "latest_checkpoint": str(latest_path),
    }
    (metrics_dir / "torch_training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["torch", "engine"], default="torch")
    ap.add_argument("--model", choices=["small", "full-nueronce"], default="small")

    # small-scale (torch or engine) args
    ap.add_argument("--ckpt", type=str, default="checkpoints/nueronce_chat.pt")
    ap.add_argument("--out", type=str, default="checkpoints/nueronce_chat_sft.pt")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--val-frac", type=float, default=0.2)

    # large-scale (engine full-nueronce) args
    ap.add_argument("--train-dir", type=str, default=None, help="directory of shard_NN.jsonl files")
    ap.add_argument("--validation", type=str, default=None)
    ap.add_argument("--test", type=str, default=None)
    ap.add_argument("--num-shards", type=int, default=10)
    ap.add_argument("--examples-per-shard", type=int, default=10_000)
    ap.add_argument("--save-dir", type=str, default="checkpoints/micro_nueronce_sft_100k")
    ap.add_argument("--metrics-dir", type=str, default="metrics")
    ap.add_argument("--max-len", type=int, default=288)
    ap.add_argument("--lr-decay-factor", type=float, default=0.7)
    ap.add_argument("--min-lr", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--grad-accum-steps", type=int, default=1)
    ap.add_argument("--periodic-val-every", type=int, default=200)
    ap.add_argument("--periodic-val-examples", type=int, default=256)
    ap.add_argument("--full-val-examples", type=int, default=None)
    ap.add_argument("--checkpoint-every-steps", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--additional-steps", type=int, default=None)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--min-delta", type=float, default=0.0,
                    help="Minimum validation-loss improvement that resets convergence patience.")
    ap.add_argument("--reset-scheduler-on-resume", action="store_true")
    ap.add_argument("--resume", action="store_true")

    # shared
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.backend == "torch" and args.model == "small":
        run_torch_small(args)
    elif args.backend == "torch" and args.model == "full-nueronce":
        run_torch_full_nueronce(args)
    elif args.backend == "engine" and args.model == "small":
        run_engine_small(args)
    elif args.backend == "engine" and args.model == "full-nueronce":
        run_engine_full_nueronce(args)
    else:
        raise SystemExit(f"unsupported combination: --backend {args.backend} --model {args.model} "
                          f"(supported: torch small/full-nueronce, engine small/full-nueronce)")


if __name__ == "__main__":
    main()
