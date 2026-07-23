#!/usr/bin/env python3
"""Resume ForgeLoop SFT and stop when validation loss converges."""
from __future__ import annotations

import argparse
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from nueronce.chat import load_checkpoint
from nueronce.reasoning import AddressableExecutionRegister
from nueronce.training.dialogue_data import make_sft_batch


def read_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def make_batch(rows: list[dict], device: torch.device, system: str, max_len: int) -> dict:
    pairs = [(row["prompt"], row["response"]) for row in rows]
    batch = make_sft_batch(pairs, system=system, max_len=max_len)
    return {
        "byte_ids": torch.from_numpy(batch["byte_ids"]).to(device),
        "target_mask": torch.from_numpy(batch["target_mask"]).to(device),
    }


@torch.no_grad()
def evaluate(model, rows, device, system, max_len, amp_enabled, max_examples):
    model.eval()
    losses = []
    if rows and all(row.get("domain") for row in rows):
        grouped = {}
        for row in rows:
            grouped.setdefault(row["domain"], []).append(row)
        selected = []
        offset = 0
        names = sorted(grouped)
        while len(selected) < min(len(rows), max_examples):
            added = False
            for name in names:
                if offset < len(grouped[name]) and len(selected) < max_examples:
                    selected.append(grouped[name][offset]); added = True
            if not added:
                break
            offset += 1
        rows = selected
    context = torch.autocast(device_type="cuda", dtype=torch.float16) if amp_enabled else nullcontext()
    for start in range(0, min(len(rows), max_examples), 4):
        batch = make_batch(rows[start:start + 4], device, system, max_len)
        if not bool(batch["target_mask"].any()):
            continue
        with context:
            logits, _ = model(batch["byte_ids"])
            loss = model.masked_token_loss(logits, batch["byte_ids"], batch["target_mask"])
        losses.append(float(loss.float().item()))
    return float(np.mean(losses)) if losses else float("nan")


def atomic_save(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--system", default="")
    parser.add_argument("--system-file", default="")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-examples", type=int, default=64)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-3)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--execution-depth", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=0,
                        help="0 keeps PyTorch's detected CPU thread count")
    parser.add_argument("--reset-convergence", action="store_true",
                        help="reset best validation state when switching datasets/objectives")
    parser.add_argument("--balanced-domain-sampling", action="store_true",
                        help="sample capability domains uniformly, then examples within each domain")
    args = parser.parse_args()
    if args.system_file:
        args.system = Path(args.system_file).read_text(encoding="utf-8").strip()
    if not args.system:
        parser.error("one of --system or --system-file is required")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    train_rows, val_rows = read_jsonl(args.train), read_jsonl(args.val)
    domain_rows = {}
    if args.balanced_domain_sampling:
        for index, row in enumerate(train_rows):
            domain_rows.setdefault(row.get("domain", "unlabeled"), []).append(index)
        if len(domain_rows) < 2:
            raise ValueError("balanced domain sampling needs at least two labeled domains")
    domain_names = sorted(domain_rows)
    out = Path(args.out)

    resume = out.exists()
    source = out if resume else Path(args.base)
    model, source_checkpoint = load_checkpoint(str(source))
    current_execution_depth = int(getattr(model.cfg, "execution_depth", 0))
    if args.execution_depth > 0 and current_execution_depth == 0:
        model.cfg.execution_depth = args.execution_depth
        model.cfg.execution_residual_scale = 1.0
        model.executor = AddressableExecutionRegister(model.cfg.d_model)
    elif args.execution_depth != current_execution_depth and resume:
        raise ValueError(
            f"resume checkpoint execution_depth={current_execution_depth}, requested {args.execution_depth}"
        )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    if resume and source_checkpoint.get("optimizer"):
        optimizer.load_state_dict(source_checkpoint["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr

    history = list(source_checkpoint.get("sft_history", []))
    step = int(source_checkpoint.get("sft_step", 0))
    best_val = float(source_checkpoint.get("best_val_loss", "inf"))
    bad_evals = int(source_checkpoint.get("bad_evals", 0))
    if args.reset_convergence and not resume:
        best_val, bad_evals = float("inf"), 0
    best_path = out.with_name(out.stem + "_best.pt")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    def payload() -> dict:
        p = {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "config": vars(model.cfg),
            "step": source_checkpoint.get("step", 0),
            "history": source_checkpoint.get("history", []),
            "sft_step": step,
            "sft_history": history,
            "sft_system": args.system,
            "best_val_loss": best_val,
            "bad_evals": bad_evals,
            "rng_state": rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state().cpu(),
            "numpy_rng_state": np.random.get_state(),
        }
        if device.type == "cuda":
            p["cuda_rng_state"] = torch.cuda.get_rng_state().cpu()
        return p

    print(json.dumps({"event": "start", "device": str(device), "resume": resume,
                      "optimizer_restored": bool(resume and source_checkpoint.get("optimizer")),
                      "step": step, "best_val_loss": best_val, "train_examples": len(train_rows),
                      "val_examples": len(val_rows), "params": model.num_params(),
                      "balanced_domain_sampling": args.balanced_domain_sampling,
                      "domains": domain_names}), flush=True)

    try:
        while step < args.max_steps and bad_evals < args.patience:
            step_started = time.perf_counter()
            if args.balanced_domain_sampling:
                selected_domains = [domain_names[int(rng.integers(0, len(domain_names)))]
                                    for _ in range(args.batch)]
                indices = [domain_rows[domain][int(rng.integers(0, len(domain_rows[domain])))]
                           for domain in selected_domains]
            else:
                indices = rng.integers(0, len(train_rows), size=args.batch)
                selected_domains = []
            batch = make_batch([train_rows[int(i)] for i in indices], device, args.system, args.max_len)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            context = torch.autocast(device_type="cuda", dtype=torch.float16) if amp_enabled else nullcontext()
            with context:
                logits, _ = model(batch["byte_ids"])
                loss = model.masked_token_loss(logits, batch["byte_ids"], batch["target_mask"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            scaler.step(optimizer)
            scaler.update()
            step += 1
            step_seconds = time.perf_counter() - step_started

            if step % args.eval_every == 0:
                val_loss = evaluate(model, val_rows, device, args.system, args.max_len,
                                    amp_enabled, args.eval_examples)
                improved = val_loss < best_val - args.min_delta
                if improved:
                    best_val, bad_evals = val_loss, 0
                else:
                    bad_evals += 1
                record = {"sft_step": step, "train_loss": float(loss.detach()), "val_loss": val_loss,
                          "grad_norm": grad_norm, "best_val_loss": best_val,
                          "bad_evals": bad_evals, "improved": improved, "time": time.time(),
                          "step_seconds": step_seconds,
                          "sequence_bytes": int(batch["byte_ids"].numel()),
                          "bytes_per_second": float(batch["byte_ids"].numel() / step_seconds),
                          "sampled_domains": selected_domains}
                history.append(record)
                print(json.dumps(record), flush=True)
                atomic_save(payload(), out)
                if improved:
                    atomic_save(payload(), best_path)
            elif step % args.checkpoint_every == 0:
                atomic_save(payload(), out)

    except KeyboardInterrupt:
        print(json.dumps({"event": "interrupted", "step": step, "best_val_loss": best_val,
                          "interrupted": True, "reason": "user_requested",
                          "save_timestamp": time.time(), "last_completed_step": step}), flush=True)
        try:
            atomic_save(payload(), out)
            print(json.dumps({"event": "interrupted_checkpoint_saved", "path": str(out),
                              "step": step}), flush=True)
        except Exception as e:
            print(json.dumps({"event": "interrupted_checkpoint_failed", "error": str(e),
                              "step": step}), flush=True)
            raise
        return

    atomic_save(payload(), out)
    print(json.dumps({"event": "converged" if bad_evals >= args.patience else "max_steps",
                      "step": step, "best_val_loss": best_val, "bad_evals": bad_evals}), flush=True)


if __name__ == "__main__":
    main()
