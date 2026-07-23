#!/usr/bin/env python3
"""Section B of FOUNDATIONAL_GENERATION_RECOVERY.md: train a tiny model to
near-zero response loss on the 32 fixed diagnostic examples
(scripts/tiny_exact_overfit_examples.py), then hand off to
scripts/eval_tiny_exact_overfit.py to check whether it can *free-running*
reproduce what it was trained on.

This is deliberately separate from broad curriculum SFT and from the sealed
proof gate: it exists only to answer one question -- is the shared
train/serialize/mask/generate pipeline mechanically correct? If a fresh tiny
model can't overfit 32 short examples it was trained on, the bug is in the
pipeline (masking, serialization, stop handling). If it can, the pipeline is
sound and the 0/8 proof-gate failure is a training-data/scale problem, not a
plumbing problem.

Uses the exact same serializer as the real trainer
(nueronce.training.dialogue_data.make_sft_batch / encode_example ->
nueronce.prompting.format_training_example) so a pass here is evidence about
the real pipeline, not a simplified stand-in for it.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from nueronce.model import ModelConfig, NUERONCEModel, chat_config
from nueronce.training.dialogue_data import make_sft_batch
from tiny_exact_overfit_examples import TINY_EXAMPLES

DEFAULT_SYSTEM = (
    "You are CFNA, a bounded software-engineering assistant. Respect authority "
    "and provenance constraints. For coding work use ForgeLoop: CONTRACT, MAP, "
    "PLAN, ACT, OBSERVE, CRITIQUE, REVISE, VERIFY, LEDGER, MEMORY. Do not claim "
    "a tool result that was not observed."
)


def atomic_save(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/tiny_exact_overfit/checkpoint.pt")
    ap.add_argument("--system-file", default="runs/forgeloop/system_prompt.txt")
    ap.add_argument("--system", default="")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--loss-threshold", type=float, default=0.05,
                     help="stop once average response-byte loss drops below this")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--fast-tooling-check", action="store_true",
                     help="use a much smaller architecture for a quick pipeline-plumbing "
                          "smoke test on CPU; NOT the real diagnostic -- the real gate uses "
                          "the actual chat_11m config (the default)")
    args = ap.parse_args()

    system = args.system
    if not system:
        sp = Path(args.system_file)
        system = sp.read_text(encoding="utf-8").strip() if sp.exists() else DEFAULT_SYSTEM

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.fast_tooling_check:
        cfg = ModelConfig(byte_embed_dim=24, d_local=32, d_model=48, p_max=24,
                          physical_blocks=1, logical_depth=1, n_heads=2, unit_window=24,
                          decoder_window=32, decoder_layers=1, d_state=8, channel_dim=12,
                          ret_byte_dim=16, min_patch=3, max_patch=24, boundary_loss_weight=0.2)
    else:
        cfg = chat_config()
    model = NUERONCEModel(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    pairs = list(TINY_EXAMPLES)
    batch = make_sft_batch(pairs, system=system, max_len=args.max_len)
    byte_ids = torch.from_numpy(batch["byte_ids"])
    target_mask = torch.from_numpy(batch["target_mask"])
    if not bool(target_mask.any()):
        raise SystemExit("target_mask is all-False -- response masking is broken upstream")

    history = []
    out = Path(args.out)
    started = time.time()
    step = 0
    loss_value = float("inf")
    while step < args.max_steps and loss_value > args.loss_threshold:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(byte_ids)
        loss = model.masked_token_loss(logits, byte_ids, target_mask)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        step += 1
        loss_value = float(loss.detach())
        if step % args.log_every == 0 or loss_value <= args.loss_threshold:
            record = {"step": step, "loss": loss_value, "grad_norm": grad_norm,
                      "elapsed_seconds": time.time() - started}
            history.append(record)
            print(json.dumps(record), flush=True)

    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": vars(model.cfg),
        "sft_step": step,
        "sft_system": system,
        "final_loss": loss_value,
        "history": history,
        "diagnostic": "tiny_exact_overfit",
        "n_examples": len(pairs),
        "seed": args.seed,
    }
    atomic_save(payload, out)
    converged = loss_value <= args.loss_threshold
    print(json.dumps({
        "event": "converged" if converged else "max_steps_reached",
        "step": step, "final_loss": loss_value, "checkpoint": str(out),
    }))
    if not converged:
        raise SystemExit(
            f"did not reach loss threshold {args.loss_threshold} within "
            f"{args.max_steps} steps (final loss {loss_value:.4f}); the "
            "exact-overfit eval will likely fail for a training-capacity "
            "reason, not a pipeline reason -- raise --max-steps first"
        )


if __name__ == "__main__":
    main()
