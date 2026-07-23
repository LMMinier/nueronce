#!/usr/bin/env python3
"""Loss-sweep extension of the section B/D diagnostics: quantify HOW MUCH
teacher-forced response loss must fall before free-running generation
switches on, instead of treating "low loss + broken generation" as a mystery.

The central arithmetic this makes visible (2026-07-23 investigation): a
response-byte loss of ~0.844 nats/byte is ~85% per-byte argmax accuracy, and
exact reproduction of a ~30-byte answer at 85%/byte is ~0.85^30 ~= 0.8% --
so a checkpoint at that loss level scoring ~0/8 on the sealed gate is the
EXPECTED value of its loss, not a generation bug. This script turns that
argument into a measured curve for the real architecture on this machine.

Method: train one fresh model on the 32 fixed tiny examples
(scripts/tiny_exact_overfit_examples.py, same production serializer as the
real trainer), and every time the training loss first crosses the next
threshold on the way down, freeze, then measure at that point:

  - response-byte teacher-forced argmax accuracy (all 32 examples)
  - first-8-assistant-bytes TF accuracy (the bytes that decide whether an
    answer starts on-topic at all)
  - free-running greedy exact-match count (the section-B condition)
  - predicted exact-match probability from per-byte accuracy alone
    (acc ** target_len, averaged) -- so the "it's just arithmetic" claim is
    checked against the observed free-run rate at every rung

Output: one JSON with a row per threshold. The curve answers "what response
loss must checkpoint selection target before a gate attempt is worth
running" -- the working target from the investigation is <= 0.05.

Uses the exact production paths: dialogue_data.make_sft_batch for training,
prompting.format_inference_prompt + NUERONCEModel.generate(greedy, STOP_SEQUENCES)
for generation. Does not touch the sealed proof gate.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from nueronce.model import ModelConfig, NUERONCEModel, chat_config
from nueronce.prompting import STOP_SEQUENCES, extract_assistant_continuation, format_inference_prompt
from nueronce.training.dialogue_data import make_sft_batch
from tiny_exact_overfit_examples import TINY_EXAMPLES

DEFAULT_SYSTEM = (
    "You are CFNA, a bounded software-engineering assistant. Respect authority "
    "and provenance constraints. For coding work use ForgeLoop: CONTRACT, MAP, "
    "PLAN, ACT, OBSERVE, CRITIQUE, REVISE, VERIFY, LEDGER, MEMORY. Do not claim "
    "a tool result that was not observed."
)


@torch.no_grad()
def teacher_forced_metrics(model: NUERONCEModel, byte_ids: torch.Tensor,
                           target_mask: torch.Tensor) -> dict:
    """Per-byte argmax accuracy on response bytes, overall and first-8."""
    model.eval()
    logits, _ = model(byte_ids)
    pred = logits[:, :-1].argmax(dim=-1)          # logits[t] predicts byte[t+1]
    tgt = byte_ids[:, 1:]
    sel = target_mask[:, 1:]
    correct = (pred == tgt) & sel
    overall = float(correct.sum()) / max(1, int(sel.sum()))
    first8_hits, first8_total = 0, 0
    for b in range(byte_ids.shape[0]):
        idx = sel[b].nonzero(as_tuple=True)[0][:8]
        first8_hits += int(correct[b, idx].sum())
        first8_total += len(idx)
    return {"response_argmax_acc": overall,
            "first8_argmax_acc": first8_hits / max(1, first8_total)}


@torch.no_grad()
def free_run_exact(model: NUERONCEModel, system: str, max_new: int, max_ctx: int) -> dict:
    """Greedy free-running exact-match over all 32 prompts, plus the
    predicted match rate from TF accuracy alone (filled in by caller)."""
    model.eval()
    hits, answers = 0, []
    for prompt, target in TINY_EXAMPLES:
        rendered = format_inference_prompt(system_message=system, user_request=prompt,
                                           trusted_evidence="", response_plan="")
        raw = model.generate(rendered.encode("utf-8"), max_new=max_new, temperature=0.0,
                             greedy=True, max_ctx=max_ctx,
                             stop_sequences=STOP_SEQUENCES, continuation_only=True)
        answer = extract_assistant_continuation(raw).strip()
        answers.append(answer)
        hits += int(answer == target)
    return {"free_run_exact_count": hits, "answers": answers}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="metrics/loss_generation_curve.json")
    ap.add_argument("--thresholds", default="2.0,1.0,0.5,0.25,0.1,0.05,0.02",
                     help="descending comma-separated response-loss thresholds to sample at")
    ap.add_argument("--system-file", default="runs/forgeloop/system_prompt.txt")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--max-ctx", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--fast-tooling-check", action="store_true",
                     help="undersized architecture for a CPU smoke test of the tooling; "
                          "NOT the real diagnostic")
    args = ap.parse_args()

    sp = Path(args.system_file)
    system = sp.read_text(encoding="utf-8").strip() if sp.exists() else DEFAULT_SYSTEM
    thresholds = sorted((float(t) for t in args.thresholds.split(",")), reverse=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.fast_tooling_check:
        cfg = ModelConfig(byte_embed_dim=24, d_local=32, d_model=48, p_max=24,
                          physical_blocks=1, logical_depth=1, n_heads=2, unit_window=24,
                          decoder_window=32, decoder_layers=1, d_state=8, channel_dim=12,
                          ret_byte_dim=16, min_patch=3, max_patch=24, boundary_loss_weight=0.2)
    else:
        cfg = chat_config()
    model = NUERONCEModel(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    batch = make_sft_batch(list(TINY_EXAMPLES), system=system, max_len=args.max_len)
    byte_ids = torch.from_numpy(batch["byte_ids"])
    target_mask = torch.from_numpy(batch["target_mask"])
    target_lens = [len(t.encode("utf-8")) for _, t in TINY_EXAMPLES]

    rows = []
    pending = list(thresholds)
    started = time.time()
    step, loss_value = 0, float("inf")
    while step < args.max_steps and pending:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(byte_ids)
        loss = model.masked_token_loss(logits, byte_ids, target_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1
        loss_value = float(loss.detach())

        if pending and loss_value <= pending[0]:
            threshold = pending.pop(0)
            tf = teacher_forced_metrics(model, byte_ids, target_mask)
            fr = free_run_exact(model, system, args.max_new, args.max_ctx)
            acc = tf["response_argmax_acc"]
            predicted = float(np.mean([acc ** n for n in target_lens]))
            row = {
                "threshold": threshold, "step": step, "train_response_loss": loss_value,
                **tf,
                "free_run_exact_count": fr["free_run_exact_count"],
                "free_run_exact_fraction": fr["free_run_exact_count"] / len(TINY_EXAMPLES),
                "predicted_exact_fraction_from_tf_acc": predicted,
                "elapsed_seconds": time.time() - started,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            model.train()

    report = {
        "diagnostic": "loss_generation_curve_v1",
        "architecture": "fast_tooling_check" if args.fast_tooling_check else "chat_11m",
        "params": model.num_params(), "seed": args.seed,
        "n_examples": len(TINY_EXAMPLES), "system_bytes": len(system.encode("utf-8")),
        "thresholds_requested": thresholds,
        "thresholds_reached": [r["threshold"] for r in rows],
        "final_step": step, "final_loss": loss_value,
        "rows": rows,
        "interpretation": (
            "free_run_exact_fraction should track "
            "predicted_exact_fraction_from_tf_acc if generation is healthy: "
            "exact reproduction is the product of per-byte accuracies, so it "
            "stays near zero until per-byte accuracy is very high. Select "
            "checkpoints on response loss / first-8-bytes metrics, not "
            "aggregate val loss, and do not attempt the sealed gate above the "
            "loss level where this curve switches on (working target: <=0.05)."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "final_loss": loss_value,
                      "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
