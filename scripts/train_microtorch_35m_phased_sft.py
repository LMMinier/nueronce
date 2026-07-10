#!/usr/bin/env python3
"""Crash-safe, response-only phased SFT for the 35M MicroTorch CFNA model.

Each invocation performs one bounded step, saves atomically, and exits. Only
assistant-response target bytes contribute to loss; system/evidence/user bytes
are context. This makes the run genuine supervised fine-tuning rather than
unmasked next-byte language modeling over the whole record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
from cfna.microtorch.optim import clip_grad_norm_

PHASES = {
    "direct": {"direct_summary", "direct_definition", "direct_procedure", "direct_explanation"},
    "grounded": {"grounded_answer", "grounded_summary", "abstain_conflict_qualified"},
    "verification": {"abstain_missing", "abstain_conflict", "abstain_conflict_qualified"},
}


def finite_gradients(params):
    count = 0
    for index, parameter in enumerate(params):
        if parameter.grad is None:
            continue
        count += 1
        if not np.isfinite(parameter.grad).all():
            raise FloatingPointError(f"nonfinite gradient at parameter {index} shape={parameter.shape}")
    return count


class PagedFactorState:
    """Momentum-free factorized second moments, compatible with the step-3 checkpoint."""

    def __init__(self, params, *, lr=1e-5, step=0, state=None, beta2=0.999,
                 eps=1e-8, tile_rows=128, clip_threshold=1.0):
        self.params = list(params)
        self.lr = float(lr)
        self.step_index = int(step)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.tile_rows = int(tile_rows)
        self.clip_threshold = float(clip_threshold)
        self.state = state if state is not None else [
            (np.zeros(p.shape[0], np.float32), np.zeros(int(np.prod(p.shape[1:])), np.float32))
            if p.ndim >= 2 else np.zeros(p.shape, np.float32)
            for p in self.params
        ]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def update(self):
        finite_gradients(self.params)
        self.step_index += 1
        correction = max(1.0 - self.beta2 ** self.step_index, self.eps)
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = np.asarray(p.grad, np.float32)
            if p.ndim >= 2:
                w = p.data.reshape(p.shape[0], -1)
                gg = g.reshape(w.shape)
                vr, vc = self.state[i]
                vr *= self.beta2
                vr += (1.0 - self.beta2) * np.mean(gg * gg, axis=1)
                vc *= self.beta2
                vc += (1.0 - self.beta2) * np.mean(gg * gg, axis=0)
                rh, ch = vr / correction, vc / correction
                normalizer = max(float(rh.mean()), self.eps)
                for start in range(0, w.shape[0], self.tile_rows):
                    stop = min(start + self.tile_rows, w.shape[0])
                    update = gg[start:stop] / (
                        np.sqrt(rh[start:stop, None] * ch[None, :] / normalizer) + self.eps
                    )
                    rms = float(np.sqrt(np.mean(update * update)))
                    if rms > self.clip_threshold:
                        update *= self.clip_threshold / (rms + self.eps)
                    w[start:stop] -= self.lr * update
            else:
                v = self.state[i]
                v *= self.beta2
                v += (1.0 - self.beta2) * g * g
                update = g / (np.sqrt(v / correction) + self.eps)
                rms = float(np.sqrt(np.mean(update * update))) if update.size else 0.0
                if rms > self.clip_threshold:
                    update *= self.clip_threshold / (rms + self.eps)
                p.data -= self.lr * update
            p.grad = None


def render_record(record):
    evidence = "\n".join(record.get("trusted_evidence") or [])
    context = (
        f"System: {record.get('system_message', 'You are CFNA.')}\n"
        f"Evidence:\n{evidence}\n"
        f"User: {record.get('user_request', '')}\n"
        "Assistant: "
    ).encode("utf-8")
    response = (record.get("assistant_response", "") + "\n").encode("utf-8")
    return context, response


def choose_window(context, response, seq_len, step):
    response_room = max(2, seq_len // 2)
    response_start = (step * response_room) % max(1, len(response))
    response_piece = response[response_start:response_start + response_room]
    if len(response_piece) < 2:
        response_start = 0
        response_piece = response[:response_room]
    context_room = seq_len - len(response_piece)
    context_piece = context[-context_room:]
    sequence = context_piece + response_piece
    if len(sequence) < 2:
        raise ValueError("record produced an unusable SFT sequence")
    mask = np.zeros(len(sequence), dtype=bool)
    mask[len(context_piece):] = True
    return sequence, mask, response_start


def load_records(root, phase):
    records = []
    allowed = PHASES[phase]
    for path in sorted((root / "data/sft_prompt_aligned/train").glob("shard_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                category = record.get("category", "")
                match = category in allowed
                if phase == "grounded":
                    match = match or "ground" in category or "qualified" in category
                if phase == "verification":
                    match = match or "abstain" in category or "conflict" in category or "verify" in category
                if match and record.get("assistant_response"):
                    records.append(record)
    if not records:
        raise RuntimeError(f"no records found for phase {phase}")
    return records


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=tuple(PHASES), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=9)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    args = parser.parse_args()

    started = time.time()
    np.random.seed(20260710)
    cfg = preset_configs()["base_35m"]
    model = MicroCFNAModel(cfg)

    with args.checkpoint.open("rb") as handle:
        checkpoint = pickle.load(handle)
    if len(checkpoint["params"]) != len(list(model.parameters())):
        raise RuntimeError("checkpoint parameter count does not match base_35m")
    for parameter, stored in zip(model.parameters(), checkpoint["params"]):
        parameter.data[...] = stored

    old_opt = checkpoint["optimizer"]
    optimizer = PagedFactorState(
        model.parameters(),
        lr=args.lr if args.lr is not None else old_opt.get("lr", 1e-5),
        step=old_opt.get("t", checkpoint["meta"].get("step", 0)),
        state=old_opt["v"],
    )
    global_step = int(checkpoint["meta"].get("step", 0)) + 1
    phase_records = load_records(ROOT, args.phase)
    phase_seen = sum(1 for item in checkpoint["meta"].get("history", []) if item.get("phase") == args.phase)
    record = phase_records[phase_seen % len(phase_records)]
    context, response = render_record(record)
    sequence, target_mask, response_offset = choose_window(context, response, args.seq_len, phase_seen)
    batch = np.frombuffer(sequence, dtype=np.uint8).astype(np.int64)[None, :]
    mask = target_mask[None, :]

    optimizer.zero_grad()
    t0 = time.time()
    logits, _ = model.forward(batch)
    first_target = int(np.nonzero(mask[0])[0][0])
    response_logits = logits[:, first_target - 1:-1].reshape(-1, 256)
    response_targets = batch[:, first_target:].reshape(-1)
    loss = F.cross_entropy(response_logits, response_targets)
    forward_s = time.time() - t0
    t0 = time.time()
    loss.backward()
    backward_s = time.time() - t0
    grad_tensors = finite_gradients(model.parameters())
    grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    t0 = time.time()
    optimizer.update()
    update_s = time.time() - t0

    record_log = {
        "step": global_step,
        "phase": args.phase,
        "objective": "assistant_response_masked_sft",
        "record_id": record.get("id"),
        "category": record.get("category"),
        "response_offset": response_offset,
        "sequence_bytes": len(sequence),
        "supervised_target_bytes": int(mask.sum()),
        "loss": float(loss.item()),
        "grad_norm": float(grad_norm),
        "grad_tensors": grad_tensors,
        "forward_s": forward_s,
        "backward_s": backward_s,
        "update_s": update_s,
    }
    history = list(checkpoint["meta"].get("history", []))
    history.append(record_log)
    payload = {
        "format": "microtorch-base35m-phased-sft-v2",
        "config": cfg.__dict__,
        "params": [p.data for p in model.parameters()],
        "optimizer": {"name": "streamfactor", "lr": optimizer.lr, "t": optimizer.step_index, "v": optimizer.state},
        "meta": {
            **checkpoint["meta"],
            "step": global_step,
            "data_cursor": global_step,
            "training_objective": "response-only masked SFT",
            "phase": args.phase,
            "history": history,
        },
    }
    temporary = args.checkpoint.with_suffix(args.checkpoint.suffix + ".tmp")
    t0 = time.time()
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(args.checkpoint)
    save_s = time.time() - t0

    output = {
        **record_log,
        "checkpoint": str(args.checkpoint),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "checkpoint_sha256": sha256(args.checkpoint),
        "save_s": save_s,
        "elapsed_s": time.time() - started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
