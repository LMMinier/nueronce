"""Conversation interface for a trained CFNA byte checkpoint.

Honest framing: this is a small byte-level model trained on public-domain books
and speeches. On its own it does not "understand" a conversation the way a
large instruct-tuned model does — it continues text in the register it learned.
The chat loop conditions generation on the running transcript and returns the
model's continuation as the reply, stopping at a turn boundary.

``scripts/train_sft.py`` (backed by ``cfna.training.sft`` /
``cfna.training.vgrft.VGRFTTrainer``) adds an actual supervised fine-tuning
pass over (prompt, response) turns in this same ``User: `` / ``Assistant: ``
layout, so a checkpoint that has been through it has real turn-taking signal —
not just next-byte continuation of prose — behind its replies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from .model import CFNAModel, ModelConfig
from .prompting import (
    ASSISTANT,
    END,
    STOP_SEQUENCES,
    SYSTEM,
    USER,
    extract_assistant_continuation,
)


def load_checkpoint(path: str) -> Tuple[CFNAModel, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["config"])
    model = CFNAModel(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def _continue(model: CFNAModel, context: bytes, max_new: int, temperature: float,
              stop_bytes: bytes, min_new: int, max_ctx: int,
              use_incremental: bool = True) -> bytes:
    del stop_bytes, min_new
    if use_incremental:
        # State-cached generation (cfna.incremental): the engine must prove
        # itself before it is trusted — on first use per model we prime it on
        # this context and require its last-position logits to match a dense
        # forward; any mismatch disables it for that model and falls back to
        # the dense path. Retrieval-free chat only.
        try:
            from .incremental import IncrementalGenerator
            if getattr(model, "_incremental_ok", None) is None:
                inc = IncrementalGenerator(model)
                inc.prime(list(context)[-max_ctx:] or [32])
                fast = inc._last_logits()
                ctx_t = torch.tensor([inc.ids], dtype=torch.long,
                                     device=next(model.parameters()).device)
                dense = model(ctx_t)[0][0, -1]
                model._incremental_ok = bool(torch.allclose(fast, dense, atol=1e-4))
            if model._incremental_ok:
                return IncrementalGenerator(model).generate(
                    context, max_new=max_new, temperature=temperature,
                    greedy=(temperature <= 0), max_ctx=max_ctx,
                    stop_sequences=STOP_SEQUENCES, continuation_only=True,
                )
        except Exception:
            model._incremental_ok = False  # never let the fast path break chat
    return model.generate(
        context,
        max_new=max_new,
        temperature=temperature,
        greedy=(temperature <= 0),
        max_ctx=max_ctx,
        stop_sequences=STOP_SEQUENCES,
        continuation_only=True,
    )


@dataclass
class Conversation:
    model: CFNAModel
    system: str = ""
    user_tag: str = USER
    bot_tag: str = ASSISTANT
    temperature: float = 0.7
    max_new: int = 80
    min_new: int = 8
    max_ctx: int = 320
    transcript: List[Tuple[str, str]] = field(default_factory=list)

    def _context(self, user_msg: str) -> bytes:
        # Match cfna.training.dialogue_data.encode_messages exactly for
        # message-SFT checkpoints: no empty evidence/plan blocks.
        turns = list(self.transcript)
        while True:
            text = f"{SYSTEM}\n{self.system.strip() if self.system else ''}\n"
            for role, content in turns:
                if role == "user":
                    text += f"{USER}\n{content}\n"
                else:
                    text += f"{ASSISTANT}\n{content}\n{END}\n"
            text += f"{USER}\n{user_msg}\n{ASSISTANT}\n"
            out = text.encode("utf-8")
            if len(out) <= self.max_ctx or not turns:
                return out[-self.max_ctx:]
            turns = turns[2:] if len(turns) >= 2 else []

    def say(self, user_msg: str) -> str:
        context = self._context(user_msg)
        raw = _continue(
            self.model, context, max_new=self.max_new, temperature=self.temperature,
            stop_bytes=b"\n", min_new=self.min_new, max_ctx=self.max_ctx,
        )
        reply = extract_assistant_continuation(raw)
        reply = _tidy(reply)
        self.transcript.append(("user", user_msg))
        self.transcript.append(("assistant", reply))
        return reply


def _tidy(text: str) -> str:
    """Trim a generated continuation to its last complete sentence for readability."""
    if "<|" in text:
        text = text.split("<|", 1)[0]
    text = text.replace("\n", " ").strip()
    for end in range(len(text) - 1, -1, -1):
        if text[end] in ".!?":
            return text[: end + 1].strip()
    return text


__all__ = ["load_checkpoint", "Conversation"]
