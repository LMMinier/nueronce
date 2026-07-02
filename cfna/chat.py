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
    STOP_SEQUENCES,
    USER,
    assemble_conversation_prompt,
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
              stop_bytes: bytes, min_new: int, max_ctx: int) -> bytes:
    del stop_bytes, min_new
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
        return assemble_conversation_prompt(
            system_message=self.system,
            current_user=user_msg,
            recent_turns=self.transcript,
            max_chars=self.max_ctx,
        ).encode("utf-8")

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
    text = text.replace("\n", " ").strip()
    for end in range(len(text) - 1, -1, -1):
        if text[end] in ".!?":
            return text[: end + 1].strip()
    return text


__all__ = ["load_checkpoint", "Conversation"]
