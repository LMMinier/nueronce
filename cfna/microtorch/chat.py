"""Conversation interface for a trained MicroCFNAModel checkpoint — the
microtorch port of ``cfna.chat`` (which needs PyTorch to load a ``CFNAModel``
checkpoint). Same honest framing applies: a small byte-level model produces
English-shaped continuations in the register it learned; real turn-taking
signal requires an SFT pass (``cfna.training.sharded_sft`` /
``cfna.training.sft``), same as the PyTorch path.

Checkpoints here are the ``cfna.training.sharded_sft.save_checkpoint`` pickle
format (reused directly, not re-implemented) rather than ``cfna.chat``'s
``torch.save`` format — the two are not interchangeable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from ..prompting import assemble_conversation_prompt
from ..training.dialogue_data import BOT_TAG, USER_TAG
from ..training.sharded_sft import load_checkpoint as _load_checkpoint_payload
from .cfna_model import MicroCFNAModel, MicroModelConfig

_STOP = b"\n"


def load_checkpoint(path: str) -> Tuple[MicroCFNAModel, dict]:
    payload = _load_checkpoint_payload(path)
    model = MicroCFNAModel(MicroModelConfig(**payload["config"]))
    for p, arr in zip(model.parameters(), payload["params"]):
        p.data = arr.copy()
    return model, payload


@dataclass
class MicroConversation:
    model: MicroCFNAModel
    system: str = ""
    user_tag: str = USER_TAG
    bot_tag: str = BOT_TAG
    temperature: float = 0.7
    max_new: int = 80
    min_new: int = 8
    max_ctx: int = 288
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
        raw = self.model.generate(
            context, max_new=self.max_new, temperature=self.temperature,
            greedy=(self.temperature <= 0), max_ctx=self.max_ctx,
            stop_bytes=_STOP, min_new=self.min_new,
        )
        reply_bytes = raw[len(context):]
        reply = reply_bytes.decode("utf-8", errors="replace").strip()
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


__all__ = ["load_checkpoint", "MicroConversation"]
