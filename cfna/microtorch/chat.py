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

from ..prompting import ASSISTANT, END, SYSTEM, USER
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
    """``prompt_format`` must match what the checkpoint was trained on:
    ``"canonical"`` = the ``<|user|>``-marker format from ``cfna.prompting``;
    ``"legacy"`` = the ``User: ``/``Assistant: `` layout every
    ``sharded_sft`` checkpoint to date was trained on. Checkpoints that do
    not record a ``prompt_format`` in their meta predate the canonical
    format, so ``from_checkpoint_payload`` resolves them to legacy — feeding
    a model marker bytes it has never seen guarantees garbage output.
    """

    model: MicroCFNAModel
    system: str = ""
    user_tag: str = USER_TAG
    bot_tag: str = BOT_TAG
    temperature: float = 0.7
    max_new: int = 80
    min_new: int = 8
    max_ctx: int = 288
    prompt_format: str = "canonical"
    use_incremental: bool = True  # state-cached generation (17x measured; byte-exact tested)
    transcript: List[Tuple[str, str]] = field(default_factory=list)

    def _generate(self, context: bytes) -> bytes:
        kwargs = dict(max_new=self.max_new, temperature=self.temperature,
                      greedy=(self.temperature <= 0), max_ctx=self.max_ctx,
                      stop_bytes=_STOP, min_new=self.min_new)
        if self.use_incremental:
            try:
                from .incremental import IncrementalGenerator
                return IncrementalGenerator(self.model).generate(context, **kwargs)
            except Exception:
                pass  # never let the fast path break chat
        return self.model.generate(context, **kwargs)

    @staticmethod
    def resolve_format(payload: dict) -> str:
        meta = payload.get("meta") or {}
        return meta.get("prompt_format", payload.get("prompt_format", "legacy"))

    def _context(self, user_msg: str) -> bytes:
        if self.prompt_format == "legacy":
            # Literal legacy tags — NOT the module-level USER_TAG/BOT_TAG, which
            # were later repointed to the canonical <|user|>/<|assistant|>
            # markers. Legacy checkpoints were trained on exactly
            # "User: <msg>\nAssistant: <reply>\n" with no system line when empty.
            u, b = "User: ", "Assistant: "
            parts = []
            if self.system:
                parts.append(self.system.strip())
            for role, text in self.transcript:
                parts.append(f"{u if role == 'user' else b}{text}")
            parts.append(f"{u}{user_msg}")
            return ("\n".join(parts) + f"\n{b}").encode("utf-8")
        text = f"{SYSTEM}\n{self.system.strip() if self.system else ''}\n"
        for role, content in self.transcript:
            if role == "user":
                text += f"{USER}\n{content}\n"
            else:
                text += f"{ASSISTANT}\n{content}\n{END}\n"
        text += f"{USER}\n{user_msg}\n{ASSISTANT}\n"
        return text.encode("utf-8")[-self.max_ctx:]

    def say(self, user_msg: str) -> str:
        context = self._context(user_msg)
        raw = self._generate(context)
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
