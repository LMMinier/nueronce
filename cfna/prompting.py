"""Canonical CFNA prompt format shared by training and inference."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

SYSTEM = "<|system|>"
USER = "<|user|>"
EVIDENCE = "<|evidence|>"
PLAN = "<|plan|>"
ASSISTANT = "<|assistant|>"
END = "<|end|>"

STOP_SEQUENCES = (END.encode("utf-8"), f"\n{USER}".encode("utf-8"), b"\nUser:")


def _block(marker: str, text: str) -> str:
    return f"{marker}\n{text.strip() if text else ''}\n"


def evidence_text(items: Iterable[object]) -> str:
    rows = []
    for i, ev in enumerate(items):
        sid = getattr(ev, "memory_id", None) or getattr(ev, "source_id", None) or f"evidence_{i}"
        content = getattr(ev, "content", None) or getattr(ev, "raw_text", None) or str(ev)
        auth = getattr(ev, "authority_level", None) or getattr(ev, "authority", "")
        provenance = getattr(ev, "authenticity_status", "")
        rows.append(f"[{sid}] {content} (authority={auth}, authenticity={provenance})")
    return "\n".join(rows)


def plan_text(plan: object) -> str:
    if plan is None:
        return ""
    if isinstance(plan, dict):
        parts = []
        for k, v in plan.items():
            if isinstance(v, (list, tuple)):
                parts.append(f"{k}: " + "; ".join(str(x) for x in v))
            else:
                parts.append(f"{k}: {v}")
        return "\n".join(parts)
    return str(plan)


def format_inference_prompt(
    *,
    system_message: str,
    user_request: str,
    trusted_evidence: str = "",
    response_plan: str = "",
) -> str:
    return (
        _block(SYSTEM, system_message)
        + _block(USER, user_request)
        + _block(EVIDENCE, trusted_evidence)
        + _block(PLAN, response_plan)
        + f"{ASSISTANT}\n"
    )


def format_training_example(
    *,
    system_message: str = "",
    user_request: str,
    assistant_response: str,
    trusted_evidence: str = "",
    response_plan: str = "",
) -> Tuple[bytes, List[bool]]:
    prefix = format_inference_prompt(
        system_message=system_message,
        user_request=user_request,
        trusted_evidence=trusted_evidence,
        response_plan=response_plan,
    )
    full = prefix + assistant_response.strip() + "\n" + END + "\n"
    pb = prefix.encode("utf-8")
    fb = full.encode("utf-8")
    return fb, [False] * len(pb) + [True] * (len(fb) - len(pb))


def format_revision_prompt(
    *,
    system_message: str,
    user_request: str,
    trusted_evidence: str,
    response_plan: str,
    first_draft: str,
    verifier_feedback: dict,
) -> str:
    feedback = "\n".join(f"{k}: {v}" for k, v in verifier_feedback.items())
    plan = (
        f"{response_plan}\n"
        "Revision instructions: remove, correct, or qualify only the listed failures.\n"
        f"First draft: {first_draft}\n"
        f"Verifier feedback:\n{feedback}"
    )
    return format_inference_prompt(
        system_message=system_message,
        user_request=user_request,
        trusted_evidence=trusted_evidence,
        response_plan=plan,
    )


def extract_assistant_continuation(text_or_bytes: str | bytes) -> str:
    text = text_or_bytes.decode("utf-8", errors="replace") if isinstance(text_or_bytes, bytes) else text_or_bytes
    if ASSISTANT in text:
        text = text.split(ASSISTANT, 1)[1]
    for stop in (END, f"\n{USER}", "\nUser:"):
        if stop in text:
            text = text.split(stop, 1)[0]
    return text.strip()


def assemble_conversation_prompt(
    *,
    system_message: str,
    current_user: str,
    recent_turns: Sequence[Tuple[str, str]] = (),
    trusted_evidence: str = "",
    response_plan: str = "",
    max_chars: Optional[int] = None,
) -> str:
    """Structured truncation: keep system/current/evidence/plan, drop old turns first."""
    turns = list(recent_turns)
    while True:
        history = "".join(_block(USER if r == "user" else ASSISTANT, t) for r, t in turns)
        prompt = (
            _block(SYSTEM, system_message)
            + history
            + _block(USER, current_user)
            + _block(EVIDENCE, trusted_evidence)
            + _block(PLAN, response_plan)
            + f"{ASSISTANT}\n"
        )
        if max_chars is None or len(prompt) <= max_chars or not turns:
            return prompt
        turns = turns[2:] if len(turns) >= 2 else []


__all__ = [
    "SYSTEM", "USER", "EVIDENCE", "PLAN", "ASSISTANT", "END", "STOP_SEQUENCES",
    "format_training_example", "format_inference_prompt", "format_revision_prompt",
    "extract_assistant_continuation", "assemble_conversation_prompt",
    "evidence_text", "plan_text",
]
