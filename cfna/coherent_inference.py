"""Practical coherent-response wrapper for CFNA checkpoints.

This module does not change model weights and does not pretend tool answers are
learned. It provides a clean inference surface:

- consistent chat prompt formatting,
- robust trimming of byte continuations,
- explicit deterministic assists for narrow computable/factual prompts,
- probe metrics that report model-only and assisted behavior separately.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .training.synthetic_dialogue import _CAPITALS

_PRINTABLE = re.compile(r"[\x09\x0A\x0D\x20-\x7E]")
_ADD = re.compile(r"(?:what is |add |calculate )?(-?\d+)\s*(?:plus|\+|and)\s*(-?\d+)", re.I)
_SUB = re.compile(r"(?:what is |subtract |calculate )?(-?\d+)\s*(?:minus|-)\s*(-?\d+)", re.I)
_MUL = re.compile(r"(?:what is |multiply |calculate )?(-?\d+)\s*(?:times|\*|by)\s*(-?\d+)", re.I)
_DIV = re.compile(r"(?:what is |divide |calculate )?(-?\d+)\s*(?:divided by|/|by)\s*(-?\d+)", re.I)
_CAPITAL = re.compile(r"capital of ([A-Za-z ]+)\??", re.I)

_CAPITAL_MAP = {country.lower(): capital for country, capital, _ in _CAPITALS}


@dataclass(frozen=True)
class Response:
    text: str
    source: str  # "model" | "tool" | "fallback"
    raw_model_text: str = ""
    warning: Optional[str] = None


def deterministic_answer(prompt: str) -> Optional[str]:
    """Return an exact answer for narrow prompts the project already treats as
    computable or table-backed. This is an adapter/tool path, not model learning."""
    text = prompt.strip()
    for regex, op_name, fn in (
        (_ADD, "plus", lambda a, b: a + b),
        (_SUB, "minus", lambda a, b: a - b),
        (_MUL, "times", lambda a, b: a * b),
    ):
        m = regex.search(text)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            return f"{a} {op_name} {b} equals {fn(a, b)}."
    m = _DIV.search(text)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if b == 0:
            return "Division by zero is undefined."
        if a % b == 0:
            return f"{a} divided by {b} equals {a // b}."
        return f"{a} divided by {b} equals {a / b:.3f}."
    m = _CAPITAL.search(text)
    if m:
        country = m.group(1).strip().lower()
        if country in _CAPITAL_MAP:
            return f"The capital of {country.title()} is {_CAPITAL_MAP[country]}."
    if "what can you do" in text.lower():
        return "I can answer simple questions, use narrow tools for exact arithmetic, and generate short text from the CFNA byte model."
    return None


def clean_reply(text: str) -> str:
    text = text.replace("\r", "\n").split("User:", 1)[0].split("Assistant:", 1)[0]
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    for end in range(len(text) - 1, -1, -1):
        if text[end] in ".!?":
            return text[: end + 1].strip()
    return text[:240].strip()


def coherence_warning(text: str) -> Optional[str]:
    if len(text.strip()) < 2:
        return "empty_or_too_short"
    printable = sum(1 for c in text if _PRINTABLE.match(c))
    if printable / max(1, len(text)) < 0.9:
        return "mostly_nonprintable"
    if text:
        most = max(text.count(c) for c in set(text))
        if most / len(text) > 0.45:
            return "repetitive_output"
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    if len(words) >= 8:
        most_word = max(words.count(w) for w in set(words))
        if most_word / len(words) > 0.6:
            return "repetitive_output"
    return None


def respond(
    prompt: str,
    model_fn: Callable[[str], str],
    *,
    assist_tools: bool = True,
    fallback: str = "I am not confident enough to answer that coherently yet.",
) -> Response:
    if assist_tools:
        ans = deterministic_answer(prompt)
        if ans is not None:
            return Response(ans, "tool")
    raw = model_fn(prompt)
    cleaned = clean_reply(raw)
    warning = coherence_warning(cleaned)
    if warning:
        return Response(fallback, "fallback", raw_model_text=raw, warning=warning)
    return Response(cleaned, "model", raw_model_text=raw)


@dataclass(frozen=True)
class Probe:
    prompt: str
    expected_substring: Optional[str] = None


DEFAULT_PROBES: List[Probe] = [
    Probe("Hello, who are you?", "assistant"),
    Probe("What is 17 plus 25?", "42"),
    Probe("Calculate 9 * 8.", "72"),
    Probe("What is the capital of France?", "Paris"),
    Probe("What can you do?", "simple questions"),
]


def run_probes(model_fn: Callable[[str], str], *, assist_tools: bool = True,
               probes: List[Probe] = DEFAULT_PROBES) -> Dict:
    rows = []
    for p in probes:
        r = respond(p.prompt, model_fn, assist_tools=assist_tools)
        ok = (p.expected_substring is None or
              p.expected_substring.lower() in r.text.lower())
        rows.append({
            "prompt": p.prompt,
            "response": r.text,
            "source": r.source,
            "warning": r.warning,
            "expected_substring": p.expected_substring,
            "passed": ok,
        })
    return {
        "n": len(rows),
        "pass_rate": sum(row["passed"] for row in rows) / max(1, len(rows)),
        "tool_rate": sum(row["source"] == "tool" for row in rows) / max(1, len(rows)),
        "fallback_rate": sum(row["source"] == "fallback" for row in rows) / max(1, len(rows)),
        "rows": rows,
    }


__all__ = [
    "Response", "Probe", "DEFAULT_PROBES", "deterministic_answer",
    "clean_reply", "coherence_warning", "respond", "run_probes",
]
