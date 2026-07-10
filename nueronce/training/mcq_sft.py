"""Convert open MCQ / QA datasets into NUERONCE SFT conversations, plus a
choice-ranking evaluator.

Two training signals from the same records ("different training methods" for
the same knowledge):

1. **Generative SFT** — each MCQ becomes a ``User:``/``Assistant:`` turn in
   exactly the schema `nueronce.training.dataset_prep` validates and
   `nueronce.training.sharded_sft` trains on, so the whole existing
   validate→dedupe→shard→train pipeline applies unchanged.
2. **Choice ranking** — score each candidate answer by the model's masked
   response loss (`masked_token_loss` over the answer bytes only) and pick the
   lowest; this evaluates *knowledge* without requiring byte-exact generation,
   sidestepping the known renderer weakness (structure generalizes before
   content — see MICRO_NUERONCE_SFT_100K_REPORT).

Normalization is pure-Python over plain dict records, so unit tests inject
records directly; the Hugging Face `datasets` dependency is only touched by
`load_and_convert`, which the local GPU machine runs.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np

from .dialogue_data import make_conversation_batch

_LETTER = "ABCDEFGH"


def _rec(rid: str, prompt: str, response: str, source: str, category: str) -> Dict:
    return {
        "id": rid,
        "messages": [{"role": "user", "content": prompt},
                     {"role": "assistant", "content": response}],
        "source": source,
        "category": category,
    }


def _mcq_prompt(question: str, choices: Sequence[str]) -> str:
    lines = [question.strip()]
    for i, c in enumerate(choices):
        lines.append(f"{_LETTER[i]}) {c.strip()}")
    lines.append("Answer with the correct choice.")
    return "\n".join(lines)


def _mcq_response(answer_idx: int, choices: Sequence[str]) -> str:
    return f"{_LETTER[answer_idx]}) {choices[answer_idx].strip()}"


def normalize_mcq(record: Dict, source: str, category: str) -> Optional[Dict]:
    """Normalize one MCQ record from any of the supported schemas:

    - ARC / OpenBookQA / CommonsenseQA: ``question``/``question_stem``,
      ``choices: {text: [...], label: [...]}``, ``answerKey``
    - MathQA: ``Problem``, ``options`` ("a ) x , b ) y , ..."), ``correct``

    Returns the normalized SFT record, or None if malformed/unanswerable.
    """
    question = record.get("question") or record.get("question_stem") or record.get("Problem")
    if not question or not str(question).strip():
        return None

    choices: List[str] = []
    answer_idx: Optional[int] = None

    ch = record.get("choices")
    if isinstance(ch, dict) and "text" in ch and "label" in ch:
        choices = [str(t) for t in ch["text"]]
        labels = [str(l).strip().upper() for l in ch["label"]]
        key = str(record.get("answerKey", "")).strip().upper()
        if key in labels:
            answer_idx = labels.index(key)
    elif record.get("options") and record.get("correct"):
        # MathQA: "a ) 38 , b ) 27.675 , c ) 30 , ..."
        parts = [p.strip() for p in str(record["options"]).split(",")]
        for p in parts:
            if ")" in p:
                choices.append(p.split(")", 1)[1].strip())
        key = str(record["correct"]).strip().lower()
        idx = ord(key) - ord("a") if len(key) == 1 and key.isalpha() else -1
        if 0 <= idx < len(choices):
            answer_idx = idx

    if answer_idx is None or len(choices) < 2 or len(choices) > len(_LETTER):
        return None
    if not all(c.strip() for c in choices):
        return None

    rid = hashlib.sha256(f"{source}|{question}|{answer_idx}".encode("utf-8")).hexdigest()[:16]
    prompt = _mcq_prompt(str(question), choices)
    response = _mcq_response(answer_idx, choices)
    rec = _rec(f"{category}-{rid}", prompt, response, source, category)
    rec["mcq"] = {"choices": choices, "answer_idx": answer_idx}
    return rec


def normalize_qa(record: Dict, source: str, category: str,
                 question_field: str = "question", answer_field: str = "answer") -> Optional[Dict]:
    """Free-form QA (e.g. GSM8K): question + full worked answer."""
    q = str(record.get(question_field) or "").strip()
    a = str(record.get(answer_field) or "").strip()
    if not q or not a:
        return None
    rid = hashlib.sha256(f"{source}|{q}".encode("utf-8")).hexdigest()[:16]
    return _rec(f"{category}-{rid}", q, a, source, category)


def convert_records(records: Iterable[Dict], template: str, source: str,
                    category: str) -> Iterator[Dict]:
    """Stream-normalize records; silently drops malformed ones (the dataset_prep
    validator downstream counts anything that still slips through)."""
    for record in records:
        out = normalize_mcq(record, source, category) if template == "mcq" \
            else normalize_qa(record, source, category)
        if out is not None:
            yield out


def load_and_convert(entry, split: str = "train", limit: Optional[int] = None) -> Iterator[Dict]:
    """Pull a `nueronce.corpus.stack.CorpusStackEntry` from Hugging Face and
    normalize it (requires the `datasets` package — the local GPU machine)."""
    from datasets import load_dataset  # local-machine dependency, kept lazy

    ds = load_dataset(entry.dataset_name, entry.dataset_config, split=split,
                      streaming=entry.streaming)
    template = entry.document_template or "qa"
    n = 0
    for record in ds:
        out = normalize_mcq(record, entry.source_id, entry.source_id) if template == "mcq" \
            else normalize_qa(record, entry.source_id, entry.source_id)
        if out is None:
            continue
        yield out
        n += 1
        if limit is not None and n >= limit:
            break


# --------------------------------------------------------------------------- #
# Choice ranking: evaluate MCQ knowledge without byte-exact generation
# --------------------------------------------------------------------------- #

def rank_choices(model, question_prompt: str, choices: Sequence[str],
                 max_len: int = 512) -> int:
    """Return the index of the choice the model finds most probable, scored by
    masked response loss (lower = better). Works with any model exposing the
    shared ``forward(byte_ids) -> (logits, ...)`` + ``masked_token_loss``
    interface (NueronceModel here, NUERONCEModel on the GPU machine)."""
    losses = []
    for i, choice in enumerate(choices):
        conv = [{"role": "user", "content": question_prompt},
                {"role": "assistant", "content": _mcq_response(i, choices)}]
        batch = make_conversation_batch([conv], max_len=max_len)
        logits = model.forward(batch["byte_ids"])[0]
        loss = model.masked_token_loss(logits, batch["byte_ids"], batch["target_mask"])
        losses.append(float(loss.item() if hasattr(loss, "item") else loss))
    return int(np.argmin(losses))


def evaluate_mcq(model, records: Sequence[Dict], max_examples: Optional[int] = None,
                 max_len: int = 512) -> Dict:
    """Choice-ranking accuracy over normalized MCQ records (must carry the
    ``mcq`` payload from :func:`normalize_mcq`)."""
    correct = total = 0
    for rec in records[:max_examples]:
        mcq = rec.get("mcq")
        if not mcq:
            continue
        prompt = rec["messages"][0]["content"]
        pred = rank_choices(model, prompt, mcq["choices"], max_len=max_len)
        correct += int(pred == mcq["answer_idx"])
        total += 1
    n_choices = [len(r["mcq"]["choices"]) for r in records[:max_examples] if r.get("mcq")]
    chance = float(np.mean([1.0 / n for n in n_choices])) if n_choices else 0.0
    return {"accuracy": correct / total if total else 0.0, "n": total, "chance": chance}


__all__ = ["normalize_mcq", "normalize_qa", "convert_records", "load_and_convert",
           "rank_choices", "evaluate_mcq"]
