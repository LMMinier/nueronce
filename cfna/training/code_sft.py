"""Code instruction-tuning data: turn real functions into (instruction, code)
pairs so the model learns to *respond* to coding requests, not just continue
code. Torch-free (AST + string work only); emits the same record schema as
``cfna.training.mixed_sft`` so it plugs straight into
``scripts/build_conversation_sft.py`` and the response-only SFT mask.

License note: only run this over PERMISSIVELY-licensed source (the corpus
stack's the-stack-smol / codeparrot dumps, or your own code). The extractor
carries no license metadata itself — that is the caller's responsibility,
same discipline as the rest of the corpus registry.

Each accepted function yields up to three complementary pairs:
  1. spec -> implementation   ("Write a Python function ... that <docstring>")
  2. complete-the-body        (signature + docstring  ->  full function)
  3. explain                  (code  ->  its docstring summary)
so the model sees code as both a target and a thing to describe.
"""

from __future__ import annotations

import ast
import hashlib
from typing import Dict, Iterable, Iterator, List, Optional

SYSTEM = "You are CFNA, a concise coding assistant."


def extract_functions(source: str, *, max_lines: int = 40,
                      min_lines: int = 2) -> List[Dict[str, str]]:
    """Parse Python source and return well-formed top-level/def functions with
    a docstring, bounded in size. Returns dicts with name, signature, doc,
    body_src (full function text). Skips functions without a docstring (no
    reliable natural-language spec) or outside the size band."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    lines = source.splitlines()
    out: List[Dict[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        doc = ast.get_docstring(node)
        if not doc:
            continue
        start = node.lineno - 1
        end = getattr(node, "end_lineno", None)
        if end is None:
            continue
        n = end - start
        if n < min_lines or n > max_lines:
            continue
        full = "\n".join(lines[start:end]).rstrip()
        if len(full) > 1800 or "\t" in full:  # keep it clean + bounded
            continue
        # signature line(s): up to the line ending with ':'
        sig_lines, i = [], start
        while i < end:
            sig_lines.append(lines[i])
            if lines[i].rstrip().endswith(":"):
                break
            i += 1
        out.append({
            "name": node.name,
            "signature": "\n".join(sig_lines).strip(),
            "doc": " ".join(doc.strip().split())[:300],
            "full": full,
        })
    return out


def _rec(idx: int, prompt: str, response: str, kind: str, source: str) -> Dict:
    return {
        "id": f"code-{kind}-{idx:06d}",
        "source": source,
        "category": f"code_{kind}",
        "system_message": SYSTEM,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
    }


def functions_to_records(funcs: Iterable[Dict[str, str]], *, source: str,
                         start: int = 0) -> Iterator[Dict]:
    """Turn extracted functions into instruction/code message records (schema
    matches cfna.training.mixed_sft; single user turn, so evidence/plan-free
    records render through prompting.format_training_example)."""
    i = start
    for f in funcs:
        doc, name, sig, full = f["doc"], f["name"], f["signature"], f["full"]
        # 1) spec -> implementation
        yield _rec(i, f"Write a Python function `{name}` that {doc}",
                   full, "impl", source); i += 1
        # 2) complete the body from signature + docstring
        stub = f'{sig}\n    """{doc}"""'
        yield _rec(i, f"Complete this Python function:\n{stub}",
                   full, "complete", source); i += 1
        # 3) explain what code does
        yield _rec(i, f"In one sentence, what does this function do?\n{full}",
                   doc if doc.endswith(".") else doc + ".", "explain", source); i += 1


def records_from_files(paths: Iterable[str], *, source: str = "permissive-code",
                       **extract_kw) -> Iterator[Dict]:
    """Convenience: read .py files and yield deduped code-SFT records. Caller
    must ensure the files are permissively licensed."""
    seen = set()
    idx = 0
    for p in paths:
        try:
            src = open(p, "r", encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        funcs = extract_functions(src, **extract_kw)
        for rec in functions_to_records(funcs, source=source, start=idx):
            key = hashlib.sha256(
                (rec["messages"][0]["content"] + rec["messages"][1]["content"])
                .encode("utf-8")).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            idx += 1
            yield rec


_PERMISSIVE = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc",
               "cc0-1.0", "unlicense", "0bsd"}


def records_from_rows(rows: Iterable[Dict], *, content_field: str = "content",
                      license_field: Optional[str] = "license",
                      allowed: Optional[set] = None, source: str = "the-stack",
                      max_functions: Optional[int] = None, **extract_kw) -> Iterator[Dict]:
    """Yield code-SFT records from an iterable of dataset rows (e.g. the
    license-filtered the-stack-smol / codeparrot Hugging Face datasets, passed
    in by the caller — this stays torch/datasets-free and dependency-injected
    so it is unit-testable without a download). Rows lacking a permissive
    license (when ``license_field`` is present) are skipped."""
    allowed = allowed or _PERMISSIVE
    seen = set()
    idx = 0
    for row in rows:
        if license_field and str(row.get(license_field, "")).lower() not in allowed:
            continue
        code = row.get(content_field) or ""
        if not isinstance(code, str) or not code.strip():
            continue
        for rec in functions_to_records(extract_functions(code, **extract_kw),
                                        source=source, start=idx):
            key = hashlib.sha256((rec["messages"][0]["content"]
                                  + rec["messages"][1]["content"]).encode("utf-8")).hexdigest()
            if key in seen:
                continue
            seen.add(key); idx += 1
            yield rec
            if max_functions and idx >= max_functions:
                return


__all__ = ["extract_functions", "functions_to_records", "records_from_files",
           "records_from_rows", "SYSTEM"]
