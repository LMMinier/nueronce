#!/usr/bin/env python3
"""Drop contaminated (prompt, response) rows from SFT JSONL files.

Two contamination classes were found in the committed OASST1-derived SFT
set (docs/RESULTS.md, 2026-07-23; e.g. train.jsonl row 1 pairs a monopsony
question with a Minecraft-modding answer that opens "Yes, that's correct."):

  1. ORPHANED CONTEXT -- the response was an assistant reply to a *deeper*
     turn of some conversation, but got paired with a root prompt it never
     answered. Fingerprint: openings that acknowledge prior context that is
     not in the prompt ("Yes, that's correct", "You're right", "I apologize
     for the confusion", ...).
  2. ROLE-SWAPPED -- the "response" is actually another user's *prompt*
     (asks a question / issues a request instead of answering).

Training on such rows directly teaches "the response need not relate to the
prompt" -- a plausible contributor to prompt-blind generation.

Modes:
  - default (heuristic): pattern-based drop, runs offline anywhere.
  - --oasst-verify: exact repair -- downloads OpenAssistant/oasst1, indexes
    message text -> (id, parent, role), and keeps a row only if its response
    text is an assistant message whose parent prompter message's text equals
    the row's prompt. Needs Hugging Face access; run this mode on the
    training machine and prefer its output over the heuristic one.

Non-OASST rows (source != OpenAssistant/oasst1) pass through untouched.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ORPHAN_OPENINGS = (
    "yes, that's correct", "yes, that is correct", "that's correct",
    "that is correct", "you're right", "you are right", "you're correct",
    "you are correct", "i apologize for the confusion",
    "i apologize for the mistake", "my apologies", "sorry for the confusion",
    "as i mentioned", "as mentioned earlier", "as i said", "as stated above",
    "glad you liked", "i'm glad you", "thank you for pointing that out",
    "thanks for pointing that out", "good catch", "here is the revised",
    "here's the revised", "sure, here is the updated", "sure, here's the updated",
)

REQUEST_OPENINGS = (
    "can you", "could you", "would you", "will you", "please write",
    "please explain", "please give", "please provide", "write me", "give me",
    "tell me", "explain to me", "i want you to", "i need you to", "help me",
    "how do i", "how can i", "what is the best way to",
)


def classify(prompt: str, response: str) -> str | None:
    """Return a drop-reason or None to keep."""
    resp = response.strip().lower()
    # orphaned context: response opens by acknowledging something the prompt
    # never said (only flag when the prompt itself doesn't invite agreement)
    for opening in ORPHAN_OPENINGS:
        if resp.startswith(opening):
            invite = ("correct?" in prompt.lower() or "right?" in prompt.lower()
                      or "agree" in prompt.lower())
            if not invite:
                return f"orphaned_context:{opening[:24]}"
    # role-swapped: the "response" is itself a request/question
    first_line = resp.splitlines()[0] if resp else ""
    if first_line.endswith("?") and len(resp) < 400:
        for opening in REQUEST_OPENINGS:
            if resp.startswith(opening):
                return "role_swapped:question_response"
    return None


def load_oasst_index():
    from datasets import load_dataset
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    by_id, assistant_by_text = {}, {}
    for row in ds:
        by_id[row["message_id"]] = row
        if row["role"] == "assistant":
            assistant_by_text.setdefault(row["text"].strip(), []).append(row)
    return by_id, assistant_by_text


def verify_against_oasst(prompt: str, response: str, by_id, assistant_by_text) -> bool:
    """True iff response is an assistant message whose parent prompter text
    is exactly this prompt."""
    for msg in assistant_by_text.get(response.strip(), []):
        parent = by_id.get(msg.get("parent_id"))
        if parent and parent.get("role") == "prompter" and parent["text"].strip() == prompt.strip():
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="JSONL files of {prompt,response,...} rows")
    ap.add_argument("--out-dir", default="data/clean_sft")
    ap.add_argument("--report", default="data/clean_sft/clean_report.json")
    ap.add_argument("--oasst-verify", action="store_true",
                     help="exact parent-child verification against OpenAssistant/oasst1 "
                          "(needs HF access; supersedes the heuristics for oasst rows)")
    args = ap.parse_args()

    by_id = assistant_by_text = None
    if args.oasst_verify:
        print("loading OpenAssistant/oasst1 for exact verification...")
        by_id, assistant_by_text = load_oasst_index()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"mode": "oasst-verify" if args.oasst_verify else "heuristic", "files": {}}

    for path_str in args.inputs:
        path = Path(path_str)
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        kept, dropped = [], []
        for i, row in enumerate(rows):
            prompt, response = row.get("prompt", ""), row.get("response", "")
            is_oasst = "oasst" in str(row.get("source", "")).lower()
            reason = None
            if is_oasst:
                if args.oasst_verify:
                    if not verify_against_oasst(prompt, response, by_id, assistant_by_text):
                        reason = "oasst_pairing_unverified"
                else:
                    reason = classify(prompt, response)
            if reason:
                dropped.append({"index": i, "reason": reason,
                                "prompt": prompt[:120], "response": response[:120]})
            else:
                kept.append(row)
        out_path = out_dir / path.name
        with out_path.open("w", encoding="utf-8") as f:
            for row in kept:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        report["files"][path.name] = {
            "input_rows": len(rows), "kept": len(kept), "dropped": len(dropped),
            "dropped_fraction": len(dropped) / max(1, len(rows)),
            "reasons": {}, "dropped_rows": dropped,
        }
        for d in dropped:
            key = d["reason"].split(":")[0]
            report["files"][path.name]["reasons"][key] = \
                report["files"][path.name]["reasons"].get(key, 0) + 1
        print(f"{path.name}: kept {len(kept)}/{len(rows)} "
              f"(dropped {len(dropped)}, {len(dropped)/max(1,len(rows))*100:.1f}%) -> {out_path}")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("report:", report_path)


if __name__ == "__main__":
    main()
