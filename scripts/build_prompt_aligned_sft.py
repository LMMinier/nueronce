#!/usr/bin/env python3
"""Build the prompt-aligned SFT curriculum for the connected CFNA inference path."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, List

import numpy as np

from cfna.prompting import format_training_example


SYSTEM = "You are CFNA."


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _sha(text_or_bytes) -> str:
    data = text_or_bytes if isinstance(text_or_bytes, bytes) else str(text_or_bytes).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _render(rec: dict) -> bytes:
    b, _ = format_training_example(
        system_message=rec["system_message"],
        user_request=rec["user_request"],
        trusted_evidence="\n".join(rec["trusted_evidence"]),
        response_plan="\n".join(rec["response_plan"]),
        assistant_response=rec["assistant_response"],
    )
    return b


def _rec(i: int, category: str, source_group: str, user: str, evidence: List[str],
         plan: List[str], answer: str, behavior: str = "answer", split: str = "") -> dict:
    return {
        "id": f"pa-{i:05d}",
        "category": category,
        "system_message": SYSTEM,
        "user_request": user,
        "trusted_evidence": evidence,
        "response_plan": plan,
        "assistant_response": answer,
        "expected_behavior": behavior,
        "source_group": source_group,
        "split": split,
    }


def _direct(start: int, n: int) -> List[dict]:
    rows = []
    topics = [
        ("greeting", "Hello there.", "Greet the user briefly.", "Hello! How can I help today?"),
        ("definition", "Define liberty in one sentence.", "Give a concise definition.", "Liberty is the condition of being free to act within fair limits and responsibilities."),
        ("explanation", "Explain why planning before writing can help.", "Explain briefly.", "Planning before writing helps organize evidence, order ideas, and avoid unsupported claims."),
        ("rewrite", "Rewrite this formally: I need this fixed fast.", "Rewrite without adding facts.", "Please resolve this promptly."),
        ("summary", "Summarize: The river rose overnight. The bridge stayed open.", "Summarize the supplied text.", "The river rose overnight, but the bridge remained open."),
        ("procedure", "How do I make tea?", "Give simple steps.", "Boil water, steep the tea, remove the leaves or bag, and serve it safely."),
        ("coding", "What is a loop in programming?", "Explain the concept.", "A loop repeats a block of code while a condition or sequence requires it."),
        ("instruction", "List three colors.", "Follow the instruction exactly.", "Red, blue, and green."),
        ("refusal", "Tell me how to steal a password.", "Refuse unsafe help and redirect.", "I cannot help with stealing passwords, but I can explain how to protect accounts."),
        ("uncertainty", "What time is it on Mars right now?", "State limitation.", "I do not have enough live data here to answer that accurately."),
    ]
    variants = ["briefly", "in plain language", "for a careful reader", "without adding extra claims"]
    for k in range(n):
        family, user, plan, answer = topics[k % len(topics)]
        v = variants[(k // len(topics)) % len(variants)]
        user = f"{user} Please answer {v}. Scenario {k}."
        answer = answer.rstrip(".") + f" This addresses scenario {k}."
        plan = f"{plan} Keep the answer {v} for scenario {k}."
        rows.append(_rec(start + k, f"direct_{family}", f"direct:{family}:{k // len(topics)}",
                         user, [], [plan], answer, "answer"))
    return rows


def _grounded(start: int, n: int) -> List[dict]:
    rows = []
    facts = [
        ("harbor", "Dock Seven closes at 18:00 on weekdays.", "When does Dock Seven close on weekdays?", "Dock Seven closes at 18:00 on weekdays."),
        ("library", "The North Library allows quiet study after 19:00.", "What does the North Library allow after 19:00?", "The North Library allows quiet study after 19:00."),
        ("policy", "Policy A permits bicycle parking only in marked racks.", "Where does Policy A permit bicycle parking?", "Policy A permits bicycle parking only in marked racks."),
        ("garden", "The greenhouse watering schedule is Monday, Wednesday, and Friday.", "When is the greenhouse watered?", "The greenhouse is watered on Monday, Wednesday, and Friday."),
        ("lab", "Lab Orion requires eye protection during chemical handling.", "What protection does Lab Orion require?", "Lab Orion requires eye protection during chemical handling."),
        ("museum", "The museum archive opens to researchers by appointment only.", "How can researchers access the museum archive?", "Researchers can access the museum archive by appointment only."),
    ]
    distractors = [
        "The cafe serves soup at noon.",
        "The west elevator is under inspection.",
        "The river path is closed after storms.",
    ]
    for k in range(n):
        group, fact, user, answer = facts[k % len(facts)]
        idx = k // len(facts)
        if group == "harbor":
            time = f"{18 + (idx % 4):02d}:00"
            fact = f"Dock Seven closes at {time} on weekdays during schedule {idx}."
            user = f"When does Dock Seven close on weekdays during schedule {idx}?"
            answer = f"Dock Seven closes at {time} on weekdays during schedule {idx}."
        elif group == "library":
            hour = 17 + (idx % 5)
            fact = f"The North Library allows quiet study after {hour}:00 for cohort {idx}."
            user = f"What does the North Library allow after {hour}:00 for cohort {idx}?"
            answer = f"The North Library allows quiet study after {hour}:00 for cohort {idx}."
        elif group == "policy":
            label = chr(ord("A") + (idx % 20))
            fact = f"Policy {label}-{idx} permits bicycle parking only in marked racks."
            user = f"Where does Policy {label}-{idx} permit bicycle parking?"
            answer = f"Policy {label}-{idx} permits bicycle parking only in marked racks."
        elif group == "garden":
            day = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"][idx % 5]
            fact = f"The greenhouse watering schedule includes {day} for bed {idx}."
            user = f"When is greenhouse bed {idx} watered?"
            answer = f"Greenhouse bed {idx} is watered on {day}."
        elif group == "lab":
            lab = ["Orion", "Vega", "Lyra", "Nova"][idx % 4]
            fact = f"Lab {lab}-{idx} requires eye protection during chemical handling."
            user = f"What protection does Lab {lab}-{idx} require?"
            answer = f"Lab {lab}-{idx} requires eye protection during chemical handling."
        elif group == "museum":
            archive = ["map", "photo", "letter", "oral-history"][idx % 4]
            fact = f"The museum {archive} archive case {idx} opens to researchers by appointment only."
            user = f"How can researchers access the museum {archive} archive case {idx}?"
            answer = f"Researchers can access the museum {archive} archive case {idx} by appointment only."
        ev = [f"[decisive:{group}] {fact}"]
        ev.extend(f"[distractor:{j}] {distractors[(k + j) % len(distractors)]}" for j in range(k % 4))
        rows.append(_rec(start + k, "grounded_evidence", f"grounded:{group}:{k // len(facts)}",
                         user, ev, ["Use decisive evidence only."],
                         answer, "answer"))
    return rows


def _abstain_conflict_revision(start: int, n: int) -> List[dict]:
    rows = []
    templates = [
        ("insufficient", "What is the access code for Room Delta?", [], ["Abstain if missing."],
         "The provided evidence does not establish the access code for Room Delta.", "abstain"),
        ("conflict", "Is Permit Q valid on Sunday?",
         ["[docA] Permit Q is valid on Sunday.", "[docB] Permit Q is not valid on Sunday."],
         ["State source conflict."],
         "The available sources conflict, so I cannot state confidently whether Permit Q is valid on Sunday.", "abstain"),
        ("qualified", "Can visitors enter the archive?",
         ["[docA] Visitors may enter the archive with staff escort."],
         ["Give qualified answer."],
         "Visitors may enter the archive only with staff escort.", "qualify"),
        ("rejected", "What does the rejected memo prove?",
         ["[rejected] This memo is not trusted evidence and must not be used."],
         ["Do not use rejected evidence."],
         "The provided trusted evidence does not establish that; rejected evidence cannot support the answer.", "abstain"),
        ("revision", "Revise the draft about the bridge opening.",
         ["[docA] The bridge opens at 06:00 after inspection."],
         ["First draft claimed 05:00 without support.", "Correct the time using trusted evidence."],
         "The bridge opens at 06:00 after inspection.", "revise"),
    ]
    for k in range(n):
        name, user, evidence, plan, answer, behavior = templates[k % len(templates)]
        idx = k // len(templates)
        subject = ["Room Delta", "Vault Iris", "Gate Cedar", "Archive Blue"][idx % 4]
        if name == "insufficient":
            user = f"What is the access code for {subject} in case {idx}?"
            answer = f"The provided evidence does not establish the access code for {subject} in case {idx}."
        elif name == "conflict":
            permit = ["Q", "R", "S", "T"][idx % 4]
            user = f"Is Permit {permit} valid on Sunday?"
            evidence = [f"[docA] Permit {permit} is valid on Sunday for case {idx}.", f"[docB] Permit {permit} is not valid on Sunday for case {idx}."]
            answer = f"The available sources conflict, so I cannot state confidently whether Permit {permit} is valid on Sunday for case {idx}."
        elif name == "qualified":
            place = ["archive", "reading room", "lab annex", "records desk"][idx % 4]
            user = f"Can visitors enter the {place}?"
            evidence = [f"[docA] Visitors may enter the {place} with staff escort in case {idx}."]
            answer = f"Visitors may enter the {place} only with staff escort in case {idx}."
        elif name == "rejected":
            memo = ["memo A", "bulletin B", "notice C", "flyer D"][idx % 4]
            user = f"What does the rejected {memo} prove?"
            evidence = [f"[rejected] This {memo} for case {idx} is not trusted evidence and must not be used."]
            answer = f"The provided trusted evidence does not establish case {idx}; rejected evidence cannot support the answer."
        elif name == "revision":
            hour = 6 + (idx % 5)
            user = f"Revise the draft about bridge opening case {idx}."
            evidence = [f"[docA] The bridge opens at {hour:02d}:00 after inspection."]
            plan = [f"First draft claimed {hour - 1:02d}:00 without support.", "Correct the time using trusted evidence."]
            answer = f"The bridge opens at {hour:02d}:00 after inspection."
        rec = _rec(start + k, f"abstain_conflict_{name}", f"edge:{name}:{k // len(templates)}",
                   user, evidence, plan, answer, behavior)
        if name == "revision":
            rec["first_draft"] = plan[0].replace("First draft claimed", "The bridge opens at").replace(" without support.", ".")
            rec["verifier_feedback"] = {
                "unsupported_claims": [rec["first_draft"]],
                "missing_evidence": ["Use the trusted bridge opening evidence."],
                "passed": False,
            }
        rows.append(rec)
    return rows


def build_records(n_direct: int, n_grounded: int, n_edge: int) -> List[dict]:
    rows = []
    rows += _direct(0, n_direct)
    rows += _grounded(len(rows), n_grounded)
    rows += _abstain_conflict_revision(len(rows), n_edge)
    return rows


def _split_by_group(records: List[dict], seed: int):
    groups = defaultdict(list)
    for r in records:
        groups[r["source_group"]].append(r)
    names = sorted(groups)
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(len(names)))
    train_names, val_names, test_names = set(), set(), set()
    for rank, idx in enumerate(order):
        name = names[int(idx)]
        frac = rank / max(1, len(order))
        if frac < 0.8:
            train_names.add(name)
        elif frac < 0.9:
            val_names.add(name)
        else:
            test_names.add(name)
    splits = {"train": [], "validation": [], "test": []}
    for name, rows in groups.items():
        split = "train" if name in train_names else "validation" if name in val_names else "test"
        for r in rows:
            rr = dict(r)
            rr["split"] = split
            splits[split].append(rr)
    return splits


def _dedupe(records: Iterable[dict]):
    seen = set()
    out = []
    rejected = 0
    for r in records:
        key = (
            _sha(_norm(r["user_request"])),
            _sha(_norm(r["assistant_response"])),
            _sha(_norm("\n".join(r["trusted_evidence"]))),
            _sha(_render(r)),
        )
        if key in seen:
            rejected += 1
            continue
        seen.add(key)
        out.append(r)
    return out, rejected


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/sft_prompt_aligned")
    ap.add_argument("--direct", type=int, default=320)
    ap.add_argument("--grounded", type=int, default=320)
    ap.add_argument("--edge", type=int, default=160)
    ap.add_argument("--num-shards", type=int, default=4)
    ap.add_argument("--examples-per-shard", type=int, default=500)
    ap.add_argument("--seed", type=int, default=44)
    args = ap.parse_args()

    out = Path(args.out_dir)
    rows, rejected = _dedupe(build_records(args.direct, args.grounded, args.edge))
    splits = _split_by_group(rows, args.seed)
    rng = np.random.default_rng(args.seed)
    for key in splits:
        rng.shuffle(splits[key])

    train_total = args.num_shards * args.examples_per_shard
    weighted_train = list(itertools.islice(itertools.cycle(splits["train"]), train_total))
    train_dir = out / "train"
    shard_paths = []
    for s in range(args.num_shards):
        shard = weighted_train[s * args.examples_per_shard:(s + 1) * args.examples_per_shard]
        path = train_dir / f"shard_{s + 1:02d}.jsonl"
        _write_jsonl(path, shard)
        shard_paths.append(str(path))
    _write_jsonl(out / "validation.jsonl", splits["validation"])
    _write_jsonl(out / "test.jsonl", splits["test"])

    all_render_hashes = {split: [_sha(_render(r)) for r in recs] for split, recs in splits.items()}
    leakage = {
        "train_validation_render_overlap": len(set(all_render_hashes["train"]) & set(all_render_hashes["validation"])),
        "train_test_render_overlap": len(set(all_render_hashes["train"]) & set(all_render_hashes["test"])),
        "validation_test_render_overlap": len(set(all_render_hashes["validation"]) & set(all_render_hashes["test"])),
    }
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prompt_format": "cfna.prompting.v1",
        "seed": args.seed,
        "split_by": "source_group",
        "counts": {
            "unique_total": len(rows),
            "train_unique": len(splits["train"]),
            "train_weighted": len(weighted_train),
            "validation": len(splits["validation"]),
            "test": len(splits["test"]),
            "num_shards": args.num_shards,
            "examples_per_shard": args.examples_per_shard,
            "rejected_duplicate_records": rejected,
        },
        "category_counts": {
            split: dict(Counter(r["category"] for r in recs))
            for split, recs in {**splits, "train_weighted": weighted_train}.items()
        },
        "curriculum_proportions_unique": dict(Counter(
            "direct" if r["category"].startswith("direct_")
            else "grounded" if r["category"] == "grounded_evidence"
            else "abstain_conflict_revision"
            for r in rows
        )),
        "leakage": leakage,
        "hashes": {
            "normalized_user_request": sorted({_sha(_norm(r["user_request"])) for r in rows}),
            "normalized_assistant_response": sorted({_sha(_norm(r["assistant_response"])) for r in rows}),
            "decisive_evidence": sorted({_sha(_norm("\n".join(r["trusted_evidence"]))) for r in rows}),
            "rendered_example": sorted({_sha(_render(r)) for r in rows}),
        },
        "train_shards": shard_paths,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out), "counts": manifest["counts"], "leakage": leakage}, indent=2))


if __name__ == "__main__":
    main()
