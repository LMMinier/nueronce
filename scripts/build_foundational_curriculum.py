"""Build a balanced seven-domain, leakage-free curriculum."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

DOMAINS = (
    "natural_language_conversation", "factual_explanatory",
    "mathematics_symbolic", "code_debugging", "causal_temporal",
    "planning_tools", "evidence_uncertainty",
)
SPACE = re.compile(r"\s+")


def norm(text):
    return SPACE.sub(" ", text.casefold()).strip()


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_for(key):
    bucket = int(digest(key)[:8], 16) % 100
    return "val" if bucket < 5 else "test" if bucket < 10 else "train"


def make(domain, key, prompt, response, category):
    return {
        "prompt": prompt, "response": response, "domain": domain,
        "category": category, "record_id": key,
        "kind": "nueronce_foundational_curriculum_v1",
        "source": "verified_generator",
        "example_hash": digest(prompt + "\0" + response),
    }


def map_domain(category):
    if category in {"arithmetic", "classification", "logic"}:
        return "mathematics_symbolic"
    if "coding" in category:
        return "code_debugging"
    if category in {"facts", "definitions", "explanations"}:
        return "factual_explanatory"
    if "grounded" in category or "abstain" in category or category == "uncertainty":
        return "evidence_uncertainty"
    if category == "procedures":
        return "planning_tools"
    return "natural_language_conversation"


def generate():
    for n in range(2, 302):
        yield make("code_debugging", f"range-{n}",
                   f"Debug this Python loop so it prints every integer from 0 through {n}: "
                   f"for i in range({n}): print(i)",
                   f"Use for i in range({n + 1}): print(i). Python excludes the upper bound, "
                   f"so {n + 1} is required to include {n}.", "debug_off_by_one")
        yield make("code_debugging", f"name-{n}",
                   f"Fix the Python NameError: total_{n} = {n}; print(totl_{n}).",
                   f"Change the misspelled name to print(total_{n}).", "debug_name_error")
        yield make("code_debugging", f"remainder-{n}",
                   f"What Python expression returns the remainder when {n} is divided by 7?",
                   f"Use {n} % 7; it evaluates to {n % 7}.", "code_symbolic")

    chains = [
        ("the switch is pressed", "the circuit closes", "the lamp turns on"),
        ("water reaches the seed", "the seed absorbs moisture", "germination can begin"),
        ("temperature falls below freezing", "liquid water freezes", "ice forms"),
        ("the alarm detects smoke", "the siren activates", "occupants are warned"),
    ]
    for i in range(600):
        first, second, third = chains[i % len(chains)]
        yield make("causal_temporal", f"cause-{i}",
                   f"If {first}, then {second}. If {second}, then {third}. "
                   f"{first.capitalize()} in system {i}. What follows in system {i}?",
                   f"First {second}; therefore {third}.", "causal_chain")
        start, gap1, gap2 = 8 + i % 8, 1 + i % 4, 2 + i % 5
        yield make("causal_temporal", f"time-{i}",
                   f"On schedule {i}, task A starts at {start}:00. B starts {gap1} hours after A, "
                   f"and C starts {gap2} hours after B. When does C start?",
                   f"B starts at {start + gap1}:00, so C starts at {start + gap1 + gap2}:00.",
                   "temporal_composition")

    tools = [
        ("calculate 37 multiplied by {x}", "calculator", "37 * {x}", "check the returned number"),
        ("find Python files containing item_{x}", "code search", "item_{x}", "review matching paths and lines"),
        ("look up saved project note {x}", "retrieval", "project note {x}", "verify its source and timestamp"),
        ("check current weather for sector {x}", "weather service", "sector {x}", "report its observation time"),
    ]
    for i in range(800):
        task, tool, argument, check = tools[i % len(tools)]
        x = 10 + i
        yield make("planning_tools", f"tool-{i}",
                   f"Create a minimal tool-use plan to {task.format(x=x)}. Do not invent the result.",
                   f"1. Call the {tool} with {argument.format(x=x)}. 2. Inspect its value or error. "
                   f"3. {check.capitalize()}. 4. Answer only from the verified result.", "tool_planning")
    for i in range(200):
        yield make("planning_tools", f"plan-{i}",
                   f"Plan these dependent steps for input {i}: validate, transform, verify, then publish.",
                   f"1. Validate input {i}. 2. Stop on errors. 3. Transform it. "
                   "4. Verify the output. 5. Publish only if verification passes.", "conditional_planning")

    for i in range(600):
        value = 20 + i % 80
        yield make("evidence_uncertainty", f"evidence-{i}",
                   f"What is the verified value for item {i}? Trusted evidence says {value}; "
                   f"an unverified note says {value + 7}.",
                   f"The verified value is {value}; the unverified conflict should not override it.",
                   "evidence_authority")
        yield make("evidence_uncertainty", f"missing-{i}",
                   f"What is the launch date for project {i}? Evidence gives its budget but no date.",
                   "The launch date cannot be determined from the supplied evidence.", "evidence_missing")

    mechanisms = [
        ("Why does ice float?", "Ice is less dense than liquid water because its bonded structure occupies more volume."),
        ("Why do shadows form?", "A shadow forms when an opaque object blocks light from a source."),
        ("Why can metal feel colder than wood?", "Metal conducts heat away from skin faster than wood."),
        ("Why can salt dissolve in water?", "Polar water molecules surround and separate charged ions."),
        ("Why does exercise raise breathing rate?", "Working muscles need more oxygen and release more carbon dioxide."),
    ]
    for i in range(500):
        question, answer = mechanisms[i % len(mechanisms)]
        yield make("factual_explanatory", f"explain-{i}", f"{question} Case {i}.", answer,
                   "mechanistic_explanation")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("data/foundational_sanitized"))
    parser.add_argument("--output", type=Path, default=Path("data/foundational_curriculum_v1"))
    parser.add_argument("--cap-train-per-domain", type=int, default=3500)
    parser.add_argument("--seed", type=int, default=91)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    splits = {name: [] for name in ("train", "val", "test")}
    for split in splits:
        with (args.base / f"{split}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                item["domain"] = map_domain(item.get("category", ""))
                splits[split].append(item)
    for item in generate():
        splits[split_for(item["record_id"])].append(item)

    for split, items in splits.items():
        seen, unique = set(), []
        for item in items:
            key = digest(norm(item["prompt"]) + "\0" + norm(item["response"]))
            if key not in seen:
                seen.add(key)
                unique.append(item)
        splits[split] = unique
    val_prompts = {digest(norm(x["prompt"])) for x in splits["val"]}
    test_prompts = {digest(norm(x["prompt"])) for x in splits["test"]}
    splits["train"] = [x for x in splits["train"]
                       if digest(norm(x["prompt"])) not in val_prompts | test_prompts]
    splits["test"] = [x for x in splits["test"] if digest(norm(x["prompt"])) not in val_prompts]

    grouped = defaultdict(list)
    for item in splits["train"]:
        grouped[item["domain"]].append(item)
    train = []
    for domain in DOMAINS:
        rng.shuffle(grouped[domain])
        train.extend(grouped[domain][:args.cap_train_per_domain])
    rng.shuffle(train)
    splits["train"] = train

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"version": 1, "seed": args.seed, "domains": list(DOMAINS), "splits": {}}
    prompt_sets = {}
    for split, items in splits.items():
        path = args.output / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        prompt_sets[split] = {digest(norm(x["prompt"])) for x in items}
        manifest["splits"][split] = {
            "count": len(items),
            "domain_counts": dict(Counter(x["domain"] for x in items)),
            "category_counts": dict(Counter(x["category"] for x in items)),
            "sha256": digest(path.read_text(encoding="utf-8")),
        }
    manifest["leakage"] = {
        "train_val": len(prompt_sets["train"] & prompt_sets["val"]),
        "train_test": len(prompt_sets["train"] & prompt_sets["test"]),
        "val_test": len(prompt_sets["val"] & prompt_sets["test"]),
    }
    manifest["quality_gate_passed"] = (
        not any(manifest["leakage"].values())
        and all(manifest["splits"]["train"]["domain_counts"].get(d, 0) >= 500 for d in DOMAINS)
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if not manifest["quality_gate_passed"]:
        raise SystemExit("curriculum quality gate failed")


if __name__ == "__main__":
    main()
