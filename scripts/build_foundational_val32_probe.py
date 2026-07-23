import json
from collections import Counter, defaultdict
from pathlib import Path

SOURCE = Path(r"data\foundational_recovery_v2\val.jsonl")
OUTPUT = Path(r"data\foundational_val32\val.jsonl")

REQUIRED_DOMAINS = [
    "causal_temporal",
    "code_debugging",
    "conversation",
    "evidence_uncertainty",
    "factual_explanation",
    "mathematics",
    "planning_tools",
]

EXTRA_DOMAINS = [
    "conversation",
    "mathematics",
    "evidence_uncertainty",
    "factual_explanation",
]

rows = [
    json.loads(line)
    for line in SOURCE.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

by_domain = defaultdict(list)

for row in rows:
    domain = row.get("domain")
    if domain:
        by_domain[domain].append(row)

print("Source validation rows by domain:")

for domain in REQUIRED_DOMAINS:
    print(f"  {domain}: {len(by_domain[domain])}")

    if len(by_domain[domain]) < 4:
        raise SystemExit(
            f"ERROR: {domain!r} has only {len(by_domain[domain])} rows."
        )

def ordering_key(row):
    return (
        str(row.get("template_id", "")),
        str(row.get("example_hash", "")),
        json.dumps(row, sort_keys=True, ensure_ascii=False),
    )

selected = []
seen = set()

def add_rows(domain, amount):
    added = 0

    for row in sorted(by_domain[domain], key=ordering_key):
        signature = json.dumps(
            row,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        if signature in seen:
            continue

        selected.append(row)
        seen.add(signature)
        added += 1

        if added == amount:
            return

    raise RuntimeError(
        f"Could not select {amount} unique rows for domain {domain!r}."
    )

# Four examples from every domain: 7 × 4 = 28.
for domain in REQUIRED_DOMAINS:
    add_rows(domain, 4)

# One additional example from four domains: 28 + 4 = 32.
for domain in EXTRA_DOMAINS:
    add_rows(domain, 1)

if len(selected) != 32:
    raise RuntimeError(f"Expected 32 rows, produced {len(selected)}.")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

OUTPUT.write_text(
    "\n".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        for row in selected
    ) + "\n",
    encoding="utf-8",
)

counts = Counter(row["domain"] for row in selected)

print(f"\nWrote {len(selected)} rows to {OUTPUT}")
print("Selected rows by domain:")

for domain in REQUIRED_DOMAINS:
    print(f"  {domain}: {counts[domain]}")
