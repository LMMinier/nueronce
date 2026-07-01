# Known Limitations — Cognitive Contract V2 Baseline

This document consolidates the honest limitations already recorded in
[`docs/COGNITIVE_V2_FINDINGS.md`](COGNITIVE_V2_FINDINGS.md) and
[`docs/COGNITIVE_CONTRACT.md`](COGNITIVE_CONTRACT.md). Read it alongside the
headline metrics in `benchmarks/cognitive_v2.json` — the numbers are strong, but
their scope is deliberately narrow.

## 1. FULL = 1.000 reflects policy ≡ spec (internal consistency, not external validity)

The FULL_COGNITIVE_LOOP composite/value accuracy of **1.000 ± 0.000** (and
adversarial-holdout accuracy 1.000) does **not** demonstrate real-world
correctness. The orchestration policy and the gold labels were authored from the
**same underlying rules**, so a perfect score measures **internal consistency** of
the policy against its own specification — not **external validity** against the
world. It establishes that the policy is internally complete across all 15
scenario families and 5 seeds, and that every mechanism is load-bearing (removing
any one module collapses ≥1 family), and nothing beyond that.

## 2. Authority labels are supplied as ground-truth metadata, not inferred from text

Every evidence item's **authority level, trusted/untrusted flag, effective/expiry
dates, and scope are provided as ground-truth metadata**. The policy is robust to
text-based attacks (adversarial holdout = 1.000) precisely because it reasons over
this metadata and **ignores the raw text** — the one baseline that reasons from
text (`KEYWORD_RULE_ENGINE`) is exactly the one that gets poisoned. Whether
authority, claims, temporal validity, or scope can be **inferred from raw text** is
**untested and unlearned** in this baseline. That inference is the binding
real-world gap.

## 3. No learned modules exist yet

The entire cognition engine is **deterministic and heuristic** — authority
ranking, supersession, scope/revocation handling, conflict surfacing, planning,
and verification are all **programmed rules, not learned behavior**. Status is
**REAL / HEURISTIC**: real, exercised structures and resolution logic, but nothing
is trained. No Phase-4 neural objectives have been started. The recommended first
learned module (source/authority classification from raw text) is identified in
`COGNITIVE_V2_FINDINGS.md` but **not built**.

## 4. The byte LM renderer is weak (~3.60 bits/byte, plateaued)

The repository also contains a separate byte-level language-model renderer, which
is **not used at all** by the cognitive evaluation. Its best checkpoint reached
~**3.604 bits/byte** on held-out data (11M-param config, ~900 steps, ~39 min CPU)
and **plateaued** there. This number applies only to that specific configuration;
it does not prove that corpus scaling or a tuned schedule would be ineffective —
only that this run plateaued. The renderer still needs optimization (deferred to a
later phase) and does not affect the heuristic V2 results.

## 5. Two systems, not yet connected

The repo holds **two separate systems that are not yet integrated**: (a) the
learned neural renderer (`corpus → training → byte generation`), and (b) this
programmed cognitive layer (`provenance → authority → supersession → plan →
verify`). Neural population of the cognitive contract is **NOT INTEGRATED**, and an
end-to-end learned cognitive loop is **NOT PROVEN**. Typed states improve
auditability and control; they do **not** establish that a decoder-only network
could not reproduce the same input/output behavior.

## 6. The branch is not merged to the default branch

This baseline lives on `research/cognitive-contract-v1` and is **intentionally NOT
merged** to the default branch. It is a research baseline frozen at commit
`d712c36` (tag `v0.2-cognitive-contract`), not a shipped result.
