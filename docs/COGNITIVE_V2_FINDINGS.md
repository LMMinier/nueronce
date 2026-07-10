# Cognitive Suite V2 — Findings & Next-Step Decision

This is the write-up required after Steps 1–6. It ties together the committed
milestone, the randomized evaluation, the baselines, the failure taxonomy, and the
decision about the **first learned module**. It deliberately does **not** start any
neural-module training (that waits on this analysis).

## 1. Branch and provenance

- Branch: `research/cognitive-contract-v1` (not merged to `main`).
- Milestone commit: `dcc681f` — provenance-aware cognitive contract + V1 ablation.
- Honesty commit: `ed95834` — corrected overstated claims + status table.
- V2 commit: recorded in the branch history (this document's commit).

## 2. Full test result

`pytest` (whole repo): **111 passed, 0 skipped, 0 failed** in ~93 s.
Environment: Python 3.13.2, torch 2.11.0+cpu, numpy 2.2.3, Windows-10.

## 3. Artifacts

| Artifact | Path |
|---|---|
| V1 raw metrics (JSON) | `benchmarks/cognitive_v1.json` |
| V1 report (Markdown) | `docs/reports/COGNITIVE_V1_REPORT.md` |
| V2 raw metrics (JSON) | `benchmarks/cognitive_v2.json` |
| V2 report (Markdown) | `docs/reports/COGNITIVE_V2_REPORT.md` |

Reproduce V2:

```bash
python scripts/eval_cognitive_v2.py --seeds 1 2 3 4 5 --n 1050 --holdout 480 \
    --json benchmarks/cognitive_v2.json --md docs/reports/COGNITIVE_V2_REPORT.md
pytest tests/test_cognitive_suite_v2.py -q
```

## 4. V2 generator design

- **Deterministic randomized generator** (`nueronce/cognition_v2.py`): seeded RNG,
  15 scenario families, entities/attributes/values/timestamps/source order/
  authority/wording/distractor-count/effective-dates/malicious-phrasing all
  randomized. 1050 trials/seed × 5 seeds (5250 in-distribution trials).
- **Families:** HIGHER_AUTHORITY_CORRECTION, LOWER_AUTHORITY_POISON,
  EQUAL_AUTHORITY_TEMPORAL_UPDATE, FUTURE_EFFECTIVE_CORRECTION, EXPIRED_FACT,
  IRRELEVANT_HIGH_AUTHORITY, AUTHORITY_IMPERSONATION, PARAPHRASED_POISON,
  MULTIPLE_COMPETING_FACTS, UNCERTAIN_TRUSTED_SOURCE, SOURCE_REVOCATION,
  SCOPE_LIMITED_FACT, MISSING_EVIDENCE, CITATION_REQUIREMENT, CONFLICT_SURFACING.
- **Adversarial holdouts** (`nueronce/cognitive_holdouts.py`): 8 templates with attack
  surfaces the policy was *not* written against — paraphrased escalation, indirect
  injection, Unicode confusables, quoted/code-fenced/citation-embedded malicious
  text, contradictory trusted sources, misleading-but-true distractors. Separate
  seed stream from the dev suite. 480 trials/seed.
- **Contract extension** (`nueronce/contract.py`): `EvidenceItem` gained
  `effective_date`, `expiry_date`, `scope`, `revoked`, `revokes`, `raw_text`,
  `uncertain`, `is_working` — all optional, so V1 is unaffected.

## 5. Baseline matrix (5 seeds, mean)

`composite` = right value AND required citation/refusal/conflict.
`value-only` = right value, ignoring citation (the fair comparison to
citation-blind baselines).

| strategy | composite | value-only | source-sel | adversarial | poison rate |
|---|---|---|---|---|---|
| FULL_COGNITIVE_LOOP | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| NEWEST_FACT_WINS | 0.034 | 0.500 | 0.538 | 0.000 | 1.000 |
| HIGHEST_AUTHORITY_ONLY | 0.034 | 0.750 | 0.769 | 0.000 | 0.000 |
| KEYWORD_RULE_ENGINE | 0.034 | 0.583 | 0.615 | 0.000 | 0.750 |
| NO_AUTHORITY | 0.634 | 0.667 | 0.692 | 0.125 | 1.000 |
| NO_SUPERSESSION | 0.800 | 0.833 | 0.846 | 0.875 | 0.000 |
| NO_RETRIEVAL | 0.067 | 0.000 | 0.000 | 0.000 | 0.000 |
| NO_PLANNING | 0.200 | 1.000 | 1.000 | 0.125 | 0.000 |
| NO_VERIFICATION | 0.200 | 1.000 | 1.000 | 0.125 | 0.000 |
| RANDOM_CHOICE | 0.034 | 0.560 | 0.593 | 0.000 | 0.460 |

## 6. Failure taxonomy (summed over 5 seeds)

| strategy | total | dominant failure modes |
|---|---|---|
| FULL_COGNITIVE_LOOP | 0 | — |
| NO_AUTHORITY | 1923 | poisoned 1400, missed_conflict 350, missed_decline 173 |
| NO_SUPERSESSION | 1050 | stale 700, missed_conflict 350 |
| NO_RETRIEVAL | 4900 | false_decline 4200, stale 350, missed_conflict 350 |
| NO_PLANNING | 4200 | missing_citation 4200 |
| NO_VERIFICATION | 4200 | missing_citation 4200 |
| NEWEST_FACT_WINS | 5073 | missing_citation 2450, poisoned 1400, wrong_value 700 |
| HIGHEST_AUTHORITY_ONLY | 5073 | missing_citation 3500, stale 700, wrong_value 350 |
| KEYWORD_RULE_ENGINE | 5073 | missing_citation 2800, poisoned 1050, wrong_value 700 |

Each module's removal maps to a specific, sensible failure mode: authority ↔
poisoning, supersession ↔ stale/conflict, retrieval ↔ false-decline, planning &
verification ↔ missing-citation.

## 7. What V2 does and does not establish

**Establishes (given ground-truth provenance labels):**
- The orchestration policy is internally complete across 15 families and 5 seeds.
- Every mechanism is load-bearing: removing any one collapses ≥1 family.
- The policy resists text-based attacks (adversarial holdout 1.000) because
  authority is metadata-derived; the text-based KEYWORD baseline is poisoned on
  75% of poison trials and fully missed by paraphrase.
- On value accuracy alone it beats HIGHEST_AUTHORITY_ONLY (1.00 vs 0.75) and
  NEWEST_FACT_WINS (1.00 vs 0.50) — real gaps from temporal/scope/revocation.

**Does NOT establish:**
- Real-world correctness. FULL = 1.000 ± 0.000 reflects that the policy and gold
  labels were authored from the same rules (internal consistency, not external
  validity).
- That authority/temporal/scope/claims can be *inferred* — every item's authority
  is supplied as ground-truth metadata. That inference is untested and unlearned.
- That a decoder could not reproduce this behavior.

## 8. Updated honest status table

| Component | Status |
|---|---|
| Cognitive contract (typed inter-stage states) | REAL / HEURISTIC |
| Authority ranking (from metadata) | REAL / HEURISTIC |
| Temporal supersession / validity | REAL / HEURISTIC |
| Scope + revocation handling | REAL / HEURISTIC |
| Conflict surfacing | REAL / HEURISTIC |
| Planner / Verifier (citation chain) | REAL / HEURISTIC |
| Randomized falsification suite + baselines | REAL / TESTED |
| Authority/claim/temporal inference from raw text | NOT BUILT (learnable) |
| Neural population of the contract | NOT INTEGRATED |
| Language renderer (byte model) | REAL / TRAINABLE, weak checkpoint (~3.60 bpb) |
| End-to-end learned cognitive loop | NOT PROVEN |

## 9. Recommendation: the first learned module

The V2 result points to one conclusion. The deterministic policy is correct **only
because it is handed ground-truth authority metadata**, and it is robust to attacks
**only because it ignores raw text**. The single baseline that reasons from text
(KEYWORD) is exactly the one that gets poisoned. So the binding real-world gap —
and the highest-leverage first learnable component — is:

> **Source / authority classification: `(raw retrieved text + source features) → AuthorityLevel` (+ a trusted/untrusted gate).**

Rationale:
- Every downstream mechanism (supersession, conflict, citation) already works
  *given* the label; the label is the unjustified assumption.
- It directly attacks the poisoning failure mode that defeats the text-based
  baseline.
- It has a clean supervised objective and can be evaluated in isolation against the
  existing metadata labels (which become the training targets).

Sequence after that (each trained and evaluated **separately**, policy kept
deterministic so the system stays auditable):

1. Source/authority classification ← **start here**
2. Claim extraction: `raw text → (entity, attribute, value)`
3. Temporal-scope extraction: effective/expiry/jurisdiction from text
4. Contradiction detection (feeds conflict surfacing)

Keep authority *resolution* and supersession *policy* deterministic. That yields a
system where **neural perception populates an explicit, auditable cognitive
contract** — the original NUERONCE architecture — rather than one decoder learning
everything invisibly.

## 10. Explicitly NOT started

No Phase-4 neural objectives were begun. This document is the input to that
decision, not the decision's execution.
