# CFNA Cognitive Contract & First Milestone

This document records the corrected direction for CFNA: **prove the cognitive
architecture first**, and treat the byte language model as one *renderer*
component — not the spine. It corresponds to Phase 1 (freeze the cognitive
contract) and the first milestone of the plan in
[`CFNA_concurrent_plan.txt`](CFNA_concurrent_plan.txt).

Subsystem label (honesty rule): **REAL / HEURISTIC** — real, exercised structures
and resolution logic; not learned.

## The claim under test

> Does separating perception, meaning, intent, memory, retrieval, reasoning,
> planning, communication, verification, and revision produce behavior that a
> single continuation network does not?

The test deliberately uses **no language-model training**, so the result is about
the *cognitive structure*, not about how fluent the byte model is.

## Phase 1 — the frozen contract (`cfna/contract.py`)

Every stage consumes and produces a named, provenance-carrying dataclass, so the
system's decision is auditable from typed intermediate state rather than an opaque
forward pass:

| Stage | Type | Carries |
|---|---|---|
| perceive | `PerceptionState` | normalized surface, segments, byte length |
| represent meaning | `SemanticState` / `Claim` | (entity, attribute, value) claims |
| determine intent | `IntentState` | task type, target, citation requirement |
| query memory | `MemoryQuery` | entity/attribute/keywords, min authority |
| retrieve | `EvidenceItem` / `EvidenceSet` | value **+ source, authority, timestamp, content hash, trusted flag** |
| reason (resolve) | `ReasoningState` | resolved value, winning evidence, **superseded**, **rejected untrusted**, rationale |
| plan | `Plan` | ordered steps, `must_cite` |
| communicate | `Draft` | text, claims, citations |
| verify | `VerificationReport` | pass/fail + typed failures |
| revise | `Revision` | before/after, reason |
| (whole pass) | `CognitiveTrace` | every stage above, logged |

## First milestone — modular necessity by ablation

`cfna/cognition.py` runs the loop; `cfna/cognitive_suite.py` defines controlled
tasks; `scripts/eval_cognitive.py` runs FULL vs each single-stage ablation.

The canonical task (`authority_overwrite`) is the plan's milestone: a user states
a fact, a **verified** source later corrects it, and an **untrusted** page tries
to overwrite it. The system must answer with the trusted, updated fact and cite
its source.

### Result

```
config                score   solves
FULL                   4/4    all four scenarios
no_retrieval           1/4    (misses trusted corrections)
no_authority           2/4    (poison document wins)
no_supersession        2/4    (keeps the stale fact)
no_planning            1/4    (drops the citation requirement)
no_verification        1/4    (never enforces citations)

MILESTONE: PASS - full loop strictly beats every ablation
```

FULL answer to the milestone task:

```
The capital of Zedland is Belport. (source: verified_primary_source:gov_gazette)
```

Each ablation fails **exactly** the tasks that need the removed module:
- remove **authority** -> the untrusted "Xtown" overwrites the verified fact;
- remove **supersession** -> the loop is stuck on the older belief;
- remove **retrieval** -> it never sees the correction;
- remove **planning** or **verification** -> the required citation is never enforced.

Reproduce:

```bash
python scripts/eval_cognitive.py --json benchmarks/cognitive_ablation.json
pytest tests/test_cognitive_authority.py -q
```

## Why this comes before corpus/LM scaling

A separate 40-minute CPU run on an expanded 30 MB public-domain corpus drove
held-out bits/byte from ~8.0 to a **plateau at ~3.60** (900 steps, 39 min) — still far from coherent
words. That is a language-*renderer* limitation, and per the corrected build order
it is deferred to Phase 7. It does **not** affect this milestone, which is exactly
the point of isolating the architecture from LM fluency.

## Next (per corrected build order)

1. Extend the controlled suite (goal preservation across turns; procedural memory
   reuse; instruction-vs-retrieved-content separation).
2. Give each module its own measurable objective (Phase 4) before integrating a
   trained renderer (Phase 3/5).
3. Only then optimize CPU streaming/conditional execution (Phase 6) and broaden
   the corpus (Phase 7).
