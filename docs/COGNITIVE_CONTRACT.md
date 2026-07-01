# CFNA Cognitive Contract & First Milestone

This document records the corrected direction for CFNA: **prove the cognitive
architecture first**, and treat the byte language model as one *renderer*
component — not the spine. It corresponds to Phase 1 (freeze the cognitive
contract) and the first milestone of the plan in
[`CFNA_concurrent_plan.txt`](CFNA_concurrent_plan.txt).

Subsystem label (honesty rule): **REAL / HEURISTIC** — real, exercised structures
and resolution logic; **not learned**.

## Scope and honest limits (read first)

This milestone is a **unit-level architecture proof, not a research result**. Be
precise about what it does and does not show:

- The cognition engine is **deterministic and heuristic**. Authority ranking and
  supersession are **programmed rules**, not learned behavior.
- It **does not use the trained language model** at all. The answer is decided by
  typed reasoning over provenance, not by the byte model.
- It proves **modular policy behavior** — that a provenance-aware control layer
  behaves correctly and degrades predictably under ablation. It does **not** prove
  *learned* cognition.
- Typed states improve **auditability and control**. They do **not** establish that
  a decoder-only network could not reproduce the same input/output behavior; a
  decoder could emit equivalent structures or imitate the result. The contract
  makes the system modular and inspectable, not provably irreducible.
- The repository currently holds **two separate systems** that are **not yet
  connected**: a learned neural renderer (`corpus -> training -> byte generation`)
  and this programmed cognitive layer (`provenance -> authority -> supersession ->
  plan -> verify`). The research goal is to connect them; this milestone does not.

## The strongest honest claim

> CFNA now has a deterministic, provenance-aware cognitive orchestration layer
> whose authority, retrieval, supersession, planning, and verification components
> are behaviorally testable through controlled ablations.

## The claim being probed (not yet established)

> Does separating perception, meaning, intent, memory, retrieval, reasoning,
> planning, communication, verification, and revision produce behavior that a
> single continuation network does not?

This four-scenario suite is only a first, small probe of that question. The test
uses **no language-model training**, which isolates the *cognitive structure* from
LM fluency — but four hand-authored scenarios are not enough to answer the
question. Cognitive Suite V2 (randomized trials, competing baselines, adversarial
holdouts) is the real test.

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

Reproduce (deterministic; `--seed` is recorded but the V1 loop has no randomness):

```bash
python scripts/eval_cognitive.py --seed 0 --json benchmarks/cognitive_v1.json --md docs/reports/COGNITIVE_V1_REPORT.md
pytest tests/test_cognitive_authority.py -q
```

## Honest component status

| Component | Status |
|---|---|
| Cognitive contract (typed inter-stage states) | REAL / HEURISTIC |
| Authority ranking | REAL / HEURISTIC |
| Temporal supersession | REAL / HEURISTIC |
| Planner | REAL / HEURISTIC |
| Verifier | REAL / HEURISTIC |
| Neural semantic population of states | NOT INTEGRATED |
| Learned authority reasoning | NOT PROVEN |
| Learned supersession | NOT PROVEN |
| Language renderer (byte model) | REAL / TRAINABLE, weak checkpoint |
| End-to-end learned cognitive loop | NOT PROVEN |

## Note on the language renderer

A separate 40-minute CPU run on an expanded 30 MB public-domain corpus drove
held-out bits/byte from ~8.0 to **~3.604** (900 steps, 39 min), where it
plateaued. **This number applies only to the tested configuration** (11M-param
config, constant learning rate, this corpus/optimizer, 900 steps). It does **not**
show that corpus scaling, a tuned learning-rate schedule, or longer optimized
training is ineffective — only that *this* run plateaued. The renderer still needs
optimization; that work is deferred to Phase 7, and it does not affect this
heuristic milestone.

## Next (per corrected build order)

1. **Cognitive Suite V2**: replace the four fixed scenarios with a deterministic
   randomized generator (15 scenario families, >=1000 trials/seed x >=5 seeds),
   competing baselines, and adversarial holdouts.
2. Use V2 failures to decide **which** deterministic module should become
   learnable first — do **not** start Phase 4 neural objectives before then.
3. Only after that: give surviving modules measurable objectives, then integrate a
   trained renderer, then optimize CPU streaming/conditional execution.
