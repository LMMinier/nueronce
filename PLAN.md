# PLAN: CFNA — from "English-shaped continuations" to a conversational model, then scale

**Mission.** Take CFNA from a model that produces English-shaped continuations
to one that reliably answers in turn-taking dialogue, then walk the parameter
ladder (11M → 35M → 90M → 337M) with a go/no-go gate at each rung.

**Ground truth first (no inflated claims).** The largest *trained* model is
the 11M chat checkpoint (plus the 2.04M demo — ~13M trained total, and the
112,709-param `micro_cfna_sft_100k` proof-of-loop at 0.906 teacher-forced byte
accuracy on its own format). The 337M config constructs and forwards but has
**random weights** — it has never been trained. Every claim below maps to a
file, test, or logged run; weak results are written down, not hidden
(`docs/reports/`, `docs/BREAKTHROUGH_MAP.md`).

This is a **live ledger**: each phase item is marked DONE / PARTIAL / TODO
against the repo as of 2026-07-02. Update marks as runs land.

---

## Phase 1 — Turn contract (make training and inference speak one format)

| Item | Status | Evidence |
|---|---|---|
| Canonical markers `<\|system\|>` `<\|user\|>` `<\|evidence\|>` `<\|plan\|>` `<\|assistant\|>` `<\|end\|>` | DONE | `cfna/prompting.py` |
| Stop sequences (`<\|end\|>`, `\n<\|user\|>`, `\nUser:`) wired into generation | DONE | `prompting.STOP_SEQUENCES`, used by chat + eval paths |
| Round-trip tests (format → extract → identical) | DONE | `tests/test_prompting.py` (5 tests) |
| Prompt-format drift guard: checkpoint stamping + legacy auto-detect | DONE | `sharded_sft.save_checkpoint` stamps `meta["prompt_format"]`; `MicroConversation.resolve_format`; `tests/test_chat_format_drift.py` (4 tests) |
| `docs/FORMAT.md` documenting the contract | DONE | `docs/FORMAT.md` (this commit) |

**Phase 1 is closed.** The one incident worth remembering: repointing the
shared tags silently cost a legacy checkpoint 38 points of byte accuracy
(0.906 → 0.524). The stamp + tests make that a hard failure now, not a
mystery.

## Phase 2 — SFT dataset in the canonical format

| Item | Status | Evidence |
|---|---|---|
| Canonical-prompt SFT builder | DONE | `scripts/build_prompt_aligned_sft.py`; `dialogue_data.encode_example`/`encode_messages` emit canonical layout + response-only masks |
| Synthetic dialogue generator (~127K unique) | DONE | `cfna/training/synthetic_dialogue.py` |
| MCQ/QA conversion (ARC, OpenBookQA, CommonsenseQA, MathQA, GSM8K) | DONE | `cfna/training/mcq_sft.py` (`load_and_convert`, `convert_records`) |
| Assembled ≥50K-record mixed SFT set, ≤25% per register | TODO (local GPU) | mix rule + poisoning lesson in `docs/CODEX_HANDOFF.md` §3; 800-record set is known-insufficient |

## Phase 3 — VGRFT (the SFT stage, implemented — not vaporware)

| Item | Status | Evidence |
|---|---|---|
| Stage-1 SFT (response-only masked CE), torch backend | DONE | `cfna/training/sft.py` |
| Stage-1 SFT, sharded/resumable + microtorch backend | DONE | `cfna/training/sharded_sft.py`; `tests/test_sft_microtorch.py` |
| Proof the loop works end-to-end | DONE | `checkpoints/micro_cfna_sft_100k` → 0.906 byte-acc; `docs/reports/MICRO_CFNA_SFT_100K_REPORT.md` |
| VGRFT stages 2–4 (verifier-guided revision, residual, continual) | TODO (by design) | `cfna/training/vgrft.py` raises `NotImplementedError` for un-injected backends; not needed until a base model clears the bpb gate |

Note: an earlier framing said VGRFT "currently raises NotImplementedError."
That is stale for stage 1 — SFT is implemented twice over. Only the later
verifier-guided stages remain scaffolding, deliberately, until there is a
base model worth revising.

## Phase 4 — Evaluation (measure the right thing)

| Item | Status | Evidence |
|---|---|---|
| Inference suite (valid / non-echo / stop-terminated) | DONE | `scripts/eval_inference_phase2.py` |
| Choice-ranking eval (masked answer loss per MCQ option, chance reported) | DONE | `mcq_sft.rank_choices` / `evaluate_mcq` |
| Teacher-forced byte accuracy as checkpoint-health metric | DONE | used to detect the format drift; always eval in the checkpoint's own format (`docs/FORMAT.md`) |
| Fresh eval numbers on current checkpoints, in `metrics/` | TODO (local GPU) | structure generalizes before content — do **not** discard checkpoints on generative exact-match alone (`docs/BREAKTHROUGH_MAP.md` §3.4) |

## Phase 5 — Scale (the only real remaining work is compute)

| Item | Status | Evidence |
|---|---|---|
| Parameter presets: chat_11m / base_35m (34.4M) / base_90m (92.1M) / large_337m | DONE | `cfna.model.CONFIG_PRESETS`; construct+forward verified |
| fp16/AMP-safe numerics (masked softmax, fp32 RMSNorm stats) | DONE | `cfna/nn.py`; `tests/test_gpu_amp.py` |
| Multi-subject corpus recipe (~400 MB, 22 sources) | DONE | `cfna/corpus/stack.py`; `docs/LOCAL_TRAINING_PLAYBOOK.md` §1 |
| base_35m pretrain to held-out bpb ≤ 1.5 | TODO (local GPU) | run order + gates in `docs/CODEX_HANDOFF.md` |
| SFT on ≥50K records, best-by-val checkpoint | TODO (local GPU) | gated on base bpb < 1.8 — SFT shapes competence, it cannot create it |
| base_90m rung | BLOCKED | unlocks only when 35M acceptance holds |
| 337M training | BLOCKED | two rungs away; anything sooner is theater |

## Acceptance criteria for "conversational model, 35M rung" (verbatim from the handoff)

- base_35m held-out bpb ≤ 1.5 on the multi-subject corpus
- choice-ranking accuracy beats chance by ≥15 points on ≥3 MCQ subjects
- phase2 inference suite: ≥60% valid, non-echo, stop-terminated answers
- 5/5 sample transcripts grammatical English addressing the question
  (fluency bar, not correctness bar)

## Division of labor

- **Local GPU session (Codex):** executes the TODO(local GPU) rows above, in
  the run order in `docs/CODEX_HANDOFF.md`; pushes `metrics/` + reports after
  every stage; architecture frozen.
- **Cloud session (Claude):** curve analysis, go/no-go per rung, code fixes
  flagged via `docs/reports/*`, keeps this ledger current.

## Standing rules

1. From scratch only — no `nn.Transformer`, no external model imports.
2. The verified neural core is frozen; all 70+ tests (especially causality)
   stay green.
3. Every doc claim maps to a test or a logged run; negative results get
   written down.
4. Never eval a checkpoint in a prompt format it wasn't trained on.
