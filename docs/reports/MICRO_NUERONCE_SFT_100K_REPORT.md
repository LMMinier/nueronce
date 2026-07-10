# MicroNUERONCE 100K-Conversation SFT Run — Report

**Status: completed.** All ten shards trained continuously (no reset), best
checkpoint selected by validation, full test-set evaluation run, and a
311-prompt novel-generalization suite run against the best checkpoint. This
document reports exactly what happened, including the parts that didn't work.

## 1. What this run is (and isn't)

This replaces the earlier 61-example SFT demo with a real, ~100,000-conversation
run on the **same, unenlarged** `NueronceModel` architecture (112,709 params —
identical config to the 61-example run), entirely on the from-scratch
`nueronce/engine` engine (no PyTorch anywhere in the training path). The
purpose was narrow and explicit: **does substantially more data produce
generalization on this architecture, holding architecture size fixed?**

The headline answer: **partially, and unevenly.** Byte-level next-token
accuracy and structural fluency improved dramatically. Genuine arithmetic
computation did not generalize past memorized examples — not even to *other
memorized* examples of the same skill, as detailed in §7. Read the whole
report before taking either "it works" or "it doesn't" as the takeaway.

## 2. Dataset

### 2.1 Provenance and license

100% programmatically generated: self-authored templates x enumerated/randomized
parameters (`nueronce/training/synthetic_dialogue.py`), not a scraped or third-party
corpus. No external license to track — every template, entity table, and
generation range is in that file. This trades organic linguistic diversity for
zero licensing risk and perfect reproducibility (the whole 126,925-record
corpus regenerates byte-for-byte from a fixed seed).

### 2.2 Cleaned counts (exactly what was used, not what was attempted)

| Stage | Count |
|---|---|
| Raw generated records | 126,950 |
| Rejected: invalid schema | 0 |
| Rejected: exact duplicate | 23 |
| Rejected: near duplicate (normalized text) | 2 |
| Rejected: duplicate prompt+response pair | 0 |
| **Accepted (clean)** | **126,925** |
| Train (10 shards x 10,000) | **100,000** |
| Validation (fixed, held out) | **5,000** |
| Test (fixed, held out) | **5,000** |

Split is a deterministic seeded shuffle (seed 42) over byte offsets into the
clean file — validation and test are carved out *first*, so every train shard
is guaranteed disjoint from both. `assert_no_leakage` (streaming, hash-based)
was run and passed with zero overlaps. See `data/sft_100k/manifest.json` for
the full manifest and `nueronce/training/dataset_prep.py` for the pipeline.

### 2.3 Category distribution — measured, not assumed

The dataset is **heavily skewed toward arithmetic and numeric classification**.
This is a direct, honest consequence of how it was built: those two skills
scale combinatorially (any operand pair is a new, legitimate example) while
categories like `explanations` or `procedures` are bounded by how many
distinct scenarios were hand-authored. The task's own example ("arithmetic
with different operands is genuine variety") is exactly why arithmetic
dominates — but that dominance turns out to matter a great deal for the
results in §7.

| Category | Train | Validation | Test |
|---|---:|---:|---:|
| arithmetic | 77,467 | 3,908 | 3,909 |
| classification | 16,730 | 830 | 815 |
| logic | 3,615 | 150 | 167 |
| coding | 387 | 14 | 14 |
| instruction_following | 792 | 44 | 42 |
| facts | 385 | 21 | 21 |
| definitions | 187 | 8 | 8 |
| rewriting | 103 | 5 | 8 |
| multiturn | 103 | 3 | 2 |
| uncertainty | 52 | 3 | — |
| greetings | 70 | 6 | 1 |
| procedures | 35 | 1 | 1 |
| summarization | 33 | 4 | 4 |
| refusals | 23 | 2 | 4 |
| explanations | 18 | 1 | 4 |

77.5% of training examples are arithmetic; 94% are arithmetic or numeric
classification combined. Every other listed category (greetings, facts,
definitions, instruction-following, rewriting, summarization, classification,
logic, coding, procedures, multi-turn, refusals, uncertainty) is present, but
several have only a few dozen to a few hundred examples — see §7.4 for what
that did to generalization on those skills specifically.

## 3. Model and architecture

Unchanged from the 61-example run, per the experiment's own constraint (test
data scale, not architecture scale):

```
NueronceConfig(byte_embed_dim=16, d_local=24, d_model=32, p_max=16,
                  physical_blocks=1, logical_depth=2, n_heads=4,
                  unit_window=12, decoder_window=16, decoder_layers=1,
                  d_state=8, channel_dim=8, ret_byte_dim=8,
                  min_patch=2, max_patch=14)
```

- **112,709 parameters** — identical to the 61-example run.
- **vocab_size = 256** — raw bytes in, raw bytes out. Never touched.
- **Context length: 288 bytes.** The task allows up to 512; 288 was chosen
  because it is the smallest power-of-convenience ceiling that covers the
  *entire* dataset's encoded length distribution with **zero truncation**
  (measured max encoded conversation length: 264 bytes, p99.9: 197 bytes).
  This is a context-length choice, not a vocabulary change.
- Full architecture used: `BytePerceptionEncoder` (causal byte CNN + boundary
  head) → dynamic patching (`nueronce/engine/segment.py`) → `UnitEmbedder` →
  `TypedRecurrentMemory` → `HybridCoreStack` (`NUERONCESelectiveSSM` + local
  attention + sparse global attention + retrieval cross-attention, router-
  merged) → `ByteDecoder`. Retrieval path exists and is wired (see
  `tests/test_nueronce_engine_nueronce_model.py::test_retrieval_path_only_gets_gradient_when_used`)
  but this SFT run passed no retrieval context — every conversation is
  self-contained, so there was nothing to retrieve.
- Loss: `masked_token_loss`, response-bytes-only (including the terminal stop
  byte), computed via `nueronce.training.dialogue_data.encode_messages`. Verified
  by `tests/test_sharded_sft.py::test_prompt_bytes_contribute_zero_gradient_response_bytes_do`.

## 4. Training

### 4.1 Command

```bash
python scripts/train_sft.py --backend engine --model full-nueronce \
  --train-dir data/sft_100k/train_shards \
  --validation data/sft_100k/validation.jsonl \
  --test data/sft_100k/test.jsonl \
  --num-shards 10 --examples-per-shard 10000 \
  --save-dir checkpoints/micro_nueronce_sft_100k --metrics-dir metrics \
  --batch 32 --lr 2e-3 --lr-decay-factor 0.7 --min-lr 1e-4 --grad-clip 1.0 \
  --periodic-val-every 200 --periodic-val-examples 256 \
  --checkpoint-every-steps 500 --log-every 100 --seed 42 --resume
```

### 4.2 Hardware / environment

Linux container, x86_64, 4 vCPUs, Python 3.11.15, NumPy only (no GPU, no
PyTorch installed) — this run is a genuine test of the Nueronce Engine's
practicality at 100K-conversation scale on CPU alone.

### 4.3 Continuity, interruption, and resume (this actually happened, not a drill)

The run was **not** a single uninterrupted process. The sandbox environment
restarted partway through shard 4, killing the background training process.
This is exactly the scenario the resumable design exists for:

1. First launch trained shards 1-4 (through examples_seen=39,936), checkpointing
   `latest.pt`/`best.pt` at shard boundaries and every 500 steps.
2. The process died (verified via `ps`; no error in the log — an external
   restart, not a training bug).
3. Re-launched with `--resume`: it loaded `latest.pt`, reported
   `resumed from checkpoints/.../latest.pt: shard 4, step_within_shard 0,
   examples_seen 39,936`, and continued shard 5 onward with the *same*
   optimizer momentum/variance state (Adam `m`/`v`/`t`), not a fresh optimizer.
4. Shards 5-10 completed in the resumed process without further interruption.

This is reflected in `metrics/shard_metrics.jsonl` (two `pre_training`-style
resume log lines are visible where the second process started) and is exactly
what `tests/test_sharded_sft.py::test_resume_continues_from_correct_shard_and_step_with_continuous_state`
verifies in miniature.

### 4.4 Shard-by-shard results

Total wall-clock training time: **1,558 seconds (~26 minutes)** across both
runs combined (149s pre-training validation + shard time; the interruption
gap itself doesn't count toward "elapsed_seconds", which only accumulates
during active training).

| Shard | Examples Seen | Train Loss | Validation Loss | Bits/Byte | Byte Acc | Best? |
|---|---:|---:|---:|---:|---:|:---:|
| (pre-training) | 0 | — | 5.8571 | 8.4499 | 0.6% | |
| 1 | 9,984 | 0.7295 | 0.7139 | 1.0299 | 77.4% | ✓ |
| 2 | 19,968 | 0.5307 | 0.5175 | 0.7466 | 82.7% | ✓ |
| 3 | 29,952 | 0.4479 | 0.4639 | 0.6692 | 84.2% | ✓ |
| 4 | 39,936 | 0.4237 | 0.4293 | 0.6194 | 85.5% | ✓ |
| 5 | 49,920 | 0.3677 | 0.3909 | 0.5639 | 86.6% | ✓ |
| 6 | 59,904 | 0.3697 | 0.4001 | 0.5773 | 86.7% | ✗ (LR decayed 2e-3→1.4e-3) |
| 7 | 69,888 | 0.3373 | 0.3460 | 0.4992 | 88.2% | ✓ |
| 8 | 79,872 | 0.3441 | 0.3390 | 0.4891 | 88.4% | ✓ |
| 9 | 89,856 | 0.3272 | 0.3285 | 0.4739 | 88.7% | ✓ |
| 10 | 99,840 | 0.2834 | 0.3145 | 0.4537 | **89.1%** | ✓ |

**Best checkpoint: shard 10** (`checkpoints/micro_nueronce_sft_100k/best.pt`),
validation loss 0.3145. Note 99,840 examples were actually processed, not
100,000 — `10,000 // batch_size(32) = 312` steps per shard, so 320 examples
per shard round down and are not visited; an honest floor-division artifact,
not a missing-data bug (0.16% of the train set).

### 4.5 Final evaluation

| Split | Loss | Bits/Byte | Byte Accuracy |
|---|---:|---:|---:|
| Validation (final, full 5,000) | 0.3145 | 0.4537 | 89.1% |
| **Test (5,000, held out, best checkpoint)** | **0.3155** | **0.4552** | **89.06%** |

Test tracks validation closely (0.3155 vs 0.3145) — no sign of overfitting to
the validation set itself from repeated checking.

### 4.6 Comparison to the 61-example run

| | 61-example run | 100,000-example run |
|---|---:|---:|
| Training examples | 61 (52 train / 9 held-out) | 100,000 (10 shards) |
| Model | Same architecture, 112,709 params | Same architecture, 112,709 params |
| Held-out loss trend | **Worsened** with more steps (2.5→4.5 on 9 examples) — textbook overfitting on a tiny set | **Improved monotonically** apart from one shard (0.71→0.31), stable generalization curve |
| Held-out byte accuracy | Not measured this way; qualitative replies to held-out prompts were garbled | **89.1%** on a genuine 5,000-example held-out set |
| Exact-match on seen prompts | "Hello"/"two plus two" reproduced correctly, most others garbled | See §7: seen-prompt structure is excellent, seen-prompt arithmetic *content* is not |

**More data unambiguously fixed the overfitting problem** that plagued the
61-example run — the validation curve here is well-behaved across all 10
shards instead of diverging after step ~100. That is a real, measured
improvement, not a training-loss artifact. What it did *not* fix is discussed
next.

## 5. Tests

New test files added for this run: `tests/test_dataset_prep.py` (12 tests),
`tests/test_sharded_sft.py` (9 tests), `tests/test_generalization_eval.py` (5
tests). Combined with the pre-existing `tests/test_nueronce_engine_nueronce_model.py`
and `tests/test_sft_engine.py`, all 16 required categories are covered:

| # | Requirement | Test |
|---|---|---|
| 1 | JSONL parsing | `test_dataset_prep.py::test_records_round_trip_through_jsonl` |
| 2 | Invalid-record rejection | `test_invalid_records_are_rejected`, `test_build_clean_dataset_drops_invalid_and_counts_reasons` |
| 3 | Exact deduplication | `test_exact_duplicates_are_removed` |
| 4 | No split leakage | `test_no_leakage_between_train_val_test`, `test_leakage_check_actually_detects_overlap` |
| 5 | Ten-shard creation | `test_shard_creation_produces_exact_requested_shards_and_sizes` |
| 6 | Deterministic shuffling | `test_split_is_deterministic_given_the_same_seed`, `test_different_seeds_give_different_shuffles` |
| 7 | Response-only loss masking | `test_sharded_sft.py::test_prompt_bytes_contribute_zero_gradient_response_bytes_do` |
| 8 | Stop-byte targets | `test_encode_messages_masks_the_trailing_stop_byte` |
| 9 | Checkpoint save/resume | `test_checkpoint_roundtrip_restores_weights_and_optimizer_state` |
| 10 | Resume from correct shard | `test_resume_continues_from_correct_shard_and_step_with_continuous_state` |
| 11 | Continuous optimizer state | (same test — asserts identical total steps/examples vs an uninterrupted run) |
| 12 | Validation-based best-checkpoint | `test_best_checkpoint_is_selected_by_validation_not_last_shard` |
| 13 | Small end-to-end engine SFT learning | `test_sft_engine.py::test_micro_dialogue_sft_loss_decreases_on_a_toy_set` |
| 14 | Full NueronceModel forward/backward | `test_nueronce_engine_nueronce_model.py::test_forward_and_loss_shapes`, `test_model_learns_on_toy_corpus` |
| 15 | Generation termination | `test_sharded_sft.py::test_generation_terminates_at_stop_byte`, `test_generation_respects_min_new_before_checking_stop_bytes` |
| 16 | Retrieval behavior | `test_nueronce_engine_nueronce_model.py::test_retrieval_path_only_gets_gradient_when_used` |

**Full suite result: 10 pre-existing failures (all `ModuleNotFoundError: No
module named 'torch'` — torch is not installed in this sandbox; identical
failures exist on a clean checkout before any of this work), 131 passed, 9
skipped (torch-gated tests).** No new failures were introduced. Nothing was
skipped to make this run look better than it is.

## 6. Generalization evaluation

`scripts/eval_generalization.py` against `checkpoints/micro_nueronce_sft_100k/best.pt`:
100 conversations sampled verbatim from training (memorization probes, gold
answers known) + 311 hand-authored novel prompts
(`nueronce/training/generalization_eval.py::build_novel_prompts`) — new operand
values (parity-guaranteed outside the training grids, or above the trained
numeric ranges), new countries/elements/words not in any training table, and
new phrasings. **Novelty is verified, not assumed**: every prompt is checked
against a hash index of every user turn actually seen in the 100,000 training
conversations. That check caught 3 of the 311 "novel" prompts as accidental
duplicates of training data (the word "keyboard" was reused in both the
training word list and my hand-written novel-prompt list) — they were
reclassified as memorized before scoring, exactly the kind of self-check this
report is trying to model.

- 100 declared memorized probes + 3 reclassified = **103 truly memorized**
- **308 truly novel** prompts

| | n | Valid UTF-8 | Stopped at stop byte | "Coherent" (heuristic) | Check-pass rate |
|---|---:|---:|---:|---:|---:|
| Overall | 411 | 100% | 94.2% | 99.5% | 10.7% |
| Memorized | 103 | 100% | 99.0% | 100% | 9.8% |
| Novel | 308 | 100% | 92.5% | 99.4% | 11.0% |

**The "coherent" heuristic is a byte-sanity check, not a correctness or
sense-making judge** — it only verifies non-trivial length, ≥90% printable
ASCII, and no single byte dominating the output. Section 7.4 shows examples
that pass this heuristic while being nonsense; do not read 99.5% as "99.5% of
answers make sense."

## 7. Memorization vs. generalization — the actual findings

This is the part of the report that matters most, and it does not point one
direction.

### 7.1 Turn-taking and structural fluency: real, and it generalizes

Every response (411/411) is valid UTF-8, uses the `User:`/`Assistant:` turn
structure correctly, and follows the trained response *template* for its
category almost every time — including on prompts the hash check confirms
were never in training. `"What is 99 plus 475?"` reliably produces something
of the shape `"<number> <operation-word> <number> equals <number>."`, not
random bytes. This is genuine generalization of *response structure*, and the
100,000-example run clearly deepened it over the 61-example run's often
garbled held-out replies.

### 7.2 Arithmetic computation: memorized structure, not learned procedure

This is the central, sobering finding, and it required checking beyond the
naive substring-match metric to see clearly:

- **On novel operand pairs** (values chosen to be outside the training
  generation grids): **2.1%** check-pass rate (6/286).
- **On the exact training pairs used as memorization probes**: **4.7%**
  check-pass rate (4/86) — the model gets arithmetic *examples it was
  literally trained on* right less than 5% of the time.

Representative memorized-probe failures (prompt was in training verbatim):

```
Calculate 36 * 52.        gold: 36 times 52 equals 1872.    model: 36 times 52 equals 1180.
Divide 583 by 53.         gold: 583 divided by 53 equals 11. model: 475 divided by 53 equals 15.
Add 200 and 116.          gold: 200 plus 116 equals 316.     model: 200 plus 192 equals 316.
```

The template ("times", "equals", operand echoing) is essentially perfect; the
*computed number* is essentially never right, even when the exact question
was trained on. At 112K parameters and ~26 minutes of training, the model
learned "what an arithmetic answer looks like" far faster and more reliably
than it learned (or memorized) the actual arithmetic table — arithmetic being
78% of the training set did not translate into arithmetic being 78% learned.
This is not a generalization failure specific to novel numbers; it's a
capacity/training-depth ceiling on the arithmetic *content* itself.

### 7.3 A metric-design lesson, reported against ourselves

The naive `check_passed` for classification (even/odd/prime) showed an
apparently strong **83.3%** pass rate on genuinely novel numbers — until a
stricter check (does the response echo *the same number it was asked about*,
not just *a* plausible-sounding label) was applied to the even/odd subset:

```
Is 4001 even or odd?   model: "4010 is odd."     (label word "odd" matches -> naive PASS; wrong number -> real FAIL)
Is 6104 even or odd?   model: "6140 is even."    (digits transposed; naive PASS; real FAIL)
Is 5017 even or odd?   model: "5017 is odd."     (fully correct)
```

Only **1 of 24** even/odd responses to genuinely novel 4-digit numbers got
*both* the number and the label right. The apparent 83% is mostly an artifact
of a 2-way label space (even/odd) plus a model that reliably reproduces
*some* nearby number with *a* plausible label — not evidence that parity
reasoning generalized to unseen numbers. **Reporting the naive number alone
would have been describing memorization/pattern-matching as generalization,
which this report is explicitly instructed not to do** — so both numbers are
given here, and the stricter one is the one that should be believed.

### 7.4 Underrepresented categories: mostly incoherent on novel prompts

Categories with only tens to low-hundreds of training examples (facts,
definitions, logic with new entities, uncertainty, refusals, procedures,
rewriting, summarization) produced visibly garbled output on novel prompts —
passing the crude byte-sanity "coherent" heuristic while being nonsense to a
human reader:

```
"What is the capital of Nepal?"     -> "The capit of of of or a or ous t a t pa a A"
"What does cheerful mean?"          -> "catis is o o o t pa pa a a a a be"
"What are the steps to make toast?" -> "The capital o o o o o a tpa a tp o o o a ime o..."
```

Note the last one drifting into "The capital of..." — a fragment from the
*facts* category bleeding into a *procedures* prompt. With only 35-488
training examples per one of these categories (vs. 77,467 for arithmetic),
there is not enough signal for the model to reliably route to the right
response family for a truly novel prompt in that category, let alone answer
it. This is a direct, measurable consequence of the category imbalance
documented in §2.3, not a separate mystery.

Some novel prompts in these categories, by design, test **recall of specific
facts the model never saw** (e.g., Nepal's capital) rather than a learned
procedure — no amount of "generalization" should be expected to produce a
byte sequence the model never encountered. That's a fair test of "did the
model memorize world facts," not of "did the model learn arithmetic," and the
answer for both is "no," for related but distinct reasons.

## 8. Failures and limitations

- **The background training process was killed once by a sandbox restart**
  mid-shard-4 and had to be resumed — reported in §4.3, not hidden. The
  resumable design handled it correctly, which is itself part of what this
  run was meant to validate.
- **Arithmetic computation does not generalize, and is barely memorized** —
  §7.2. This is the single most important limitation of this run.
- **Naive substring-based correctness checks overstate apparent
  generalization** for classification — §7.3. Any future eval scripts reusing
  `EvalPrompt.check` callables should echo-check numeric identity, not just
  label presence.
- **Category imbalance (77.5% arithmetic) leaves low-volume skills
  under-trained** — §7.4. Facts/definitions/procedures/etc. are present but
  too sparse (tens to low-hundreds of examples) for this small model to
  generalize on.
- **99,840 of 100,000 nominal training examples were actually used**
  (floor-division of 10,000 by batch size 32 per shard) — a rounding
  artifact, disclosed rather than rounded away.
- **max_len=288 was chosen because it covers 100% of this dataset**, not
  because 512 was tested and found unnecessary elsewhere; a future dataset
  with longer conversations would need to revisit that number.
- **The "coherent" heuristic is byte-sanity only** (§6) — it should not be
  read as a semantic-quality metric, and this report does not treat it as one.

## 9. Honest bottom line

- **Pipeline correctness: yes.** Streaming dataset generation, dedup, leakage-free
  splitting, exact 10-shard creation, continuous resumable sharded training
  with real interruption-and-recovery, validation-based best-checkpoint
  selection, and a hash-verified novel-prompt evaluation suite all work as
  designed and are tested (§5).
- **Memorization: strong for structure, weak for arithmetic content.** The
  model reliably reproduces the *shape* of a trained response category, even
  on prompts it never saw, but does not reliably reproduce *exact
  computed numbers* even for prompts it was trained on directly (§7.2).
- **Narrow generalization: real but shallow.** Turn-taking/response-template
  structure generalizes convincingly to unseen prompts across categories
  (§7.1). Parity/primality classification shows some signal but far less than
  the naive metric suggested once checked strictly (§7.3).
- **Broad generalization: not demonstrated.** No claim here that the model
  learned an arithmetic algorithm, "understood" definitions, or would
  generalize to skills outside this dataset's templates.
- **Architectural capability vs. dataset limitation:** the *same* 112,709-parameter
  architecture went from an overfitting 61-example run to a stably-improving
  100,000-example run (§4.6) — the architecture can clearly use more data.
  What it could not do, at this size and this ~26-minute training budget, is
  learn 4-digit arithmetic from 77,467 examples of it. Whether more capacity,
  more training time, or a different data mix (e.g., explicit place-value
  supervision) fixes that is future work, not something this run answers.
- **Did the 100,000-example run improve over the 61-example run?** Yes,
  unambiguously, on the axis it was designed to test: the severe overfitting
  of the 61-example run (held-out loss climbing from 2.5 to 4.5) is gone,
  replaced by a validation curve that improves for 9 of 10 shards and ends at
  89.1% byte accuracy on a real 5,000-example held-out set. It did not,
  however, turn the model into something that reliably does arithmetic —
  that remains unsolved and is now precisely characterized rather than
  assumed away.
