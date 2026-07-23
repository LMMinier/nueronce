# NUERONCE Foundational Generation Recovery

> **STATUS 2026-07-23 — investigation closed, verdicts in.** The loss sweep
> answered the core question: 0.844 nats/byte response loss ≈ ~85% per-byte
> argmax accuracy, and 0.85^30 ≈ 0.8% exact reproduction of a 30-byte answer
> — the 0/8 gate score is the *expected value* of that loss level, not a
> generation bug. H1–H4 cleared, H5 (preset drift) found and fixed, H6
> confirmed as consequence, H7 (insufficient training) confirmed as primary
> cause. Full verdicts, secondary findings, and the 5-step fix ladder:
> `docs/RESULTS.md` (2026-07-23 entry). Measurement tool:
> `scripts/eval_loss_generation_curve.py`. Checkpoint selection must move to
> response-byte/first-8-bytes metrics with target ≤ 0.05 before any gate
> attempt.

## Cloud-session status (earlier pass)

Implemented, from the cloud side (no GPU here, so nothing below claims the
sealed gate now passes -- that verdict can only come from running the real
diagnostic on the actual `chat_11m` architecture, on the machine that holds
the checkpoints):

- **Section A (safe stopping)** -- already done before this pass
  (`tests/test_safe_interruption.py`).
- **Section B (tiny exact-overfit)** -- implemented:
  `scripts/tiny_exact_overfit_examples.py` (the 32 fixed items, 8 categories
  x 4), `scripts/train_tiny_exact_overfit.py`, `scripts/eval_tiny_exact_overfit.py`.
  Both scripts use the *actual* production serializer path
  (`nueronce.training.dialogue_data.make_sft_batch`/`encode_example` for
  training, `nueronce.prompting.format_inference_prompt` +
  `NUERONCEModel.generate(greedy=True, stop_sequences=STOP_SEQUENCES)` for
  eval) -- the same functions `scripts/train_forgeloop_sft.py` and
  `scripts/eval_foundational_proof_gate.py` use, confirmed by reading both
  files. Run order:
  ```
  python scripts/train_tiny_exact_overfit.py       # fresh chat_11m, ~2-10 min on a real GPU
  python scripts/eval_tiny_exact_overfit.py         # writes runs/tiny_exact_overfit/eval_report.json
  ```
  A CPU-only smoke test of the tooling itself (not the real diagnostic --
  intentionally undersized architecture via `--fast-tooling-check`: 48-dim,
  1 physical block, 1 logical-depth reuse, vs. `chat_11m`'s 256-dim/3
  blocks/4 logical depth) ran to full convergence in the cloud sandbox: 697
  steps, final training loss 0.149 (below the 0.15 threshold), both scripts
  executed without error, and the JSON report has the documented shape
  (`runs/tiny_exact_overfit_smoketest/fast_tooling_check_report.json`).
  **The eval result itself is instructive, with the caveat that this is the
  crippled smoke-test config, not the real architecture:** the model
  converged to a *constant* output -- it answered every one of the 32
  prompts (copying, arithmetic, polite-rewrite, ..., all 8 categories) with
  the literal bytes `"12"`, scoring 1/32 only because one target
  legitimately is `"12"`. Category scores: everything 0.0 except
  `evidence_extraction` at 0.25 (1 of its 4 items). This is precisely the
  failure signature under investigation -- low *aggregate* response-byte
  loss coexisting with output that ignores the prompt entirely -- reproduced
  on a toy scale where it can be inspected cheaply. It is very plausibly a
  genuine capacity artifact of squeezing this architecture down to 1
  physical block / 1 logical-depth reuse (dynamic patching + typed memory +
  hybrid-core routing may need more reuse depth to actually carry
  prompt-conditioning information to the decoder at all), not evidence about
  the real `chat_11m`. **The required next step is unchanged: run both
  scripts with the real `chat_11m` config (the default, no
  `--fast-tooling-check` flag) on real compute.** If the real run *also*
  collapses to a constant output, that reframes hypothesis 7 (insufficient
  training) as something structural instead -- worth watching for
  specifically now that the failure mode has a name and a cheap reproduction.
- **One negative result already established** (hypothesis 1, prompt/target
  misalignment): `tests/test_foundational_recovery_pipeline.py::test_encode_example_prefix_matches_inference_prompt`
  proves the *actual* trainer path (`encode_example` -> `format_training_example`)
  and the *actual* eval path (`format_inference_prompt`, used verbatim by
  `eval_foundational_proof_gate.py`) render byte-identical prompt prefixes
  for the same `(system, user_request)`. `scripts/start_foundational_curriculum_phase.ps1`
  confirms `runs/foundational_executor/latest_best.pt` (the checkpoint the
  current 0/8 gate result is against) and the proof gate both go through
  this exact code path -- so hypothesis 1 is not the cause of the current
  0/8 result. 9 tests in all added, covering required-test items 1-7 (byte
  prefix equality, mask start position, next-byte-shift indexing, EOS
  inside the mask, `STOP_SEQUENCES` catching a bare `<|`, no cross-call
  state leak, and mismatched-architecture load failing loudly). Items 8 and
  9 are covered by the scripts above and `tests/test_incremental_torch.py`
  respectively (both need real training time, not run in CI).
- **Not yet implemented**: sections C, D, E (a single canonical serializer
  is *mostly* already true in practice -- `dialogue_data.encode_example`'s
  canonical branch and `prompting.format_inference_prompt` agree, verified
  above -- but `dialogue_data.encode_messages`/`nueronce.chat.Conversation`
  use a *different* layout with no `<|evidence|>`/`<|plan|>` blocks; this
  divergence is real and should be resolved or explicitly documented, not
  assumed harmless), F (isolation is proven for `generate()`'s own call
  boundary; not yet proven for `IncrementalGenerator`'s cache across
  examples), G, H (per-field loss reporting and full checkpoint-provenance
  logging beyond the architecture-mismatch test).
- **One more lineage fact worth recording**: `scripts/start_foundational_curriculum_phase.ps1`
  shows `runs/foundational_curriculum` was launched with `--execution-depth 2`
  against a base (`foundational_executor/latest_best.pt`) that itself has
  `execution_depth=0` -- i.e. a *freshly random-initialized*
  `AddressableExecutionRegister` module gets spliced into an already-trained
  checkpoint at the start of that run. The current sealed 0/8 gate result is
  against the pre-splice `foundational_executor` checkpoint, so this isn't
  implicated in the *current* number, but it is one more variable in the
  next stage and worth a dedicated ablation (gate with `execution_depth=0`
  vs `2` at matched step count) once section B is green.

## Current verified result

The current approximately 11.85M-parameter checkpoint achieves very low
teacher-forced validation loss but fails the deterministic foundational proof
gate at 0/8.

Do not resume broad curriculum SFT.

Do not weaken, change, or train directly against the sealed eight-item proof
gate.

## Objective

Determine whether the failure comes from:

1. prompt or target misalignment;
2. response-loss masking;
3. chat-boundary or EOS handling;
4. recurrent-state contamination;
5. checkpoint-loading mismatch;
6. inability to autoregressively reproduce training examples;
7. insufficient generalization or insufficient base language training.

## Required implementation

### A. Safe stopping behavior

Ensure the curriculum trainer:

- handles KeyboardInterrupt;
- saves latest.pt atomically;
- preserves best.pt;
- saves optimizer, scheduler, step, RNG, and architecture configuration;
- exits without launching the proof gate after a manual interruption.

### B. Exact tiny-overfit diagnostic

Add:

scripts/eval_tiny_exact_overfit.py

and, if needed:

scripts/train_tiny_exact_overfit.py

Use 32 fixed, very short examples covering:

- copying;
- arithmetic;
- polite rewriting;
- evidence extraction;
- abstention;
- simple debugging;
- temporal ordering;
- one-step planning.

Requirements:

- separate diagnostic output directory;
- deterministic seed;
- no modification of the sealed proof gate;
- response-only loss;
- greedy model-only inference;
- exact same prompt serializer used by normal training and inference;
- train until near-zero response loss or a fixed maximum;
- evaluate free-running generation on the exact training prompts;
- save every prompt, target, response, score, and first mismatch.

Pass condition:

- at least 31/32 exact training prompts correct;
- no malformed delimiter output;
- state reset confirmed between every example.

If this fails, block all broad SFT.

### C. Prompt-byte audit

Add a diagnostic that writes:

- serialized prompt bytes;
- assistant response boundary;
- target bytes;
- loss-mask indices;
- BOS/EOS handling;
- first generation position;
- generated byte IDs;
- decoded response.

For the same example, assert that training and inference use byte-identical
prompt prefixes.

### D. Teacher-forced versus free-running audit

For every tiny diagnostic item, record:

- teacher-forced next-byte accuracy;
- first incorrect byte;
- correct-byte probability;
- correct-byte rank;
- free-running exact match;
- normalized match;
- EOS behavior.

This must identify whether failure begins before generation or compounds after
the first generated error.

### E. Delimiter and termination repair

Investigate outputs ending in fragments such as:

<|

Confirm whether chat delimiters are:

- literal byte sequences;
- reserved IDs;
- or ordinary training text.

Implement one canonical chat serializer shared by:

- curriculum construction;
- validation;
- dense generation;
- incremental generation;
- proof-gate generation.

Generation must stop on the complete canonical assistant terminator. Partial
delimiter fragments must not leak into the user-facing answer.

Do not hide general incoherence with postprocessing.

### F. State isolation

Before every independent example:

- clear recurrent memory;
- clear incremental caches;
- clear retrieval state;
- reset position counters;
- reset any dynamic patching state that is conversation-specific.

Add tests proving that evaluating prompt B after prompt A produces the same
logits as evaluating prompt B in a fresh process.

### G. Loss reporting

Report validation metrics separately for:

- prompt bytes;
- assistant-response bytes;
- first 8 assistant bytes;
- remaining assistant bytes;
- numeric answer bytes;
- EOS or terminator bytes.

Do not use aggregate sequence loss as the sole checkpoint-selection metric.

### H. Checkpoint verification

On every evaluation, record:

- checkpoint path;
- checkpoint SHA-256;
- saved training step;
- architecture preset;
- measured parameter count;
- tokenizer or byte vocabulary configuration;
- serializer version;
- optimizer restoration status where applicable.

Fail loudly if the loaded architecture does not exactly match the checkpoint.

## Required tests

Add automated tests for:

1. training and inference prompt-prefix byte equality;
2. response mask beginning at the correct byte;
3. target shifted exactly one byte;
4. EOS or terminator generation;
5. no partial delimiter leakage;
6. state isolation between prompts;
7. checkpoint architecture agreement;
8. exact-overfit reproduction;
9. dense versus incremental agreement;
10. proof gate remaining unchanged.

## Advancement rules

Do not resume broad curriculum training until:

1. tiny exact-overfit passes at least 31/32;
2. exact training prompts generate coherently;
3. prompt-prefix equality passes;
4. state isolation passes;
5. delimiter handling passes;
6. dense and incremental inference remain equivalent.

After that, build a new development set of at least 100 unseen atomic examples.
Keep the original eight proof-gate cases sealed.

Do not claim foundational coherence until the unchanged sealed gate passes.
