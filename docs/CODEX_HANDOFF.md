# Handoff: cloud session → local machine

*You (on the owner's local machine, presumably with real GPU power) and the
cloud Claude session are working the same repo/branch
(`claude/new-session-tkjr3b`). This file is the coordination channel — pull
it first, everything below reflects the live state as of 2026-07-24.*

## Current priority (2026-07-24) — supersedes everything below until closed

**Status: Section B (tiny exact-overfit) is done and passed (31/32,
`docs/RESULTS.md`). Base pretrain and curriculum SFT are IN PROGRESS on a
CPU-only cloud sandbox and have hit real compute limits — please pick this
up and finish it with real hardware.** Everything needed to continue is
already pushed; nothing needs to be rebuilt from scratch.

### Where things stand right now

1. **Base pretrain** (`checkpoints/chat_11m_base_local_slim.pt`, resumed
   from step 7009/bpb 1.971 left by an earlier parallel session): ran an
   LR ladder on CPU —
   - LR 5e-4: step 7009 → 8300, plateaued at bpb ~1.94
   - LR 1.67e-4: step 8300 → 9400, plateaued at bpb ~1.858
   - LR 5.56e-5: step 9400 → 10900, plateaued at bpb ~1.843
   - LR 1.85e-5: step 10900 → 11300 (paused here), best bpb **1.8414**
   Each drop bought less than the previous one (0.08 → 0.015 → ~0.009) —
   this looks like it may be a real floor for this corpus/scale, not just
   "needs more LR-ladder patience." The corpus used
   (`scripts/build_local_corpus.py`, ~24MB of owner-curated public-domain
   texts already in the repo — not the original 400MB HF-streamed plan)
   is almost certainly too small to push much past this. **If you have
   reliable Hugging Face access, the highest-value thing you can do is
   rebuild `corpus_local` with the real 400MB pull** (see "Corpus" below)
   and resume base pretrain on that instead of the 24MB fallback — the
   cloud sandbox couldn't do this because `bigcode/the-stack-smol` is a
   **gated** HF dataset it couldn't authenticate to non-interactively;
   your machine likely can (`huggingface-cli login` or `HF_TOKEN` env var),
   or just drop that one source and use the other 5.

2. **Curriculum SFT** (`checkpoints/chat_11m_v31_sft_best_slim.pt`): started
   from the base checkpoint above **despite it being above the 1.8 bpb
   threshold** — that threshold turned out to be an unvalidated heuristic
   (grep the repo: it appears in exactly one place, stated with no
   derivation), so rather than block on it indefinitely, we're testing
   empirically whether SFT can still work. At sft_step 550: val_loss 1.842,
   response-byte argmax accuracy 46.6%, **still improving every single
   eval** (0 bad_evals) — genuinely still converging, not plateaued.
   Two sealed-gate checks so far (both expected to fail this early,
   logged for the record): step 150 → 0/8, mostly empty/near-empty answers;
   step 500 → 0/8, but qualitatively different — repetitive stuttering
   loops ("the the the the", "cor cor cor cor") instead of emptiness. That
   progression (empty → stutter → hopefully coherent) is worth tracking as
   you keep training, not just the raw pass/fail.

### Exact commands to resume

```bash
# Curriculum SFT (auto-resumes from --out if it exists; only the slim,
# no-optimizer checkpoint is in git, so this restarts the optimizer state
# but keeps the trained weights):
cp checkpoints/chat_11m_v31_sft_best_slim.pt checkpoints/chat_11m_v31_sft.pt
python scripts/train_forgeloop_sft.py \
    --base checkpoints/chat_11m_base_local.pt \
    --train data/foundational_recovery_v3_1/train.jsonl \
    --val data/foundational_recovery_v3_1/val.jsonl \
    --out checkpoints/chat_11m_v31_sft.pt \
    --system-file runs/forgeloop/system_prompt.txt \
    --batch 4 --max-len 768 --lr 2e-5 --eval-every 25 --patience 20 \
    --balanced-domain-sampling
# (auto-detects CUDA and enables AMP if present -- no flag needed)

# Periodically test the real thing, not just gate_ready (which requires an
# almost-impossible val_loss<=0.05 -- the sealed gate is the actual test):
python scripts/eval_foundational_proof_gate.py \
    --checkpoint checkpoints/chat_11m_v31_sft_best.pt \
    --output metrics/foundational_proof_gate_current.json --no-fail-exit
```

```bash
# If SFT plateaus well short of passing, the next lever is base pretrain
# on the REAL corpus rather than the 24MB fallback:
cp checkpoints/chat_11m_base_local_slim.pt checkpoints/chat_11m_base_local.pt
python scripts/dump_corpus_stack.py --out corpus_large \
    --sources cosmopedia_100k,open_web_math,project_gutenberg,english_wikipedia_latest,pmc_oa_comm \
    --target-bytes 400000000 --val-every 20
    # add the_stack_smol back in --sources if you have HF auth for gated datasets
python scripts/train_checkpoint.py --preset chat_11m --corpus corpus_large \
    --minutes 240 --seq 192 --batch 16 --lr 5e-4 --amp --resume \
    --out checkpoints/chat_11m_base_local.pt
# (--amp only takes effect on CUDA; drop it on CPU. Same LR-ladder rule
# applies: resume at 1/3 LR each time held-out bpb plateaus 18+ evals.)
```

Push `metrics/` + a slim checkpoint after every session (see the git
commands in the cloud session's commit history on this branch for the
exact pattern — `git add -f checkpoints/*_slim.pt`, since `checkpoints/`
is gitignored by default).

### One code fix already made, safe to keep

`scripts/build_foundational_recovery_v3_1.py` had hardcoded Windows
backslash paths (`r"artifacts\..."`), which `Path()` on Linux treats as a
literal filename character, not a separator — fixed to forward slashes.
Output (`data/foundational_recovery_v3_1/{train,val,test}.jsonl`,
`artifacts/foundational_v3_1_manifest.json`) is byte-identical to what was
already committed, confirming the build is fully deterministic; no need to
re-run it unless you're regenerating from scratch.

---

## Prior priority (closed) — kept for context

`FOUNDATIONAL_GENERATION_RECOVERY.md` (repo root) is now the live investigation:
the `foundational_executor`/`foundational_curriculum` checkpoints reach very
low SFT val loss but score **0/8, 0.0 overall** on the sealed
`scripts/eval_foundational_proof_gate.py` gate — every answer is incoherent
even though loss looks converged. **Do not resume broad curriculum training**
(`scripts/start_foundational_curriculum_phase.ps1` or equivalent) until
Section B below passes.

Section B (tiny exact-overfit) is now implemented and needs to be run on
your machine, against the real architecture, before anything else:

```bash
python scripts/train_tiny_exact_overfit.py \
    --out runs/tiny_exact_overfit/checkpoint.pt \
    --system-file runs/forgeloop/system_prompt.txt
python scripts/eval_tiny_exact_overfit.py \
    --checkpoint runs/tiny_exact_overfit/checkpoint.pt \
    --output runs/tiny_exact_overfit/eval_report.json
```

This trains a **fresh** `chat_11m` (not your sealed checkpoint — this never
touches `foundational_executor`/`foundational_curriculum`) on 32 fixed short
examples until near-zero response loss, then checks whether it can
free-running (greedy, no teacher forcing) reproduce what it just memorized.
Read `runs/tiny_exact_overfit/eval_report.json`'s `gate_passed` and
`exact_match_count` (need ≥31/32) plus `delimiter_leaks` and
`state_isolation_ok`.

- **If it fails** (< 31/32, or delimiter leaks, or isolation breaks): the bug
  is in the shared train/serialize/generate pipeline, not in the amount of
  curriculum data or training steps. Report exactly which of the 32 items
  failed and their `first_mismatch_char` — that's the next thing to fix, and
  it blocks everything downstream (per section E/F, `KNOWN_LIMITATIONS.md`
  item 4: generation has no incremental cache during teacher-forced training,
  but `nueronce.incremental.IncrementalGenerator` is what the proof gate
  actually runs inference through — if the fresh-model overfit test also
  produces garbage, suspect that path first since it's the one variable
  neither the trainer nor a plain `model.generate()` test exercises).
- **If it passes**: the pipeline is mechanically sound, and the 0/8 result on
  the real checkpoints is a training-scale/data problem (recovery doc
  hypothesis 7), not a plumbing bug. In that case the original diagnosis
  below (base bpb too high before SFT, too little unique data, wrong
  eval metric during the transition) is very likely still the live issue —
  `runs/foundational_recovery_v2_probe500/FROZEN_DO_NOT_RESUME.txt` already
  shows a directly-tested base checkpoint (`cfna_base_large_best.pt`, step
  73,199) producing empty/repetitive garbage with **zero** SFT applied, which
  points hard at base pretraining never having converged, independent of
  everything built on top of it since. Cross-check its held-out bpb before
  building another curriculum on top of it.

One negative result already established from the cloud side without a GPU:
`tests/test_foundational_recovery_pipeline.py::test_encode_example_prefix_matches_inference_prompt`
proves `scripts/train_forgeloop_sft.py`'s trainer path and
`scripts/eval_foundational_proof_gate.py`'s eval path render byte-identical
prompt prefixes for the same input — so prompt/target misalignment between
*those two specific scripts* is ruled out. It does not rule out a bug inside
`IncrementalGenerator` itself, which only the tiny-overfit run (real
`generate()` call, not a unit test of the byte-formatting alone) can surface.

---

## Diagnosis (from your own artifacts + this repo's prior experiments)

1. **You are training on CPU while a GPU sits idle.** Your checkpoint audit
   records `torch 2.11.0+cpu`, `torch.version.cuda = None`
   (`docs/reports/PROMPT_ALIGNED_GROUNDED_SFT_REPORT.md`). Every conclusion
   you drew about "not feasible in this session" follows from that one
   installation. Fix first:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   pytest tests/test_gpu_amp.py -v   # 2 CUDA tests green = fp16/AMP path is safe (already hardened)
   ```
2. **The base checkpoint is too weak to SFT.** `nueronce_chat.pt` is at step 923,
   held-out bpb **3.60**. This repo has already proven (61-example vs 100K
   runs, `docs/reports/MICRO_NUERONCE_SFT_100K_REPORT.md`) that SFT *shapes*
   competence but cannot *create* it. Your 500-step SFT improving val loss
   2.39→2.00 while output stays unusable is exactly that result again.
   **Gate: do not run SFT until base held-out bpb < 1.8** (11M rung) —
   prefer moving to `base_35m` (34.4M, `nueronce.model.CONFIG_PRESETS`) with the
   corpus from `docs/LOCAL_TRAINING_PLAYBOOK.md` §1.
3. **800 unique SFT records will memorize, not generalize.** Feed your
   canonical-prompt builder (`scripts/build_prompt_aligned_sft.py`) from the
   generators already in-repo until you have **≥50,000 unique** records:
   `nueronce.training.synthetic_dialogue` (~127K), `nueronce.training.mcq_sft`
   (ARC/OpenBookQA/CommonsenseQA/MathQA/GSM8K via `load_and_convert`),
   OASST1/Dolly from the stack. Cap any single register at ~25% of records
   (the 77%-arithmetic poisoning lesson).
4. **You are measuring knowledge with the wrong metric during the transition.**
   Structure generalizes before content at these scales, so generative
   exact-match stays near zero long after knowledge is present. Report
   **choice-ranking accuracy** (`nueronce.training.mcq_sft.evaluate_mcq`: scores
   each option by masked answer loss; chance level reported alongside) next
   to your `eval_inference_phase2` pass rate. Do not discard checkpoints on
   generative metrics alone (`docs/BREAKTHROUGH_MAP.md` §3.4).

## Run order (all on GPU)

1. CUDA torch + `tests/test_gpu_amp.py` green.
2. Corpus: playbook §1 (`dump_corpus_stack` multi-subject pull, ~400 MB,
   `--val-every 20`).
3. Base pretrain: `base_35m` preset, AMP, seq 192 × batch 16, LR 3e-4,
   resume-capable, until held-out bpb plateaus (target ≤ 1.5). Push
   `metrics/` after every session.
4. SFT: ≥50K unique canonical-prompt records (mix per §3 above), early stop
   on validation, keep best-by-val (never last-step).
5. Report per checkpoint: base bpb curve, SFT val curve, choice-ranking acc
   vs chance per subject, phase2 suite pass rate, and 5 raw transcripts
   (unedited). Commit to `metrics/` + push.

## Division of labor

- **Codex (you):** execute the runs above; push metrics/manifests/reports
  after each stage; keep architecture frozen (no new modules — the
  concurrent plan's final directive stands).
- **Cloud session:** pulls your metrics, does curve analysis and go/no-go on
  the next parameter rung, fixes any code defects you flag in commit
  messages or `docs/reports/*`, and pushes back. Anything you need changed
  in the shared code: write it in a report file and push — that is the
  channel.

## Acceptance for "frontier model finished" (this phase)

- base_35m held-out bpb ≤ 1.5 on the multi-subject corpus
- choice-ranking accuracy beats chance by ≥15 points on ≥3 MCQ subjects
- phase2 inference suite: ≥60% valid, non-echo, stop-terminated answers
- 5/5 sample transcripts are grammatical English addressing the question
  (fluency bar, not correctness bar)

When those hold, the next rung (`base_90m`) unlocks per the playbook rules.
