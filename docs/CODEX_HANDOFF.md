# Handoff: cloud session → local Codex session

*You (Codex, on the owner's GPU machine) and the cloud Claude session are
working the same repo. This file is the coordination channel: it diagnoses
why your runs plateau at "loss improves but no usable natural language," and
gives the exact next runs. Pull the default branch first — your
`codex/prompt-aligned-grounded-sft` tip is one commit behind and everything
below is already merged.*

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
2. **The base checkpoint is too weak to SFT.** `cfna_chat.pt` is at step 923,
   held-out bpb **3.60**. This repo has already proven (61-example vs 100K
   runs, `docs/reports/MICRO_CFNA_SFT_100K_REPORT.md`) that SFT *shapes*
   competence but cannot *create* it. Your 500-step SFT improving val loss
   2.39→2.00 while output stays unusable is exactly that result again.
   **Gate: do not run SFT until base held-out bpb < 1.8** (11M rung) —
   prefer moving to `base_35m` (34.4M, `cfna.model.CONFIG_PRESETS`) with the
   corpus from `docs/LOCAL_TRAINING_PLAYBOOK.md` §1.
3. **800 unique SFT records will memorize, not generalize.** Feed your
   canonical-prompt builder (`scripts/build_prompt_aligned_sft.py`) from the
   generators already in-repo until you have **≥50,000 unique** records:
   `cfna.training.synthetic_dialogue` (~127K), `cfna.training.mcq_sft`
   (ARC/OpenBookQA/CommonsenseQA/MathQA/GSM8K via `load_and_convert`),
   OASST1/Dolly from the stack. Cap any single register at ~25% of records
   (the 77%-arithmetic poisoning lesson).
4. **You are measuring knowledge with the wrong metric during the transition.**
   Structure generalizes before content at these scales, so generative
   exact-match stays near zero long after knowledge is present. Report
   **choice-ranking accuracy** (`cfna.training.mcq_sft.evaluate_mcq`: scores
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
