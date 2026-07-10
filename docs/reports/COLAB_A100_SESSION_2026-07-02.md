# Colab A100 session — 2026-07-02: base pretrain clears the SFT gate

First GPU training session of the forgeloop pipeline (`notebooks/
nueronce_large_corpus_forgeloop.ipynb`), run by the owner on Colab Pro,
NVIDIA A100-SXM4-40GB, torch 2.11.0+cu128, AMP fp16. Numbers below are
transcribed from the live training log (checkpoints + full history JSON in
the owner's Drive under `NUERONCE_checkpoints/`; not committed — *.pt is
gitignored).

## Corpus (validated, cell 5)

142 shards / 0.367 GB, whole-document + per-repository holdout, all
license checks green: code 151.9 MB, Cosmopedia 120.0 MB, books 54.8 MB
(21 owner-committed public-domain texts = 23.5 MB + downloaded classics),
ForgeLoop traces 40 MB.

## Base pretrain (11.1M params, seq 256, batch 64, lr 5e-4, 55 min)

Held-out bpb, from scratch: **4.515 → 1.785** in 7,524 steps / 55 min at a
steady ~37K byte-targets/s. **The < 1.8 SFT gate fell in-session at step
~7000 (1.797), minute 51** — the prior estimate was 1–2 sessions. No
crashes; atomic best-checkpoint saves fired throughout (final best 1.785
vs final latest 1.787).

Milestones observed in live greedy/sampled probes (cell 7b):
- bpb 2.67: greedy loops ("the the of the"), code prompt → newline only
- bpb 2.11 (sampled): morphologically-legal invented words ("probacle",
  "packagragition"), English sentence rhythm
- bpb 2.03: code prompt → indented Python with `self.assertEqual(`;
  prose registers separating (Cosmopedia expository vs book narrative)

## SFT stage (20 min) — worked, then overfit, as the repo predicted

`SFT_MAX_LEN=384` + ~300-byte system prompt filtered ~20K OpenAssistant
pairs down to **429 examples** (8 val). Val loss bottomed at **1.246 @
step 225 (~1.6 min)** then climbed to ~1.94 by step 2425 while train loss
fell to ~0.05 — textbook memorization. The session-era SFT trainer saved
only final weights, so the deployed checkpoint is the overfit endpoint.

Chat test (cell 12): "Hello NUERONCE. Introduce yourself" → **"Hello! Whow w
are you?"** — first turn-taking response (greeting answered with a
greeting) from a real-corpus base in this project. Long out-of-
distribution technical prompts → empty responses (expected at 429
examples).

Fixes already pushed for the next session (commit 52d4bcf):
`SFT_MAX_LEN` 384→1024 (pairs survive the filter), SFT best-by-val
`*_best.pt` + atomic saves, early stop after 20 evals without
improvement, test/backup cells prefer/include best checkpoints.

## Ledger updates

- H-gate confirmed at GPU scale: base quality, not SFT step count, was
  the binding constraint all along (consistent with
  `MICRO_NUERONCE_SFT_100K_REPORT.md`).
- 11M capacity is NOT yet the bottleneck at 0.367 GB corpus: train bpb
  ~1.2 vs held-out 1.785 with the curve still descending at cutoff.
  Plateau estimate 1.6–1.75 after ~4–8 more A100-hours.

## Next session (in order)

1. Restore `checkpoints/` from Drive; rerun cells 1–3.
2. Rerun cell 9 (rebuilds SFT pairs at 1024 → expect ~15–20K examples),
   then cell 11 (early-stops at the val bottom on its own).
3. Chat (cell 12, loads `*_best.pt`), rerun backup (cell 13).
4. Optional: continue base pretrain toward plateau, or start the 35M rung
   (cell 7b `RUN_35M=True`) if an L4/A100 is available.
