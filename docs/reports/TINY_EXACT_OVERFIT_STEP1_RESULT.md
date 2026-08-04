# Section B tiny exact-overfit result — real `chat_11m`, cloud CPU sandbox

Run: 2026-07-23, cloud session, CPU-only Codespace (no GPU; `torch==2.13.0+cpu`).

## Commands run

```
python scripts/train_tiny_exact_overfit.py \
    --out runs/tiny_exact_overfit/checkpoint.pt \
    --system-file runs/forgeloop/system_prompt.txt \
    --micro-batch 8
python scripts/eval_tiny_exact_overfit.py \
    --checkpoint runs/tiny_exact_overfit/checkpoint.pt \
    --output runs/tiny_exact_overfit/eval_report.json
```

`--micro-batch 8` is a new, additive flag on the training script (see
"Infrastructure fix" below) that changes nothing about what is being
measured — see that section for why it was necessary on this sandbox and
why it is mathematically a no-op on the gradient that gets computed.

Training used the **real, default `chat_11m` architecture** (the same
`nueronce.model.chat_config()` used everywhere else in this repo,
11,131,477 params) — not `--fast-tooling-check`'s crippled 48-dim stand-in
from the prior cloud-session smoke test. Converged at **step 115**,
training loss **0.0471** (below the script's 0.05 stop threshold), in
~2220 s wall-clock on 2 vCPUs.

## Result

```json
{
  "gate_passed": false,
  "exact_match_count": 27,
  "n_examples": 32,
  "exact_match_fraction": 0.84375,
  "delimiter_leaks": 0,
  "state_isolation_ok": true
}
```

**27/32 — below the required 31/32 threshold. Gate does not pass.**

Category scores (all categories nonzero):

| Category | Score |
|---|---:|
| copying | 0.75 |
| arithmetic | 0.75 |
| polite_rewriting | 0.75 |
| evidence_extraction | 1.0 |
| abstention | 0.75 |
| simple_debugging | 1.0 |
| temporal_ordering | 1.0 |
| one_step_planning | 0.75 |

## The 5 failed items, in full

| # | Category | Target | Answer | first_mismatch_char |
|---|---|---|---|---:|
| 0 | copying | `apple` | `aple` | 2 |
| 6 | arithmetic | `6` | `6\n6` | 1 |
| 11 | polite_rewriting | `Could you please answer the question?` | `Could you please anser the quesesestion?` | 20 |
| 18 | abstention | `Not provided in the evidence.` | `Not provided in theviden e.` | 19 |
| 29 | one_step_planning | `List the folder and keep files ending in .py.` | `List the folder and kep files ending in .py.` | 22 |

State isolation checks (replaying items 0, 15, 31 after all 32 have run)
were all identical to their first-pass answers — no cross-example state
leak. Zero `<|` delimiter fragments leaked into any answer.

## Reading the failure pattern

This is **not** the sealed proof gate's failure signature. The 0/8 sealed
result (`metrics/foundational_proof_gate_current.json`, still on disk from
the prior local/Codex session) is total incoherence: answers like `"The
web to a languer of a for a standent of the protect.\n<|"` — off-topic,
grammatically broken, and trailing a bare, unstopped delimiter fragment.

Here, every one of the 32 answers is topically and grammatically correct,
and 4 of the 5 misses are a **single dropped or substituted character** in
an otherwise byte-perfect reproduction (`aple`/`apple`, `anser`/`answer`,
`kep`/`keep`, one extra letter dropped from `evidence`/`theviden e`). The
5th (`6\n6`) reproduces the correct digit and then echoes it again instead
of stopping. None of these are delimiter corruption, off-topic drift, or
garbage — they read like a checkpoint that stopped training *just* past
the edge of the 0.05 threshold (final loss 0.0471, i.e. barely under it)
rather than a broken serialize/mask/generate/stop pipeline.

This is worth stating plainly against the task's own prior framing: a
<31/32 result was expected to mean "the pipeline itself is broken, not
that the model needs more training." The evidence gathered here does not
fit that pattern — zero delimiter leaks, isolation holds, every category
nonzero, and every miss is a localized near-miss rather than nonsense.
**I did not treat this as license to just retrain past it and declare
success** — per the recovery doc's advancement rules, the sealed
requirement is `>=31/32`, unmodified, and this run does not meet it. I am
stopping here and reporting rather than unilaterally deciding the shortfall
is "close enough" or retraining until it happens to clear the bar.

One caveat for whoever picks this up: `eval_tiny_exact_overfit.py` calls
`NUERONCEModel.generate()` directly (the dense path), not
`nueronce.incremental.IncrementalGenerator` — the class the sealed
`eval_foundational_proof_gate.py` actually runs inference through
(`docs/CODEX_HANDOFF.md`'s section 9, "dense versus incremental
agreement", is explicitly still open). A pass here would leave that
question open; this near-pass leaves it *more* open, not less, since we
don't yet know whether `IncrementalGenerator` would reproduce these same
5 items identically, worse, or better.

## Infrastructure fix made along the way (not a pipeline/science change)

The original `train_tiny_exact_overfit.py`, run unmodified on this
sandbox, could not complete even one training step: a `torch.no_grad()`
forward pass on the real 32-example batch (shape `[32, 456]`) finished in
9.4 s, but the identical **grad-enabled** forward + `backward()` got
`SIGTERM`'d (exit 143, not the kernel OOM-killer's `SIGKILL`/137) every
time, in under a minute. A `free -m` sampler running once/second during a
repro confirmed the cause directly: system memory used climbed from
~2.7 GB to ~7 GB (of 7.8 GB total, no swap) over the course of the
grad-enabled forward+backward, then dropped straight back to ~2.8 GB the
instant the process was killed — a soft memory guard, not an infinite
loop or an O(n²) blowup in the dynamic-patching code (both of which were
suspected and ruled out first: isolated forward passes at seq
64/96/128/160/192 all completed in 1-4s once run without concurrent
sandbox load).

Fix: added an optional `--micro-batch N` flag to
`train_tiny_exact_overfit.py` that splits the 32-example batch into
gradient-accumulation chunks. Each chunk's cross-entropy sum is divided by
the **global** (whole-batch) valid-target-byte count before its own
`backward()` call, so summing the per-chunk backward passes produces
*exactly* the same gradient as one `model.masked_token_loss()` call over
the full batch — this is standard gradient accumulation, not an
approximation, and the default (`--micro-batch 0`) preserves the original
single-batch behavior unchanged for anyone running this on a machine with
enough RAM/VRAM. `--micro-batch 8` kept peak memory under ~4.9 GB and let
training complete.

## Recommendation (not a decision made unilaterally)

Given the near-miss character-level pattern and that training stopped
right at the 0.05 boundary, the most likely next experiment is simply
lowering `--loss-threshold` (e.g. to 0.01) or raising `--max-steps` on this
*same* diagnostic and re-running both scripts unchanged otherwise — cheap
on this hardware (~115 steps took ~37 minutes; a few dozen more steps is
a small addition). That would directly test whether this is a
convergence-boundary artifact rather than anything structural. Per the
recovery doc's own rule ("If this fails, block all broad SFT") I have not
done that automatically and have not proceeded to Step 2 (base
pretraining) — reporting back per the task's explicit instruction to stop
and report before continuing.
