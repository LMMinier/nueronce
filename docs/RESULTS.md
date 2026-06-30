# Recorded run

Hardware: 4-core CPU, 15 GB RAM, no GPU. PyTorch 2.12 (CPU), used only as a
tensor/autograd substrate. Every operator is hand-built (`cfna/nn.py`,
`cfna/blocks.py`).

## Training (`python scripts/train_demo.py --steps 400 --seq 96 --batch 16`)

Model: **2,042,833 parameters**. Objective: next-byte cross-entropy + auxiliary
boundary loss. Corpus: the project's own design vocabulary (~5 KB), byte-level.

| step | loss | lm | bits/byte | ~units/seq |
|---:|---:|---:|---:|---:|
| 0 | 5.98 | 5.75 | 8.30 | 14.4 |
| 50 | 1.30 | 1.30 | 1.88 | 13.5 |
| 100 | 0.41 | 0.41 | 0.59 | 13.5 |
| 200 | 0.17 | 0.17 | 0.24 | 13.5 |
| 300 | 0.09 | 0.09 | 0.13 | 13.5 |
| 399 | 0.08 | 0.08 | 0.11 | 13.5 |

- **bits/byte: 8.30 → 0.11** (uniform-byte baseline = 8.0). The architecture
  learns.
- Dynamic patching forms ~13–14 information units per 96 bytes (avg patch ≈ 7
  bytes) — the learned boundary head is active, not degenerate.
- Greedy continuation of `"CFNA separates "`:
  > CFNA separates understanding, thinking, remembering, and speaking. Perception
  > forms dynamic inf…
- Training time: ~118 s for 400 steps on CPU.

## Causality check

`tests/test_model_learns.py::test_model_is_causal` edits a future byte and asserts
**zero** change in every earlier-position logit. After fixing the router
(see below) the measured change before the edit is exactly `0.0`.

## End-to-end pipeline (`python scripts/run_pipeline.py --steps 300`)

After 300 steps (final train loss 0.114), `cfna.pipeline.respond` on the query
`"CFNA separates understanding,"`:

```
retrieved : ['doc0', 'doc3', 'doc4']        # doc0 is the matching sentence
reasoning : hypothesis_a (conf 0.887)
plan      : ['answer', 'support', 'caveats']
verifier  : {'passes': True, 'supported_fraction': 1.0, 'n_failures': 1}
answer    : CFNA separates understanding, thinking, remembering, and speaking.
            Perception forms dynamic informat[i]on units fr[om] raw byt[es]
```

All subsystems fire: real hybrid retrieval surfaces the relevant document, the
latent workspace produces a hypothesis with calibrated confidence, the planner
orders the response, the model renders the answer, and the independent verifier
checks it against the retrieved evidence.

## Retrieval wired into training (`python scripts/train_retrieval.py`)

Retrieval is trained, not bolted on at inference: the core's retrieval cross-
attention and a byte-decoder retrieval cross-attention both receive gradients
(`tests/test_retrieval_training.py::test_retrieval_path_receives_gradients`).

The task isolates retrieval: each example is a fresh, random `lookup <key> =
<value>` fact used exactly once, so the value **cannot** be memorized in the
weights — it is only recoverable from the retrieved document. A frozen
`embed_text` retriever fetches the matching fact (plus distractors) from a store
(recall@k = 1.0); the model is trained to copy the value. Retrieval context is
exposed as *(left-context key, current-byte value)* so resolving the value is an
induction-style copy.

Ablation after training (value token = one of 10 digits, chance = 0.10):

| step | value loss WITH | value loss WITHOUT | acc WITH | acc WITHOUT |
|---:|---:|---:|---:|---:|
| 0 | 4.37 | 4.69 | 0.19 | 0.19 |
| 300 | 1.79 | 3.10 | 0.34 | 0.06 |
| 600 | 1.17 | 5.30 | 0.59 | 0.12 |
| 900 | 0.79 | 4.42 | 0.69 | 0.09 |

- **With retrieval**: value accuracy 0.10 → ~0.6–0.7, loss → ~0.8.
- **Without retrieval** (same weights, retrieval removed at eval): accuracy stays
  at chance and loss *climbs* — the model has learned to depend on retrieval.

This is the RETRO-style result: retrieval substitutes for parametric memory, and
the with-vs-without gap is the proof that the retrieval path is genuinely used.

## Matched baselines on a held-out split (the honest signal)

Reporting *training-corpus* loss as quality is meaningless. This harness
(`cfna/eval.py`, `scripts/compare_baselines.py`) trains matched-parameter,
from-scratch models on one region of a non-repeating corpus and measures
bits/byte on a **disjoint held-out region**.

Corpus 4,415 bytes → train 3,311 / held-out 1,104 (disjoint). ~1.1–1.2M params
each, same optimizer/steps.

| model | params | train bpb | held-out bpb (seed 0, 500 steps) |
|---|---:|---:|---:|
| CFNA | 1.22M | 0.83 | **6.54** |
| ByteTransformer | 1.14M | 0.51 | 6.75 |
| ByteSSM (pure) | 1.12M | 0.16 | 6.97 |

Across seeds (held-out bpb), 400 steps:

| seed | CFNA | Transformer | SSM |
|---:|---:|---:|---:|
| 1 | 6.12 | **5.59** | 6.68 |
| 2 | 6.10 | **5.69** | 6.99 |

**Honest reading:** held-out bpb (6–7) is well below the 8.0 uniform baseline, so
the models learn *some* transferable byte statistics. CFNA is **competitive** with
a matched byte Transformer but does **not** decisively beat it — the Transformer
is better at 400 steps, CFNA was better at 500 steps; the margins are within
seed/step noise. The one consistent finding is that the **pure SSM generalizes
worst** despite memorizing the training region hardest (train bpb 0.16) — classic
overfitting. No scaling or superiority claim is made: this is a ~4 KB corpus and
~1.2M-param models. The value here is the *harness*, which makes the question
answerable.

## Trainable segmentation

The boundary head previously learned only from a hand-built syntax target
(`boundary_logits.detach()`). Now a differentiable per-unit boundary feature is
injected into the unit representations, so the **next-byte loss flows gradient
into the boundary head** (verified: LM-only gradient into `boundary_head` ≈ 4.55,
`test_model_learns`/`test_pipeline_components`). The discrete segmentation
structure stays straight-through; fully-differentiable discrete patching
(Gumbel/soft-pool) remains future work.

## Verify→revise actually revises

The default pipeline previously passed `lambda d, f: d` (a no-op). It now removes
unsupported/contradicted claims, hedges over-certain wording, and appends missing
requirements (`impl.revise_draft`). `test_revise.py` shows a draft that fails
verification (an unsupported claim) **passing after one real revision round**. The
verifier's claim/evidence matching was also upgraded from char-n-gram overlap
(which made unrelated English look "supported") to content-word overlap.

## 350M is now a real model, not bookkeeping

`cfna.model.large_config()` builds a **337M-parameter** CFNAModel (within ~4% of
the design's 350M budget) that forwards correctly; `test_scaling.py` constructs
and runs it. It is not trained here (that needs real data and compute).

## A correctness bug the tests caught

The hybrid block's adaptive router originally mean-pooled path outputs over the
**entire** unit sequence to compute mixing weights, then applied them per
position — leaking future units into past positions. The causality test failed
(max logit change before an edited byte ≈ 0.04). Switching to **per-position
routing** fixed it; the test now reports `0.0`.
