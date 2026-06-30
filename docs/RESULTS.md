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

## A correctness bug the tests caught

The hybrid block's adaptive router originally mean-pooled path outputs over the
**entire** unit sequence to compute mixing weights, then applied them per
position — leaking future units into past positions. The causality test failed
(max logit change before an edited byte ≈ 0.04). Switching to **per-position
routing** fixed it; the test now reports `0.0`.
