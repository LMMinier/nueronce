# RFT → NUERONCE 355M: controlled integration plan

## Decision

Do **not** replace NUERONCE weights, StreamFactor, or full gradients with RFT by default.
QuantoniumOS explicitly records that general neural-network weight phi-sparsity is unproven.
The existing spectral fine-tuning prototype also uses a simplified projection-gradient estimate,
not full end-to-end backpropagation.

The first defensible integration is an opt-in **Phi-RoPE self-attention ablation**:

- rotate Q and K feature pairs with `f_k = frac((k + 1) * phi)`;
- preserve vector norms exactly;
- add no learned parameters;
- keep checkpoint tensor shapes and the measured 352,993,825-parameter count unchanged;
- apply only to self-attention, not retrieval/unit cross-attention;
- compare against a baseline fork from the exact same checkpoint and batch order.

## Why this maps to NUERONCE

The Nueronce Engine currently has causal self-attention but no explicit rotary or absolute
position encoding inside Q/K. Order information reaches the model through causal masks,
causal byte convolutions, dynamic segmentation, and the selective SSM. Phi-RoPE tests whether
the RFT positional geometry adds useful position discrimination without replacing those paths.

QuantoniumOS's reproducible geometry test reports lower cross-window positional confusion than
standard RoPE. That result is evidence for testing the geometry, not proof that it will improve
NUERONCE loss or generation.

## Training-stage correction

`scripts/train_nueronce_engine_355m.py` runs response-only conversation SFT. A 355M checkpoint
near 5.37 held-out BPB has not cleared the base-language gate and should continue ordinary
next-byte base pretraining first. The new base launcher uses `model.loss()` on corpus windows.

## A/B protocol

Create two output directories from the same verified source checkpoint:

```bash
# Baseline
python scripts/train_nueronce_engine_355m_base_rft.py \
  --corpus corpus \
  --resume-from checkpoints/nueronce_engine_355m_protocol_step270/source_step270.pkl \
  --save-dir checkpoints/nueronce_355m_ab_baseline \
  --metrics-dir metrics/nueronce_355m_ab_baseline \
  --position-mode baseline \
  --seq 16 --batch 1 --lr 1e-5 --additional-steps 30

# Phi-RoPE, identical source and data order
python scripts/train_nueronce_engine_355m_base_rft.py \
  --corpus corpus \
  --resume-from checkpoints/nueronce_engine_355m_protocol_step270/source_step270.pkl \
  --save-dir checkpoints/nueronce_355m_ab_phi_rope \
  --metrics-dir metrics/nueronce_355m_ab_phi_rope \
  --position-mode phi_rope \
  --seq 16 --batch 1 --lr 1e-5 --additional-steps 30
```

Do not compare runs with different source hashes, corpus manifests, batch ordering, learning
rates, or validation windows.

## Acceptance gates

Phi-RoPE advances only when all are true:

1. all gradients remain finite;
2. no causal-leakage test fails;
3. held-out BPB is lower than baseline across at least three matched checkpoints;
4. the gain is larger than run-to-run noise;
5. wall time and memory overhead are recorded;
6. generation quality does not regress on the fixed probe set.

If the result is neutral or worse, retain the baseline and move RFT to inference-only KV/state
compression after NUERONCE receives a real incremental cache.

## Deferred experiments

- Row-wise RFT filtering of **real full gradients** before StreamFactor, tested first on 11M.
- Spectral-entropy features for the existing SSM/local/global/retrieval router.
- RFT KV-cache compression after incremental decoding is implemented.
- SmartWeights-style checkpoint manifests and integrity receipts; useful operationally but not
  a model-quality claim.
