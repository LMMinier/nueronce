# Coherent Inference And SFT Finalization

Branch: `codex/finalize-coherent-inference`

## What Changed

This pass adds two practical pieces around the existing CFNA training stack:

1. A coherent inference wrapper:
   - consistent chat probing;
   - deterministic narrow tools for exact arithmetic and known synthetic facts;
   - repetition/non-printable detection;
   - explicit fallback instead of presenting loops as coherent replies.

2. A balanced SFT dataset builder:
   - reserves unique validation/test records before any train weighting;
   - repeats small categories only inside training as explicit weighting;
   - keeps leakage checks against validation/test.

No new cryptographic mechanisms were added. No claim-extraction training was
started.

## Local Checkpoint Probe

Command:

```bash
python scripts/coherent_chat.py --backend torch --ckpt checkpoints/cfna_chat.pt \
  --probe --json metrics/coherent_probe_assisted.json
```

Result:

- assisted pass rate: `0.800`
- tool rate: `0.800`
- fallback rate: `0.200`

Model-only probe:

```bash
python scripts/coherent_chat.py --backend torch --ckpt checkpoints/cfna_chat.pt \
  --probe --no-assist-tools --json metrics/coherent_probe_model_only.json
```

Result:

- pass rate: `0.000`
- fallback rate: `1.000`
- observed failure: repeated-token continuations such as `the the the...`

Interpretation: the local PyTorch checkpoint is not yet a coherent assistant.
The wrapper prevents bad continuations from being surfaced and lets exact narrow
tools answer when appropriate.

## SFT Smoke Run

Command:

```bash
python scripts/train_sft.py --backend torch --model small \
  --ckpt checkpoints/cfna_chat.pt \
  --out checkpoints/cfna_chat_sft_smoke.pt \
  --steps 20 --batch 4 --lr 5e-4 --seed 0
```

Measured result:

- held-out response-byte loss improved from `2.859` to `2.474`;
- qualitative replies remained repetitive;
- coherent probe still needed fallback on open-ended chat.

Interpretation: the fine-tuning path works and receives gradient, but 20 steps
on the tiny hand-written SFT set is not enough to make the model conversational.
Loss movement alone should not be reported as coherent behavior.

## Balanced Fine-Tuning Recipe

Build a balanced curriculum:

```bash
python scripts/build_balanced_sft_dataset.py \
  --out-dir data/sft_balanced \
  --train-examples-per-category 500 \
  --val-per-category 4 \
  --test-per-category 4 \
  --num-shards 5 \
  --examples-per-shard 1500 \
  --seed 43
```

Then run microtorch full-CFNA SFT:

```bash
python scripts/train_sft.py \
  --backend microtorch --model full-cfna \
  --train-dir data/sft_balanced/train_shards \
  --validation data/sft_balanced/validation.jsonl \
  --test data/sft_balanced/test.jsonl \
  --num-shards 5 --examples-per-shard 1500 \
  --save-dir checkpoints/micro_cfna_sft_balanced \
  --metrics-dir metrics/balanced_sft \
  --batch 32 --lr 1e-3 --seed 43 --resume
```

Evaluate:

```bash
python scripts/coherent_chat.py --backend microtorch \
  --ckpt checkpoints/micro_cfna_sft_balanced/best.pt \
  --probe --json metrics/coherent_probe_balanced.json
```

## Honest Bottom Line

The inference path is now safer and more usable, but coherent model-only
conversation is still not demonstrated on the local checkpoint. The next real
milestone is a balanced SFT run whose model-only probe improves without relying
on deterministic tools or fallback.
