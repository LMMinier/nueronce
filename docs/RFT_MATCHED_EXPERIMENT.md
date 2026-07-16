# Matched RFT Training Experiment

The benchmark harness is `scripts/benchmark_rft_matched.py`.

It compares four model variants using the same NUERONCE configuration, corpus,
training batches, seed, optimizer, learning rate, sequence length, batch size,
step count, evaluation batches, and generation prompts:

1. `dense` — unchanged NUERONCE hybrid-core feed-forward network;
2. `ordinary_sparse` — direct-coordinate fixed sparse connectivity;
3. `dct_sparse` — the same sparse budget in an orthonormal DCT basis;
4. `rft_sparse` — the same sparse budget in the canonical RFT basis.

The three sparse variants use the same row/column connectivity and two
trainable real scalars per active edge. A `dense_budget` replacement is also
implemented in `nueronce.spectral_baselines` for a reduced-width dense control,
though it is not yet included in the default four-way CLI table.

## Measurements

Each variant runs in its own subprocess and records:

| Question | Measurement |
|---|---|
| Does it lower parameter memory? | trainable parameters, trainable bytes, model storage bytes |
| Does it lower optimizer memory? | actual allocated AdamW state bytes after training |
| Does it lower total process memory? | isolated-process peak RSS and RSS growth |
| Does it lower GPU memory? | peak CUDA allocated bytes when run on CUDA |
| Is it faster? | train seconds, steps/second, tokens/second |
| Does it learn? | first and final training BPB |
| Does it generalize? | mean validation BPB over fixed held-out batches |
| Does generation degrade? | saved outputs for four conversation-style prompts |
| Is output mechanically healthier? | printable ratio, unique-byte ratio, four-gram repetition, non-empty length |

The conversation proxy is deliberately weak. It is not called a coherence score
because printable, non-repetitive text can still be meaningless. The generated
samples must be reviewed manually, and a serious conversational comparison
requires matched response-only SFT after base training.

## Outputs

The harness creates:

```text
metrics/rft_matched/
  dense.json
  ordinary_sparse.json
  dct_sparse.json
  rft_sparse.json
  matched_results.json
  matched_results.csv
  matched_results.md
```

## Commands

Smoke test:

```bash
python scripts/benchmark_rft_matched.py \
  --tiny --steps 20 --seq 48 --batch 2 --eval-batches 2
```

CPU comparison:

```bash
python scripts/benchmark_rft_matched.py \
  --steps 400 --seq 96 --batch 8 --eval-batches 16 \
  --fan-in 8 --device cpu --out-dir metrics/rft_matched_cpu
```

CUDA comparison:

```bash
python scripts/benchmark_rft_matched.py \
  --steps 1000 --seq 256 --batch 16 --eval-batches 32 \
  --fan-in 8 --device cuda --out-dir metrics/rft_matched_cuda
```

## Decision table

After a run, interpret results using this table:

| Claim | Passing condition |
|---|---|
| Lower peak memory | RFT peak RSS/CUDA memory is below both dense and ordinary sparse controls |
| Faster training | RFT tokens/s exceeds controls without worse numerical stability |
| Dense-quality match | RFT validation BPB is within the predeclared tolerance of dense |
| RFT-specific advantage | RFT beats ordinary sparse and DCT sparse at the same active-edge budget |
| Better conversation | Human review after matched SFT prefers RFT outputs; proxy metrics alone are insufficient |

No claim should be marked proven from a single seed. A publishable comparison
should use at least three seeds and report mean, standard deviation, and raw
per-seed results.