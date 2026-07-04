# CFNA CPU-First Streaming Training

## Purpose

This pipeline is the first implementation aimed directly at CFNA's original
hardware thesis: train by streaming small byte windows on CPU, retain no
full-corpus or full-sequence graph, and update only one active subsystem at a
time.

It is an experimental training algorithm, not a claim that local learning has
already matched end-to-end backpropagation.

## Training schedule

The trainer rotates through four local stages:

1. **Perception** — the causal byte encoder predicts the next byte and learned
   boundaries. Only the perception encoder and a small local head receive
   gradients.
2. **Units** — frozen byte features are pooled through the model's real dynamic
   segmentation. The unit embedder, boundary projection, and recurrent memory
   predict the byte distribution following each unit.
3. **Core** — frozen unit states feed the real routed hybrid core. Only the core
   and its local prediction head receive gradients.
4. **Decoder** — the upstream stack is produced under `no_grad`; only the byte
   decoder is trained with ordinary next-byte cross entropy.

After `steps_per_stage`, the next subsystem becomes active. No stage holds a
backward graph through the entire CFNA model.

## Memory contract

For every step:

- device is CPU;
- input length is hard-capped by `chunk_size`;
- corpus files are read incrementally;
- frozen stages execute under `torch.no_grad()`;
- only active parameters allocate gradients;
- plain SGD allocates no momentum or Adam state;
- the computation graph is discarded after the local update.

The trainer reports:

- total model parameters;
- active parameters for the current stage;
- estimated fp32 gradient bytes;
- optimizer-state bytes;
- current chunk bound.

This bounds activation memory by the active window and active subsystem. Weight
memory still scales with total parameters. Very large models will therefore
need parameter paging, quantized frozen weights, or SSD/RAM block loading as a
later layer.

## Run

```bash
python scripts/train_cpu_streaming.py corpus/ books/ \
  --steps 10000 \
  --chunk-size 128 \
  --batch-size 1 \
  --steps-per-stage 32 \
  --checkpoint checkpoints/cfna_cpu_streaming.pt
```

CUDA is ignored even when present.

## Scientific comparison required

The pipeline must be evaluated against standard end-to-end CFNA and a matched
Transformer using identical data and CPU limits. Record:

- peak resident RAM;
- bytes processed per second;
- wall-clock time;
- held-out bits per byte;
- active/total parameter ratio;
- energy where available;
- quality per GB of peak memory.

The central gate is not merely whether loss decreases. It is whether local
streaming CFNA reaches useful held-out quality at substantially lower peak
training memory.

## Current limitations

- Local objectives are proxies for global language-model loss.
- The stages are block-coordinate rather than simultaneous.
- Total model weights remain resident in system RAM.
- No quantized weight paging or SSD offload is implemented yet.
- No benchmark has yet established equal quality versus full backpropagation.
- Retrieval and provenance objectives are not included in this first trainer.

These limitations are intentional and explicit. This commit creates the
runnable experimental foundation needed to test and improve the non-VRAM
training thesis instead of continuing to assume it.
