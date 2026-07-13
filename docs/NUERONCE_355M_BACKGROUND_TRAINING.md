# NUERONCE 355M unattended base → conversation training

This workflow runs on the machine that owns the corpus and checkpoint. ChatGPT
cannot keep a compute process alive after a response ends; the repository uses a
detached local process instead.

## What the controller does

1. verifies the source checkpoint and records its SHA-256;
2. resumes 352,993,825-parameter base pretraining in atomic chunks;
3. evaluates held-out bits per byte after each chunk;
4. enters conversational SFT only after held-out BPB reaches the configured gate;
5. initializes SFT from the converged base weights, not random weights;
6. trains response-only conversational loss with StreamFactor;
7. keeps `best.pt` by validation loss and stops on a measured validation plateau;
8. evaluates the selected checkpoint and runs a six-prompt deterministic chat probe;
9. writes a restart-safe `state.json`, checkpoints, metrics, PID metadata, and log.

The default position mode is the opt-in `phi_rope` RFT attention ablation. Use
`--position-mode baseline` for the original attention geometry.

## Start on Windows

Run this as one line in PowerShell from the repository root:

```powershell
python scripts/nueronce_355m_job.py start -- --source-checkpoint checkpoints/nueronce_engine_355m_protocol_step270/source_step270.pkl --corpus corpus --position-mode phi_rope --build-sft-if-missing --base-target-bpb 1.8 --base-max-steps 50000 --sft-min-steps 1000 --sft-max-steps 50000
```

## Start on Linux, macOS, or a Colab shell

```bash
python scripts/nueronce_355m_job.py start -- \
  --source-checkpoint checkpoints/nueronce_engine_355m_protocol_step270/source_step270.pkl \
  --corpus corpus \
  --position-mode phi_rope \
  --build-sft-if-missing \
  --base-target-bpb 1.8 \
  --base-max-steps 50000 \
  --sft-min-steps 1000 \
  --sft-max-steps 50000
```

The launcher returns immediately after creating a detached process. Training
continues only while that computer/runtime remains powered on. Colab disconnects
can still stop the process; rerunning the same command resumes from atomic
checkpoints.

## Status

```bash
python scripts/nueronce_355m_job.py status
```

Default outputs:

```text
runs/nueronce_355m_convergence/job.json
runs/nueronce_355m_convergence/state.json
runs/nueronce_355m_convergence/training.log
runs/nueronce_355m_convergence/base/checkpoints/latest.pkl
runs/nueronce_355m_convergence/base/metrics/base_metrics.jsonl
runs/nueronce_355m_convergence/conversation/checkpoints/latest.pt
runs/nueronce_355m_convergence/conversation/checkpoints/best.pt
runs/nueronce_355m_convergence/conversation/metrics/conversation_metrics.jsonl
runs/nueronce_355m_convergence/conversation/metrics/conversation_summary.json
runs/nueronce_355m_convergence/conversation/metrics/chat_probe.json
```

## Convergence rules

Base training does not silently proceed into SFT merely because it ran for a
long time. It must reach `--base-target-bpb` (default `1.8`). If held-out BPB
fails to improve by `--base-min-delta` for `--base-patience` checks, the
controller stops and records that the language gate failed.

Conversational training stops when validation loss fails to improve by
`--sft-min-delta` for `--sft-patience` evaluations after `--sft-min-steps`.
`--sft-max-steps` remains a hard safety bound. A plateau is evidence that the
current data/protocol has stopped improving, not proof of general intelligence.

## Resource warning

The 355M Nueronce Engine path uses float32 parameters and StreamFactor state.
Checkpoint serialization temporarily needs additional RAM and several gigabytes
of free disk. Start with sequence length 16 and batch size 1. Do not raise batch
or context length until a one-step smoke run and checkpoint save both succeed.
