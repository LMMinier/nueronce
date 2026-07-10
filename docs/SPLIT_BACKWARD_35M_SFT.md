# Split-Run 35M SFT Workflow

The 35M MicroTorch SFT runner can divide one optimization step into two separate operating-system processes.

## Why the graph is rebuilt

MicroTorch autograd nodes contain Python backward closures. Those closures are not a stable checkpoint format and should not be serialized. The prepare process therefore stores the exact deterministic training inputs and checkpoint identity, not the live graph. The backward process verifies the same checkpoint and reconstructs the graph from the saved bytes before running backward.

This gives backward its own uninterrupted process budget while preserving mathematical correctness.

## 1. Prepare a direct-response step

```bash
python scripts/train_microtorch_35m_split_step.py prepare \
  --phase direct \
  --checkpoint checkpoints/microtorch_base35m.pkl \
  --plan checkpoints/pending_step.pkl \
  --seq-len 9 \
  --lr 1e-5 \
  --max-grad-norm 1.0
```

The prepare run:

- hashes the current checkpoint,
- selects the exact phase record,
- stores the byte sequence and response-only target span,
- evaluates a no-graph forward loss,
- atomically writes the pending plan,
- exits without changing model parameters.

## 2. Run backward in its own process

```bash
python scripts/train_microtorch_35m_split_step.py backward \
  --plan checkpoints/pending_step.pkl
```

The backward run:

- verifies that the source checkpoint hash is unchanged,
- rebuilds the exact forward graph,
- verifies that reconstructed loss matches prepared loss,
- performs backward,
- rejects nonfinite gradients before mutation,
- clips the global gradient norm,
- applies the factorized StreamFactor update,
- atomically replaces the model checkpoint,
- marks the step plan completed.

## Safety properties

A plan cannot be safely applied twice because its status becomes `completed` after a successful update. A stale plan is rejected when the checkpoint hash or step differs from the prepared source. An interrupted backward run leaves the previous checkpoint intact because replacement occurs only after a complete update and successful temporary-file flush.

## Phase rotation

Recommended initial rotation:

```text
4 direct steps
2 grounded steps
2 verification steps
validation
repeat
```

Create a new plan after every completed backward run. Do not reuse an old prepared plan after the checkpoint advances.

## Important limitation

Splitting the process does not reduce the arithmetic required by backward. It gives backward a dedicated runtime window and removes data-selection/checkpoint-planning work from that process. Further speed gains require native fused kernels, more efficient block-local backward, or faster hardware.
