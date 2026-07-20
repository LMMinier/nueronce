# NUERONCE RFT Integration — Stage 1

This branch begins the RFT-native neural-model experiment without changing the default NUERONCE architecture or invalidating existing checkpoints.

## Implemented

### Canonical differentiable RFT

`nueronce.rft.CanonicalRFT` constructs the canonical unitary basis

\[
U = \Phi(\Phi^\dagger\Phi)^{-1/2}
\]

from a golden-ratio-spaced complex dictionary. Analysis and synthesis are

\[
z = U^\dagger x, \qquad x = Uz.
\]

The basis is fixed and stored as a non-trainable buffer. Large model dimensions are divided into smaller RFT blocks so basis construction does not require a large eigendecomposition for every layer.

### Sparse spectral linear operator

`RFTSparseLinear` implements

\[
y = \operatorname{Re}(U_{out} S U_{in}^\dagger x) + b.
\]

Only active entries of `S` are trainable. Inactive dense weights, gradients, and optimizer moments are not allocated. Connectivity is currently deterministic and phi-strided, with a fixed `fan_in` budget per output channel.

### Sparse RFT SwiGLU

`RFTGatedMLP` replaces the three dense projections in the standard NUERONCE feed-forward path:

```text
up   : RFTSparseLinear(dim, hidden)
gate : RFTSparseLinear(dim, hidden)
down : RFTSparseLinear(hidden, dim)
```

### Opt-in model integration

Existing models remain unchanged.

```python
from nueronce.model import ModelConfig
from nueronce.rft_model import RFTNUERONCEModel

model = RFTNUERONCEModel(
    ModelConfig(),
    rft_fan_in=8,
    rft_block_size=64,
    rft_ffn_mult=3,
)
```

An existing model can also be converted in place:

```python
from nueronce.rft_model import enable_rft_ffn

enable_rft_ffn(model, fan_in=8, block_size=64)
```

Only the hybrid-core FFNs are replaced. Byte perception, segmentation, typed memory, SSM, local attention, sparse global attention, retrieval, byte decoder, and output head remain conventional. This isolates the RFT variable for fair experiments.

## Validation completed locally

The focused primitive suite passed:

```text
5 passed in 28.40s
```

It verified:

- canonical-basis unitarity;
- exact analysis/synthesis reconstruction within floating-point tolerance;
- norm preservation;
- gradients through complex sparse spectral weights;
- optimizer learning on a small regression problem;
- lower trainable-scalar count than the matched dense FFN.

Repository integration tests are included in `tests/test_rft.py` and should run in the full NUERONCE environment.

## What this does not prove

This first stage does not yet prove that RFT sparsity improves language-model quality, speed, or memory in a full training run. It establishes a trainable implementation and an isolated integration point.

## Required next experiment

Train matched models using the same corpus and training budget:

1. standard dense NUERONCE;
2. ordinary sparse NUERONCE;
3. DCT-coordinate sparse NUERONCE;
4. canonical-RFT sparse NUERONCE.

Record:

- active parameters;
- optimizer-state bytes;
- peak RAM/VRAM;
- step time and tokens per second;
- held-out bits per byte;
- generation probes;
- gradient norms and numerical failures.

The RFT-specific hypothesis is supported only if it beats ordinary sparsity or another fixed orthogonal basis at the same active parameter and memory budget.
