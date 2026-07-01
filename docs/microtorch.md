# microtorch — a from-scratch autograd engine for CFNA

`cfna/microtorch/` is a minimal, NumPy-only automatic-differentiation framework —
"my own PyTorch" — built to serve the CFNA architecture from first principles. No
PyTorch, no autograd libraries: every gradient is hand-derived code you can read.

## Why

The main CFNA stack uses PyTorch purely as a tensor/autograd substrate. This
package answers the deeper question — *what is that substrate?* — by implementing
it: tensors, reverse-mode backprop, the architecture's operators, and optimizers.
It is for **correctness, portability, and understanding**, not speed (pure-NumPy
autograd is far slower than PyTorch, so it is not used to train the 11M model).

## What's in it

| Module | Contents |
|---|---|
| `tensor.py` | `Tensor` with broadcasting-correct reverse-mode autograd: `+ - * / ** @`, `sum/mean`, `exp/log/tanh/relu`, `reshape/transpose`, gather (`__getitem__`), `cat/stack`, `backward()`, `no_grad()` |
| `functional.py` | `sigmoid`, `silu`, `gelu`, `softmax`, `masked_softmax`, `cross_entropy`, `conv1d_causal` — all composed from primitives (no hand-written backward) |
| `nn.py` | `Module`/`Parameter`, `Linear`, `Embedding`, `RMSNorm`, `MLP`, `GatedMLP`, `MultiHeadAttention` (causal + windowed), `SelectiveSSM` (selective scan) |
| `optim.py` | `SGD`, `AdamW`, `clip_grad_norm_` |
| `models.py` | `MicroByteLM` (a hybrid attention+SSM byte LM) + `train_overfit` |

## How it works

Each op records a closure that pushes its output's gradient to its inputs. A
topological sort from the loss runs those closures in reverse (`Tensor.backward`).
Broadcasting is handled by summing a gradient back down to its input's shape
(`_unbroadcast`). Because `functional`/`nn` are *composed* from primitives, the
engine differentiates them automatically — the fact that attention, the SSM scan,
and causal convolution all train is itself evidence the primitive set is complete.

## Verification (not stubs)

`tests/test_microtorch.py`:
- **Finite-difference gradient checks** on every op (max error ~1e-8).
- **PyTorch parity**: identical gradients to `torch` on the same inputs (~1e-8).
- **Causality**: `conv1d_causal` output at `t` is independent of inputs `> t`.
- **Optimizers**: AdamW minimizes a quadratic; grad-norm clipping is exact.
- **End-to-end**: `MicroByteLM` overfits a string (loss 6.0 → ~0.001).

## Run it

```bash
python scripts/microtorch_demo.py
```

```python
import numpy as np
from cfna.microtorch import Tensor, functional as F, nn, optim

x = Tensor(np.random.randn(2, 6, 16), requires_grad=True)
attn = nn.MultiHeadAttention(16, n_heads=4, window=3)
y = attn(x); y.sum().backward()          # gradients flow into attn.parameters()
```

## Relationship to the main stack

`cfna/nn.py` (PyTorch primitives) and `cfna/microtorch/nn.py` intentionally mirror
each other. The PyTorch path is for real training speed; microtorch is the
transparent reference implementation of the same operators. Bringing the full
CFNA model onto microtorch would require the performance work listed in
`docs/STATUS.md` (fused scans, real sparse compute); it is not a goal today.
