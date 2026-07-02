"""microtorch — a from-scratch tensor + reverse-mode autograd engine for CFNA.

A minimal, dependency-light (NumPy only) automatic-differentiation framework that
implements the machinery CFNA needs, hand-built from first principles:

- :class:`~cfna.microtorch.tensor.Tensor` — an n-d array with broadcasting-correct
  reverse-mode autodiff (``.backward()``).
- :mod:`~cfna.microtorch.functional` — softmax (masked), cross-entropy, gelu/silu,
  causal conv1d — the ops the architecture's operators are built from.
- :mod:`~cfna.microtorch.nn` — Module/Parameter plus Linear, Embedding, RMSNorm,
  MLP/GatedMLP, hand-written attention, and a selective state-space scan, so the
  engine demonstrably *serves the CFNA architecture and its optimizations*.
- :mod:`~cfna.microtorch.optim` — SGD, AdamW, grad-norm clipping.

It is verified two ways (see ``tests/test_microtorch.py``): finite-difference
gradient checks, and parity with PyTorch on the same random inputs. It trades
speed for transparency — every gradient is code you can read.

:mod:`cfna.microtorch.cfna_model` builds on this engine to port the *full*
CFNA architecture (not just the ``MicroByteLM`` demo below) — see
``tests/test_microtorch_cfna_model.py``.
"""

from .tensor import Tensor, tensor, no_grad
from . import functional, nn, optim

__all__ = ["Tensor", "tensor", "no_grad", "functional", "nn", "optim"]
