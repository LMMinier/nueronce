"""engine — a from-scratch tensor + reverse-mode autograd engine for NUERONCE.

A minimal, dependency-light (NumPy only) automatic-differentiation framework that
implements the machinery NUERONCE needs, hand-built from first principles:

- :class:`~nueronce.engine.tensor.Tensor` — an n-d array with broadcasting-correct
  reverse-mode autodiff (``.backward()``).
- :mod:`~nueronce.engine.functional` — softmax (masked), cross-entropy, gelu/silu,
  causal conv1d — the ops the architecture's operators are built from.
- :mod:`~nueronce.engine.nn` — Module/Parameter plus Linear, Embedding, RMSNorm,
  MLP/GatedMLP, hand-written attention, and a selective state-space scan, so the
  engine demonstrably *serves the NUERONCE architecture and its optimizations*.
- :mod:`~nueronce.engine.optim` — SGD, AdamW, grad-norm clipping.

It is verified two ways (see ``tests/test_nueronce_engine.py``): finite-difference
gradient checks, and parity with PyTorch on the same random inputs. It trades
speed for transparency — every gradient is code you can read.

:mod:`nueronce.engine.nueronce_model` builds on this engine to port the *full*
NUERONCE architecture (not just the ``MicroByteLM`` demo below) — see
``tests/test_nueronce_engine_nueronce_model.py``.
"""

from .tensor import Tensor, tensor, no_grad
from . import functional, nn, optim

__all__ = ["Tensor", "tensor", "no_grad", "functional", "nn", "optim"]
