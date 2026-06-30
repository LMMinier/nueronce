"""Pytest setup: pin thread counts so the suite is fast and deterministic.

Without this, torch/OMP spin up many threads and the small CPU tensors in these
tests run slower, not faster (oversubscription). Set before torch does any work.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

try:
    import torch

    torch.manual_seed(0)
    torch.set_num_threads(1)
except Exception:  # torch optional for the pure-logic tests
    pass
