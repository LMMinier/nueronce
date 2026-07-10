"""Large-scale MicroTorch configuration and runtime controls."""
from __future__ import annotations

from typing import Any

from .backend import DTypePolicy, set_dtype_policy, xp
from .cfna_model import MicroModelConfig
from .tensor import Tensor


def base_355m_config() -> MicroModelConfig:
    """Return the measured 352,993,825-parameter CFNA configuration."""
    return MicroModelConfig(
        byte_embed_dim=128, d_local=524, d_model=1048, p_max=64,
        physical_blocks=6, logical_depth=12, n_heads=8,
        unit_window=256, decoder_window=256, decoder_layers=6,
        d_state=16, channel_dim=64, ret_byte_dim=64,
        min_patch=4, max_patch=128,
    )


def enable_training_dtype(dtype: str = "float32") -> None:
    """Make all subsequently-created Tensor values use the selected dtype.

    This compatibility bridge lets the large-scale launcher use float32 before
    every Tensor call site has migrated to the backend policy. It patches the
    constructor once, while preserving the existing autograd behavior.
    """
    if dtype not in {"float32", "float64"}:
        raise ValueError("dtype must be float32 or float64")
    target = getattr(xp, dtype)
    set_dtype_policy(DTypePolicy(param=target, activation=target, grad=target,
                                 optimizer=xp.float32))
    if getattr(Tensor, "_dtype_policy_patched", False):
        Tensor._runtime_dtype = target
        return

    original_init = Tensor.__init__

    def policy_init(self, data: Any, requires_grad: bool = False,
                    _children=(), _op: str = ""):
        if isinstance(data, Tensor):
            converted = data.data
        else:
            converted = xp.asarray(data, dtype=Tensor._runtime_dtype)
        original_init(self, converted, requires_grad=requires_grad,
                      _children=_children, _op=_op)
        if self.data.dtype != Tensor._runtime_dtype:
            self.data = self.data.astype(Tensor._runtime_dtype, copy=False)

    Tensor._runtime_dtype = target
    Tensor.__init__ = policy_init
    Tensor._dtype_policy_patched = True


__all__ = ["base_355m_config", "enable_training_dtype"]
