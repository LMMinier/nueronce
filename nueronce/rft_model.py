"""Opt-in NUERONCE integration for sparse canonical-RFT feed-forward layers.

Existing NUERONCE checkpoints remain compatible because the base model is not
modified. Construct :class:`RFTNUERONCEModel` or call :func:`enable_rft_ffn`
to replace only the hybrid core feed-forward layers.
"""
from __future__ import annotations

from typing import Any

from .model import ModelConfig, NUERONCEModel
from .rft import RFTGatedMLP


def enable_rft_ffn(
    model: NUERONCEModel,
    *,
    fan_in: int = 8,
    block_size: int = 64,
    ffn_mult: int = 3,
) -> NUERONCEModel:
    """Replace each physical hybrid block's dense FFN with an RFT sparse FFN.

    This mutates ``model`` in place. It intentionally leaves byte perception,
    typed memory, SSM, attention, decoder, and output head unchanged so the
    first experiment isolates the effect of RFT-coordinate feed-forward
    sparsity.
    """
    dim = int(model.cfg.d_model)
    hidden = int(ffn_mult * dim)
    for block in model.core.blocks:
        block.ffn = RFTGatedMLP(
            dim,
            hidden,
            fan_in=fan_in,
            block_size=block_size,
        )
    model.rft_config = {
        "enabled": True,
        "fan_in": int(fan_in),
        "block_size": int(block_size),
        "ffn_mult": int(ffn_mult),
    }
    return model


class RFTNUERONCEModel(NUERONCEModel):
    """NUERONCE with sparse canonical-RFT FFNs in the hybrid core."""

    def __init__(
        self,
        cfg: ModelConfig | None = None,
        *,
        rft_fan_in: int = 8,
        rft_block_size: int = 64,
        rft_ffn_mult: int = 3,
    ):
        super().__init__(cfg)
        enable_rft_ffn(
            self,
            fan_in=rft_fan_in,
            block_size=rft_block_size,
            ffn_mult=rft_ffn_mult,
        )

    def rft_stats(self) -> dict[str, Any]:
        layers = []
        for block_index, block in enumerate(self.core.blocks):
            ffn = block.ffn
            layers.append({
                "block": block_index,
                "active_parameters": ffn.active_parameter_count(),
                "up_density": ffn.up.spectral_density,
                "gate_density": ffn.gate.spectral_density,
                "down_density": ffn.down.spectral_density,
            })
        return {
            "config": dict(self.rft_config),
            "layers": layers,
            "total_rft_parameters": sum(
                item["active_parameters"] for item in layers
            ),
        }


__all__ = ["enable_rft_ffn", "RFTNUERONCEModel"]
