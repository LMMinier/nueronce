"""Configuration defaults for the CFNA-350M prototype.

These values are *implementation defaults for a minimal reproducible prototype*,
not claims that the source notes pinned them down. Where the notes are explicit
(dynamic units, hybrid blocks, typed memory, operating modes, latent workspace,
verifier loop) the structure reflects that; the exact dimensionality is a
default and is marked as such in the design doc's assumptions table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

# Operating modes drive logical recurrent depth and reasoning effort.
Mode = str  # "FAST" | "DELIBERATE" | "RESEARCH"
MODES: Tuple[str, ...] = ("FAST", "DELIBERATE", "RESEARCH")


@dataclass
class PerceptionConfig:
    byte_embed_dim: int = 128
    d_local: int = 512
    min_patch: int = 4
    max_patch: int = 128
    boundary_tau: float = 0.5
    # boundary score = w_learned*sigmoid(logit) + w_shift*||Δfeat|| + w_syntax*spike
    w_learned: float = 0.5
    w_shift: float = 0.3
    w_syntax: float = 0.2


@dataclass
class EmbeddingConfig:
    d_sem: int = 768
    d_struct: int = 256
    d_hier: int = 128
    d_time: int = 64
    d_prov: int = 64
    belief_classes: int = 3  # support / contradict / unresolved


@dataclass
class MemoryConfig:
    # Per-channel recurrent state width.
    channel_dim: int = 256
    # Default retention timescales (lambda) per typed channel.
    retention: Dict[str, float] = field(
        default_factory=lambda: {
            "sem": 0.98,
            "str": 0.97,
            "goal": 0.95,
            "evid": 0.99,
            "unc": 0.90,
            "auth": 0.999,
            "proc": 0.98,
        }
    )
    authority_write_threshold: float = 0.5


@dataclass
class CoreConfig:
    d_model: int = 1536
    physical_blocks: int = 6
    logical_depth: Dict[str, int] = field(
        default_factory=lambda: {"FAST": 4, "DELIBERATE": 12, "RESEARCH": 24}
    )
    n_local_heads: int = 8
    local_window: int = 256
    sparse_global_heads: int = 8
    sparse_global_topk: int = 64
    ffn_mult: int = 4


@dataclass
class RetrievalConfig:
    dense_dim: int = 768
    k_recall: int = 50
    k_rerank: int = 10
    # Combined-score weights (must mirror cfna.retrieval.combine_scores).
    w_dense: float = 0.30
    w_sparse: float = 0.20
    w_late: float = 0.25
    w_temporal: float = 0.10
    w_provenance: float = 0.10
    w_contradiction: float = 0.05  # subtracted


@dataclass
class WorkspaceConfig:
    n_slots: int = 32
    reasoning_steps: Dict[str, int] = field(
        default_factory=lambda: {"FAST": 1, "DELIBERATE": 6, "RESEARCH": 16}
    )


@dataclass
class VerifierConfig:
    max_revision_rounds: int = 3
    fast_max_rounds: int = 1
    unsupported_claim_threshold: float = 0.5
    blocking_severity: float = 0.8
    calibration_tolerance: float = 0.15


@dataclass
class AdapterConfig:
    lora_rank: int = 16
    lora_alpha: float = 16.0
    quantization_bits: int = 8  # base weights; activations in bf16/fp16


@dataclass
class CFNAConfig:
    """Top-level config aggregating all subsystem defaults (CFNA-350M)."""

    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    core: CoreConfig = field(default_factory=CoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)

    # Approximate parameter budget by subsystem (millions), for bookkeeping.
    param_budget_m: Dict[str, float] = field(
        default_factory=lambda: {
            "perception": 20.0,
            "patcher": 5.0,
            "shared_core": 280.0,
            "heads_planners_verifier": 35.0,
            "embeddings_output": 10.0,
        }
    )

    @property
    def total_param_budget_m(self) -> float:
        return float(sum(self.param_budget_m.values()))


DEFAULT_CONFIG = CFNAConfig()


__all__ = [
    "Mode",
    "MODES",
    "PerceptionConfig",
    "EmbeddingConfig",
    "MemoryConfig",
    "CoreConfig",
    "RetrievalConfig",
    "WorkspaceConfig",
    "VerifierConfig",
    "AdapterConfig",
    "CFNAConfig",
    "DEFAULT_CONFIG",
]
