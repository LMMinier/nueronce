"""Verifier-Guided Residual Fine-Tuning (VGRFT) and continual learning.

Post-foundation phase: structured instruction tuning, tool-grounded tuning,
verifier training, and residual correction experts that patch specific failure
categories instead of rewriting the whole model. Also includes the controlled
continual-learning loop (index first → episodic staging → scheduled adapter
updates with regression gating and rollback) and counterfactual/disagreement
training generators.

These are training drivers whose inner steps (forward passes, backprop) require a
backend; the structure and control flow (including the rollback gate) are laid out
explicitly. Methods raise NotImplementedError where a backend/optimizer is needed.

Stage 1 (``supervised_instruction_tune``) is real when given a backend that
implements ``.train(dataset, **kwargs)`` — e.g.
:class:`nueronce.training.sft.TorchSFTBackend` (PyTorch, fine-tunes the real
``NUERONCEModel``) or :class:`nueronce.engine.models.MicroSFTBackend` (the
from-scratch NumPy-only autograd engine). Either runs an actual masked-loss SFT
pass over (prompt, response) turns from :mod:`nueronce.training.dialogue_data`.
Stages 2-4 still need tool traces / verifier ground truth that don't exist
yet, so they remain stubs.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional


class VGRFTTrainer:
    """Drives the four VGRFT sub-stages. Inject a ``backend`` exposing
    forward/loss/backprop primitives to make the steps runnable.
    """

    def __init__(self, backend: Optional[Any] = None):
        self.backend = backend

    def _require_backend(self, stage: str):
        if self.backend is None:
            raise NotImplementedError(
                f"VGRFTTrainer.{stage} needs a training backend "
                f"(forward_task/propose_action/generate + loss + backprop)."
            )
        return self.backend

    def supervised_instruction_tune(self, dataset, **train_kwargs) -> List[dict]:
        """Run SFT over (prompt, response) turns, e.g.
        ``nueronce.training.dialogue_data.SFT_DATASET``.

        Requires a backend implementing ``.train(dataset, **kwargs) -> history``
        — see ``nueronce.training.sft.TorchSFTBackend`` (PyTorch) or
        ``nueronce.engine.models.MicroSFTBackend`` (from-scratch, NumPy-only).
        Deliberately backend-agnostic: this module never imports torch or
        numpy itself, only whichever autograd engine the backend brings.
        """
        backend = self._require_backend("supervised_instruction_tune")
        train_fn = getattr(backend, "train", None)
        if train_fn is None:
            raise NotImplementedError(
                "supervised_instruction_tune needs a backend with a "
                ".train(dataset, **kwargs) method; see nueronce.training.sft.TorchSFTBackend "
                "or nueronce.engine.models.MicroSFTBackend."
            )
        return train_fn(dataset, **train_kwargs)

    def tool_grounded_tune(self, tool_dataset) -> None:
        self._require_backend("tool_grounded_tune")
        raise NotImplementedError(
            "Per trace step: propose_action -> action_loss; execute/replay tool -> "
            "update_state_with_observation -> observation_read_loss; backprop sum."
        )

    def verifier_train(self, examples) -> None:
        self._require_backend("verifier_train")
        raise NotImplementedError(
            "Per example: verifier.verify(candidate, plan, evidence, tools); "
            "loss = verifier_report_loss(report, target_failures); backprop."
        )

    def residual_expert_train(self, examples) -> None:
        self._require_backend("residual_expert_train")
        raise NotImplementedError(
            "Per example: generate candidate; verify; for each failure, route to a "
            "residual expert that proposes a patch; loss vs gold_patch; backprop (H8)."
        )


# --------------------------------------------------------------------------- #
# Continual learning (control flow is real; the steps are injected)
# --------------------------------------------------------------------------- #

class ContinualLearner:
    """Index first, stage episodic, then *scheduled* adapter updates with a
    regression gate and rollback — not unbounded base-weight rewriting.
    """

    def __init__(
        self,
        ingest_and_validate: Callable[[Any], Optional[Any]],
        parse_source: Callable[[Any], Any],
        compile_units: Callable[[Any], list],
        add_to_indexes: Callable[[list], None],
        stage_as_episodic: Callable[[Any], None],
        enough_verified_new_data: Callable[[], bool],
        train_small_adapters: Callable[[], None],
        run_regression_suite: Callable[[], None],
        regression_passes: Callable[[], bool],
        promote_adapter_version: Callable[[], None],
        rollback_adapter_version: Callable[[], None],
    ):
        self._ingest = ingest_and_validate
        self._parse = parse_source
        self._compile = compile_units
        self._index = add_to_indexes
        self._stage = stage_as_episodic
        self._enough = enough_verified_new_data
        self._train = train_small_adapters
        self._regress = run_regression_suite
        self._passes = regression_passes
        self._promote = promote_adapter_version
        self._rollback = rollback_adapter_version

    def update(self, new_sources: List[Any]) -> str:
        accepted = [r for r in (self._ingest(s) for s in new_sources) if r is not None]

        # retrieval/index first
        for rec in accepted:
            self._index(self._compile(self._parse(rec)))

        # episodic staging only
        for rec in accepted:
            self._stage(rec)

        # scheduled adapter updates, not immediate base-model rewrite
        if self._enough():
            self._train()
            self._regress()
            if self._passes():
                self._promote()
                return "promoted"
            self._rollback()
            return "rolled_back"
        return "staged"


__all__ = ["VGRFTTrainer", "ContinualLearner"]
