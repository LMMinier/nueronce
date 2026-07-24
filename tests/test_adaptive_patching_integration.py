"""Integration + safety tests for wiring entropy patching into NUERONCEModel.

The load-bearing guarantees (any of these failing = the live 35M run is at
risk): the default (syntax) path is byte-identical to before, an old-config
checkpoint still resumes, and the entropy path actually trains its head.
"""
import pytest

torch = pytest.importorskip("torch")

from nueronce.model import ModelConfig, NUERONCEModel


def _tiny(**kw):
    base = dict(byte_embed_dim=16, d_local=24, d_model=32, p_max=8,
                physical_blocks=2, logical_depth=2, n_heads=2, unit_window=8,
                decoder_window=8, decoder_layers=1, d_state=4, channel_dim=8,
                ret_byte_dim=8, min_patch=2, max_patch=8)
    base.update(kw)
    return ModelConfig(**base)


def test_default_mode_has_no_entropy_head_and_unchanged_params():
    """Default (syntax) model must carry ZERO new parameters — otherwise an
    existing checkpoint's state_dict would fail to load."""
    m = NUERONCEModel(_tiny())
    assert m.entropy_head is None
    names = [n for n, _ in m.named_parameters()]
    assert not any("entropy_head" in n for n in names)


def test_default_mode_loss_is_byte_identical_to_syntax_baseline():
    """The default loss path must be numerically identical to the pre-adaptive
    computation (lm + boundary_loss_weight * bnd, syntactic target)."""
    import torch.nn.functional as F
    from nueronce.segment import boundary_targets

    torch.manual_seed(0)
    m = NUERONCEModel(_tiny())
    ids = torch.randint(0, 256, (2, 48))

    total, stats = m.loss(ids)

    # recompute the legacy way and compare exactly
    logits, boundary_logits = m.forward(ids)
    lm = F.cross_entropy(logits[:, :-1].reshape(-1, 256), ids[:, 1:].reshape(-1))
    b_target = boundary_targets(ids, m._syntax)
    bnd = F.binary_cross_entropy_with_logits(boundary_logits, b_target)
    expected = lm + m.cfg.boundary_loss_weight * bnd

    assert torch.allclose(total, expected, atol=0, rtol=0), "default loss drifted from syntax baseline"
    assert "entropy_lm" not in stats


def test_old_checkpoint_config_still_resumes():
    """Simulate the exact resume guard from train_checkpoint.py: a checkpoint
    saved with the OLD (pre-adaptive) config dict must pass the shared-field
    check against the NEW config, and a real preset change must still fail."""
    cur_cfg = vars(_tiny())                          # has the 4 new fields
    old_cfg = {k: v for k, v in cur_cfg.items()
               if k not in ("boundary_target_mode", "entropy_head_dim",
                            "entropy_global_theta", "entropy_relative_theta")}

    def shared_mismatch(saved, cur):
        return {k: (saved[k], cur[k]) for k in (set(saved) & set(cur)) if saved[k] != cur[k]}

    assert shared_mismatch(old_cfg, cur_cfg) == {}, "old checkpoint should resume"

    changed = dict(old_cfg); changed["d_model"] = 999   # genuine preset change
    assert shared_mismatch(changed, cur_cfg) != {}, "a real preset change must still be refused"


def test_entropy_mode_builds_head_and_trains_it():
    """Entropy mode must construct the head, run loss, and flow gradient into
    BOTH the entropy head and the boundary head."""
    torch.manual_seed(0)
    m = NUERONCEModel(_tiny(boundary_target_mode="entropy_global"))
    assert m.entropy_head is not None
    ids = torch.randint(0, 256, (2, 48))
    total, stats = m.loss(ids)
    assert "entropy_lm" in stats
    m.zero_grad(); total.backward()
    eh = [p.grad for n, p in m.named_parameters() if "entropy_head" in n and p.grad is not None]
    bh = [p.grad for n, p in m.named_parameters() if "boundary_head" in n and p.grad is not None]
    assert eh and any(g.abs().max() > 0 for g in eh), "entropy head received no gradient"
    assert bh and any(g.abs().max() > 0 for g in bh), "boundary head received no gradient"


def test_entropy_relative_mode_also_builds_and_runs():
    torch.manual_seed(0)
    m = NUERONCEModel(_tiny(boundary_target_mode="entropy_relative"))
    ids = torch.randint(0, 256, (2, 48))
    total, _ = m.loss(ids)
    total.backward()  # must not raise
