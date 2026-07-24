"""Falsifiable tests for entropy-driven adaptive patching.

Every test includes the negative control that proves the check can actually
fail — an entropy patcher that fired everywhere (or nowhere) would be useless,
so we prove it fires exactly where surprise is and nowhere else.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from nueronce.adaptive_patching import (
    ByteEntropyHead,
    entropy_boundary_targets,
    patch_stats,
    predictive_entropy,
)


def _sharp(idx: int) -> torch.Tensor:
    v = torch.full((256,), -30.0)
    v[idx] = 30.0
    return v  # near-zero entropy


def _flat() -> torch.Tensor:
    return torch.zeros(256)  # uniform -> max entropy ln256


def test_predictive_entropy_bounds():
    logits = torch.stack([_sharp(0), _flat()])[None]  # [1, 2, 256]
    H = predictive_entropy(logits)[0]
    assert H[0] < 0.01                       # sharp -> ~0
    assert abs(H[1] - math.log(256)) < 0.01  # flat -> ln256


def test_boundaries_fire_exactly_at_surprise_onsets():
    # predicting-distributions per position; H high at positions 2 and 4.
    # surprise of byte i = H[i-1], so boundaries should land at i=3 and i=5.
    rows = [_sharp(1), _sharp(2), _flat(), _sharp(3), _flat(), _sharp(4)]
    logits = torch.stack(rows)[None]  # [1, 6, 256]
    tgt = entropy_boundary_targets(logits, mode="global", global_theta=2.5)[0]
    fired = set(torch.nonzero(tgt).flatten().tolist())
    assert fired == {3, 5}, f"expected boundaries at 3,5; got {sorted(fired)}"


def test_negative_control_all_predictable_gives_no_boundaries():
    # every position sharp -> no surprise -> no informational boundaries.
    logits = torch.stack([_sharp(i % 256) for i in range(8)])[None]
    tgt = entropy_boundary_targets(logits, mode="global", global_theta=2.5)[0]
    assert tgt.sum() == 0


def test_negative_control_all_surprising_gives_boundaries_everywhere():
    # every position flat -> every byte (i>=1) is a fresh boundary.
    logits = torch.stack([_flat() for _ in range(8)])[None]
    tgt = entropy_boundary_targets(logits, mode="global", global_theta=2.5)[0]
    assert tgt[1:].sum() == 7 and tgt[0] == 0


def test_relative_mode_fires_on_rising_entropy_only():
    # H profile: low, low, high, high -> rise happens once (low->high), so the
    # relative rule fires at the byte after the jump, not on the sustained high.
    rows = [_sharp(1), _sharp(2), _flat(), _flat(), _sharp(3)]
    logits = torch.stack(rows)[None]  # H = [~0,~0,hi,hi,~0]
    tgt = entropy_boundary_targets(logits, mode="relative", relative_theta=1.0)[0]
    fired = set(torch.nonzero(tgt).flatten().tolist())
    # surprise(byte i)=H[i-1]: rise at i=3 (H2-H1 = hi-0). i=4 (H3-H2=0) no. i=5 negative.
    assert fired == {3}, f"expected only the onset boundary at 3; got {sorted(fired)}"


def test_patch_stats_counts_correctly():
    # 8 bytes, 4 units -> 2 bytes/unit, 0.5 units/byte.
    seg = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3]])
    stats = patch_stats(seg)
    assert abs(stats["bytes_per_unit"] - 2.0) < 1e-6
    assert abs(stats["units_per_byte"] - 0.5) < 1e-6


def test_entropy_head_produces_useful_entropy_after_overfit():
    """The honest-risk guard: is a *cheap* head good enough that its entropy is
    actually informative? Overfit a tiny head on a periodic pattern and assert
    it is meaningfully less surprised by the pattern than by novel bytes."""
    torch.manual_seed(0)
    head = ByteEntropyHead(dim=48, kernel=5, layers=2)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-3)
    pattern = torch.tensor([[ord("a"), ord("b")] * 32])  # "abab..." length 64
    for _ in range(300):
        opt.zero_grad()
        loss = head.lm_loss(pattern)
        loss.backward()
        opt.step()

    head.eval()
    with torch.no_grad():
        # entropy continuing the learned pattern vs. an unlearned novel run
        learned = torch.tensor([[ord("a"), ord("b")] * 8])
        novel = torch.tensor([[ord("z"), ord("q")] * 8])
        H_learned = predictive_entropy(head(learned)).mean().item()
        H_novel = predictive_entropy(head(novel)).mean().item()

    assert loss.item() < 0.5, f"head failed to overfit the pattern (loss {loss.item():.3f})"
    assert H_learned < H_novel - 0.3, (
        f"entropy not informative: learned {H_learned:.3f} vs novel {H_novel:.3f}"
    )
