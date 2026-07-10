"""Tests for the mixed conversational SFT layer (nueronce.training.mixed_sft).

Torch-free: these pin the data-side invariants the desktop/Colab torch loop
relies on — canonical rendering, the response-only mask contract, register
caps, and batch packing."""

import numpy as np

from nueronce.prompting import ASSISTANT, END, USER
from nueronce.training.mixed_sft import (
    build_batches,
    dataset_summary,
    enforce_category_caps,
    render_record,
    stride_sample,
)


def _plain(i=0):
    return {"id": f"p-{i}", "source": "t", "category": "plain",
            "messages": [{"role": "user", "content": f"Question {i}?"},
                         {"role": "assistant", "content": f"Answer {i}."}]}


def _evidence(i=0):
    return {"id": f"e-{i}", "source": "t", "category": "grounded",
            "messages": [{"role": "user", "content": f"When does dock {i} close?"},
                         {"role": "assistant", "content": f"Dock {i} closes at 18:00."}],
            "system_message": "You are NUERONCE.",
            "trusted_evidence": [f"[doc] Dock {i} closes at 18:00."],
            "response_plan": ["Use decisive evidence only."]}


def test_render_plain_is_canonical_with_response_mask():
    b, m = render_record(_plain())
    text = b.decode("utf-8")
    assert USER in text and ASSISTANT in text and END in text
    assert len(b) == len(m)
    masked = bytes(v for v, keep in zip(b, m) if keep).decode("utf-8")
    assert "Answer 0." in masked and END in masked
    assert "Question 0?" not in masked  # prompt bytes are never loss targets


def test_render_evidence_record_includes_blocks():
    b, m = render_record(_evidence())
    text = b.decode("utf-8")
    assert "<|evidence|>" in text and "<|plan|>" in text and "You are NUERONCE." in text
    masked = bytes(v for v, keep in zip(b, m) if keep).decode("utf-8")
    assert masked.startswith("Dock 0 closes")
    assert "decisive evidence" not in masked  # plan/evidence stay unmasked


def test_enforce_category_caps():
    records = [dict(_plain(i), category="spam") for i in range(300)]
    records += [dict(_plain(1000 + i), category=f"c{i % 5}") for i in range(100)]
    capped = enforce_category_caps(records, cap_frac=0.25, seed=0)
    s = dataset_summary(capped)
    assert s["max_category_frac"] <= 0.2501
    # non-offending categories are untouched
    assert sum(1 for r in capped if r["category"].startswith("c")) == 100


def test_stride_sample_spreads_across_stream():
    picked = list(stride_sample(iter(_plain(i) for i in range(1000)), 10, 1000))
    ids = [int(r["id"].split("-")[1]) for r in picked]
    assert len(picked) == 10
    assert max(ids) > 800  # not a prefix slice


def test_build_batches_shapes_and_padding():
    records = [_plain(i) for i in range(10)] + [_evidence(i) for i in range(6)]
    batches = list(build_batches(records, batch_size=4, max_len=400, seed=0))
    assert sum(b["byte_ids"].shape[0] for b in batches) == 16
    for b in batches:
        ids, mask = b["byte_ids"], b["target_mask"]
        assert ids.dtype == np.int64 and mask.dtype == bool
        assert ids.shape == mask.shape
        assert not mask[ids == 0].any()  # padding is never a target


def test_build_batches_full_loss_targets_all_real_bytes():
    records = [_plain(i) for i in range(4)]
    (batch,) = list(build_batches(records, batch_size=4, max_len=400, seed=0, loss="full"))
    ids, mask = batch["byte_ids"], batch["target_mask"]
    assert bool(mask[ids > 0].all()) and not bool(mask[ids == 0].any())


def test_build_batches_skips_overlong_instead_of_truncating():
    long_rec = _plain(0)
    long_rec["messages"][1]["content"] = "x" * 500
    batches = list(build_batches([long_rec, _plain(1)], batch_size=2, max_len=200, seed=0))
    assert sum(b["byte_ids"].shape[0] for b in batches) == 1
