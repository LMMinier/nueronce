"""Corpus pipeline: cleaning, manifest/provenance, document-uniform batching.

Network-free: the cleaning/parse functions are pure, and the dataset test builds a
tiny synthetic corpus directory (no download).
"""

import json
from pathlib import Path

import pytest

from cfna.corpus.build import clean_text, parse_header, quality_score
from cfna.corpus.sources import EXCLUDED_KINDS, safe_commercial_sources


def test_only_safe_commercial_sources_are_used():
    for s in safe_commercial_sources():
        assert s.bucket == "safe_commercial"
        assert s.commercial_use is True
        assert "public-domain" in s.license_id
    # the guardrail list documents what is intentionally excluded
    assert "social_media" in EXCLUDED_KINDS and "blogs" in EXCLUDED_KINDS


def test_parse_header_and_clean_strip_bracket_line():
    raw = "[Alice's Adventures in Wonderland by Lewis Carroll 1865]\n\nCHAPTER I\n\n\n\nAlice was beginning.\n"
    meta = parse_header(raw)
    assert meta["title"].startswith("Alice")
    assert meta["author"] == "Lewis Carroll"
    assert meta["year"] == 1865
    cleaned = clean_text(raw)
    assert "[Alice" not in cleaned          # header removed
    assert "\n\n\n" not in cleaned          # blank runs collapsed
    assert "Alice was beginning." in cleaned


def test_quality_score_range():
    assert 0.0 <= quality_score("") <= 1.0
    good = quality_score("A clean English paragraph. " * 50)
    assert 0.0 < good <= 1.0


def _write_doc(d: Path, doc_id: str, text: str, split: str) -> dict:
    sub = d / "safe_commercial" / "public-domain-us"
    sub.mkdir(parents=True, exist_ok=True)
    (sub / f"{doc_id}.txt").write_text(text, encoding="utf-8")
    return {"document_id": doc_id, "title": doc_id, "author": "x", "document_type": "book",
            "source_collection": "t", "source_locator": "u", "license": "pd",
            "license_id": "public-domain-us", "commercial_use": True,
            "attribution_required": False, "language": "en", "publication_year": 1900,
            "retrieved_at": "2026", "content_hash": "sha256:x", "quality_score": 0.9,
            "n_bytes": len(text.encode()), "split": split, "bucket": "safe_commercial",
            "path": f"safe_commercial/public-domain-us/{doc_id}.txt"}


def test_byte_corpus_document_uniform_sampling(tmp_path):
    from cfna.corpus.dataset import ByteCorpus

    records = [
        _write_doc(tmp_path, "small_book", "small text here. " * 50, "train"),
        _write_doc(tmp_path, "huge_book", "huge text everywhere. " * 5000, "train"),
        _write_doc(tmp_path, "heldout_book", "validation prose abounds. " * 60, "val"),
    ]
    (tmp_path / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in records))

    import numpy as np
    train = ByteCorpus(tmp_path, "train")
    assert len(train.docs) == 2
    val = ByteCorpus(tmp_path, "val")
    assert len(val.docs) == 1 and val.titles == ["heldout_book"]
    assert train.sample_batch(32, 4, rng=np.random.default_rng(0)).shape == (4, 32)

    # Document-uniform sampling: despite a ~100x size difference, the *small* book
    # must still be sampled often. "small" is unique to it; size-proportional
    # sampling would surface it ~1% of the time, document-uniform ~50%.
    rng = np.random.default_rng(0)
    text = b" ".join(bytes(train.sample_batch(32, 1, rng)[0].tolist()) for _ in range(200))
    assert b"small" in text          # the tiny document is well represented
    assert b"everywhere" in text     # the huge document too
