import numpy as np

from nueronce.config import PerceptionConfig
from nueronce.perception import dynamic_patching, encode_information_units


def _feats_for(data: bytes, dim: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((len(data), dim))


def test_patching_covers_whole_stream_without_overlap():
    data = b"The quick brown fox. Jumps! Over (the) lazy dog.\n" * 3
    feats = _feats_for(data)
    logits = np.zeros(len(data))
    spans = dynamic_patching(list(data), feats, logits)
    assert spans[0][0] == 0
    assert spans[-1][1] == len(data)
    # contiguous, non-overlapping
    for (s0, e0), (s1, e1) in zip(spans, spans[1:]):
        assert e0 == s1
        assert e0 > s0


def test_patch_length_bounds_respected():
    data = bytes([65]) * 500  # no syntax bytes, flat features
    feats = np.zeros((len(data), 4))
    logits = np.full(len(data), -10.0)  # learned boundary ~0
    cfg = PerceptionConfig(min_patch=4, max_patch=32)
    spans = dynamic_patching(list(data), feats, logits, cfg)
    lengths = [e - s for s, e in spans]
    # With no positive boundary signal, max_patch forces the cut.
    assert max(lengths) <= 32
    assert sum(lengths) == len(data)


def test_empty_input():
    assert dynamic_patching([], np.zeros((0, 4)), np.zeros(0)) == []


def test_encode_information_units_exact_and_local():
    data = b"alpha beta gamma delta"
    feats = _feats_for(data, dim=6)
    spans = dynamic_patching(list(data), feats, np.zeros(len(data)))
    units = encode_information_units(list(data), spans, feats)
    assert len(units) == len(spans)
    for u in units:
        assert u["span"][1] > u["span"][0]
        assert isinstance(u["exact"], dict)  # sparse lexical features
        assert u["local"].shape == (6,)
