import numpy as np

from cfna import ops


def test_cosine_basic():
    assert ops.cosine([1, 0], [1, 0]) == 1.0
    assert ops.cosine([1, 0], [0, 1]) == 0.0
    assert abs(ops.cosine([1, 0], [-1, 0]) + 1.0) < 1e-9
    assert ops.cosine([0, 0], [1, 1]) == 0.0  # zero vector guard


def test_softmax_sums_to_one():
    p = ops.softmax([1.0, 2.0, 3.0])
    assert abs(float(np.sum(p)) - 1.0) < 1e-9
    assert np.argmax(p) == 2


def test_sigmoid_monotone():
    assert ops.sigmoid(0.0) == 0.5
    assert ops.sigmoid(10.0) > 0.99
    assert ops.sigmoid(-10.0) < 0.01


def test_sparse_dot():
    a = {1: 2.0, 3: 1.0}
    b = {3: 4.0, 5: 1.0}
    assert ops.sparse_dot(a, b) == 4.0
    assert ops.sparse_dot(a, {}) == 0.0


def test_hash_ngram_features_stable_and_overlapping():
    f1 = ops.hash_ngram_features(b"parser.py")
    f2 = ops.hash_ngram_features(b"parser.py")
    assert f1 == f2 and len(f1) > 0
    # Identical strings score higher with each other than with unrelated text.
    f3 = ops.hash_ngram_features(b"totally different content here")
    assert ops.sparse_dot(f1, f2) > ops.sparse_dot(f1, f3)


def test_sha256_and_iso():
    assert len(ops.sha256_bytes(b"abc")) == 64
    assert ops.now_iso().endswith("Z")


def test_l2_normalize_sparse():
    f = ops.l2_normalize_sparse({1: 3.0, 2: 4.0})
    mag = (f[1] ** 2 + f[2] ** 2) ** 0.5
    assert abs(mag - 1.0) < 1e-9
