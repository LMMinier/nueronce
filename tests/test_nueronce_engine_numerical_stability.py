import numpy as np
import pytest

from nueronce.engine import functional as F
from nueronce.engine.optim import NonFiniteGradientError, StreamFactor
from nueronce.engine.tensor import Tensor


def test_sigmoid_extremes_have_finite_values_and_gradients():
    x = Tensor(np.array([-1e4, -100.0, 0.0, 100.0, 1e4]), requires_grad=True)
    y = F.sigmoid(x).sum()
    y.backward()
    assert np.isfinite(y.data).all()
    assert np.isfinite(x.grad).all()
    assert 0.0 <= y.data <= 5.0


def test_fully_masked_softmax_has_zero_finite_gradient_float32():
    x = Tensor(np.zeros((2, 4), dtype=np.float32), requires_grad=True)
    mask = np.zeros((2, 4), dtype=bool)
    y = F.masked_softmax(x, mask).sum()
    y.backward()
    np.testing.assert_array_equal(y.data, 0.0)
    np.testing.assert_array_equal(x.grad, 0.0)
    assert np.isfinite(x.grad).all()


def test_streamfactor_rejects_nonfinite_step_before_mutation():
    p = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    p.grad = np.array([np.nan, 1.0])
    opt = StreamFactor([p], lr=0.1, momentum=False)
    before = p.data.copy()
    with pytest.raises(NonFiniteGradientError):
        opt.step()
    np.testing.assert_array_equal(p.data, before)
    assert opt.t == 0
