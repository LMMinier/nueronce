"""GPU/AMP finalization checks for local CUDA training.

These validate the fixes that matter when CFNAModel trains under
``torch.autocast(dtype=torch.float16)`` on a real GPU (as
``notebooks/cfna_large_corpus_forgeloop.ipynb`` does):

- ``masked_softmax`` must stay NaN-free in fp16 — a fixed -1e30 fill overflows
  to -inf in half precision, and fully-masked rows (which occur by design:
  first-segment bytes attend zero units in the decoder) then produce
  ``-inf - (-inf) = NaN``.
- ``RMSNorm`` must compute its statistics in fp32 — squaring fp16 activations
  overflows past |x| ~ 16.
- ``cfna.chat.Conversation`` must follow the model's device instead of
  building CPU context tensors against a CUDA model.

The fp16 tests run anywhere torch is installed; the CUDA tests skip cleanly
on CPU-only machines.
"""

import pytest

torch = pytest.importorskip("torch")

from cfna import nn as cnn
from cfna.chat import Conversation
from cfna.model import CFNAModel, ModelConfig


def test_masked_softmax_is_nan_free_in_fp16():
    torch.manual_seed(0)
    scores = torch.randn(1, 1, 3, 4).half()
    mask = torch.ones(1, 1, 3, 4, dtype=torch.bool)
    mask[0, 0, 0] = False        # fully-masked row: the fp16 NaN trap
    w = cnn.masked_softmax(scores, mask)
    assert torch.isfinite(w).all()
    assert float(w[0, 0, 0].sum()) == 0.0            # masked row -> zeros, not NaN
    assert abs(float(w[0, 0, 1].sum()) - 1.0) < 1e-2  # fp16 tolerance


def test_masked_softmax_unchanged_in_fp32():
    torch.manual_seed(0)
    scores = torch.randn(2, 1, 5, 5)
    mask = torch.ones(2, 1, 5, 5, dtype=torch.bool).tril()
    w = cnn.masked_softmax(scores, mask)
    assert torch.isfinite(w).all()
    row_sums = w.sum(-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_rmsnorm_survives_large_fp16_inputs_and_matches_fp32():
    torch.manual_seed(0)
    norm = cnn.RMSNorm(8)
    x16 = (torch.randn(2, 4, 8) * 60).half()   # squaring these overflows raw fp16
    y = norm(x16)
    assert torch.isfinite(y).all()
    y32 = norm(x16.float())
    assert (y.float() - y32).abs().max() < 1e-2


def _tiny() -> ModelConfig:
    return ModelConfig(
        byte_embed_dim=16, d_local=32, d_model=48, p_max=16, physical_blocks=1,
        logical_depth=1, n_heads=4, unit_window=12, decoder_window=16,
        decoder_layers=1, d_state=8, channel_dim=8)


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
def test_model_loss_and_backward_under_amp_autocast():
    torch.manual_seed(0)
    model = CFNAModel(_tiny()).cuda()
    ids = torch.randint(0, 256, (2, 64), device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss, parts = model.loss(ids)
    assert torch.isfinite(loss)
    loss.backward()
    grads_finite = all(p.grad is None or torch.isfinite(p.grad).all()
                       for p in model.parameters())
    assert grads_finite


@cuda
def test_conversation_generates_on_cuda_model():
    torch.manual_seed(0)
    model = CFNAModel(_tiny()).cuda()
    convo = Conversation(model, temperature=0.0, max_new=8, min_new=2)
    reply = convo.say("Hello")
    assert isinstance(reply, str)
