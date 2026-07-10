import types

import pytest

torch = pytest.importorskip("torch")

from nueronce.model import NUERONCEModel, ModelConfig


def _tiny():
    return NUERONCEModel(ModelConfig(
        byte_embed_dim=16, d_local=32, d_model=48, p_max=16, physical_blocks=1,
        logical_depth=1, n_heads=4, unit_window=12, decoder_window=16,
        decoder_layers=1, d_state=8, channel_dim=8))


def _patch_forward(model, chooser):
    calls = []

    def forward(self, byte_ids, neighbor_ids=None, neighbor_mask=None):
        calls.append((byte_ids.detach().clone(), neighbor_ids, neighbor_mask))
        logits = torch.full((byte_ids.shape[0], byte_ids.shape[1], 256), -1000.0,
                            device=byte_ids.device)
        idx = chooser(byte_ids, neighbor_ids, len(calls))
        logits[:, -1, idx] = 1000.0
        return logits, torch.zeros(byte_ids.shape, device=byte_ids.device)

    model.forward = types.MethodType(forward, model)
    return calls


def test_generate_returns_continuation_only_and_no_prompt_echo():
    model = _tiny()
    _patch_forward(model, lambda ids, n, step: ord("A"))
    out = model.generate(b"What is liberty?", max_new=3, greedy=True)
    assert out == b"AAA"
    assert not out.startswith(b"What is liberty?")


def test_generate_honors_multibyte_stop_sequence():
    model = _tiny()
    seq = [ord("O"), ord("K"), ord("<"), ord("|"), ord("e"), ord("n"), ord("d"), ord("|"), ord(">")]
    _patch_forward(model, lambda ids, n, step: seq[step - 1])
    out = model.generate(b"prompt", max_new=len(seq), greedy=True, stop_sequences=[b"<|end|>"])
    assert out == b"OK"


def test_generate_passes_retrieval_tensors_every_step_and_evidence_can_change_output():
    model = _tiny()
    calls = _patch_forward(
        model,
        lambda ids, neighbor_ids, step: ord("X") if int(neighbor_ids.sum().item()) == 1 else ord("Y"),
    )
    n1 = torch.tensor([[[1]]], dtype=torch.long)
    m1 = torch.tensor([[[True]]])
    n2 = torch.tensor([[[2]]], dtype=torch.long)
    out1 = model.generate(b"p", neighbor_ids=n1, neighbor_mask=m1, max_new=2, greedy=True)
    out2 = model.generate(b"p", neighbor_ids=n2, neighbor_mask=m1, max_new=2, greedy=True)
    assert out1 == b"XX"
    assert out2 == b"YY"
    assert all(call[1] is not None and call[2] is not None for call in calls)


def test_generate_top_k_top_p_and_repetition_penalty_are_safe():
    model = _tiny()

    def forward(self, byte_ids, neighbor_ids=None, neighbor_mask=None):
        logits = torch.zeros((byte_ids.shape[0], byte_ids.shape[1], 256), device=byte_ids.device)
        logits[:, -1, ord("A")] = 10.0
        logits[:, -1, ord("B")] = 9.0
        return logits, torch.zeros(byte_ids.shape, device=byte_ids.device)

    model.forward = types.MethodType(forward, model)
    assert model.generate(b"p", max_new=1, top_k=1, temperature=1.0) == b"A"
    assert model.generate(b"p", max_new=1, top_p=0.8, temperature=1.0) in (b"A", b"B")
    assert model.generate(b"A", max_new=1, greedy=True, repetition_penalty=2.0) == b"B"


def test_generate_return_scores():
    model = _tiny()
    _patch_forward(model, lambda ids, n, step: ord("A"))
    scores = model.generate(b"p", max_new=2, greedy=True, return_scores=True)
    assert scores["bytes"] == b"AA"
    assert len(scores["logprobs"]) == 2
    assert "avg_logprob" in scores and "avg_entropy" in scores
