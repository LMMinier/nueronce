"""The wired-up real components: perception adapter, typed memory, hybrid core,
embeddings, routers, workspace, retrieval, and the end-to-end pipeline."""

import pytest

torch = pytest.importorskip("torch")

import numpy as np

from cfna import data, impl, pipeline
from cfna.core import CFNACore
from cfna.embeddings import CognitiveEmbeddingCompiler
from cfna.memory import TypedRecurrentMemoryCell
from cfna.model import CFNAModel, ModelConfig
from cfna.perception import ByteCharPerception, dynamic_patching, encode_information_units
from cfna.routers import RelationRouters
from cfna.workspace import GlobalWorkspace


def _tiny():
    return ModelConfig(byte_embed_dim=24, d_local=48, d_model=64, p_max=24,
                       physical_blocks=1, logical_depth=2, n_heads=4, unit_window=16,
                       decoder_window=24, decoder_layers=1, d_state=8, channel_dim=16)


def test_perception_adapter_real_forward():
    enc = ByteCharPerception()
    ids = torch.randint(0, 256, (2, 40))
    feats, boundary = enc.forward(ids)
    assert feats.shape[0] == 2 and feats.shape[1] == 40
    assert boundary.shape == (2, 40)


def test_typed_memory_step_and_sequence():
    cell = TypedRecurrentMemoryCell(d_model=64)
    state = cell.init_state(batch=2)
    x = torch.randn(2, 64)
    s2 = cell.step(x, state)
    assert s2.hidden.shape == state.hidden.shape
    # authority mask of zeros blocks writes -> cell stays at zero
    s_blocked = cell.step(x, state, authority_mask=torch.zeros(2, 7))
    assert float(s_blocked.cell.detach().abs().max()) == 0.0
    seq = cell.forward(torch.randn(2, 5, 64))
    assert seq.shape == (2, 5, 64)


def test_hybrid_core_runs_over_modes():
    from cfna.config import CoreConfig
    core = CFNACore(CoreConfig(d_model=64, physical_blocks=2, local_window=8,
                               sparse_global_topk=8))
    x = torch.randn(2, 12, 64)
    y = core.run(x, mode="FAST")
    assert y.shape == x.shape


def test_embeddings_and_routers_real():
    comp = CognitiveEmbeddingCompiler(d_local=48)
    ids = torch.randint(0, 256, (1, 32))
    enc = ByteCharPerception()
    feats, blogit = enc.forward(ids)
    spans = dynamic_patching(list(ids[0].numpy()),
                             feats[0].detach().numpy(), blogit[0].detach().numpy())
    units = encode_information_units(list(ids[0].numpy()), spans, feats[0].detach().numpy())
    b1 = comp.compile(units[0], {"section_path": ["1"], "unit_id": "u1"})
    b2 = comp.compile(units[-1], {"section_path": ["1"], "unit_id": "u2"})
    assert b1.dense_semantic.shape[0] == 768
    rr = RelationRouters(d_sem=768)
    rel = rr.evidence_relation(b1, b2)
    assert abs(sum(rel.values()) - 1.0) < 1e-5  # real classifier -> probabilities


def test_workspace_iterates_and_extracts():
    ws = GlobalWorkspace(d_model=64)
    from cfna.types import TaskState
    task = TaskState("q", "q", "answer", 0.5, 0.5, 0.3)
    ws.initialize(task, [], core_state=np.ones(64, dtype=np.float32))
    c0 = [s.confidence for s in ws.slots]
    ws.iterate()
    result = ws.extract_reasoning_result()
    assert "best_hypothesis" in result
    assert len(result["slot_confidence"]) == ws.cfg.n_slots


def test_retrieval_surfaces_relevant_doc():
    corpus = [s.strip() for s in data.CORPUS.split(".") if s.strip()]
    cands = impl.build_corpus_candidates(corpus)
    dense = impl.InMemoryDenseIndex(cands)
    q = impl.build_corpus_candidates(["dense sparse late interaction retrieval"])[0].bundle
    hits = dense.search(q.dense_semantic, topk=3)
    joined = " ".join(corpus[int(h.bundle.source_id[3:])] for h in hits).lower()
    assert "retrieval" in joined


def test_pipeline_end_to_end_runs():
    torch.manual_seed(0)
    m = CFNAModel(_tiny())
    corpus = [s.strip() for s in data.CORPUS.split(".") if s.strip()]
    text, report, trace = pipeline.respond(m, "what does the verifier check?", corpus,
                                           mode="FAST", max_rounds=1)
    assert isinstance(text, str)
    assert "retrieved" in trace and trace["retrieved"]
    assert 0.0 <= report.supported_claim_fraction <= 1.0
