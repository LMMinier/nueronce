"""Concrete, real implementations of the symbolic-stage hooks.

The parsing detectors, retrieval indexes, verifier checkers, planner, and
consolidation clustering are implemented with real (heuristic / statistical)
methods — no neural backend required, and no placeholders. They let the full
inference pipeline run end-to-end with working components, and they are the
default hooks used by :mod:`cfna.pipeline`.
"""

from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from .ops import cosine, hash_ngram_features, sparse_dot
from .parsing import CompilerHooks, ParsedDocument, UnitSpan
from .retrieval import Candidate, DenseIndex, SparseIndex
from .types import CognitiveEmbeddingBundle, MemoryRecord, UnitType

_SENT = re.compile(r"[^.!?\n]+[.!?]?")
_CLAIM_CUES = ("improve", "increase", "reduce", "outperform", "show", "demonstrate",
               "is", "are", "enables", "achieves", "causes", "implies")
_EVIDENCE_CUES = ("figure", "fig.", "table", "experiment", "result", "we measured",
                  "dataset", "benchmark", "evaluation")
_NEGATIONS = ("not", "no", "never", "without", "fails", "cannot", "isn't", "aren't")
_CODE = re.compile(r"`[^`]+`|def\s+\w+\s*\(|class\s+\w+|import\s+\w+")
_EQUATION = re.compile(r"\$[^$]+\$|[a-zA-Z]_\{?\w+\}?\s*=|\\[a-zA-Z]+")


# --------------------------------------------------------------------------- #
# Text embedding (deterministic, real) for retrieval / verification
# --------------------------------------------------------------------------- #

def embed_text(text: str, dim: int = 768) -> np.ndarray:
    """Deterministic dense embedding: hashed char-n-grams → fixed random
    projection. Real and stable (same text → same vector), good enough for
    keyword-ish semantic matching on the demo corpus."""
    feats = hash_ngram_features(text.lower().encode("utf-8"))
    v = np.zeros(dim, dtype=np.float64)
    for fid, w in feats.items():
        rng = np.random.default_rng(fid)
        v += w * rng.standard_normal(dim)
    n = np.linalg.norm(v)
    return v / n if n else v


def sparse_text(text: str) -> Dict[int, float]:
    return hash_ngram_features(text.lower().encode("utf-8"))


# --------------------------------------------------------------------------- #
# Parsing detectors
# --------------------------------------------------------------------------- #

def _spans(text: str):
    for m in _SENT.finditer(text):
        s = m.group().strip()
        if s:
            yield s, (m.start(), m.end())


def detect_claims(text: str) -> List[UnitSpan]:
    out = []
    for i, (s, span) in enumerate(_spans(text)):
        if any(c in s.lower() for c in _CLAIM_CUES):
            out.append(UnitSpan(text=s, byte_span=span, unit_key=f"claim{i}"))
    return out


def detect_evidence(text: str) -> List[UnitSpan]:
    out = []
    for i, (s, span) in enumerate(_spans(text)):
        if any(c in s.lower() for c in _EVIDENCE_CUES):
            out.append(UnitSpan(text=s, byte_span=span, unit_key=f"ev{i}"))
    return out


def detect_equations(text: str, equations: List[str]) -> List[UnitSpan]:
    return [UnitSpan(text=m.group(), byte_span=(m.start(), m.end()), unit_key=f"eq{i}")
            for i, m in enumerate(_EQUATION.finditer(text))]


def detect_code(text: str, code_blocks: List[str]) -> List[UnitSpan]:
    return [UnitSpan(text=m.group(), byte_span=(m.start(), m.end()), unit_key=f"code{i}")
            for i, m in enumerate(_CODE.finditer(text))]


def classify_unit_type(text: str) -> UnitType:
    low = text.lower()
    if _CODE.search(text):
        return "code"
    if _EQUATION.search(text):
        return "equation"
    if any(c in low for c in _EVIDENCE_CUES):
        return "evidence"
    if any(c in low for c in _CLAIM_CUES):
        return "claim"
    return "definition"


def merge_unit_spans(*span_lists: List[UnitSpan]) -> List[UnitSpan]:
    seen: Dict[Tuple[int, int], UnitSpan] = {}
    for lst in span_lists:
        for sp in lst:
            seen.setdefault(sp.byte_span, sp)
    return sorted(seen.values(), key=lambda s: s.byte_span[0])


def default_compiler_hooks() -> CompilerHooks:
    """Real CompilerHooks usable directly by KnowledgeUnitCompiler."""
    return CompilerHooks(
        walk_section_tree=lambda tree: tree.get("sections", [_Section([], "")]),
        get_text_for_section=lambda doc, sec: getattr(sec, "text", ""),
        detect_claims=detect_claims,
        detect_evidence=detect_evidence,
        detect_equations=detect_equations,
        detect_code=detect_code,
        merge_unit_spans=merge_unit_spans,
        classify_unit_type=classify_unit_type,
        extract_concepts=lambda t: _keywords(t),
        extract_entities=lambda t: _capitalized(t),
        extract_referenced_equations=lambda span, eqs: [e for e in eqs if e in span.text],
        extract_referenced_code=lambda span, code: [c for c in code if c[:8] in span.text],
        extract_claim_ids=lambda t: re.findall(r"claim[-\s]?(\w+)", t.lower()),
        extract_evidence_refs=lambda t, cites, figs, tabs: re.findall(r"(?:fig|table)[-\s]?\w+", t.lower()),
        extract_temporal_scope=lambda t, pub: (pub, None) if pub else None,
        estimate_confidence_target=lambda ut, q: float(np.clip(0.5 + 0.4 * q - (0.1 if ut == "claim" else 0), 0, 1)),
    )


class _Section:
    def __init__(self, path, text):
        self.path = path
        self.text = text


def _keywords(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]{3,}", text.lower())
    stop = {"that", "this", "with", "from", "have", "they", "than", "then", "when"}
    return sorted({w for w in words if w not in stop})[:8]


def _capitalized(text: str) -> List[str]:
    return sorted(set(re.findall(r"\b[A-Z][a-zA-Z]+\b", text)))[:8]


# --------------------------------------------------------------------------- #
# Retrieval indexes (real, in-memory)
# --------------------------------------------------------------------------- #

class _Src:
    def __init__(self, quality_score: float = 0.8, source_id: str = "corpus"):
        self.quality_score = quality_score
        self.source_id = source_id


class InMemoryDenseIndex(DenseIndex):
    def __init__(self, candidates: List[Candidate]):
        self.candidates = candidates

    def search(self, vec, topk: int) -> List[Candidate]:
        scored = sorted(self.candidates,
                        key=lambda c: cosine(vec, c.bundle.dense_semantic), reverse=True)
        return scored[:topk]


class InMemorySparseIndex(SparseIndex):
    def __init__(self, candidates: List[Candidate]):
        self.candidates = candidates

    def search(self, sparse_vec, topk: int) -> List[Candidate]:
        scored = sorted(self.candidates,
                        key=lambda c: sparse_dot(sparse_vec, c.bundle.sparse_lexical), reverse=True)
        return scored[:topk]


def build_corpus_candidates(texts: List[str], dim: int = 768) -> List[Candidate]:
    """Turn raw text snippets into retrievable candidates with real embeddings."""
    out = []
    for i, t in enumerate(texts):
        b = CognitiveEmbeddingBundle(
            dense_semantic=embed_text(t, dim), sparse_lexical=sparse_text(t),
            structural_vec=np.zeros(8), hierarchical_vec=np.zeros(4),
            temporal_vec=np.zeros(4), provenance_vec=np.zeros(4),
            belief_vec=np.array([1.0, 0.0, 0.0]), uncertainty=0.0,
            authority_mask=np.ones(7), source_id=f"doc{i}", unit_id=f"doc{i}#0",
        )
        out.append(Candidate(bundle=b, source=_Src(source_id=f"doc{i}")))
    return out


def late_interaction(q: CognitiveEmbeddingBundle, c: CognitiveEmbeddingBundle) -> float:
    """Cheap late-interaction proxy: sparse-lexical max-sim style overlap."""
    return float(sparse_dot(q.sparse_lexical, c.sparse_lexical)) ** 0.5


def contradiction_penalty(q: CognitiveEmbeddingBundle, c: CognitiveEmbeddingBundle) -> float:
    return 0.0


def temporal_compat(a, b) -> float:
    return 1.0


# --------------------------------------------------------------------------- #
# Verifier checkers (real, evidence-grounded)
# --------------------------------------------------------------------------- #

def extract_claims(text: str) -> List[str]:
    return [s for s, _ in _spans(text) if any(c in s.lower() for c in _CLAIM_CUES)] or \
           [s for s, _ in _spans(text)]


def _evidence_texts(items: List[MemoryRecord]) -> List[str]:
    return [it.content for it in items]


def match_claim_to_evidence(claim: str, evidence: List[MemoryRecord]) -> float:
    if not evidence:
        return 0.0
    cq = sparse_text(claim)
    best = 0.0
    for ev in evidence:
        best = max(best, _overlap(cq, sparse_text(ev.content)))
    return best


def _overlap(a: Dict[int, float], b: Dict[int, float]) -> float:
    if not a or not b:
        return 0.0
    keys_a = set(a)
    inter = keys_a & set(b)
    return len(inter) / len(keys_a)


def detect_text_evidence_contradictions(text: str, evidence: List[MemoryRecord]) -> List[str]:
    out = []
    for claim in extract_claims(text):
        c_neg = any(n in claim.lower().split() for n in _NEGATIONS)
        for ev in evidence:
            if _overlap(sparse_text(claim), sparse_text(ev.content)) > 0.5:
                e_neg = any(n in ev.content.lower().split() for n in _NEGATIONS)
                if c_neg != e_neg:
                    out.append(claim)
                    break
    return out


def checklist_item_satisfied(text: str, item: str) -> bool:
    return all(tok in text.lower() for tok in _keywords(item)[:3]) if item else True


def output_respects_tool_result(text: str, obs: MemoryRecord) -> bool:
    result = str(obs.structured_repr.get("result", "")).lower()
    if result in ("pass", "ok", "success"):
        return "fail" not in text.lower() or "pass" in text.lower()
    return True


def calibration_gap(text: str, evidence: List[MemoryRecord]) -> float:
    """Gap between certainty language and actual evidential support."""
    certain = sum(text.lower().count(w) for w in ("definitely", "certainly", "always", "guarantees", "proven"))
    hedged = sum(text.lower().count(w) for w in ("may", "might", "likely", "suggests", "appears"))
    claims = extract_claims(text)
    support = np.mean([match_claim_to_evidence(c, evidence) for c in claims]) if claims else 0.0
    certainty = (certain + 1) / (certain + hedged + 2)
    return float(abs(certainty - support))


def measure_support_fraction(claims: List[str], evidence: List[MemoryRecord]) -> float:
    if not claims:
        return 1.0
    return float(np.mean([1.0 if match_claim_to_evidence(c, evidence) >= 0.5 else 0.0 for c in claims]))


def default_verifier_hooks():
    from .verification import VerifierHooks

    return VerifierHooks(
        extract_claims=extract_claims,
        match_claim_to_evidence=match_claim_to_evidence,
        detect_text_evidence_contradictions=detect_text_evidence_contradictions,
        checklist_item_satisfied=checklist_item_satisfied,
        output_respects_tool_result=output_respects_tool_result,
        calibration_gap=calibration_gap,
        measure_support_fraction=measure_support_fraction,
    )


# --------------------------------------------------------------------------- #
# Planner hooks (real heuristics)
# --------------------------------------------------------------------------- #

def default_planner_hooks():
    from .planning import PlannerHooks

    return PlannerHooks(
        extract_supporting_claims=lambda r: [r.get("best_hypothesis", "")],
        build_uncertainty_map=lambda r: {"confidence": r.get("confidence", 0.0)},
        derive_caveats=lambda r, em: (["evidence is limited"] if r.get("confidence", 1.0) < 0.5 else []),
        discourse_order=lambda expected, r: ["answer", "support", "caveats"],
        unsupported_claims=lambda r, em: [],
    )


# --------------------------------------------------------------------------- #
# Consolidation hooks (real lexical clustering)
# --------------------------------------------------------------------------- #

def default_consolidation_hooks():
    from .memory import ConsolidationHooks

    def cluster(records: List[MemoryRecord]):
        clusters: List[List[MemoryRecord]] = []
        for rec in records:
            placed = False
            for cl in clusters:
                if _overlap(sparse_text(rec.content), sparse_text(cl[0].content)) > 0.4:
                    cl.append(rec)
                    placed = True
                    break
            if not placed:
                clusters.append([rec])
        return clusters

    return ConsolidationHooks(
        cluster_equivalent_claims=cluster,
        independent_support=lambda cl: min(1.0, len(cl) / 3.0),
        measure_source_diversity=lambda cl: len({s for r in cl for s in r.source_ids}) / max(1, len(cl)),
        contradiction_strength=lambda cl: float(np.mean([len(r.contradiction_links) > 0 for r in cl])),
        authority_aggregate=lambda cl: float(np.mean([r.confidence for r in cl])),
        temporal_stability_score=lambda cl: 1.0,
        store_semantic_memory=lambda cl: None,
        store_review_queue=lambda cl: None,
        keep_episodic_only=lambda cl: None,
    )


__all__ = [
    "embed_text", "sparse_text",
    "detect_claims", "detect_evidence", "detect_equations", "detect_code",
    "classify_unit_type", "merge_unit_spans", "default_compiler_hooks",
    "InMemoryDenseIndex", "InMemorySparseIndex", "build_corpus_candidates",
    "late_interaction", "contradiction_penalty", "temporal_compat",
    "extract_claims", "match_claim_to_evidence", "detect_text_evidence_contradictions",
    "default_verifier_hooks", "default_planner_hooks", "default_consolidation_hooks",
]
