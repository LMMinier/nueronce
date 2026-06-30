"""Document parser and knowledge-unit compiler.

Converts raw HTML/PDF/code into *structured content units* rather than flattening
everything into plain text, then compiles those into :class:`KnowledgeUnit`
records carrying claims, evidence refs, equations, code, temporal scope, and
provenance links.

The structural control flow is implemented; the format-specific extractors
(``parse_html_document``, ``parse_pdf_or_epub``, ``parse_code_and_repo_structure``)
and the linguistic detectors (claim/evidence/equation/code detection, entity and
concept extraction) are dependency-heavy and left as injectable hooks that raise
``NotImplementedError`` with a clear message until wired.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .types import KnowledgeUnit, SourceRecord, UnitType


@dataclass
class ParsedDocument:
    source: SourceRecord
    text_blocks: List[str] = field(default_factory=list)
    code_blocks: List[str] = field(default_factory=list)
    equations: List[str] = field(default_factory=list)
    tables: List[dict] = field(default_factory=list)
    figures: List[dict] = field(default_factory=list)
    citations: List[dict] = field(default_factory=list)
    section_tree: dict = field(default_factory=dict)
    layout_spans: List[Tuple[int, int, str]] = field(default_factory=list)


@dataclass
class UnitSpan:
    text: str
    byte_span: Tuple[int, int]
    unit_key: str


# Hooks the host application supplies; each maps (source, raw_bytes) -> ParsedDocument
DocFormatParser = Callable[[SourceRecord, bytes], ParsedDocument]


class DocumentParser:
    """Dispatches to a format-specific parser based on ``source.source_type``."""

    HTML_TYPES = {"html", "documentation", "forum", "news"}
    DOC_TYPES = {"journal_article", "book", "report"}
    CODE_TYPES = {"code_repository", "code_file"}

    def __init__(
        self,
        html_parser: Optional[DocFormatParser] = None,
        doc_parser: Optional[DocFormatParser] = None,
        code_parser: Optional[DocFormatParser] = None,
    ):
        self._html_parser = html_parser
        self._doc_parser = doc_parser
        self._code_parser = code_parser

    def parse(self, source: SourceRecord, raw_bytes: bytes) -> ParsedDocument:
        if source.source_type in self.HTML_TYPES:
            return self._dispatch(self._html_parser, "html", source, raw_bytes)
        if source.source_type in self.DOC_TYPES:
            return self._dispatch(self._doc_parser, "pdf/epub", source, raw_bytes)
        if source.source_type in self.CODE_TYPES:
            return self._dispatch(self._code_parser, "code", source, raw_bytes)
        raise ValueError(f"unsupported source_type: {source.source_type!r}")

    @staticmethod
    def _dispatch(parser, label, source, raw_bytes):
        if parser is None:
            raise NotImplementedError(
                f"No {label} parser injected. Provide a DocFormatParser to "
                f"DocumentParser(...) to handle source_type={source.source_type!r}."
            )
        return parser(source, raw_bytes)


@dataclass
class CompilerHooks:
    """Pluggable linguistic detectors used by the knowledge-unit compiler."""

    walk_section_tree: Callable[[dict], List[Any]]
    get_text_for_section: Callable[[ParsedDocument, Any], str]
    detect_claims: Callable[[str], List[UnitSpan]]
    detect_evidence: Callable[[str], List[UnitSpan]]
    detect_equations: Callable[[str, List[str]], List[UnitSpan]]
    detect_code: Callable[[str, List[str]], List[UnitSpan]]
    merge_unit_spans: Callable[..., List[UnitSpan]]
    classify_unit_type: Callable[[str], UnitType]
    extract_concepts: Callable[[str], List[str]]
    extract_entities: Callable[[str], List[str]]
    extract_referenced_equations: Callable[[UnitSpan, List[str]], List[str]]
    extract_referenced_code: Callable[[UnitSpan, List[str]], List[str]]
    extract_claim_ids: Callable[[str], List[str]]
    extract_evidence_refs: Callable[..., List[str]]
    extract_temporal_scope: Callable[[str, Optional[str]], Optional[Tuple[str, Optional[str]]]]
    estimate_confidence_target: Callable[[UnitType, float], float]


class KnowledgeUnitCompiler:
    def __init__(self, hooks: Optional[CompilerHooks] = None):
        self.hooks = hooks

    def compile(self, doc: ParsedDocument) -> List[KnowledgeUnit]:
        h = self.hooks
        if h is None:
            raise NotImplementedError(
                "KnowledgeUnitCompiler needs CompilerHooks (claim/evidence/"
                "equation/code detectors, etc.). Inject them to enable compilation."
            )
        units: List[KnowledgeUnit] = []
        for sec in h.walk_section_tree(doc.section_tree):
            sec_text = h.get_text_for_section(doc, sec)
            spans = h.merge_unit_spans(
                h.detect_claims(sec_text),
                h.detect_evidence(sec_text),
                h.detect_equations(sec_text, doc.equations),
                h.detect_code(sec_text, doc.code_blocks),
            )
            for span in spans:
                unit_type = h.classify_unit_type(span.text)
                units.append(
                    KnowledgeUnit(
                        unit_id=f"{doc.source.source_id}#{span.unit_key}",
                        source_id=doc.source.source_id,
                        unit_type=unit_type,
                        text=span.text,
                        byte_span=span.byte_span,
                        section_path=getattr(sec, "path", []),
                        concepts=h.extract_concepts(span.text),
                        entities=h.extract_entities(span.text),
                        equations=h.extract_referenced_equations(span, doc.equations),
                        code_blocks=h.extract_referenced_code(span, doc.code_blocks),
                        claim_ids=h.extract_claim_ids(span.text),
                        evidence_refs=h.extract_evidence_refs(
                            span.text, doc.citations, doc.figures, doc.tables
                        ),
                        contradiction_group=None,
                        temporal_scope=h.extract_temporal_scope(
                            span.text, doc.source.publication_date
                        ),
                        confidence_target=h.estimate_confidence_target(
                            unit_type, doc.source.quality_score
                        ),
                    )
                )
        return units


__all__ = [
    "ParsedDocument",
    "UnitSpan",
    "DocFormatParser",
    "DocumentParser",
    "CompilerHooks",
    "KnowledgeUnitCompiler",
]
