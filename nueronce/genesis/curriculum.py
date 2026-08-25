"""Verified multi-domain curriculum for NUERONCE Genesis."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Iterable
from .plastic_memory import PlasticMemory

@dataclass
class Lesson:
    domain: str
    concept: str
    claim: str
    procedure: str
    source: str
    verifier: str
    test_input: object
    expected: object
    license_class: str = "approved"
    commercial_ok: bool = True

class CurriculumEngine:
    """Promotes knowledge only after an executable/verifiable gate."""
    def __init__(self, memory: PlasticMemory):
        self.memory=memory
        self.verifiers: Dict[str, Callable[[object], object]] = {}

    def register_verifier(self, name: str, fn: Callable[[object], object]):
        self.verifiers[name]=fn

    def study(self, lesson: Lesson):
        if lesson.license_class != "approved":
            return {"ok":False,"reason":"source_policy_block"}
        self.memory.write(
            lesson.concept, "procedure", lesson.procedure, source=lesson.source,
            confidence=.55, verified=False, evidence=lesson.claim, kind="procedural")
        fn=self.verifiers.get(lesson.verifier)
        if fn is None:
            return {"ok":False,"reason":"missing_verifier"}
        try:
            got=fn(lesson.test_input)
            ok=got == lesson.expected
        except Exception as e:
            return {"ok":False,"reason":type(e).__name__,"error":str(e)}
        if ok:
            self.memory.mark_verified(lesson.concept,"procedure",.96)
        return {"ok":ok,"got":got,"expected":lesson.expected}

    def readiness(self, concepts: Iterable[str]):
        rows=[]
        for c in concepts:
            r=self.memory.read(c,"procedure")
            rows.append(bool(r and r.verified and r.confidence >= .9))
        return sum(rows),len(rows)
