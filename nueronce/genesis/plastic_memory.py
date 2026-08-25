"""Persistent continual-learning memory for NUERONCE Genesis.

This module deliberately keeps acquired facts/procedures outside gradient-trained
parameters. The neural substrate learns how to interpret/use knowledge; this
store holds knowledge acquired after deployment with provenance and verification.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List
import hashlib, json, time

def symbol_address(text: str) -> str:
    return hashlib.blake2b(
        text.strip().lower().encode("utf-8"),
        digest_size=16,
        person=b"NUERONCE-GEN",
    ).hexdigest()

@dataclass
class MemoryRecord:
    subject: str
    relation: str
    object: str
    source: str
    confidence: float = 0.60
    verified: bool = False
    version: int = 1
    acquired_at: float = 0.0
    evidence: str = ""
    kind: str = "semantic"

class PlasticMemory:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.records: Dict[str, MemoryRecord] = {}
        self.history: List[dict] = []
        if self.path and self.path.exists():
            self.load()

    @staticmethod
    def _key(subject: str, relation: str) -> str:
        return f"{relation}:{symbol_address(subject)}"

    def write(self, subject: str, relation: str, object: str, *, source: str,
              confidence: float = 0.60, verified: bool = False,
              evidence: str = "", kind: str = "semantic") -> MemoryRecord:
        key = self._key(subject, relation)
        old = self.records.get(key)
        version = 1
        if old:
            self.history.append({"event": "superseded", **asdict(old)})
            version = old.version + 1
        rec = MemoryRecord(
            subject=subject, relation=relation, object=object, source=source,
            confidence=float(confidence), verified=bool(verified),
            version=version, acquired_at=time.time(), evidence=evidence, kind=kind,
        )
        self.records[key] = rec
        self.save()
        return rec

    def read(self, subject: str, relation: str, min_confidence: float = 0.0):
        rec = self.records.get(self._key(subject, relation))
        if rec is None or rec.confidence < min_confidence:
            return None
        return rec

    def mark_verified(self, subject: str, relation: str, confidence: float = 0.96):
        rec = self.read(subject, relation)
        if rec is None:
            raise KeyError((subject, relation))
        rec.verified = True
        rec.confidence = max(rec.confidence, float(confidence))
        self.save()
        return rec

    def save(self):
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "nueronce_genesis_memory_v1",
            "records": {k: asdict(v) for k, v in self.records.items()},
            "history": self.history,
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)

    def load(self):
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.records = {k: MemoryRecord(**v) for k, v in data.get("records", {}).items()}
        self.history = list(data.get("history", []))
