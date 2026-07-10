"""JSON schema + example access for NUERONCE's wire records.

The schema and example ``.json`` files live alongside this module so they can be
shipped as package data and reused by validators, fixtures, and docs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

_HERE = Path(__file__).parent
_EXAMPLES = _HERE / "examples"

RECORDS = ("source_record", "knowledge_unit", "retrieval_record", "memory_record")


def load_schema(name: str) -> Dict[str, Any]:
    path = _HERE / f"{name}.schema.json"
    if not path.exists():
        raise FileNotFoundError(f"no schema for {name!r} ({path})")
    return json.loads(path.read_text())


def load_example(name: str) -> Dict[str, Any]:
    path = _EXAMPLES / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"no example for {name!r} ({path})")
    return json.loads(path.read_text())


__all__ = ["RECORDS", "load_schema", "load_example"]
