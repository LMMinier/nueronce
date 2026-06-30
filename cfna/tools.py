"""Tool execution with authority-aware safety.

Tool outputs are authoritative *observations* that update state, but external or
retrieved content is treated as data, not instruction, unless explicitly
authorized. The executor refuses to run when the authority context withholds
permission, and records results as ``tool_observation`` memories.

The sandboxed tool runner and result embedding are injected; the authority check,
permission gate, and record construction are real.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from .ops import now_iso
from .types import MemoryRecord


def _new_id(prefix: str) -> str:
    import uuid

    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class ToolExecutor:
    def __init__(
        self,
        run_tool_safely: Callable[[dict], dict],
        embed_tool_result: Optional[Callable[[dict], Dict[str, Any]]] = None,
        serialize_tool_result: Callable[[dict], str] = str,
    ):
        self._run = run_tool_safely
        self._embed = embed_tool_result
        self._serialize = serialize_tool_result

    def execute(self, tool_call: dict, authority_context: dict) -> MemoryRecord:
        if not authority_context.get("may_execute_tools", False):
            raise PermissionError("tool execution not authorized")

        result = self._run(tool_call)
        ok = result.get("status") == "ok"
        embeddings = self._embed(result) if self._embed is not None else {}

        return MemoryRecord(
            memory_id=_new_id("tool"),
            memory_type="episodic",
            content=self._serialize(result),
            source_ids=[f"tool:{tool_call['tool']}"],
            embeddings=embeddings,
            structured_repr=result,
            authority_level="tool_observation",
            creation_time=now_iso(),
            last_verified_time=now_iso(),
            confidence=0.98 if ok else 0.85,
            contradiction_links=[],
            evidence_links=[f"tool:{tool_call['tool']}"],
            user_scope=None,
            privacy_scope="session",
            expiration_time=None,
            review_status="verified",
            consolidation_status="episodic_only",
        )


__all__ = ["ToolExecutor"]
