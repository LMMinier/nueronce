#!/usr/bin/env python3
"""research_mcp — a purpose-built MCP server for storing and searching research.

Backed by the Notion API. Exposes research-shaped workflow tools (capture a
note/finding, log an experiment, save a source, search/list/get/update/archive
entries) on top of a single Notion database that acts as your research corpus.

Environment variables:
    NOTION_TOKEN          (required) Notion internal integration token.
    NOTION_RESEARCH_DB_ID (optional) ID of the research database. If unset, call
                          the `research_setup_database` tool once to create it,
                          then set this var to the returned database_id.

Run (stdio):  python research_mcp.py
"""

from __future__ import annotations

import json
import os
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field
from mcp.server.fastmcp import FastMCP

# --------------------------------------------------------------------------- #
# Constants & configuration
# --------------------------------------------------------------------------- #

NOTION_API_BASE = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"
NOTION_RICH_TEXT_LIMIT = 2000          # Notion's per-rich-text-item char cap.
REQUEST_TIMEOUT = 30.0
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100

# Property names used in the research database schema.
PROP_NAME = "Name"
PROP_TYPE = "Type"
PROP_PROJECT = "Project"
PROP_TAGS = "Tags"
PROP_STATUS = "Status"
PROP_SOURCE = "Source"
PROP_SUMMARY = "Summary"   # short, searchable plaintext copy of the body

ENTRY_TYPES = ["Note", "Finding", "Experiment", "Source", "Question", "Decision"]
STATUS_VALUES = ["Open", "In Progress", "Verified", "Refuted", "Archived"]

mcp = FastMCP("research_mcp")


# --------------------------------------------------------------------------- #
# Enums shared across tool inputs
# --------------------------------------------------------------------------- #

class ResponseFormat(str, Enum):
    MARKDOWN = "markdown"
    JSON = "json"


class EntryType(str, Enum):
    NOTE = "Note"
    FINDING = "Finding"
    EXPERIMENT = "Experiment"
    SOURCE = "Source"
    QUESTION = "Question"
    DECISION = "Decision"


class EntryStatus(str, Enum):
    OPEN = "Open"
    IN_PROGRESS = "In Progress"
    VERIFIED = "Verified"
    REFUTED = "Refuted"
    ARCHIVED = "Archived"


# --------------------------------------------------------------------------- #
# Low-level Notion client helpers (shared by every tool)
# --------------------------------------------------------------------------- #

def _token() -> str:
    token = os.environ.get("NOTION_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "NOTION_TOKEN is not set. Create an internal integration at "
            "https://www.notion.so/my-integrations, copy its token, and export "
            "NOTION_TOKEN before starting the server."
        )
    return token


def _db_id() -> str:
    db = os.environ.get("NOTION_RESEARCH_DB_ID", "").strip()
    if not db:
        raise RuntimeError(
            "NOTION_RESEARCH_DB_ID is not set. Run the `research_setup_database` "
            "tool once with a parent page ID, then export the returned database_id "
            "as NOTION_RESEARCH_DB_ID."
        )
    return db


def _headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {_token()}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }


async def _notion(method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Single entry point for all Notion API calls, with shared error handling."""
    url = f"{NOTION_API_BASE}{path}"
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        resp = await client.request(method, url, headers=_headers(), json=payload)
        resp.raise_for_status()
        return resp.json()


def _handle_error(e: Exception) -> str:
    """Turn an exception into an actionable, agent-readable error string."""
    if isinstance(e, httpx.HTTPStatusError):
        code = e.response.status_code
        try:
            body = e.response.json()
            detail = body.get("message", e.response.text)
        except Exception:
            detail = e.response.text
        hints = {
            400: "Bad request — check the property values and IDs you passed.",
            401: "Unauthorized — NOTION_TOKEN is missing or invalid.",
            403: "Forbidden — share the page/database with your integration in Notion (••• → Connections).",
            404: "Not found — check the ID, and confirm the integration has access to it.",
            409: "Conflict — the resource was modified concurrently; retry.",
            429: "Rate limited — wait a moment and retry.",
        }
        return f"Error {code}: {detail}. {hints.get(code, '')}".strip()
    if isinstance(e, httpx.TimeoutException):
        return "Error: Notion request timed out. Please try again."
    if isinstance(e, RuntimeError):
        return f"Error: {e}"
    return f"Error: {type(e).__name__}: {e}"


# --------------------------------------------------------------------------- #
# Rich-text / block helpers
# --------------------------------------------------------------------------- #

def _rich_text(content: str) -> List[Dict[str, Any]]:
    """Build a rich_text array, capped at one item under Notion's char limit."""
    return [{"type": "text", "text": {"content": (content or "")[:NOTION_RICH_TEXT_LIMIT]}}]


def _chunk(text: str, size: int = NOTION_RICH_TEXT_LIMIT) -> List[str]:
    return [text[i:i + size] for i in range(0, len(text), size)] or [""]


def _body_blocks(body: str) -> List[Dict[str, Any]]:
    """Convert a long body string into a list of paragraph blocks."""
    blocks: List[Dict[str, Any]] = []
    for line in body.split("\n"):
        for piece in _chunk(line):
            blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": _rich_text(piece) if piece else []},
            })
    return blocks


def _plain(rich: List[Dict[str, Any]]) -> str:
    return "".join(part.get("plain_text", "") for part in (rich or []))


def _select(value: Optional[str]) -> Optional[Dict[str, Any]]:
    return {"select": {"name": value}} if value else None


def _multi(values: Optional[List[str]]) -> Dict[str, Any]:
    return {"multi_select": [{"name": v} for v in (values or [])]}


# --------------------------------------------------------------------------- #
# Property builders & extractors
# --------------------------------------------------------------------------- #

def _build_properties(
    title: Optional[str] = None,
    entry_type: Optional[str] = None,
    project: Optional[str] = None,
    tags: Optional[List[str]] = None,
    status: Optional[str] = None,
    source: Optional[str] = None,
    summary: Optional[str] = None,
) -> Dict[str, Any]:
    props: Dict[str, Any] = {}
    if title is not None:
        props[PROP_NAME] = {"title": _rich_text(title)}
    if entry_type is not None:
        props[PROP_TYPE] = _select(entry_type)
    if project is not None:
        props[PROP_PROJECT] = _select(project)
    if tags is not None:
        props[PROP_TAGS] = _multi(tags)
    if status is not None:
        props[PROP_STATUS] = _select(status)
    if source is not None:
        props[PROP_SOURCE] = {"url": source or None}
    if summary is not None:
        props[PROP_SUMMARY] = {"rich_text": _rich_text(summary)}
    return props


def _extract_entry(page: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a Notion page object into a compact research-entry dict."""
    p = page.get("properties", {})

    def _title(prop: str) -> str:
        return _plain(p.get(prop, {}).get("title", []))

    def _sel(prop: str) -> Optional[str]:
        sel = p.get(prop, {}).get("select")
        return sel.get("name") if sel else None

    def _ms(prop: str) -> List[str]:
        return [x["name"] for x in p.get(prop, {}).get("multi_select", [])]

    def _txt(prop: str) -> str:
        return _plain(p.get(prop, {}).get("rich_text", []))

    return {
        "id": page.get("id"),
        "url": page.get("url"),
        "title": _title(PROP_NAME),
        "type": _sel(PROP_TYPE),
        "project": _sel(PROP_PROJECT),
        "tags": _ms(PROP_TAGS),
        "status": _sel(PROP_STATUS),
        "source": p.get(PROP_SOURCE, {}).get("url"),
        "summary": _txt(PROP_SUMMARY),
        "created": page.get("created_time"),
        "updated": page.get("last_edited_time"),
    }


def _format_entries(entries: List[Dict[str, Any]], fmt: ResponseFormat) -> str:
    if fmt == ResponseFormat.JSON:
        return json.dumps(entries, indent=2)
    if not entries:
        return "_No matching research entries._"
    lines: List[str] = []
    for e in entries:
        tags = f" · tags: {', '.join(e['tags'])}" if e["tags"] else ""
        src = f"\n  source: {e['source']}" if e.get("source") else ""
        summ = f"\n  {e['summary']}" if e.get("summary") else ""
        lines.append(
            f"### {e['title']}  ({e.get('type') or 'Note'})\n"
            f"  id: `{e['id']}` · project: {e.get('project') or '—'} · "
            f"status: {e.get('status') or '—'}{tags}{src}{summ}"
        )
    return "\n\n".join(lines)


# --------------------------------------------------------------------------- #
# Input models
# --------------------------------------------------------------------------- #

class _Base(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")


class SetupInput(_Base):
    parent_page_id: str = Field(
        ..., min_length=1,
        description="ID of an existing Notion page to create the research database under "
                    "(e.g. '24f1a2b3c4d5e6f7...'). The integration must have access to it.",
    )
    title: str = Field(default="Research Corpus", max_length=200,
                       description="Title for the new research database.")


class CaptureInput(_Base):
    title: str = Field(..., min_length=1, max_length=200,
                       description="Short title for the entry (e.g. 'Fibonacci-N conditioning κ≈2.85').")
    body: str = Field(default="", max_length=100_000,
                      description="Full content / notes. Stored as the page body; first 2000 chars are also "
                                  "indexed in a searchable Summary property.")
    type: EntryType = Field(default=EntryType.NOTE, description="Kind of entry.")
    project: Optional[str] = Field(default=None, max_length=100,
                                   description="Research thread this belongs to (e.g. 'New AI Infra', 'RFT', 'AACP').")
    tags: Optional[List[str]] = Field(default_factory=list, max_items=25,
                                      description="Free-form tags for filtering.")
    status: EntryStatus = Field(default=EntryStatus.OPEN, description="Verification/progress status.")
    source: Optional[str] = Field(default=None, max_length=2000,
                                  description="Optional URL/DOI/citation link for the source.")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class SearchInput(_Base):
    query: str = Field(..., min_length=1, max_length=500,
                       description="Keyword(s) to match against entry titles and summaries.")
    limit: int = Field(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class ListInput(_Base):
    project: Optional[str] = Field(default=None, max_length=100, description="Filter by project.")
    type: Optional[EntryType] = Field(default=None, description="Filter by entry type.")
    status: Optional[EntryStatus] = Field(default=None, description="Filter by status.")
    tag: Optional[str] = Field(default=None, max_length=100, description="Filter by a single tag.")
    limit: int = Field(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE)
    start_cursor: Optional[str] = Field(default=None, description="Pagination cursor from a previous call's next_cursor.")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class GetInput(_Base):
    entry_id: str = Field(..., min_length=1, description="Notion page ID of the entry.")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class UpdateInput(_Base):
    entry_id: str = Field(..., min_length=1, description="Notion page ID of the entry to update.")
    title: Optional[str] = Field(default=None, max_length=200)
    type: Optional[EntryType] = Field(default=None)
    project: Optional[str] = Field(default=None, max_length=100)
    tags: Optional[List[str]] = Field(default=None, max_items=25, description="Replaces existing tags if provided.")
    status: Optional[EntryStatus] = Field(default=None)
    source: Optional[str] = Field(default=None, max_length=2000)
    append_body: Optional[str] = Field(default=None, max_length=100_000,
                                       description="Text appended to the page body as new paragraph(s).")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class ArchiveInput(_Base):
    entry_id: str = Field(..., min_length=1, description="Notion page ID of the entry to archive.")


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #

@mcp.tool(
    name="research_setup_database",
    annotations={"title": "Set up research database", "readOnlyHint": False,
                 "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def research_setup_database(params: SetupInput) -> str:
    """Create the Notion database that backs the research corpus. Run this ONCE.

    Creates a database under `parent_page_id` with the schema this server expects
    (Name, Type, Project, Tags, Status, Source, Summary). After it returns, set the
    environment variable NOTION_RESEARCH_DB_ID to the returned database_id and
    restart the server.

    Returns: JSON string {"database_id": str, "url": str, "message": str}.
    """
    try:
        payload = {
            "parent": {"type": "page_id", "page_id": params.parent_page_id},
            "title": [{"type": "text", "text": {"content": params.title}}],
            "properties": {
                PROP_NAME: {"title": {}},
                PROP_TYPE: {"select": {"options": [{"name": t} for t in ENTRY_TYPES]}},
                PROP_PROJECT: {"select": {}},
                PROP_TAGS: {"multi_select": {}},
                PROP_STATUS: {"select": {"options": [{"name": s} for s in STATUS_VALUES]}},
                PROP_SOURCE: {"url": {}},
                PROP_SUMMARY: {"rich_text": {}},
            },
        }
        db = await _notion("POST", "/databases", payload)
        return json.dumps({
            "database_id": db["id"],
            "url": db.get("url"),
            "message": "Database created. Set NOTION_RESEARCH_DB_ID to database_id and restart the server.",
        }, indent=2)
    except Exception as e:  # noqa: BLE001
        return _handle_error(e)


@mcp.tool(
    name="research_capture",
    annotations={"title": "Capture a research entry", "readOnlyHint": False,
                 "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def research_capture(params: CaptureInput) -> str:
    """Store a new research entry (note, finding, experiment, source, question, decision).

    The full `body` is saved as the page content; its first 2000 chars are also
    written to a searchable Summary property so `research_search` can find it.

    Returns: the created entry as markdown or JSON (per response_format), including its id.
    """
    try:
        props = _build_properties(
            title=params.title, entry_type=params.type.value, project=params.project,
            tags=params.tags, status=params.status.value, source=params.source,
            summary=params.body[:NOTION_RICH_TEXT_LIMIT],
        )
        payload: Dict[str, Any] = {"parent": {"database_id": _db_id()}, "properties": props}
        if params.body:
            payload["children"] = _body_blocks(params.body)
        page = await _notion("POST", "/pages", payload)
        entry = _extract_entry(page)
        if params.response_format == ResponseFormat.JSON:
            return json.dumps(entry, indent=2)
        return f"Captured entry **{entry['title']}** (id: `{entry['id']}`).\n\n" + _format_entries([entry], ResponseFormat.MARKDOWN)
    except Exception as e:  # noqa: BLE001
        return _handle_error(e)


@mcp.tool(
    name="research_search",
    annotations={"title": "Search research entries", "readOnlyHint": True,
                 "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def research_search(params: SearchInput) -> str:
    """Keyword search across entry titles and summaries in the research corpus.

    Queries the database, OR-matching the keyword against the Name (title) and
    Summary properties. For deeper matches, capture entries with informative titles
    and let the first 2000 chars of the body populate the Summary.

    Returns: matching entries (most recently edited first) as markdown or JSON.
    """
    try:
        payload = {
            "filter": {"or": [
                {"property": PROP_NAME, "title": {"contains": params.query}},
                {"property": PROP_SUMMARY, "rich_text": {"contains": params.query}},
            ]},
            "sorts": [{"timestamp": "last_edited_time", "direction": "descending"}],
            "page_size": params.limit,
        }
        data = await _notion("POST", f"/databases/{_db_id()}/query", payload)
        entries = [_extract_entry(pg) for pg in data.get("results", [])]
        return _format_entries(entries, params.response_format)
    except Exception as e:  # noqa: BLE001
        return _handle_error(e)


@mcp.tool(
    name="research_list",
    annotations={"title": "List/filter research entries", "readOnlyHint": True,
                 "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def research_list(params: ListInput) -> str:
    """List research entries, optionally filtered by project, type, status, and/or tag.

    Supports pagination via start_cursor. The JSON response includes a `next_cursor`
    (pass it back as start_cursor) and `has_more` flag.

    Returns: markdown list of entries, or JSON {entries, count, has_more, next_cursor}.
    """
    try:
        conditions: List[Dict[str, Any]] = []
        if params.project:
            conditions.append({"property": PROP_PROJECT, "select": {"equals": params.project}})
        if params.type:
            conditions.append({"property": PROP_TYPE, "select": {"equals": params.type.value}})
        if params.status:
            conditions.append({"property": PROP_STATUS, "select": {"equals": params.status.value}})
        if params.tag:
            conditions.append({"property": PROP_TAGS, "multi_select": {"contains": params.tag}})

        payload: Dict[str, Any] = {
            "sorts": [{"timestamp": "last_edited_time", "direction": "descending"}],
            "page_size": params.limit,
        }
        if conditions:
            payload["filter"] = {"and": conditions} if len(conditions) > 1 else conditions[0]
        if params.start_cursor:
            payload["start_cursor"] = params.start_cursor

        data = await _notion("POST", f"/databases/{_db_id()}/query", payload)
        entries = [_extract_entry(pg) for pg in data.get("results", [])]

        if params.response_format == ResponseFormat.JSON:
            return json.dumps({
                "entries": entries,
                "count": len(entries),
                "has_more": data.get("has_more", False),
                "next_cursor": data.get("next_cursor"),
            }, indent=2)
        footer = "\n\n_More results available — pass next_cursor to continue._" if data.get("has_more") else ""
        return _format_entries(entries, ResponseFormat.MARKDOWN) + footer
    except Exception as e:  # noqa: BLE001
        return _handle_error(e)


@mcp.tool(
    name="research_get",
    annotations={"title": "Get a research entry with full body", "readOnlyHint": True,
                 "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def research_get(params: GetInput) -> str:
    """Fetch a single research entry including its full page body text.

    Returns: the entry's properties plus the reconstructed body text, as markdown or JSON.
    """
    try:
        page = await _notion("GET", f"/pages/{params.entry_id}")
        entry = _extract_entry(page)
        blocks = await _notion("GET", f"/blocks/{params.entry_id}/children?page_size=100")
        body_parts: List[str] = []
        for b in blocks.get("results", []):
            btype = b.get("type")
            rich = b.get(btype, {}).get("rich_text", []) if btype else []
            if rich:
                body_parts.append(_plain(rich))
        entry["body"] = "\n".join(body_parts)
        if params.response_format == ResponseFormat.JSON:
            return json.dumps(entry, indent=2)
        head = _format_entries([entry], ResponseFormat.MARKDOWN)
        return f"{head}\n\n---\n\n{entry['body']}" if entry["body"] else head
    except Exception as e:  # noqa: BLE001
        return _handle_error(e)


@mcp.tool(
    name="research_update",
    annotations={"title": "Update a research entry", "readOnlyHint": False,
                 "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def research_update(params: UpdateInput) -> str:
    """Update an entry's properties and/or append text to its body.

    Only provided fields change. If `tags` is given it REPLACES the existing tags.
    `append_body` adds new paragraph block(s) to the page without touching existing content.

    Returns: the updated entry as markdown or JSON.
    """
    try:
        props = _build_properties(
            title=params.title,
            entry_type=params.type.value if params.type else None,
            project=params.project,
            tags=params.tags,
            status=params.status.value if params.status else None,
            source=params.source,
        )
        if props:
            await _notion("PATCH", f"/pages/{params.entry_id}", {"properties": props})
        if params.append_body:
            await _notion("PATCH", f"/blocks/{params.entry_id}/children",
                          {"children": _body_blocks(params.append_body)})
        page = await _notion("GET", f"/pages/{params.entry_id}")
        entry = _extract_entry(page)
        if params.response_format == ResponseFormat.JSON:
            return json.dumps(entry, indent=2)
        return "Updated.\n\n" + _format_entries([entry], ResponseFormat.MARKDOWN)
    except Exception as e:  # noqa: BLE001
        return _handle_error(e)


@mcp.tool(
    name="research_archive",
    annotations={"title": "Archive a research entry", "readOnlyHint": False,
                 "destructiveHint": True, "idempotentHint": True, "openWorldHint": True},
)
async def research_archive(params: ArchiveInput) -> str:
    """Archive (soft-delete) a research entry. It is removed from lists but recoverable in Notion's trash.

    Returns: a confirmation string with the archived entry's id.
    """
    try:
        await _notion("PATCH", f"/pages/{params.entry_id}", {"archived": True})
        return f"Archived entry `{params.entry_id}`. It can be restored from Notion's trash."
    except Exception as e:  # noqa: BLE001
        return _handle_error(e)


if __name__ == "__main__":
    mcp.run()
