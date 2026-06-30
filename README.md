# research_mcp

A purpose-built MCP server for storing and searching all the research behind a
project — backed by your Notion workspace. Instead of generic Notion CRUD, it
exposes **research-shaped** tools: capture a note/finding/experiment/source,
search the corpus, filter by project/tag/status, and update or archive entries.

One Notion database = your research corpus. Every entry has:

| Property | Type | Purpose |
|----------|------|---------|
| Name | title | Entry title |
| Type | select | Note · Finding · Experiment · Source · Question · Decision |
| Project | select | Which research thread (e.g. "New AI Infra", "RFT", "AACP") |
| Tags | multi-select | Free-form tags |
| Status | select | Open · In Progress · Verified · Refuted · Archived |
| Source | url | DOI / link / citation |
| Summary | rich_text | First 2000 chars of the body, kept searchable |

The full body lives in the Notion page itself; the Summary mirror is what
`research_search` keyword-matches against (Notion's API can't full-text the page
body directly, so the searchable mirror is how we get keyword search with zero
extra dependencies).

## Tools

| Tool | What it does |
|------|--------------|
| `research_setup_database` | Creates the backing database once, under a page you choose |
| `research_capture` | Store a new entry (title, body, type, project, tags, status, source) |
| `research_search` | Keyword search across titles + summaries |
| `research_list` | Filter by project / type / status / tag, with pagination |
| `research_get` | Fetch one entry including its full page body |
| `research_update` | Edit properties and/or append text to the body |
| `research_archive` | Soft-delete (recoverable from Notion trash) |

## Setup

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Create a Notion integration

1. Go to <https://www.notion.so/my-integrations> → **New integration** (internal).
2. Copy the **Internal Integration Token** (starts with `ntn_` or `secret_`).
3. Open the Notion page you want the research database to live under, click the
   **•••** menu → **Connections** → add your integration. (The integration can
   only touch pages/databases you explicitly share with it.)

### 3. Create the database (once)

Set the token and run the setup tool. The easiest path is to start the server,
then in your MCP client call `research_setup_database` with the parent page ID
(the 32-char hex string in the page URL):

```bash
export NOTION_TOKEN="ntn_xxx..."
python research_mcp.py        # then call research_setup_database from the client
```

It returns a `database_id`. Set it and restart:

```bash
export NOTION_RESEARCH_DB_ID="the-returned-id"
```

> Already have a database you want to reuse? Make sure it has the seven
> properties above (exact names), share it with the integration, and just set
> `NOTION_RESEARCH_DB_ID` to its ID — you can skip `research_setup_database`.

### 4. Register with Claude Desktop

Add this to `claude_desktop_config.json`
(macOS: `~/Library/Application Support/Claude/`,
Windows: `%APPDATA%\Claude\`):

```json
{
  "mcpServers": {
    "research": {
      "command": "python",
      "args": ["/absolute/path/to/research_mcp.py"],
      "env": {
        "NOTION_TOKEN": "ntn_xxx...",
        "NOTION_RESEARCH_DB_ID": "your-database-id"
      }
    }
  }
}
```

Restart Claude Desktop. You should see the seven `research_*` tools available.

## Test it standalone

```bash
npx @modelcontextprotocol/inspector python research_mcp.py
```

## Notes

- Transport is **stdio** (local). To run it as a remote HTTP service instead,
  change the last line to `mcp.run(transport="streamable_http", port=8000)`.
- Search is keyword/full-text over titles + summaries — no embeddings, no Ollama,
  nothing else to keep running. If you later want semantic search, the Summary
  property is the natural place to hang an embedding index.
- Errors come back as actionable strings (e.g. a 403 reminds you to share the
  page with the integration), so the agent can self-correct.
