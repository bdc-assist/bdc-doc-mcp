# BDC Doc RAG

The documentation RAG MCP of [bdc-assist](https://github.com/bdc-assist/bdc-assist).

Serving side only: the API/MCP servers and the vector DB. Content is built and pushed
by [bdc-doc-builder](https://github.com/bdc-assist/bdc-doc-builder), which lives outside
this security boundary and talks to the ingest API — nothing here scrapes, chunks, or
calls a completion LLM.

```
bdc_doc_mcp/config.py      env-driven query embeddings + DB settings
bdc_doc_mcp/db.py          vector-DB backends behind one interface (chroma today; DB_BACKEND selects)
bdc_doc_mcp/api.py         FastAPI: /health /search + token-guarded /ingest/upsert /ingest/reset
bdc_doc_mcp/mcp_server.py  search_docs MCP tool for AI agents — self-contained, same search as the API
tests/                     self-checks + API / agent notebooks
```

## Setup

```bash
uv sync
cp .env.example .env    # then fill in keys/URLs
```

Embeddings (for queries) use Ollama on Sterling (connect via RENCI VPN):

```bash
kubectl -n ner port-forward svc/ollama 11434:11434
```

Or a local Ollama with `groonga/bge-m3-Q4_K_M-GGUF`. **This must be the same model
bdc-doc-builder embedded the documents with** — vectors from different models don't mix
(`bge-m3` is 1024-dim, `text-embedding-3-small` 1536); switching models means a full
re-push from the builder.

## API

```bash
uv run uvicorn bdc_doc_mcp.api:app --port 8000     # docs at /docs
```

| Endpoint | Body | Returns |
|---|---|---|
| `GET /health` | — | `{status, documents}` |
| `POST /search` | `{query, k, mode?, doc_type?, date_from?, date_to?}` | ranked chunks + metadata + score |
| `POST /ingest/upsert` | `[{id, content, embedding, metadata}]` | `{upserted, documents}` |
| `POST /ingest/reset` | — | `{status}` |

`mode` is `embedding` (default; semantic similarity, score = distance, lower is better)
or `keyword` (fuzzy literal word matching — ignores case/punctuation and tolerates
small typos, so `picsure` finds "PIC-SURE"; score = occurrence count, higher is
better — use for exact names/acronyms).
`doc_type` is a CSV of types to search (e.g. `page,faq`). When omitted, only `docs`,
`page`, `faq`, and `video` are searched — name `fellow`, `update`, or `event`
explicitly to search them.
`date_from`/`date_to` (`YYYY-MM-DD`, inclusive) filter by date; only event and update docs
carry a date, so a date filter implicitly narrows to those types.

The `/ingest/*` endpoints are the write path for bdc-doc-builder: they take finished
records (embeddings pre-computed on the builder side) and require
`Authorization: Bearer $INGEST_TOKEN`; with `INGEST_TOKEN` unset, ingest is disabled.
Answering is the caller's job — an agent brings its own LLM.

## DB backends

`bdc_doc_mcp/db.py` keeps the vector DB behind a five-method interface
(`count/search/scan/upsert/reset`); everything chroma-specific — filter syntax,
`DB_PATH`, the collection — lives in its `ChromaDB` class. To swap in a remote DB
(postgres/pgvector, qdrant, ...), implement the same methods, register the class in
`BACKENDS`, and set `DB_BACKEND`.

## MCP

```bash
uv run python -m bdc_doc_mcp.mcp_server           # stdio
uv run python -m bdc_doc_mcp.mcp_server --http    # streamable HTTP, port MCP_PORT (default 8001)
```

Exposes one tool, `search_docs` — same search as the API but queries the DB directly,
so the API service doesn't need to run. Needs a pushed DB + embeddings.

Stdio clients (Claude Desktop/Code, Cursor) launch the server themselves — register it:

```json
{"mcpServers": {"bdc-doc-mcp": {
  "command": "uv",
  "args": ["--directory", "/path/to/bdc-doc-mcp", "run", "python", "-m", "bdc_doc_mcp.mcp_server"]
}}}
```

Network clients: run `--http` and point them at `http://host:8001/mcp` instead.

Smoke test: `uv run python tests/test_mcp.py`

## Tests

```bash
uv run python tests/test_api.py       # ingest+search round-trip over a temp DB, auth — no network
uv run python tests/test_keyword.py   # keyword ranking, pure function, no DB or API
uv run python tests/test_mcp.py       # starts the server over stdio and exercises its tools; needs a pushed DB + embeddings
```

Notebooks (each starts the API on a free port and shuts it down at the end; both need a
pushed DB):

- `tests/api_test.ipynb` — plain API walkthrough: `/health`, `/search`, `doc_type` filter.
  Only needs the local embeddings.
- `tests/agent_test.ipynb` — a tool-calling agent (`deepagents`): the configured LLM gets
  `search_docs` as a LangChain tool and decides when to call it. Also needs the completion
  provider reachable.
