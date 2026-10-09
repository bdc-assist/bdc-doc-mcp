# r-doc-mcp

The documentation RAG MCP of r-assist.

Serving side only: the API/MCP servers and the vector DB. Content is built and pushed
by [r-doc-builder](../r-doc-builder), which lives outside this security boundary and
talks to the ingest API — nothing here scrapes, chunks, or calls a completion LLM.

```
r_doc_mcp/config.py      env-driven query embeddings + DB settings
r_doc_mcp/db.py          vector-DB backends behind one interface (chroma today; DB_BACKEND selects)
r_doc_mcp/api.py         FastAPI: /health /search + token-guarded /ingest/upsert /ingest/reset
r_doc_mcp/mcp_server.py  search_docs MCP tool for AI agents — self-contained, same search as the API
tests/                   self-checks + API / agent notebooks
```

## Setup

```bash
uv sync
cp .env.example .env    # then fill in keys/URLs
```

Embeddings (for queries) use a local Ollama, e.g. `bge-m3`. **This must be the same model
r-doc-builder embedded the documents with** — vectors from different models don't mix
(`bge-m3` is 1024-dim, `text-embedding-3-small` 1536); switching models means a full
re-push from the builder.

To run the three repos together from scratch, follow "From clone to chat" in r-assist's README.

## API

```bash
uv run uvicorn r_doc_mcp.api:app --port 8000     # docs at /docs (or: python -m r_doc_mcp.api, binds API_HOST:API_PORT)
```

| Endpoint | Body | Returns |
|---|---|---|
| `GET /health` | — | `{status, documents}` |
| `POST /search` | `{query, k, mode?, doc_type?, date_from?, date_to?}` | ranked chunks + metadata + score |
| `POST /ingest/upsert` | `[{id, content, embedding, metadata}]` | `{upserted, documents}` |
| `POST /ingest/lookup` | `{sources: [str]}` | `{id: metadata}` of the stored chunks with those sources |
| `POST /ingest/update` | `[{id, metadata}]` | `{updated}` — metadata only, content and vectors stay |
| `POST /ingest/delete` | `[id]` | `{deleted, documents}` |
| `POST /ingest/reset` | — | `{status}` |

`mode` is `embedding` (default; semantic similarity, score = distance, lower is better)
or `keyword` (fuzzy literal word matching — ignores case/punctuation and tolerates
small typos, so `picsure` finds "PIC-SURE"; score = occurrence count, higher is
better — use for exact names/acronyms).
`doc_type` is a CSV of types to search (e.g. `page,faq`). Types and the default scope
come from <CONFIG_DIR>/doc_types.yaml (default config/; examples/bdc/doc_types.yaml is a
filled-in one). When doc_type is omitted only the types flagged default are searched;
with no defaults declared, everything is.
`date_from`/`date_to` (`YYYY-MM-DD`, inclusive) filter by date; only chunks that carry
a date match, so a date filter implicitly narrows to those.

The `/ingest/*` endpoints are the write path for r-doc-builder: they take finished
records (embeddings pre-computed on the builder side) and require
`Authorization: Bearer $INGEST_TOKEN`; with `INGEST_TOKEN` unset, ingest is disabled.
r-doc-builder looks up what is stored for a file's sources before embedding, so a re-push
embeds only new or changed chunks. Metadata merges into what is stored on upsert and update;
a `null` value deletes that key.
Answering is the caller's job — an agent brings its own LLM.

## Doc types

`config/doc_types.yaml` declares the doc_type values r-doc-builder pushes for this
project — it drives the default search scope and the search_docs tool description,
so a new deployment only needs a new YAML file, not a code change. Each type has a
`description` (shown to the agent) and a `default` flag (searched when the caller
gives no doc_type). Set `CONFIG_DIR=examples/bdc` (or `examples/fastapi`) to see a
filled-in example. `examples/fixture` serves r-doc-builder's fake, BDC-shaped test corpus: the
same doc_types and default scope as `examples/bdc` (`tests/test_doc_types.py` keeps them in step). Each `CONFIG_DIR` gets its own collection in the DB, named after the folder
(`config`, `bdc`, ...; `COLLECTION_NAME` overrides it), so switching examples never mixes or
overwrites another's chunks, and `--reset` only clears the current one.

The wording of the search_docs tool description lives in `config/prompts.yaml`, a template
with `{project}`, `{k}`, `{types}` and `{scope}` filled from doc_types.yaml and `SEARCH_K`.
Edit it freely (literal braces as `{{ }}`); the server reads it once at startup.

## DB backends

`r_doc_mcp/db.py` keeps the vector DB behind a five-method interface
(`count/search/scan/upsert/reset`); everything chroma-specific — filter syntax,
`DB_PATH`, the collection — lives in its `ChromaDB` class. To swap in a remote DB
(postgres/pgvector, qdrant, ...), implement the same methods, register the class in
`BACKENDS`, and set `DB_BACKEND`.

## MCP

```bash
uv run python -m r_doc_mcp.mcp_server           # stdio
uv run python -m r_doc_mcp.mcp_server --http    # streamable HTTP on MCP_HOST:MCP_PORT (default 127.0.0.1:8001)
```

Exposes one tool, `search_docs` — same search as the API but queries the DB directly,
so the API service doesn't need to run. Needs a pushed DB + embeddings.

Stdio clients (Claude Desktop/Code, Cursor) launch the server themselves — register it:

```json
{"mcpServers": {"r-doc-mcp": {
  "command": "uv",
  "args": ["--directory", "/path/to/r-doc-mcp", "run", "python", "-m", "r_doc_mcp.mcp_server"]
}}}
```

The `mcpServers` key above is the client's own label for the server; the name the server
reports for itself is `MCP_SERVER_NAME` (default `r-doc-mcp`).

Network clients: run `--http` and point them at `http://host:8001/mcp` instead.

Smoke test: `uv run python tests/test_mcp.py`

## Tests

```bash
uv run python tests/test_doc_types.py # doc_types.yaml + prompts.yaml -> default scope + tool description — no network
uv run python tests/test_api.py       # ingest+search round-trip over a temp DB, auth — no network
uv run python tests/test_compare_db.py  # the DB comparison below, on throwaway DBs — no network
uv run python tests/compare_db.py DIR_A DIR_B [--collection C] [--atol X]  # do two DB dirs hold the same chunks? (servers stopped)
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
