# BDC Doc RAG

The documentation RAG MCP for of [bdc-assist](https://github.com/bdc-assist)


```
bdc_doc_mcp/config.py      env-driven embeddings/LLM/Chroma (replaces utils/__init__.set_emb_llm)
bdc_doc_mcp/ingest.py      .pkl/.md/.mdx/.txt/.pdf → embeddings → Chroma (replaces utils/chroma/utils.py)
bdc_doc_mcp/api.py         FastAPI: /health /search
bdc_doc_mcp/mcp_server.py  search_docs MCP tool for AI agents — self-contained, same search as the API
bdc_doc_mcp/preproc/       source-specific preprocessing pipeline
tests/                     self-checks + API / agent notebooks
data/                      preproc output (*.pkl), ingest input
```


## Setup

```bash
uv sync
cp .env.example .env    # then fill in keys/URLs
```

### Source repos

Only needed for preprocessing (`--sources all`); the API/MCP server and ingesting
existing `.pkl` files work without them. Clone next to this repo (or point the env
vars at them):

```bash
git clone https://github.com/stagecc/interim-bdc-website ../interim-bdc-website   # BDC_WEBSITE_DIR
git clone https://github.com/stagecc/bdc-gitbook ../bdc-gitbook                   # BDC_GITBOOK_DIR
```

### Models

For completion, use the OpenAI API on Azure (`gpt-4o-mini` by default)

For embeddings, use Ollama on Sterling (connect via RENCI VPN)
```bash
kubectl -n ner port-forward svc/ollama 11434:11434
```
Or using local Ollama with `groonga/bge-m3-Q4_K_M-GGUF` model. 



## Ingest

Full rebuild from every source (needs the two source repos cloned — see Setup;
writes `data/*.pkl`, then loads them):

```bash
uv run python -m bdc_doc_mcp.preproc.pipeline --sources all --ingest --reset
```

Individual files or directories:

```bash
uv run python -m bdc_doc_mcp.ingest ./data/docs.pkl --doc-type docs   # BDC_Chatbot preproc .pkl
uv run python -m bdc_doc_mcp.ingest ../interim-bdc-website/src/pages --doc-type page --reset
```

Embedding models are not interchangeable within a collection — `bge-m3` is 1024-dim,
`text-embedding-3-small` 1536. Switching models means `--reset` and a full re-ingest.

## API

```bash
uv run uvicorn bdc_doc_mcp.api:app --port 8000     # docs at /docs
```

| Endpoint | Body | Returns |
|---|---|---|
| `GET /health` | — | `{status, documents}` |
| `POST /search` | `{query, k, mode?, doc_type?, date_from?, date_to?}` | ranked chunks + metadata + score |

`mode` is `embedding` (default; semantic similarity, score = distance, lower is better)
or `keyword` (fuzzy literal word matching — ignores case/punctuation and tolerates
small typos, so `picsure` finds "PIC-SURE"; score = occurrence count, higher is
better — use for exact names/acronyms).
`doc_type` is a CSV of types to search (e.g. `page,faq`). When omitted, only `docs`,
`page`, `faq`, and `video` are searched — name `fellow`, `update`, or `event`
explicitly to search them.
`date_from`/`date_to` (`YYYY-MM-DD`, inclusive) filter by date; only event and update docs
carry a date, so a date filter implicitly narrows to those types.

The service is search-only by design; ingestion happens offline via the CLI (see Ingest)
and answering is the caller's job — an agent brings its own LLM.

## MCP

```bash
uv run python -m bdc_doc_mcp.mcp_server           # stdio
uv run python -m bdc_doc_mcp.mcp_server --http    # streamable HTTP, port MCP_PORT (default 8001)
```

Self-contained — searches the Chroma DB directly with the same `.env`/config as the API,
so the API service doesn't need to run. Tools: `search_docs`. Register with any MCP
client, e.g.

```json
{"mcpServers": {"bdc-doc-mcp": {
  "command": "uv",
  "args": ["--directory", "/path/to/bdc-doc-mcp", "run", "python", "-m", "bdc_doc_mcp.mcp_server"]
}}}
```

Self-check (needs an ingested `.chroma_db` + embeddings): `uv run python tests/test_mcp.py`

## Preprocessing

`bdc_doc_mcp/preproc/` is the BDC_Chatbot pipeline, ported:

| Module | Source | Ported from (BDC_Chatbot) | Notes |
|---|---|---|---|
| `bdc_repo.py` | interim-bdc-website MDX | `utils/preproc/proc_BDC_repo.py` (verbatim-ish) | fellows, events, latest-updates, pages |
| `bdc_docs.py` | bdc-gitbook markdown | `utils/preproc/proc_BDC_docs.py` (module-level LLM init removed) | chunked by header hierarchy; needs the repo cloned |
| `freshdesk.py` | bdcatalyst.freshdesk.com | `utils/preproc/proc_freshdesk.py` | live scrape |
| `vids.py` | Google Sheet + Drive SRT | `utils/preproc/proc_BDC_vids.py` (GoogleSheetsReader class flattened) | video transcripts with timestamp URLs |
| `utils.py` | — | — | LLM chunk contextualizer + summarizer |
| `pipeline.py` | — | `utils/preproc_doc.py` | orchestrator |

`--no-contextualize` skips the per-chunk LLM call (much faster, weaker retrieval).
Source paths come from `BDC_WEBSITE_DIR` / `BDC_GITBOOK_DIR`.

## Tests

```bash
uv run python tests/test_ingest.py                             # batching + chunk-id logic, no network
uv run python tests/test_keyword.py                            # keyword ranking, pure function, no DB or API
uv run python tests/test_mcp.py                                # starts the server over stdio and exercises its tools; needs .chroma_db + embeddings
```

Notebooks (each starts the API on a free port and shuts it down at the end; both need an
ingested `.chroma_db`):

- `tests/api_test.ipynb` — plain API walkthrough: `/health`, `/search`, `doc_type` filter.
  Only needs the local embeddings.
- `tests/agent_test.ipynb` — a tool-calling agent (`deepagents`): the configured LLM gets
  `search_docs` as a LangChain tool and decides when to call it. Also needs the completion
  provider reachable.


