import datetime
import difflib
import os
import re
import secrets
from collections import Counter
from typing import Literal

from fastapi import Body, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from openai import APIError
from pydantic import BaseModel, Field

from .config import API_HOST, API_PORT, KEYWORD_FUZZY_CUTOFF, SEARCH_K, doc_types, get_emb
from .db import get_db

app = FastAPI(title=f"{doc_types()['project']} Doc RAG", version="0.1", description="Search + ingest doc RAG service")


async def _as_json_error(request, exc):
    # unhandled => plain-text 500, and every client dies on .json() instead of seeing the cause.
    # APIError: embedding provider unreachable, expired key, rate limit (openai/azure/vllm — not ollama).
    # ValueError: a bad *_MODEL_PROVIDER / DB_BACKEND in .env, which otherwise reads as a mystery 500.
    # Exception: anything else, e.g. the DB rejecting a query embedding of the wrong size.
    status = 502 if isinstance(exc, APIError) else 500
    return JSONResponse(status_code=status, content={"detail": f"{type(exc).__name__}: {exc}"})


app.add_exception_handler(APIError, _as_json_error)
app.add_exception_handler(ValueError, _as_json_error)
app.add_exception_handler(Exception, _as_json_error)


class SearchRequest(BaseModel):
    query: str
    k: int = Field(SEARCH_K, ge=1)  # 0 or less: a 422 the agent can correct, not "no results"
    # embedding = semantic similarity (score is a distance, lower = better);
    # keyword = case-insensitive word-match count (score is a count, higher = better)
    mode: Literal["embedding", "keyword"] = "embedding"
    doc_type: str | None = None  # CSV of types, e.g. "page,faq"; None = default_types()
    # inclusive date range; only chunks whose source carried a date match, so a date filter
    # implicitly narrows to those
    date_from: datetime.date | None = None
    date_to: datetime.date | None = None


def default_types():
    # default search scope: the doc_types.yaml entries flagged default. Time-sensitive or niche
    # types (events, people) pollute general search — leave them default: false and ask by name
    return [name for name, t in doc_types()["types"].items() if t["default"]]


def _norm(text: str) -> str:
    # lowercase and drop punctuation inside words, so "picsure" matches "PIC-SURE"
    return re.sub(r"[^a-z0-9\s]", "", text.lower())


def _keyword_rank(query: str, docs: list, metas: list, k: int) -> list:
    """Rank docs by fuzzy word matches — case-, punctuation- and small-typo-
    insensitive. Docs matching more distinct query terms rank first, then by
    total occurrences; score = total occurrences."""
    terms = set(_norm(query).split())
    texts = [_norm(content) for content in docs]
    # typo tolerance: also count corpus words spelled close to each term
    # ponytail: difflib over the whole filtered vocab per query, on top of the
    # full-collection scan — swap in a real FTS/fuzzy index if this gets slow
    vocab = list({w for text in texts for w in text.split()})
    variants = {t: {t, *difflib.get_close_matches(t, vocab, n=5, cutoff=KEYWORD_FUZZY_CUTOFF)} for t in terms}
    scored = []
    for text, content, meta in zip(texts, docs, metas):
        words = Counter(text.split())  # whole words: "api" must not count inside "rapid"
        counts = [sum(words[v] for v in variants[t]) for t in terms]
        matched = sum(1 for c in counts if c)
        if matched:
            scored.append((matched, sum(counts), content, meta))
    scored.sort(key=lambda s: (-s[0], -s[1]))
    return [{"content": c, "metadata": m, "score": float(total)}
            for matched, total, c, m in scored[:k]]


@app.get("/health")
def health():
    return {"status": "ok", "documents": get_db().count()}


@app.post("/search")
def search(req: SearchRequest):
    types = [t.strip() for t in (req.doc_type or "").split(",") if t.strip()]
    if not types and not (req.date_from or req.date_to):
        # a date filter already narrows to the dated types (event/update) — keeping the
        # default scope there would contradict it and match nothing
        types = default_types()
    date_from = int(req.date_from.strftime("%Y%m%d")) if req.date_from else None
    date_to = int(req.date_to.strftime("%Y%m%d")) if req.date_to else None
    if req.mode == "keyword":
        # ponytail: full scan of the filtered collection per query — move to a real
        # FTS index if the collection outgrows memory
        docs, metas = get_db().scan(types, date_from, date_to)
        return _keyword_rank(req.query, docs, metas, req.k)
    return get_db().search(get_emb().embed_query(req.query), req.k, types, date_from, date_to)


# --- ingest API: r-doc-builder pushes finished {id, content, embedding, metadata}
# records here; this service never builds content itself ---

class Chunk(BaseModel):
    id: str
    content: str
    embedding: list[float]
    metadata: dict = {}


def _require_ingest_token(authorization: str | None):
    token = os.getenv("INGEST_TOKEN")
    if not token:
        raise HTTPException(status_code=403, detail="ingest disabled: set INGEST_TOKEN on the server")
    # bytes: compare_digest raises on non-ASCII str, which a crafted header turned into a 500
    if not secrets.compare_digest((authorization or "").encode(), f"Bearer {token}".encode()):
        raise HTTPException(status_code=401, detail="bad ingest token")


def _scalar_meta(meta: dict) -> dict:
    # backends only accept scalar metadata values; None passes: it deletes that key (see db.py)
    return {k: v for k, v in meta.items() if v is None or isinstance(v, (str, int, float, bool))}


@app.post("/ingest/upsert")
def ingest_upsert(chunks: list[Chunk], authorization: str | None = Header(default=None)):
    _require_ingest_token(authorization)
    if not chunks:
        return {"upserted": 0, "documents": get_db().count()}
    get_db().upsert(
        ids=[c.id for c in chunks],
        contents=[c.content for c in chunks],
        embeddings=[c.embedding for c in chunks],
        metadatas=[_scalar_meta(c.metadata) for c in chunks],
    )
    return {"upserted": len(chunks), "documents": get_db().count()}


class LookupRequest(BaseModel):
    sources: list[str]


class MetadataUpdate(BaseModel):
    id: str
    metadata: dict


@app.post("/ingest/lookup")
def ingest_lookup(req: LookupRequest, authorization: str | None = Header(default=None)):
    """Stored {id: metadata} of every chunk whose source is listed. r-doc-builder diffs a file
    against it: unchanged chunks are skipped, metadata-only changes updated, dropped chunks deleted."""
    _require_ingest_token(authorization)
    return get_db().lookup(req.sources) if req.sources else {}


@app.post("/ingest/update")
def ingest_update(updates: list[MetadataUpdate], authorization: str | None = Header(default=None)):
    """Metadata only: content and vectors stay, so a re-labelled chunk costs no embedding."""
    _require_ingest_token(authorization)
    if updates:
        get_db().update_metadata([u.id for u in updates], [_scalar_meta(u.metadata) for u in updates])
    return {"updated": len(updates)}


@app.post("/ingest/delete")
def ingest_delete(ids: list[str] = Body(...), authorization: str | None = Header(default=None)):
    _require_ingest_token(authorization)
    before = get_db().count()
    if ids:
        get_db().delete(ids)
    documents = get_db().count()
    return {"deleted": before - documents, "documents": documents}


@app.post("/ingest/reset")
def ingest_reset(authorization: str | None = Header(default=None)):
    _require_ingest_token(authorization)
    get_db().reset()
    return {"status": "reset"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=API_HOST, port=API_PORT)
