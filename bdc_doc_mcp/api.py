import datetime
import difflib
import re
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from openai import APIError
from pydantic import BaseModel

from .config import get_collection, get_vectorstore

app = FastAPI(title="BDC Doc RAG", version="0.1", description="Search-only doc RAG service")


async def _as_json_error(request, exc):
    # unhandled => plain-text 500, and every client dies on .json() instead of seeing the cause.
    # APIError: embedding provider unreachable, expired key, rate limit (openai/azure/vllm — not ollama).
    # ValueError: a bad *_MODEL_PROVIDER in .env, which otherwise reads as a mystery 500.
    status = 502 if isinstance(exc, APIError) else 500
    return JSONResponse(status_code=status, content={"detail": f"{type(exc).__name__}: {exc}"})


app.add_exception_handler(APIError, _as_json_error)
app.add_exception_handler(ValueError, _as_json_error)


class SearchRequest(BaseModel):
    query: str
    k: int = 5
    # embedding = semantic similarity (score is a distance, lower = better);
    # keyword = case-insensitive word-match count (score is a count, higher = better)
    mode: Literal["embedding", "keyword"] = "embedding"
    doc_type: str | None = None  # CSV of types, e.g. "page,faq"; None = DEFAULT_TYPES
    # inclusive date range; only event/update docs carry a date, so a date filter
    # implicitly narrows the search to those types
    date_from: datetime.date | None = None
    date_to: datetime.date | None = None


# default search scope; fellow and the time-sensitive update/event types
# pollute general doc search — ask for them explicitly via doc_type
DEFAULT_TYPES = ["docs", "page", "faq", "video"]


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
    variants = {t: {t, *difflib.get_close_matches(t, vocab, n=5, cutoff=0.8)} for t in terms}
    scored = []
    for text, content, meta in zip(texts, docs, metas):
        counts = [sum(text.count(v) for v in variants[t]) for t in terms]
        matched = sum(1 for c in counts if c)
        if matched:
            scored.append((matched, sum(counts), content, meta))
    scored.sort(key=lambda s: (-s[0], -s[1]))
    return [{"content": c, "metadata": m, "score": float(total)}
            for matched, total, c, m in scored[:k]]


@app.get("/health")
def health():
    return {"status": "ok", "documents": get_collection().count()}


@app.post("/search")
def search(req: SearchRequest):
    types = [t.strip() for t in (req.doc_type or "").split(",") if t.strip()]
    if types:
        conds = [{"doc_type": {"$in": types}}]
    elif req.date_from or req.date_to:
        # only event/update docs carry date_num, so the date condition already narrows the
        # search — keeping the default scope here would contradict it and match nothing
        conds = []
    else:
        conds = [{"doc_type": {"$in": DEFAULT_TYPES}}]
    # date_num is the int form (YYYYMMDD) of `date` — chroma range operators are numeric-only
    if req.date_from:
        conds.append({"date_num": {"$gte": int(req.date_from.strftime("%Y%m%d"))}})
    if req.date_to:
        conds.append({"date_num": {"$lte": int(req.date_to.strftime("%Y%m%d"))}})
    flt = conds[0] if len(conds) == 1 else {"$and": conds}
    if req.mode == "keyword":
        # ponytail: full scan of the filtered collection per query — chroma's $contains
        # is case-sensitive; move to a real FTS index if the collection outgrows memory
        data = get_collection().get(where=flt, include=["documents", "metadatas"])
        return _keyword_rank(req.query, data["documents"], data["metadatas"], req.k)
    hits = get_vectorstore().similarity_search_with_score(req.query, k=req.k, filter=flt)
    return [
        {"content": doc.page_content, "metadata": doc.metadata, "score": float(score)}
        for doc, score in hits
    ]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
