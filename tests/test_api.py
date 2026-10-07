"""Ingest + search round-trip through the HTTP API against a throwaway chroma DB.
No network: embeddings are pushed pre-computed (as r-doc-builder does), and embedding-mode
searches use a fake embedder. test_mcp.py covers the real one against the pushed DB."""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# must be set before r_doc_mcp.config is imported (real env wins over .env)
_tmp = tempfile.mkdtemp(prefix="r_test_chroma_")
os.environ["DB_PATH"] = _tmp
os.environ["COLLECTION_NAME"] = "test"
os.environ["INGEST_TOKEN"] = "test-token"
os.environ["CONFIG_DIR"] = str(Path(__file__).resolve().parent.parent / "examples" / "bdc")

from fastapi.testclient import TestClient

from r_doc_mcp.api import app

client = TestClient(app)
AUTH = {"Authorization": "Bearer test-token"}

CHUNKS = [
    {"id": "1", "content": "PIC-SURE offers an API for querying data.",
     "embedding": [0.1, 0.2], "metadata": {"doc_type": "docs", "nested": {"dropped": True}}},
    {"id": "2", "content": "Joining BDC as a new user.",
     "embedding": [0.3, 0.4], "metadata": {"doc_type": "page"}},
    {"id": "3", "content": "BDC webinar on PIC-SURE.",
     "embedding": [0.5, 0.6], "metadata": {"doc_type": "event", "date_num": 20250115}},
]


def test_ingest_requires_token():
    assert client.post("/ingest/reset").status_code == 401
    assert client.post("/ingest/upsert", json=[], headers={"Authorization": "Bearer wrong"}).status_code == 401
    token = os.environ.pop("INGEST_TOKEN")
    try:
        assert client.post("/ingest/reset", headers=AUTH).status_code == 403, "no token set => ingest disabled"
    finally:
        os.environ["INGEST_TOKEN"] = token


def test_ingest_and_search_roundtrip():
    assert client.post("/ingest/reset", headers=AUTH).status_code == 200

    res = client.post("/ingest/upsert", json=CHUNKS, headers=AUTH)
    assert res.status_code == 200, res.text
    assert res.json() == {"upserted": 3, "documents": 3}

    assert client.get("/health").json() == {"status": "ok", "documents": 3}

    # keyword search honors the default scope: the event chunk is excluded
    hits = client.post("/search", json={"query": "PIC-SURE", "mode": "keyword"}).json()
    assert [h["metadata"]["doc_type"] for h in hits] == ["docs"], hits
    assert "nested" not in hits[0]["metadata"], "non-scalar metadata must be dropped at the boundary"

    # explicit doc_type reaches the event; date filter narrows to dated docs
    hits = client.post("/search", json={"query": "PIC-SURE", "mode": "keyword", "doc_type": "event"}).json()
    assert [h["metadata"]["doc_type"] for h in hits] == ["event"], hits
    hits = client.post("/search", json={"query": "BDC", "mode": "keyword", "date_from": "2025-01-01"}).json()
    assert [h["metadata"]["doc_type"] for h in hits] == ["event"], hits
    assert client.post("/search", json={"query": "BDC", "mode": "keyword",
                                        "date_to": "2024-12-31"}).json() == []

    # re-pushing the same ids updates, not duplicates
    res = client.post("/ingest/upsert", json=CHUNKS, headers=AUTH)
    assert res.json() == {"upserted": 3, "documents": 3}

    assert client.post("/ingest/reset", headers=AUTH).status_code == 200
    assert client.get("/health").json()["documents"] == 0


def test_search_k_default_from_config():
    import inspect

    from r_doc_mcp import config
    from r_doc_mcp.api import SearchRequest
    from r_doc_mcp.mcp_server import search_docs, tool_description

    assert SearchRequest(query="q").k == config.SEARCH_K
    assert f"(default {config.SEARCH_K})" in tool_description()
    fn = getattr(search_docs, "fn", search_docs)  # plain function, or the mcp wrapper's original
    assert inspect.signature(fn).parameters["k"].default == config.SEARCH_K


# made-up chunks: two in the default scope (page, docs), three dated ones outside it, at both
# ends and the middle of January so inclusive bounds and doc_type + date combinations show
FILTERED = [
    {"id": "a", "content": "alpha", "embedding": [1.0, 0.0],
     "metadata": {"doc_type": "page", "page_url": "https://x/a", "title": "A", "hierarchy": "A, Intro"}},
    {"id": "b", "content": "beta", "embedding": [0.0, 1.0], "metadata": {"doc_type": "docs", "page_url": "https://x/b"}},
    {"id": "e1", "content": "event one", "embedding": [0.9, 0.1],
     "metadata": {"doc_type": "event", "date": "2025-01-01", "date_num": 20250101}},
    {"id": "u1", "content": "update", "embedding": [0.8, 0.2],
     "metadata": {"doc_type": "update", "date": "2025-01-15", "date_num": 20250115}},
    {"id": "e2", "content": "event two", "embedding": [0.7, 0.3],
     "metadata": {"doc_type": "event", "date": "2025-01-31", "date_num": 20250131}},
]


def test_embedding_search_filters_and_metadata():
    """Embedding mode (the agent's default) offline, with a fake embedder: nearest first, each hit
    {content, metadata, score} with the citation metadata exactly as pushed, and doc_type plus
    date bounds combining (both modes share the filter, keyword mode via scan)."""
    from r_doc_mcp import api

    class FakeEmb:
        def embed_query(self, text):
            return [1.0, 0.0]

    original, api.get_emb = api.get_emb, lambda: FakeEmb()
    try:
        assert client.post("/ingest/reset", headers=AUTH).status_code == 200
        assert client.post("/ingest/upsert", json=FILTERED, headers=AUTH).status_code == 200

        def contents(**body):
            res = client.post("/search", json={"query": "q", **body})
            assert res.status_code == 200, res.text
            return [h["content"] for h in res.json()]

        hits = client.post("/search", json={"query": "q"}).json()
        assert [h["content"] for h in hits] == ["alpha", "beta"], "default scope, nearest first"
        assert all(set(h) == {"content", "metadata", "score"} for h in hits) and hits[0]["score"] < hits[1]["score"]
        assert hits[0]["metadata"] == FILTERED[0]["metadata"], "citation fields come back as pushed"

        jan = {"date_from": "2025-01-01", "date_to": "2025-01-31"}
        assert sorted(contents(doc_type="event", **jan)) == ["event one", "event two"], "bounds are inclusive"
        assert contents(doc_type=" event , update,", date_from="2025-01-02", date_to="2025-01-30") == ["update"]
        assert sorted(contents(mode="keyword", query="event", doc_type="event", date_from="2025-01-02")) == ["event two"]
        assert contents(**jan) == ["event one", "update", "event two"], "a date filter alone drops the default scope"
    finally:
        api.get_emb = original
        client.post("/ingest/reset", headers=AUTH)


def test_ingest_token_edge_cases():
    """Both ingest routes answer every wrong Authorization with a 401 and write nothing (a non-ASCII
    header used to crash compare_digest into a 500); an empty INGEST_TOKEN disables ingest."""
    assert client.post("/ingest/reset", headers=AUTH).status_code == 200
    wrong = [{}, {"Authorization": "Bearer "}, {"Authorization": "Token test-token"},
             {"Authorization": "Bearer wrong"}, {"Authorization": "Bearer tést".encode("latin-1")}]
    for headers in wrong:
        for path, body in (("/ingest/upsert", CHUNKS), ("/ingest/reset", None)):
            res = client.post(path, json=body, headers=headers)
            assert res.status_code == 401, (path, headers, res.status_code)
    assert client.get("/health").json()["documents"] == 0, "a refused upsert writes nothing"

    os.environ["INGEST_TOKEN"] = ""
    try:
        assert client.post("/ingest/upsert", json=CHUNKS, headers=AUTH).status_code == 403
    finally:
        os.environ["INGEST_TOKEN"] = "test-token"


def test_unexpected_errors_are_json():
    """Any server error reaches the client as JSON {"detail"}: a query embedding of the wrong size
    (switched embedding model, see README) came back as a plain-text 500 every client dies on."""
    from r_doc_mcp import api

    class WrongSizeEmb:
        def embed_query(self, text):
            return [1.0, 0.0, 0.0]  # the collection holds 2-dim vectors

    original, api.get_emb = api.get_emb, lambda: WrongSizeEmb()
    try:
        assert client.post("/ingest/upsert", json=CHUNKS, headers=AUTH).status_code == 200
        res = TestClient(app, raise_server_exceptions=False).post("/search", json={"query": "q"})
        assert res.status_code == 500 and res.json()["detail"], (res.status_code, res.text)
    finally:
        api.get_emb = original
        client.post("/ingest/reset", headers=AUTH)


def test_mcp_search_docs_maps_its_arguments():
    """The MCP tool the agent calls hands each argument to the right SearchRequest field (swapped
    date bounds or a dropped doc_type would pass every HTTP test) and rejects bad ones."""
    from pydantic import ValidationError

    from r_doc_mcp.mcp_server import search_docs

    fn = getattr(search_docs, "fn", search_docs)  # plain function, or the mcp wrapper's original
    assert client.post("/ingest/upsert", json=FILTERED, headers=AUTH).status_code == 200
    try:
        def contents(**kw):
            return [h["content"] for h in fn("event", mode="keyword", doc_type="event", **kw)]

        assert contents(date_from="2025-01-02") == ["event two"]
        assert contents(date_to="2025-01-02") == ["event one"]
        assert len(contents(k=1)) == 1
        for bad in ({"date_from": "January"}, {"mode": "semantic"}):
            try:
                fn("q", **bad)
                raise AssertionError(f"{bad} must be rejected")
            except ValidationError:
                pass
    finally:
        client.post("/ingest/reset", headers=AUTH)


def test_search_rejects_k_below_1():
    # k=0 returned [] in keyword mode (a 500 in embedding mode) and k=-1 dropped the last hit:
    # an agent's bad k read as "no results" instead of an error it can correct
    for k in (0, -1):
        res = client.post("/search", json={"query": "x", "k": k, "mode": "keyword"})
        assert res.status_code == 422, (k, res.status_code, res.text)


def test_mcp_server_name_from_config():
    from r_doc_mcp import config
    from r_doc_mcp.mcp_server import mcp

    assert mcp.name == config.MCP_SERVER_NAME


if __name__ == "__main__":
    test_search_k_default_from_config()
    test_mcp_server_name_from_config()
    test_ingest_requires_token()
    test_ingest_and_search_roundtrip()
    test_search_rejects_k_below_1()
    test_embedding_search_filters_and_metadata()
    test_ingest_token_edge_cases()
    test_unexpected_errors_are_json()
    test_mcp_search_docs_maps_its_arguments()
    print("api ingest/search self-check passed")
