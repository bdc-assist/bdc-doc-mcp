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
    """Every ingest route answers every wrong Authorization with a 401 and writes nothing (a non-ASCII
    header used to crash compare_digest into a 500); an empty INGEST_TOKEN disables ingest."""
    routes = (("/ingest/upsert", CHUNKS), ("/ingest/reset", None), ("/ingest/lookup", {"sources": ["s"]}),
              ("/ingest/update", [{"id": "1", "metadata": {}}]), ("/ingest/delete", ["1"]))
    assert client.post("/ingest/reset", headers=AUTH).status_code == 200
    wrong = [{}, {"Authorization": "Bearer "}, {"Authorization": "Token test-token"},
             {"Authorization": "Bearer wrong"}, {"Authorization": "Bearer tést".encode("latin-1")}]
    for headers in wrong:
        for path, body in routes:
            res = client.post(path, json=body, headers=headers)
            assert res.status_code == 401, (path, headers, res.status_code)
    assert client.get("/health").json()["documents"] == 0, "a refused upsert writes nothing"

    os.environ["INGEST_TOKEN"] = ""
    try:
        for path, body in routes:
            assert client.post(path, json=body, headers=AUTH).status_code == 403, path
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


def _lookup(*sources):
    res = client.post("/ingest/lookup", json={"sources": list(sources)}, headers=AUTH)
    assert res.status_code == 200, res.text
    return res.json()


def test_lookup_update_delete_roundtrip():
    """r-doc-builder's diff loop: lookup returns stored {id: metadata} for the listed sources only,
    update rewrites metadata without touching content or vectors, delete removes ids."""
    from r_doc_mcp.db import get_db

    rows = [
        {"id": "a1", "content": "alpha one", "embedding": [1.0, 0.0], "metadata": {"source": "a", "page_url": "ua", "title": "A"}},
        {"id": "a2", "content": "alpha two", "embedding": [0.9, 0.1], "metadata": {"source": "a", "page_url": "ua", "title": "A"}},
        {"id": "b1", "content": "beta", "embedding": [0.0, 1.0], "metadata": {"source": "b", "page_url": "ub"}},
    ]
    try:
        assert client.post("/ingest/reset", headers=AUTH).status_code == 200
        assert client.post("/ingest/upsert", json=rows, headers=AUTH).status_code == 200

        assert _lookup("a", "nope") == {"a1": rows[0]["metadata"], "a2": rows[1]["metadata"]}, "only the listed sources"

        res = client.post("/ingest/update", json=[{"id": "a1", "metadata": {"title": "A2"}}], headers=AUTH)
        assert res.status_code == 200 and res.json() == {"updated": 1}, res.text
        assert _lookup("a")["a1"] == {"source": "a", "page_url": "ua", "title": "A2"}, "metadata merges; other keys stay"
        got = get_db().coll.get(ids=["a1"], include=["documents", "embeddings"])
        assert got["documents"] == ["alpha one"] and list(got["embeddings"][0]) == [1.0, 0.0], "update never touches content or vectors"

        res = client.post("/ingest/delete", json=["a2", "missing"], headers=AUTH)
        assert res.status_code == 200 and res.json() == {"deleted": 1, "documents": 2}, res.text
        assert set(_lookup("a", "b")) == {"a1", "b1"}
    finally:
        client.post("/ingest/reset", headers=AUTH)


def test_none_metadata_value_removes_the_key():
    """Chroma merges metadata on upsert and update, so a key a record dropped would live on forever;
    r-doc-builder sends it as None, which must reach Chroma and delete it."""
    row = {"id": "n1", "content": "c", "embedding": [1.0, 0.0],
           "metadata": {"source": "n", "date": "2025-01-01", "date_num": 20250101}}
    try:
        client.post("/ingest/reset", headers=AUTH)
        client.post("/ingest/upsert", json=[row], headers=AUTH)
        client.post("/ingest/update", json=[{"id": "n1", "metadata": {"date": None}}], headers=AUTH)
        assert _lookup("n")["n1"] == {"source": "n", "date_num": 20250101}, "update with None deletes the key"
        client.post("/ingest/upsert", json=[{**row, "metadata": {"source": "n", "date_num": None}}], headers=AUTH)
        assert _lookup("n")["n1"] == {"source": "n"}, "upsert with None deletes the key too"
    finally:
        client.post("/ingest/reset", headers=AUTH)


def test_metadata_types_survive_a_lookup():
    """r-doc-builder skips a chunk only when stored metadata == new metadata, so every value must
    come back with the type it was pushed with: an int returning as a float would re-push forever.
    Floats float32 can't hold exactly (0.1, a video's start_seconds) must come back bit-for-bit too."""
    meta = {"source": "t", "n": 3, "ratio": 0.5, "whole": 2.0, "tenth": 0.1, "start_seconds": 12.345,
            "flag": True, "off": False, "s": "x"}
    try:
        client.post("/ingest/reset", headers=AUTH)
        client.post("/ingest/upsert", json=[{"id": "t1", "content": "c", "embedding": [1.0, 0.0], "metadata": meta}], headers=AUTH)
        got = _lookup("t")["t1"]
        assert got == meta and {k: type(v) for k, v in got.items()} == {k: type(v) for k, v in meta.items()}, got
    finally:
        client.post("/ingest/reset", headers=AUTH)


def test_new_ingest_routes_accept_empty_bodies():
    client.post("/ingest/reset", headers=AUTH)
    assert _lookup() == {}
    assert client.post("/ingest/update", json=[], headers=AUTH).json() == {"updated": 0}
    assert client.post("/ingest/delete", json=[], headers=AUTH).json() == {"deleted": 0, "documents": 0}


if __name__ == "__main__":
    test_search_k_default_from_config()
    test_mcp_server_name_from_config()
    test_ingest_requires_token()
    test_ingest_and_search_roundtrip()
    test_search_rejects_k_below_1()
    test_embedding_search_filters_and_metadata()
    test_ingest_token_edge_cases()
    test_lookup_update_delete_roundtrip()
    test_none_metadata_value_removes_the_key()
    test_metadata_types_survive_a_lookup()
    test_new_ingest_routes_accept_empty_bodies()
    test_unexpected_errors_are_json()
    test_mcp_search_docs_maps_its_arguments()
    print("api ingest/search self-check passed")
