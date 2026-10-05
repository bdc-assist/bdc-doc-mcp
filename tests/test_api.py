"""Ingest + search round-trip through the HTTP API against a throwaway chroma DB.
No network: embeddings are pushed pre-computed (as r-doc-builder does) and the
searches use keyword mode. Embedding-mode search needs a live embedding provider —
covered by test_mcp.py."""
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


def test_mcp_server_name_from_config():
    from r_doc_mcp import config
    from r_doc_mcp.mcp_server import mcp

    assert mcp.name == config.MCP_SERVER_NAME


if __name__ == "__main__":
    test_search_k_default_from_config()
    test_mcp_server_name_from_config()
    test_ingest_requires_token()
    test_ingest_and_search_roundtrip()
    print("api ingest/search self-check passed")
