import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # not installed: package sits at the repo root

from bdc_doc_mcp.ingest import _chunk_ids, _embed_batched


class FakeEmb:
    """Records each request so we can assert on how texts were split."""

    def __init__(self):
        self.requests = []

    def embed_documents(self, texts):
        self.requests.append(list(texts))
        return [[0.0] for _ in texts]


def test_batching_respects_token_budget():
    budget_chars = 6000 * 4  # EMBEDDING_BATCH_TOKENS default, ~4 chars/token
    emb = FakeEmb()
    texts = ["x" * 10_000] * 5

    vectors = _embed_batched(emb, texts)

    assert len(vectors) == 5, "one vector per input"
    assert len(emb.requests) > 1, "should split into several requests"
    for req in emb.requests:
        assert sum(len(t) for t in req) <= budget_chars, "request exceeded the budget"


def test_oversized_single_text_is_truncated():
    emb = FakeEmb()
    _embed_batched(emb, ["x" * 500_000])
    assert len(emb.requests) == 1
    assert len(emb.requests[0][0]) == 6000 * 4, "a lone huge chunk must still fit one request"


def test_ids_are_stable_and_unique():
    contents = ["same text", "same text", "other"]
    metas = [{"source": "a.pkl"}, {"source": "a.pkl"}, {"source": "a.pkl"}]

    ids = _chunk_ids(contents, metas)
    assert len(set(ids)) == 3, "duplicate content in one file must not collide"
    assert ids == _chunk_ids(contents, metas), "ids must be stable across runs (upsert, not duplicate)"

    # different source => different id, so files don't overwrite each other
    other = _chunk_ids(["same text"], [{"source": "b.pkl"}])
    assert other[0] != ids[0]


if __name__ == "__main__":
    test_batching_respects_token_budget()
    test_oversized_single_text_is_truncated()
    test_ids_are_stable_and_unique()
    print("ingest self-check passed")
