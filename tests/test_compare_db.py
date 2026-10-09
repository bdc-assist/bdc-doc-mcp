"""compare_db.py, the oracle check of r-doc-builder's incremental-ingest tests: finds every kind
of difference, including a value that changed type, and passes identical DBs. Throwaway chroma
dirs, no network."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import chromadb

from compare_db import diff, dump, main


def _db(rows):
    path = tempfile.mkdtemp(prefix="r_test_cmp_")  # not removed: chroma keeps its files open on Windows
    coll = chromadb.PersistentClient(path=path).get_or_create_collection("cmp")
    coll.add(ids=[r[0] for r in rows], documents=[r[1] for r in rows],
             metadatas=[r[2] for r in rows], embeddings=[r[3] for r in rows])
    return path


BASE = [("same", "doc", {"n": 1}, [1.0, 0.0]), ("doc", "text", {"n": 1}, [1.0, 0.0]),
        ("meta", "text", {"n": 1}, [1.0, 0.0]), ("type", "text", {"n": 1}, [1.0, 0.0]),
        ("vec", "text", {"n": 1}, [1.0, 0.0]), ("gone", "text", {"n": 1}, [1.0, 0.0])]
OTHER = [("same", "doc", {"n": 1}, [1.0, 0.0]), ("doc", "changed", {"n": 1}, [1.0, 0.0]),
         ("meta", "text", {"n": 2}, [1.0, 0.0]), ("type", "text", {"n": 1.0}, [1.0, 0.0]),
         ("vec", "text", {"n": 1}, [1.0, 0.001]), ("new", "text", {"n": 1}, [1.0, 0.0])]


def test_compare_db_finds_every_kind_of_difference():
    a, b = _db(BASE), _db(OTHER)
    assert diff(dump(a, "cmp"), dump(b, "cmp")) == {
        "only in A": ["gone"], "only in B": ["new"], "document": ["doc"],
        "metadata": ["meta", "type"], "vector": ["vec"]}, "1 vs 1.0 is a difference: types are compared"
    assert "vector" not in diff(dump(a, "cmp"), dump(b, "cmp"), atol=0.01), "--atol absorbs float noise"
    assert main([a, a, "--collection", "cmp"]) == 0
    assert main([a, b, "--collection", "cmp"]) == 1


if __name__ == "__main__":
    test_compare_db_finds_every_kind_of_difference()
    print("compare_db self-check passed")
