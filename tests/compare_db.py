"""Do two r-doc-mcp Chroma DB dirs hold the same chunks? Compares ids, documents, metadata (exact,
value types included) and vectors (within --atol). Exits 1 on any difference.

The oracle of r-doc-builder's incremental ingest: after any sequence of pushes the DB must equal
a fresh --reset push of the same inputs. Stop the servers using both dirs first, and compare
copies of any DB that matters (chroma may write when it opens a dir).

    uv run python tests/compare_db.py DIR_A DIR_B [--collection NAME] [--atol 0]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb

EXAMPLES = 20


def dump(path, collection):
    data = chromadb.PersistentClient(path=str(path)).get_collection(collection).get(
        include=["documents", "metadatas", "embeddings"])
    return {cid: (doc, meta or {}, [float(x) for x in emb])
            for cid, doc, meta, emb in zip(data["ids"], data["documents"], data["metadatas"], data["embeddings"])}


def _typed(meta):
    # 1 == 1.0 == True in Python: compare types too, a round trip that turns ints into floats is a difference
    return {k: (type(v).__name__, v) for k, v in meta.items()}


def diff(a, b, atol=0.0):
    """-> {category: sorted ids} for every difference between two dumps; equal dumps give {}."""
    out = {"only in A": sorted(a.keys() - b.keys()), "only in B": sorted(b.keys() - a.keys()),
           "document": [], "metadata": [], "vector": []}
    for cid in sorted(a.keys() & b.keys()):
        (doc_a, meta_a, vec_a), (doc_b, meta_b, vec_b) = a[cid], b[cid]
        if doc_a != doc_b:
            out["document"].append(cid)
        if _typed(meta_a) != _typed(meta_b):
            out["metadata"].append(cid)
        if len(vec_a) != len(vec_b) or any(abs(x - y) > atol for x, y in zip(vec_a, vec_b)):
            out["vector"].append(cid)
    return {k: v for k, v in out.items() if v}


def main(argv=None):
    from r_doc_mcp.config import COLLECTION_NAME

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("a", help="first Chroma DB dir")
    parser.add_argument("b", help="second Chroma DB dir")
    parser.add_argument("--collection", default=COLLECTION_NAME, help="collection to compare (default: %(default)s)")
    parser.add_argument("--atol", type=float, default=0.0,
                        help="tolerance per vector component; real embedders can differ in the last digits (default: exact)")
    args = parser.parse_args(argv)
    a, b = dump(args.a, args.collection), dump(args.b, args.collection)
    found = diff(a, b, args.atol)
    print(f"A: {len(a)} chunks in {args.a}\nB: {len(b)} chunks in {args.b}")
    for category, ids in found.items():
        print(f"{category}: {len(ids)}")
        for cid in ids[:EXAMPLES]:
            if category == "metadata":
                ma, mb = a[cid][1], b[cid][1]
                keys = sorted(k for k in ma.keys() | mb.keys() if _typed({k: ma.get(k)}) != _typed({k: mb.get(k)}))
                print(f"  {cid}: " + "; ".join(f"{k}: {ma.get(k)!r} vs {mb.get(k)!r}" for k in keys))
            else:
                print(f"  {cid}")
    print("same" if not found else "DIFFERENT")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
