"""Vector-DB backends behind one small interface.

A backend implements: count() -> int; search(embedding, k, types, date_from, date_to)
-> [{content, metadata, score}] (score = distance, lower is better); scan(types,
date_from, date_to) -> (documents, metadatas); upsert(ids, contents, embeddings,
metadatas); lookup(sources) -> {id: metadata} of the chunks whose metadata source is
listed; update_metadata(ids, metadatas); delete(ids); reset(). upsert and
update_metadata merge metadata into what is stored, and a None value deletes that key.
types is a list of doc_type values (None/[] = no type filter);
date_from/date_to are inclusive YYYYMMDD ints matched against metadata date_num.
Add a backend (postgres/pgvector, qdrant, ...) by writing such a class and
registering it in BACKENDS; select with DB_BACKEND.
"""
import os
from functools import lru_cache

from .config import COLLECTION_NAME, DB_PATH


class ChromaDB:
    def __init__(self):
        import chromadb
        self._client = chromadb.PersistentClient(path=DB_PATH)
        self._coll = None

    @property
    def coll(self):
        if self._coll is None:
            self._coll = self._client.get_or_create_collection(COLLECTION_NAME)
        return self._coll

    @staticmethod
    def _where(types, date_from, date_to):
        conds = []
        if types:
            conds.append({"doc_type": {"$in": list(types)}})
        # chroma range operators are numeric-only, hence the date_num ints
        if date_from is not None:
            conds.append({"date_num": {"$gte": date_from}})
        if date_to is not None:
            conds.append({"date_num": {"$lte": date_to}})
        if not conds:
            return None
        return conds[0] if len(conds) == 1 else {"$and": conds}

    def count(self):
        return self.coll.count()

    def search(self, embedding, k, types=None, date_from=None, date_to=None):
        res = self.coll.query(query_embeddings=[embedding], n_results=k,
                              where=self._where(types, date_from, date_to),
                              include=["documents", "metadatas", "distances"])
        return [{"content": doc, "metadata": meta or {}, "score": float(dist)}
                for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]

    def scan(self, types=None, date_from=None, date_to=None):
        data = self.coll.get(where=self._where(types, date_from, date_to),
                             include=["documents", "metadatas"])
        return data["documents"], [m or {} for m in data["metadatas"]]

    def upsert(self, ids, contents, embeddings, metadatas):
        self.coll.upsert(ids=ids, documents=contents, embeddings=embeddings, metadatas=metadatas)

    def lookup(self, sources):
        data = self.coll.get(where={"source": {"$in": list(sources)}}, include=["metadatas"])
        return {cid: meta or {} for cid, meta in zip(data["ids"], data["metadatas"])}

    def update_metadata(self, ids, metadatas):
        self.coll.update(ids=ids, metadatas=metadatas)

    def delete(self, ids):
        self.coll.delete(ids=ids)

    def reset(self):
        if COLLECTION_NAME in [c.name for c in self._client.list_collections()]:
            self._client.delete_collection(COLLECTION_NAME)
        self._coll = None


BACKENDS = {"chroma": ChromaDB}


@lru_cache
def get_db():
    backend = os.getenv("DB_BACKEND", "chroma")
    if backend not in BACKENDS:
        raise ValueError(f"Unsupported DB_BACKEND: {backend} (have: {', '.join(BACKENDS)})")
    return BACKENDS[backend]()
