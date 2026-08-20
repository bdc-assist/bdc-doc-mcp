import argparse
import os
import pickle
import re
import uuid
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from .config import COLLECTION_NAME, DB_PATH, get_collection, get_emb

SUPPORTED = {".pkl", ".md", ".mdx", ".txt", ".pdf"}

_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

_FRONTMATTER = re.compile(r"\A---\n.*?\n---\n", re.S)
_MDX_IMPORTS = re.compile(r"^\s*(?:import|export)\s.*$", re.M)
_JSX_TAGS = re.compile(r"</?[A-Z][^>]*>")


def _scalar_meta(meta: dict) -> dict:
    # chroma only accepts scalar metadata values
    return {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}


def load_pkl(path: Path, doc_type: str, use_summary: bool = False):
    """BDC_Chatbot preproc output: list of {'content': str, 'metadata': dict}.
    Embeds metadata['contextualized_chunk'] / ['text_to_embed'] when present, else
    an LLM summary if use_summary — same precedence as the original loadPKL."""
    with open(path, "rb") as f:
        rows = pickle.load(f)
    contents, metas, embed_texts = [], [], []
    for row in tqdm(rows, desc=f"preparing {path.name}"):
        meta = dict(row["metadata"], doc_type=doc_type)
        meta.setdefault("source", str(path.name))  # docs.pkl records carry their own source path
        embed_text = meta.get("contextualized_chunk") or meta.get("text_to_embed")
        if not embed_text and use_summary:
            from .preproc.utils import get_summary
            embed_text = meta["summary"] = get_summary(row["content"])
        contents.append(row["content"])
        embed_texts.append(embed_text or row["content"])
        metas.append(_scalar_meta(meta))
    return contents, metas, embed_texts


def load_text(path: Path, doc_type: str):
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".mdx":
        text = _FRONTMATTER.sub("", text)
        text = _MDX_IMPORTS.sub("", text)
        text = _JSX_TAGS.sub("", text)
    chunks = _splitter.split_text(text)
    metas = [{"source": str(path), "doc_type": doc_type} for _ in chunks]
    return chunks, metas, list(chunks)


def load_pdf(path: Path, doc_type: str):
    from pypdf import PdfReader
    contents, metas = [], []
    for page_num, page in enumerate(PdfReader(path).pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        for chunk in _splitter.split_text(text):
            contents.append(chunk)
            metas.append({"source": str(path.name), "page": page_num, "doc_type": doc_type})
    return contents, metas, list(contents)


def iter_files(paths):
    for p in paths:
        p = Path(p)
        if p.is_dir():
            yield from (f for f in sorted(p.rglob("*")) if f.suffix.lower() in SUPPORTED)
        elif p.suffix.lower() in SUPPORTED:
            yield p
        else:
            print(f"skipping unsupported: {p}")


def _chunk_ids(contents, metas):
    """Content-derived ids so re-ingesting a file updates chunks instead of duplicating.
    Repeated text within one file (identical boilerplate sections) gets an occurrence
    suffix — deterministic, and chroma rejects duplicate ids in a single upsert."""
    ids, seen = [], {}
    for content, meta in zip(contents, metas):
        key = f"{meta.get('source')}::{meta.get('page_url', '')}::{content}"
        count = seen.get(key, 0)
        seen[key] = count + 1
        ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, key if not count else f"{key}::{count}")))
    return ids


def _embed_batched(emb, texts):
    """Embed in request-sized batches. Endpoints cap tokens per request (8k on some
    gateways); a whole file at once blows that. Budget is estimated at ~4 chars/token."""
    budget = int(os.getenv("EMBEDDING_BATCH_TOKENS", "6000")) * 4
    vectors, batch, batch_chars = [], [], 0
    for text in texts:
        text = text[:budget]  # a single oversized chunk still has to fit one request
        if batch and batch_chars + len(text) > budget:
            vectors.extend(emb.embed_documents(batch))
            batch, batch_chars = [], 0
        batch.append(text)
        batch_chars += len(text)
    if batch:
        vectors.extend(emb.embed_documents(batch))
    return vectors


def reset_collection():
    """Drop the collection so the next ingest starts empty."""
    import chromadb
    client = chromadb.PersistentClient(path=DB_PATH)
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    get_collection.cache_clear()


def ingest_paths(paths, doc_type: str = "docs", use_summary: bool = False) -> int:
    """Load, embed, and store all supported files under the given paths."""
    emb = get_emb()
    collection = get_collection()
    total = 0
    for f in iter_files(paths):
        if f.suffix == ".pkl":
            contents, metas, embed_texts = load_pkl(f, doc_type, use_summary)
        else:
            loader = load_pdf if f.suffix == ".pdf" else load_text
            contents, metas, embed_texts = loader(f, doc_type)
        if not contents:
            continue
        embeddings = _embed_batched(emb, embed_texts)
        ids = _chunk_ids(contents, metas)
        collection.upsert(ids=ids, documents=contents, embeddings=embeddings, metadatas=metas)
        total += len(contents)
        print(f"ingested {len(contents):4d} chunks from {f}")
    return total


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the doc RAG chroma DB")
    parser.add_argument("paths", nargs="+", help="files or directories (.pkl .md .mdx .txt .pdf)")
    parser.add_argument("--doc-type", default="docs")
    parser.add_argument("--reset", action="store_true", help="drop the collection first")
    parser.add_argument("--summarize", action="store_true",
                        help="embed an LLM summary for .pkl records lacking a contextualized chunk")
    args = parser.parse_args()

    if args.reset:
        reset_collection()

    total = ingest_paths(args.paths, args.doc_type, args.summarize)
    print(f"done: {total} chunks in {DB_PATH} ({get_collection().count()} total)")


if __name__ == "__main__":
    main()
