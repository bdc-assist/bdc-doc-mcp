import os
import sys
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()  # real env vars win over .env, so shell overrides work as expected

DB_PATH = os.getenv("DB_PATH", "./.chroma_db")
CONFIG_DIR = os.getenv("CONFIG_DIR", "config")  # folder holding doc_types.yaml and prompts.yaml; examples/bdc for BDC
# one collection per CONFIG_DIR (its folder name) in the shared DB, so examples don't overwrite each other;
# change it only together with a full re-push from r-doc-builder
COLLECTION_NAME = os.getenv("COLLECTION_NAME") or Path(CONFIG_DIR).resolve().name
SEARCH_K = int(os.getenv("SEARCH_K", "5"))  # chunks a search returns when the caller does not pass k
# keyword mode: how similar a corpus word must be to a query term to count as a fuzzy match
# (difflib ratio, 0-1). Lower = more typo tolerance and more noise
KEYWORD_FUZZY_CUTOFF = float(os.getenv("KEYWORD_FUZZY_CUTOFF", "0.8"))
# ports: MCP_PORT for `mcp_server.py --http`, API_HOST/API_PORT for the `python -m r_doc_mcp.api` dev server
MCP_PORT = int(os.getenv("MCP_PORT", "8001"))
# 0.0.0.0 = reachable from other machines/containers; the MCP server has no auth, so only on a trusted network
MCP_HOST = os.getenv("MCP_HOST", "127.0.0.1")
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
MCP_SERVER_NAME = os.getenv("MCP_SERVER_NAME", "r-doc-mcp")  # the name the MCP server reports to clients


@lru_cache
def doc_types():
    """<CONFIG_DIR>/doc_types.yaml (default config/doc_types.yaml) -> {'project': str, 'types': {name: {description, default}}}.
    Declares the doc_type values r-doc-builder pushes, for the default search scope and the tool description."""
    path = os.path.join(CONFIG_DIR, "doc_types.yaml")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    types = {str(name): {"description": str((spec or {}).get("description", "")).strip(),
                         "default": bool((spec or {}).get("default", False))}
             for name, spec in (data.get("types") or {}).items()}
    return {"project": str(data.get("project") or "the project"), "types": types}


@lru_cache
def prompts():
    """<CONFIG_DIR>/prompts.yaml -> {name: text}: the search_docs tool description template
    and its scope sentences (filled by mcp_server.tool_description)."""
    with open(os.path.join(CONFIG_DIR, "prompts.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def _self_hosted_key():
    # self-hosted vLLM ignores the key, but the openai client refuses to start without one
    return os.getenv("OPENAI_API_KEY") or "EMPTY"


def _provider(kind: str) -> str:
    explicit = os.getenv(f"{kind}_MODEL_PROVIDER")
    if explicit:
        # split on '#': a trailing .env comment can survive into the value and is unreadable as an error
        return explicit.split("#")[0].strip().lower()
    url = os.getenv(f"{kind}_URL")
    if url:
        # ponytail: URL substring heuristic; set *_MODEL_PROVIDER explicitly to override
        if "ollama" in url:
            return "ollama"
        return "azure" if "azure.com" in url else "vllm"
    return "openai"


@lru_cache
def get_emb():
    """Query-side embeddings — must be the same model r-doc-builder embeds documents with."""
    provider = _provider("EMBEDDING")
    url = os.getenv("EMBEDDING_URL")
    model = os.getenv("EMBEDDING_MODEL")
    # stderr: under stdio MCP, stdout is the protocol channel — a print there corrupts it
    print(f"embeddings: provider={provider} model={model} url={url}", file=sys.stderr)
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        # check_embedding_ctx_length=False sends raw strings, not token arrays —
        # required by OpenAI-compatible gateways; chunks are small so no length risk
        return OpenAIEmbeddings(model=model or "text-embedding-3-small", check_embedding_ctx_length=False)
    if provider == "vllm":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(base_url=url, model=model, api_key=_self_hosted_key(),
                                check_embedding_ctx_length=False)
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(base_url=url, model=model)
    raise ValueError(f"Unsupported EMBEDDING_MODEL_PROVIDER: {provider}")
