import os
import sys
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()  # real env vars win over .env, so shell overrides work as expected

DB_PATH = os.getenv("DB_PATH", "./.chroma_db")
# keep the original collection name so existing BDC_Chatbot chroma DBs work as-is
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "langchain")


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
    """Query-side embeddings — must be the same model bdc-doc-builder embeds documents with."""
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
