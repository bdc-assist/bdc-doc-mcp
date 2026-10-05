from mcp.server.fastmcp import FastMCP

from .api import SearchRequest, default_types, search
from .config import MCP_HOST, MCP_PORT, MCP_SERVER_NAME, SEARCH_K, doc_types, prompts

# host goes to the constructor, not settings: FastMCP derives its allowed Host headers from it
# there (localhost-only for 127.0.0.1), so setting it later would bind but reject other hosts
mcp = FastMCP(MCP_SERVER_NAME, host=MCP_HOST, port=MCP_PORT)


def tool_description():
    """search_docs help text: the prompts.yaml template, filled with the project name and
    doc types from doc_types.yaml."""
    cfg, p = doc_types(), prompts()
    defaults = default_types()
    types = "\n".join(f"  - {n}: {t['description']}" for n, t in cfg["types"].items()) or "  (none declared)"
    scope = p["scope_defaults"].format(defaults=", ".join(defaults)) if defaults else p["scope_all"]
    return p["search_docs"].format(project=cfg["project"], k=SEARCH_K, types=types, scope=scope)


@mcp.tool(description=tool_description())
def search_docs(query: str, k: int = SEARCH_K, mode: str = "embedding",
                doc_type: str | None = None,
                date_from: str | None = None, date_to: str | None = None) -> list:
    """Search the documentation database (see the tool description)."""
    return search(SearchRequest(query=query, k=k, mode=mode, doc_type=doc_type,
                                date_from=date_from, date_to=date_to))


if __name__ == "__main__":
    import sys

    from .db import get_db

    # open the DB (and import chromadb/numpy) here, on the main thread: doing it lazily inside the
    # first tool call hangs the stdio server on Windows until the client sends its next message
    get_db()
    if "--http" in sys.argv:
        # serve at http://<MCP_HOST>:<MCP_PORT>/mcp for network clients (e.g. r-assist)
        mcp.run(transport="streamable-http")
    else:
        mcp.run()
