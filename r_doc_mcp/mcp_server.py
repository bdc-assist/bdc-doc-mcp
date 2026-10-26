import asyncio

from mcp.server.fastmcp import FastMCP

from .api import SearchRequest, app, default_types, search
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
async def search_docs(query: str, k: int = SEARCH_K, mode: str = "embedding",
                      doc_type: str | None = None,
                      date_from: str | None = None, date_to: str | None = None) -> list:
    """Search the documentation database (see the tool description)."""
    req = SearchRequest(query=query, k=k, mode=mode, doc_type=doc_type, date_from=date_from, date_to=date_to)
    # in a thread: FastMCP runs a sync tool on the event loop, so each search (embedding call + DB
    # query) would hold up every other user's search, and with --http the API routes too
    return await asyncio.to_thread(search, req)


def http_app():
    """api.py's app with the MCP endpoint mounted at /mcp: /health /search /ingest/* and the MCP
    server in one process. Chroma isn't process-safe, so a separate ingest process would write
    under this one's in-memory index."""
    mcp_app = mcp.streamable_http_app()  # creates mcp.session_manager
    # a mounted app's own lifespan never runs, and /mcp needs the session manager's task group
    app.router.lifespan_context = lambda _: mcp.session_manager.run()
    app.mount("/", mcp_app)  # after api's routes, so those still match first
    return app


if __name__ == "__main__":
    import sys

    from .db import get_db

    # open the DB (and import chromadb/numpy) here, on the main thread: doing it lazily inside the
    # first tool call hangs the stdio server on Windows until the client sends its next message
    get_db()
    if "--http" in sys.argv:
        import uvicorn

        # serve at http://<MCP_HOST>:<MCP_PORT>/mcp for network clients (e.g. r-assist), plus the API
        uvicorn.run(http_app(), host=MCP_HOST, port=MCP_PORT)
    else:
        mcp.run()
