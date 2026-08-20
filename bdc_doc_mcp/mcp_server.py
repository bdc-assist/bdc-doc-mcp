import os

from mcp.server.fastmcp import FastMCP

from .api import SearchRequest, search

mcp = FastMCP("bdc-doc-mcp")


@mcp.tool()
def search_docs(query: str, k: int = 5, mode: str = "embedding",
                doc_type: str | None = None,
                date_from: str | None = None, date_to: str | None = None) -> list:
    """Search the BDC (NHLBI BioData Catalyst) documentation database.

    Returns the top-k matching chunks with content, metadata (source, doc_type,
    datetime when available), and a score.

    query is the search text. In embedding mode phrase it as a question or topic
    (e.g. "how do I bring my own data"); in keyword mode give the literal terms
    to match.

    k is the number of chunks to return (default 5). Raise it (10-20) for broad
    or multi-part questions; each chunk is a small section of a document.

    mode toggles the search engine:
      - "embedding" (default): semantic similarity — best for questions, topics,
        and paraphrased wording. score is a distance (lower = more similar).
      - "keyword": fuzzy literal word matching — ignores case and punctuation
        ("picsure" finds "PIC-SURE") and tolerates small typos — best for exact
        names, acronyms, tool names, or error messages the embedding may blur.
        Chunks matching more of the query terms rank first; score is the total
        number of occurrences (higher = better).

    doc_type is a CSV string of types to search (e.g. "page,faq" or "video").
    Available types:
      - docs:   BDC GitBook platform documentation — user guides, how-tos, and
                technical reference (bdcatalyst.gitbook.io)
      - page:   key pages of the BDC website — about/overview, joining BDC,
                analyzing & sharing data, usage costs and terms
      - faq:    Freshdesk help-desk FAQ articles (support questions & answers)
      - video:  transcripts of BDC YouTube tutorials/webinars, with timestamped
                links into the video
      - fellow: BDC Fellows profiles — fellowship recipients and their research
                projects
      - update: dated news posts ("latest updates") from the BDC website
      - event:  dated BDC events — webinars, workshops, deadlines
    When doc_type is omitted, only docs, page, faq, and video are searched —
    name fellow, update, or event explicitly to search them.

    date_from / date_to ("YYYY-MM-DD", inclusive) filter by date. Only event and
    update docs carry a date, so a date filter implicitly narrows to those types.
    Results are ranked by relevance, NOT date — for "recent"/"latest" questions,
    always set date_from to bound the range, then compare the dates returned.
    """
    return search(SearchRequest(query=query, k=k, mode=mode, doc_type=doc_type,
                                date_from=date_from, date_to=date_to))


if __name__ == "__main__":
    import sys

    if "--http" in sys.argv:
        # serve at http://127.0.0.1:<MCP_PORT>/mcp for network clients (e.g. bdcbot)
        mcp.settings.port = int(os.getenv("MCP_PORT", "8001"))
        mcp.run(transport="streamable-http")
    else:
        mcp.run()
