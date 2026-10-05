import asyncio
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parent.parent  # -m resolves the package from the server's cwd


async def main():
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "r_doc_mcp.mcp_server"],
        env={**os.environ},
        cwd=str(ROOT),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            names = {t.name for t in (await session.list_tools()).tools}
            assert names == {"search_docs"}, names

            res = await session.call_tool("search_docs", {"query": "PIC-SURE", "k": 2})
            assert not res.isError, res.content
            assert res.content and res.content[0].text.strip(), "search returned nothing"
            print("search_docs ->", res.content[0].text[:200])

    print("MCP self-check passed")


if __name__ == "__main__":
    asyncio.run(main())
