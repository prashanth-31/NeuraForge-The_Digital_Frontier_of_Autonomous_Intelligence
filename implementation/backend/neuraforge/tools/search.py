"""Search tools wrapping DuckDuckGo HTML results."""
from __future__ import annotations

from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

from ..services import search


class WebSearchInput(BaseModel):
    query: str = Field(..., description="Web search query text")
    max_results: int = Field(5, ge=1, le=10)


async def _web_search_impl(query: str, max_results: int = 5) -> str:
    results = await search.ddg_search(query, max_results=max_results)
    if not results:
        return "No results"
    lines = []
    for r in results:
        lines.append(f"- {r['title']}\n  {r['url']}\n  {r['snippet']}")
    return "\n".join(lines)


web_search_tool = StructuredTool.from_function(
    coroutine=_web_search_impl,
    name="web_search",
    description=(
        "Search the web for relevant pages (DuckDuckGo HTML). Useful to discover sources before scraping."
    ),
    args_schema=WebSearchInput,
)
