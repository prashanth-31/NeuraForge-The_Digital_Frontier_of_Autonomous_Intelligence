"""Search tools: primary uses LangChain community DuckDuckGoSearchRun; fallback custom HTML parser."""
from __future__ import annotations

import logging
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun

from ..services import search as custom_search_service

logger = logging.getLogger(__name__)


class WebSearchInput(BaseModel):
    query: str = Field(..., description="Web search query text")
    max_results: int = Field(5, ge=1, le=25)


_ddg_tool = DuckDuckGoSearchRun(name="duckduckgo_search")


async def _web_search_impl(query: str, max_results: int = 5) -> str:
    """Primary: use DuckDuckGoSearchRun (synchronous) inside thread; fallback to custom HTML scraper."""
    # DuckDuckGoSearchRun has only a synchronous .run
    try:
        from asyncio import to_thread
        raw = await to_thread(_ddg_tool.run, query)
        # raw is a string with newline separated results; truncate to max_results lines groups if needed
        lines = [l for l in raw.splitlines() if l.strip()]
        if len(lines) > max_results:
            lines = lines[: max_results]
        return "\n".join(lines)
    except Exception as e:
        logger.warning("Primary DDG search tool failed (%s); falling back to HTML scraper", e)
        try:
            results = await custom_search_service.ddg_search(query, max_results=max_results)
        except Exception as inner:
            return f"Search failed: {inner}"
        if not results:
            return "No results"
        formatted = []
        for r in results:
            formatted.append(f"- {r['title']}\n  {r['url']}\n  {r['snippet']}")
        return "\n".join(formatted)


web_search_tool = StructuredTool.from_function(
    coroutine=_web_search_impl,
    name="web_search",
    description=(
        "Search the web for relevant pages (DuckDuckGo). Returns brief lines. Use before web_scrape to gather sources."
    ),
    args_schema=WebSearchInput,
)
