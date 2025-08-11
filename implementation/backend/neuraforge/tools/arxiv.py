"""LangChain tool for arXiv search."""
from __future__ import annotations

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from ..services import arxiv


class ArxivSearchInput(BaseModel):
    query: str = Field(..., description="Search keywords, e.g., 'transformers RLHF'")
    max_results: int = Field(5, ge=1, le=20)


async def _arxiv_search(query: str, max_results: int = 5) -> str:
    items = await arxiv.search_papers(query, max_results=max_results)
    lines = []
    for it in items:
        authors = ", ".join(it.get("authors") or [])
        lines.append(f"- {it.get('title')} ({it.get('published')[:10]}) by {authors}\n  {it.get('link')}")
    return "\n".join(lines) if lines else "No results"


arxiv_search_tool = StructuredTool.from_function(
    coroutine=_arxiv_search,
    name="arxiv_search",
    description="Search arXiv for research papers (returns titles, dates, links).",
    args_schema=ArxivSearchInput,
)
