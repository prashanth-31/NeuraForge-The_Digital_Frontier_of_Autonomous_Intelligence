"""Research tools wrapping Wikipedia and scholarly APIs."""
from __future__ import annotations

from typing import Any, Dict, List
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from ..services import wikipedia, crossref, arxiv


class WikiSummaryInput(BaseModel):
    query: str = Field(..., description="Topic to look up on Wikipedia")
    lang: str = Field("en", description="Language code (e.g., en, de, fr)")


async def _wiki_summary_impl(query: str, lang: str = "en") -> str:
    data = await wikipedia.search_summary(query, lang=lang)
    title = data.get("title") or query
    extract = data.get("extract") or ""
    url = data.get("url") or ""
    return f"{title}: {extract}\nSource: {url}".strip()


wikipedia_summary_tool = StructuredTool.from_function(
    coroutine=_wiki_summary_impl,
    name="wikipedia_summary",
    description="Get a concise summary of a topic from Wikipedia.",
    args_schema=WikiSummaryInput,
)


# Crossref
class CrossrefSearchInput(BaseModel):
    query: str = Field(..., description="Scholarly search query")
    rows: int = Field(5, ge=1, le=20)


async def _crossref_search_impl(query: str, rows: int = 5) -> str:
    items = await crossref.search_works(query, rows=rows)
    lines = []
    for it in items:
        title = it.get("title") or "Untitled"
        year = it.get("issued") or ""
        url = it.get("URL") or ""
        doi = it.get("DOI") or ""
        lines.append(f"- {title} ({year}) DOI:{doi} {url}")
    return "\n".join(lines) if lines else "No results"


crossref_search_tool = StructuredTool.from_function(
    coroutine=_crossref_search_impl,
    name="crossref_search",
    description="Search Crossref for scholarly works (DOIs/metadata).",
    args_schema=CrossrefSearchInput,
)


# Combined papers search (arXiv + Crossref)
class PapersSearchInput(BaseModel):
    query: str = Field(..., description="Topic to find recent papers for")
    max_results: int = Field(10, ge=1, le=20)


def _score_item(title: str, year: int | None, tokens: List[str]) -> float:
    score = 0.0
    if year:
        # recency boost
        score += max(0.0, (year - 2000) / 30.0)
    t = (title or "").lower()
    for tok in tokens:
        if tok and tok in t:
            score += 0.5
    return score


async def _papers_search_impl(query: str, max_results: int = 10) -> str:
    # Fetch from both sources
    ax = await arxiv.search_papers(query, max_results=min(5, max_results))
    cx = await crossref.search_works(query, rows=min(5, max_results))

    tokens = [w.strip().lower() for w in query.split() if len(w) > 2]
    combined: List[Dict[str, Any]] = []

    for it in ax:
        year = None
        pub = (it.get("published") or "")[:4]
        try:
            year = int(pub)
        except Exception:
            year = None
        combined.append(
            {
                "title": it.get("title"),
                "link": it.get("link"),
                "year": year,
                "source": "arXiv",
                "authors": it.get("authors") or [],
                "score": _score_item(it.get("title") or "", year, tokens),
            }
        )

    for it in cx:
        year = it.get("issued") if isinstance(it.get("issued"), int) else None
        combined.append(
            {
                "title": it.get("title"),
                "link": it.get("URL"),
                "year": year,
                "source": "Crossref",
                "authors": it.get("author") or [],
                "score": _score_item(it.get("title") or "", year, tokens),
            }
        )

    combined.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    top = combined[:max_results]

    lines = []
    for it in top:
        year = it.get("year") or ""
        src = it.get("source")
        title = it.get("title") or "Untitled"
        link = it.get("link") or ""
        lines.append(f"- [{src}] {title} ({year})\n  {link}")
    return "\n".join(lines) if lines else "No results"


papers_search_tool = StructuredTool.from_function(
    coroutine=_papers_search_impl,
    name="papers_search",
    description="Search both arXiv and Crossref and return ranked paper links.",
    args_schema=PapersSearchInput,
)
