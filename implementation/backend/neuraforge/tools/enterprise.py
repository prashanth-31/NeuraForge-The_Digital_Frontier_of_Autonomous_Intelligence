"""Enterprise tools wrapping Wikidata."""
from __future__ import annotations

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from ..services import wikidata


class WikidataSearchInput(BaseModel):
    query: str = Field(..., description="Entity or company search query")
    language: str = Field("en", description="Language code")
    limit: int = Field(5, ge=1, le=20)


async def _wd_search(query: str, language: str = "en", limit: int = 5) -> str:
    items = await wikidata.search_entities(query, language=language, limit=limit)
    lines = [f"{it.get('id')}: {it.get('label')} — {it.get('description')}" for it in items]
    return "\n".join(lines) if lines else "No results"


wikidata_search_tool = StructuredTool.from_function(
    coroutine=_wd_search,
    name="wikidata_search",
    description="Search entities on Wikidata (companies, orgs, etc.).",
    args_schema=WikidataSearchInput,
)
