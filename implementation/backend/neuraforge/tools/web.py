"""Web tools such as respectful scraping."""
from __future__ import annotations

from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

from ..services import scraper


class WebScrapeInput(BaseModel):
    url: str = Field(..., description="HTTP/HTTPS URL to scrape")
    selector: str | None = Field(None, description="Optional CSS selector to narrow content")
    include_links: bool = Field(False, description="Whether to include top links found on the page")
    max_chars: int = Field(2000, ge=500, le=8000, description="Max characters of extracted text")


async def _web_scrape_impl(url: str, selector: str | None = None, include_links: bool = False, max_chars: int = 2000) -> str:
    return await scraper.scrape_url(url=url, selector=selector, include_links=include_links, max_chars=max_chars)


web_scrape_tool = StructuredTool.from_function(
    coroutine=_web_scrape_impl,
    name="web_scrape",
    description=(
        "Scrape and summarize readable text from a public web page, respecting robots.txt. "
        "Use when the user provides a URL or asks to pull info from a specific site."
    ),
    args_schema=WebScrapeInput,
)
