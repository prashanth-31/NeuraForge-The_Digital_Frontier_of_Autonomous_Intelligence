"""Lightweight web search via DuckDuckGo HTML endpoint."""
from __future__ import annotations

from typing import List, Dict
from urllib.parse import quote_plus

import httpx
from bs4 import BeautifulSoup

DEFAULT_TIMEOUT = 10.0
DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


async def ddg_search(query: str, max_results: int = 5, region: str = "us-en") -> List[Dict[str, str]]:
    """Return a list of {title, url, snippet} for a query using DuckDuckGo HTML results.

    Note: This uses the /html endpoint which serves server-rendered results.
    """
    q = quote_plus(query)
    url = f"https://duckduckgo.com/html/?q={q}&kl={region}"
    headers = {
        "User-Agent": DEFAULT_UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, headers=headers, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        html = resp.text

    soup = BeautifulSoup(html, "html.parser")
    results: List[Dict[str, str]] = []

    # DuckDuckGo HTML uses div.result for each item (but classes can vary)
    for res in soup.select("div.result, div.web-result"):
        a = res.select_one("a.result__a, a.result__url, a[href]")
        if not a:
            continue
        link = a.get("href")
        title = a.get_text(" ", strip=True)
        snippet_el = (
            res.select_one("a.result__snippet, div.result__snippet, div.snippet") or res.find("p")
        )
        snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""
        if link and title:
            results.append({"title": title, "url": link, "snippet": snippet})
        if len(results) >= max_results:
            break

    return results
