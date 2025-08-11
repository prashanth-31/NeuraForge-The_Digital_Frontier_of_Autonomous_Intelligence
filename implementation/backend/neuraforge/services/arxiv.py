"""arXiv API adapter (Atom feed)."""
from __future__ import annotations

from typing import Any, Dict, List
import xml.etree.ElementTree as ET
from urllib.parse import urlencode

from .http import get_text

BASE = "http://export.arxiv.org/api/query"


def _text(elem: ET.Element | None) -> str:
    return (elem.text or "").strip() if elem is not None else ""


async def search_papers(query: str, max_results: int = 5, start: int = 0) -> List[Dict[str, Any]]:
    params = {
        "search_query": f"all:{query}",
        "start": start,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    text = await get_text(BASE, params=params, headers={"User-Agent": "NeuraForge/1.0"})
    # Parse Atom
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    root = ET.fromstring(text)
    results: List[Dict[str, Any]] = []
    for entry in root.findall("atom:entry", ns):
        title = _text(entry.find("atom:title", ns))
        summary = _text(entry.find("atom:summary", ns))
        published = _text(entry.find("atom:published", ns))
        link = ""
        for l in entry.findall("atom:link", ns):
            if l.get("rel") == "alternate":
                link = l.get("href") or link
        authors = [
            _text(a.find("atom:name", ns)) for a in entry.findall("atom:author", ns)
        ]
        doi_el = entry.find("arxiv:doi", ns)
        doi = _text(doi_el) if doi_el is not None else None
        results.append(
            {
                "title": title,
                "summary": summary,
                "published": published,
                "link": link,
                "authors": authors,
                "doi": doi,
            }
        )
    return results
