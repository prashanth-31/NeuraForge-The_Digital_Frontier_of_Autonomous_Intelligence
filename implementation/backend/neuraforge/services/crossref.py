"""Crossref Works API adapter."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .http import get_json

BASE = "https://api.crossref.org"


async def search_works(query: str, rows: int = 5) -> List[Dict[str, Any]]:
    url = f"{BASE}/works"
    data = await get_json(url, params={"query": query, "rows": rows})
    items = (data or {}).get("message", {}).get("items", [])
    results: List[Dict[str, Any]] = []
    for it in items:
        results.append(
            {
                "title": (it.get("title") or [None])[0],
                "DOI": it.get("DOI"),
                "URL": it.get("URL"),
                "author": [f"{a.get('given','')} {a.get('family','')}".strip() for a in it.get("author", [])],
                "issued": (it.get("issued", {}).get("date-parts", [[None]])[0][0]),
                "publisher": it.get("publisher"),
            }
        )
    return results
