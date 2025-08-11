"""Wikidata search adapter (no key)."""
from __future__ import annotations

from typing import Any, Dict, List

from .http import get_json

BASE = "https://www.wikidata.org/w/api.php"


async def search_entities(query: str, language: str = "en", limit: int = 5) -> List[Dict[str, Any]]:
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "search": query,
        "language": language,
        "limit": limit,
    }
    data = await get_json(BASE, params=params)
    results = []
    for it in (data or {}).get("search", [])[:limit]:
        results.append(
            {
                "id": it.get("id"),
                "label": it.get("label"),
                "description": it.get("description"),
            }
        )
    return results
