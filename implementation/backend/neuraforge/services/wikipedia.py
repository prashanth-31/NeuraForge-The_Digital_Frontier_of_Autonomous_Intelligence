"""Wikipedia REST API adapter."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from urllib.parse import quote

from .http import get_json

BASE = "https://en.wikipedia.org/api/rest_v1"


async def search_summary(query: str, lang: str = "en") -> Dict[str, Any]:
    # Use page summary endpoint with title guess
    title = quote(query)
    url = f"{BASE}/page/summary/{title}"
    data = await get_json(url, headers={"Accept-Language": lang})
    return {
        "title": data.get("title"),
        "extract": data.get("extract"),
        "url": data.get("content_urls", {}).get("desktop", {}).get("page"),
    }


async def search_titles(query: str, limit: int = 5, lang: str = "en") -> List[Dict[str, Any]]:
    url = f"{BASE}/page/title/{quote(query)}"
    data = await get_json(url, headers={"Accept-Language": lang})
    items = data.get("items", [])[:limit]
    return [{"title": it.get("title"), "description": it.get("description")} for it in items]
