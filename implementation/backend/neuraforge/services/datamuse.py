"""Datamuse API adapter."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .http import get_json

BASE = "https://api.datamuse.com"


async def similar_meaning(words: str, max_results: int = 10) -> List[str]:
    data = await get_json(f"{BASE}/words", params={"ml": words, "max": max_results})
    return [d.get("word") for d in data or []]


async def rhymes(word: str, max_results: int = 10) -> List[str]:
    data = await get_json(f"{BASE}/words", params={"rel_rhy": word, "max": max_results})
    return [d.get("word") for d in data or []]
