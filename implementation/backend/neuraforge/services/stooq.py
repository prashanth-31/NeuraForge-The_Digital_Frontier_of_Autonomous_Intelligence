"""Stooq CSV price adapter (no key)."""
from __future__ import annotations

from typing import Any, Dict, List
import csv
import io

from .http import get_text

BASE = "https://stooq.com"


async def daily_ohlc(symbol: str) -> List[Dict[str, Any]]:
    # Stooq symbols are typically lowercase. Example: aapl
    s = symbol.lower()
    url = f"{BASE}/q/d/l/"  # daily list
    text = await get_text(url, params={"s": s, "i": "d"})
    # CSV columns: Date,Open,High,Low,Close,Volume
    reader = csv.DictReader(io.StringIO(text))
    rows: List[Dict[str, Any]] = []
    for r in reader:
        rows.append(
            {
                "date": r.get("Date"),
                "open": float(r.get("Open") or 0) if r.get("Open") else None,
                "high": float(r.get("High") or 0) if r.get("High") else None,
                "low": float(r.get("Low") or 0) if r.get("Low") else None,
                "close": float(r.get("Close") or 0) if r.get("Close") else None,
                "volume": int(r.get("Volume") or 0) if r.get("Volume") else None,
            }
        )
    return rows
