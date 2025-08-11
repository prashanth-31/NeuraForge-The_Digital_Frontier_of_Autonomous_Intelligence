"""ExchangeRate.host adapter (no key required)."""
from __future__ import annotations

from typing import Any, Dict

from .http import get_json

BASE = "https://api.exchangerate.host"


async def convert(amount: float, from_code: str, to_code: str) -> Dict[str, Any]:
    data = await get_json(
        f"{BASE}/convert",
        params={"from": from_code.upper(), "to": to_code.upper(), "amount": amount},
    )
    return {
        "from": from_code.upper(),
        "to": to_code.upper(),
        "amount": amount,
        "result": (data or {}).get("result"),
        "info": (data or {}).get("info"),
    }
