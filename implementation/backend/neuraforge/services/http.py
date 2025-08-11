"""HTTP client utilities with retries and timeouts."""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

DEFAULT_TIMEOUT = 10.0


class HttpError(Exception):
    pass


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, HttpError)),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
    stop=stop_after_attempt(3),
    reraise=True,
)
async def get_json(url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Any:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        resp = await client.get(url, params=params, headers=headers)
        if resp.status_code >= 400:
            raise HttpError(f"GET {url} -> {resp.status_code}: {resp.text[:200]}")
        if 'application/json' in resp.headers.get('content-type', ''):
            return resp.json()
        # try parse JSON
        try:
            return resp.json()
        except Exception:
            return resp.text


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, HttpError)),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
    stop=stop_after_attempt(3),
    reraise=True,
)
async def get_text(url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> str:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        resp = await client.get(url, params=params, headers=headers)
        if resp.status_code >= 400:
            raise HttpError(f"GET {url} -> {resp.status_code}: {resp.text[:200]}")
        return resp.text
