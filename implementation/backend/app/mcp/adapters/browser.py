from __future__ import annotations

import re
from datetime import UTC, datetime
from html import unescape
from typing import Any, Mapping

import httpx
from pydantic import BaseModel, Field, HttpUrl, model_validator

from .base import MCPToolAdapter


class BrowserOpenInput(BaseModel):
    url: HttpUrl | None = Field(default=None, description="Target URL to fetch.")
    method: str = Field("GET", pattern=r"^(GET|HEAD)$")
    timeout_seconds: float = Field(8.0, ge=0.5, le=30.0)
    headers: Mapping[str, str] | None = Field(default=None)
    max_bytes: int = Field(250_000, ge=1024, le=2_000_000)
    html: str | None = Field(default=None, description="Optional literal HTML payload (bypasses network call).")

    @model_validator(mode="after")
    def ensure_source(self) -> "BrowserOpenInput":
        if not self.url and not (self.html and self.html.strip()):
            raise ValueError("Either url or html must be supplied")
        return self

    model_config = {"extra": "forbid"}


class BrowserOpenOutput(BaseModel):
    url: str | None
    status_code: int | None
    headers: Mapping[str, str]
    content: str | None
    encoding: str | None
    fetched_at: datetime
    from_cache: bool


class BrowserOpenAdapter(MCPToolAdapter):
    name = "browser/open"
    description = "Fetches remote content or returns supplied HTML payload without external dependencies."
    labels = ("browser", "http")
    aliases = ("browser.open",)
    capabilities = ("browser", "fetch")
    InputModel = BrowserOpenInput
    OutputModel = BrowserOpenOutput

    async def _invoke(self, payload_model: BrowserOpenInput) -> Mapping[str, Any]:
        if payload_model.html is not None:
            content = payload_model.html[: payload_model.max_bytes]
            return {
                "url": str(payload_model.url) if payload_model.url else None,
                "status_code": None,
                "headers": {},
                "content": content,
                "encoding": "utf-8" if content.isascii() else None,
                "fetched_at": datetime.now(UTC),
                "from_cache": True,
            }

        assert payload_model.url is not None
        timeout = httpx.Timeout(payload_model.timeout_seconds)
        headers = dict(payload_model.headers or {})
        method = payload_model.method.upper()
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.request(method, str(payload_model.url), headers=headers)
        content_bytes = response.content[: payload_model.max_bytes]
        try:
            content_text = content_bytes.decode(response.encoding or "utf-8", errors="replace")
        except LookupError:
            content_text = content_bytes.decode("utf-8", errors="replace")
        limited_headers = {key: value for key, value in response.headers.items() if len(value) < 2048}
        return {
            "url": str(payload_model.url),
            "status_code": response.status_code,
            "headers": limited_headers,
            "content": content_text if method != "HEAD" else None,
            "encoding": response.encoding,
            "fetched_at": datetime.now(UTC),
            "from_cache": False,
        }


class BrowserExtractInput(BaseModel):
    html: str = Field(..., min_length=5, description="HTML markup to normalise.")
    collapse_whitespace: bool = Field(True)
    max_length: int = Field(20_000, ge=128, le=200_000)

    model_config = {"extra": "forbid"}


class BrowserExtractOutput(BaseModel):
    text: str
    length: int
    truncated: bool


class BrowserExtractTextAdapter(MCPToolAdapter):
    name = "browser/extract_text"
    description = "Strips HTML tags and normalises whitespace for downstream processing."
    labels = ("browser", "text")
    aliases = ("browser.extract_text",)
    capabilities = ("browser", "text")
    InputModel = BrowserExtractInput
    OutputModel = BrowserExtractOutput

    _TAG_PATTERN = re.compile(r"<[^>]+>")
    _SCRIPT_STYLE_PATTERN = re.compile(r"<(script|style)(.|\n)*?</(script|style)>", re.IGNORECASE)

    async def _invoke(self, payload_model: BrowserExtractInput) -> Mapping[str, Any]:
        sanitized = self._SCRIPT_STYLE_PATTERN.sub(" ", payload_model.html)
        sanitized = self._TAG_PATTERN.sub(" ", sanitized)
        normalized = unescape(sanitized)
        if payload_model.collapse_whitespace:
            normalized = re.sub(r"\s+", " ", normalized).strip()
        truncated = len(normalized) > payload_model.max_length
        if truncated:
            normalized = normalized[: payload_model.max_length].rstrip()
        return {
            "text": normalized,
            "length": len(normalized),
            "truncated": truncated,
        }


BROWSER_ADAPTER_CLASSES: tuple[type[MCPToolAdapter], ...] = (
    BrowserOpenAdapter,
    BrowserExtractTextAdapter,
)


__all__ = [
    "BrowserOpenAdapter",
    "BrowserExtractTextAdapter",
    "BROWSER_ADAPTER_CLASSES",
]
