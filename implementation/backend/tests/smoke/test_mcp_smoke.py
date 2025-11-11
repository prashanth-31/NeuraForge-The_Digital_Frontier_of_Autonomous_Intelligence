import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest

from app.core import metrics
from app.core.config import MCPToolSettings, ToolRateLimitSettings
from app.mcp.adapters.finance import YahooFinanceSnapshotAdapter
from app.services.tools import (
    DEFAULT_TOOL_ALIASES,
    MCPToolDescriptor,
    ToolService,
)


@pytest.mark.asyncio
async def test_mcp_aliases_route_to_resolved_tools(monkeypatch: pytest.MonkeyPatch, httpx_mock) -> None:
    settings = MCPToolSettings(
        enabled=True,
        endpoint="http://localhost:6111",
        timeout_seconds=1.0,
        cache_ttl_seconds=0,
        rate_limit=ToolRateLimitSettings(max_calls=100, period_seconds=60),
        healthcheck_path="",
        catalog_path="/tools",
        invoke_path_template="/tools/{tool}/invoke",
        catalog_refresh_seconds=0,
        verify_ssl=False,
        max_retries=0,
        retry_backoff_seconds=0.01,
        retry_jitter_seconds=0.0,
        circuit_breaker_threshold=3,
        circuit_breaker_reset_seconds=1.0,
    )
    service = ToolService(settings)

    monkeypatch.setattr(metrics, "observe_tool_invocation", lambda **_: None)
    monkeypatch.setattr(metrics, "increment_tool_error", lambda **_: None)

    resolved_map = {}
    for alias in DEFAULT_TOOL_ALIASES:
        if alias == "enterprise.playbook":
            continue  # composite tool handled separately
        resolved_map[alias] = service._resolve_tool_identifier(alias)

    descriptors = [MCPToolDescriptor(name=name) for name in sorted(set(resolved_map.values()))]
    monkeypatch.setattr(service, "_fetch_catalog", AsyncMock(return_value=descriptors))
    await service.refresh_catalog(force=True)

    for alias, resolved in resolved_map.items():
        httpx_mock.add_response(
            method="POST",
            url=f"http://localhost:6111/tools/{resolved}/invoke",
            json={"tool": resolved},
        )
        result = await service.invoke(alias, {"prompt": f"smoke:{alias}"})
        assert result.tool == alias
        assert result.resolved_tool == resolved
        assert result.response == {"tool": resolved}

    await service.aclose()


@pytest.mark.asyncio
async def test_yfinance_fallback_emits_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = YahooFinanceSnapshotAdapter()

    recorded: list[tuple[str, str]] = []

    def capture_fallback(*, provider: str, reason: str) -> None:
        recorded.append((provider, reason))

    monkeypatch.setattr(metrics, "increment_finance_quote_fallback", capture_fallback)
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    async def fake_fetch_yfinance(self, symbols):
        return [{"symbol": symbol, "regularMarketPrice": 42.0} for symbol in symbols]

    async def passthrough_attach(self, quotes, symbols):
        return quotes

    async def fake_session(self, force_refresh: bool = False):
        return None, None

    monkeypatch.setattr(YahooFinanceSnapshotAdapter, "_fetch_quotes_via_yfinance", fake_fetch_yfinance)
    monkeypatch.setattr(YahooFinanceSnapshotAdapter, "_fetch_quotes_via_stooq", AsyncMock(return_value=[]))
    monkeypatch.setattr(YahooFinanceSnapshotAdapter, "_attach_fundamentals", passthrough_attach)
    monkeypatch.setattr(YahooFinanceSnapshotAdapter, "_get_cached_quotes", AsyncMock(return_value=None))
    monkeypatch.setattr(YahooFinanceSnapshotAdapter, "_set_quote_cache", AsyncMock())
    monkeypatch.setattr(YahooFinanceSnapshotAdapter, "_ensure_quote_session", fake_session)

    class _ThrottledClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, params=None, headers=None):
            request = httpx.Request("GET", url, params=params, headers=headers)
            return httpx.Response(429, request=request)

    monkeypatch.setattr(httpx, "AsyncClient", _ThrottledClient)

    response = await adapter.invoke({"symbols": ["NEURA"]})

    assert recorded == [("yfinance", "rate_limited")]
    assert response["requested"] == ["NEURA"]
    assert response["metrics"]
    assert response["metrics"][0]["symbol"] == "NEURA"