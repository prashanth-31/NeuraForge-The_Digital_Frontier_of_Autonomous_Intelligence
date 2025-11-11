import inspect

import httpx
import pytest
from unittest.mock import AsyncMock

from app.core.config import MCPToolSettings, ToolRateLimitSettings
from app.core import metrics
from app.services.tools import (
    ToolDisabledError,
    ToolInvocationError,
    ToolService,
    MCPToolDescriptor,
)


@pytest.fixture()
def tool_settings() -> MCPToolSettings:
    return MCPToolSettings(
        enabled=True,
        endpoint="http://localhost:9999",
        api_key=None,
        api_key_header="Authorization",
        auth_scheme="Bearer",
        client_id=None,
        client_secret=None,
        timeout_seconds=0.5,
        cache_ttl_seconds=30,
        rate_limit=ToolRateLimitSettings(max_calls=100, period_seconds=60),
        healthcheck_path="",
        catalog_path="/tools",
        invoke_path_template="/tools/{tool}/invoke",
        catalog_refresh_seconds=0,
        verify_ssl=False,
        extra_headers={},
        aliases={},
        max_retries=3,
        retry_backoff_seconds=1.0,
        retry_jitter_seconds=0.5,
        circuit_breaker_threshold=5,
        circuit_breaker_reset_seconds=60,
        signing_secret=None,
        signing_header="X-MCP-Signature",
        signing_algorithm="hmac-sha256",
    )


@pytest.mark.asyncio
async def test_invoke_uses_cache(monkeypatch: pytest.MonkeyPatch, tool_settings: MCPToolSettings) -> None:
    service = ToolService(tool_settings)
    call_count = {"value": 0}
    observed = []

    async def fake_dispatch(self: ToolService, resolved_tool: str, payload: dict) -> dict:
        call_count["value"] += 1
        return {"ok": True, "tool": resolved_tool, "payload": payload}

    def fake_observe_tool_invocation(*, tool: str, latency: float, cached: bool) -> None:
        observed.append((tool, cached))

    monkeypatch.setattr(ToolService, "_dispatch", fake_dispatch, raising=False)
    monkeypatch.setattr(service, "_fetch_catalog", AsyncMock(return_value=[]))
    monkeypatch.setattr(metrics, "observe_tool_invocation", fake_observe_tool_invocation)
    monkeypatch.setattr(metrics, "increment_tool_error", lambda **_: None)

    payload = {"query": "test"}
    first = await service.invoke("search", payload)
    second = await service.invoke("search", payload)

    assert call_count["value"] == 1
    assert first.cached is False
    assert second.cached is True
    assert observed == [("search", False), ("search", True)]
    assert first.resolved_tool == "search"
    await service.aclose()


@pytest.mark.asyncio
async def test_invoke_disabled(monkeypatch: pytest.MonkeyPatch, tool_settings: MCPToolSettings) -> None:
    tool_settings.enabled = False
    service = ToolService(tool_settings)

    with pytest.raises(ToolDisabledError):
        await service.invoke("search", {"query": "test"})

    await service.aclose()


@pytest.mark.asyncio
async def test_invoke_wraps_http_errors(monkeypatch: pytest.MonkeyPatch, tool_settings: MCPToolSettings) -> None:
    service = ToolService(tool_settings)
    error_logged = {"value": False}

    async def fake_dispatch(self: ToolService, resolved_tool: str, payload: dict) -> dict:
        raise httpx.HTTPError("boom")

    def fake_increment_tool_error(*, tool: str) -> None:
        error_logged["value"] = True

    monkeypatch.setattr(ToolService, "_dispatch", fake_dispatch, raising=False)
    monkeypatch.setattr(service, "_fetch_catalog", AsyncMock(return_value=[]))
    monkeypatch.setattr(metrics, "observe_tool_invocation", lambda **_: None)
    monkeypatch.setattr(metrics, "increment_tool_error", fake_increment_tool_error)

    with pytest.raises(ToolInvocationError):
        await service.invoke("search", {"query": "test"})

    assert error_logged["value"] is True
    await service.aclose()


@pytest.mark.asyncio
async def test_invoke_resolves_alias(monkeypatch: pytest.MonkeyPatch, tool_settings: MCPToolSettings) -> None:
    tool_settings.aliases = {"finance.snapshot": "finance/yfinance"}
    service = ToolService(tool_settings)

    async def fake_dispatch(self: ToolService, resolved_tool: str, payload: dict) -> dict:
        assert resolved_tool == "finance/yfinance"
        return {"metrics": []}

    monkeypatch.setattr(ToolService, "_dispatch", fake_dispatch, raising=False)
    monkeypatch.setattr(service, "_fetch_catalog", AsyncMock(return_value=[]))
    monkeypatch.setattr(metrics, "observe_tool_invocation", lambda **_: None)
    monkeypatch.setattr(metrics, "increment_tool_error", lambda **_: None)

    result = await service.invoke("finance.snapshot", {"symbol": "NEURA"})

    assert result.tool == "finance.snapshot"
    assert result.resolved_tool == "finance/yfinance"
    await service.aclose()


@pytest.mark.asyncio
async def test_tool_service_auth_header_provider(tool_settings: MCPToolSettings) -> None:
    tool_settings.api_key = "token-123"
    service = ToolService(tool_settings)
    provider = service._client._config.auth_header_provider
    assert provider is not None
    maybe_headers = provider()
    headers = await _resolve_maybe_awaitable(maybe_headers)
    assert headers[tool_settings.api_key_header] == "Bearer token-123"
    await service.aclose()


@pytest.mark.asyncio
async def test_tool_service_request_signer(tool_settings: MCPToolSettings) -> None:
    tool_settings.signing_secret = "super-secret"
    service = ToolService(tool_settings)
    signer = service._client._config.request_signer
    assert signer is not None
    maybe_headers = signer("POST", "/foo", {"a": 1}, {"page": 2})
    headers = await _resolve_maybe_awaitable(maybe_headers)
    assert tool_settings.signing_header in headers
    assert isinstance(headers[tool_settings.signing_header], str)
    await service.aclose()


@pytest.mark.asyncio
async def test_tool_service_onboarding_status(tool_settings: MCPToolSettings) -> None:
    service = ToolService(tool_settings)
    service._catalog = {
        "search/duckduckgo": MCPToolDescriptor(name="search/duckduckgo"),
        "finance/yfinance": MCPToolDescriptor(name="finance/yfinance"),
        "creative/stylizer": MCPToolDescriptor(name="creative/stylizer"),
    }
    diagnostics = service.get_diagnostics()
    research = {entry["alias"]: entry for entry in diagnostics["onboarding"]["research"]}
    finance = {entry["alias"]: entry for entry in diagnostics["onboarding"]["finance"]}
    creative = {entry["alias"]: entry for entry in diagnostics["onboarding"].get("creative", [])}
    enterprise = {entry["alias"]: entry for entry in diagnostics["onboarding"].get("enterprise", [])}

    assert research["research.search"]["catalog_present"] is True
    assert research["research.arxiv"]["catalog_present"] is False
    assert finance["finance.snapshot"]["catalog_present"] is True
    assert finance["finance.sentiment"]["catalog_present"] is False
    assert creative["creative.tonecheck"]["catalog_present"] is True
    assert creative["creative.image"]["catalog_present"] is False
    assert enterprise["enterprise.policy"]["catalog_present"] is False

    await service.aclose()


async def _resolve_maybe_awaitable(value):
    if inspect.isawaitable(value):
        return await value
    return value


@pytest.mark.asyncio
async def test_tool_service_emits_events(monkeypatch: pytest.MonkeyPatch, tool_settings: MCPToolSettings) -> None:
    service = ToolService(tool_settings)
    events: list[dict] = []

    async def fake_dispatch(self: ToolService, resolved_tool: str, payload: dict) -> dict:
        assert resolved_tool == "search/duckduckgo"
        return {"results": [{"title": "example"}]}

    monkeypatch.setattr(ToolService, "_dispatch", fake_dispatch, raising=False)
    monkeypatch.setattr(service, "_fetch_catalog", AsyncMock(return_value=[]))
    monkeypatch.setattr(metrics, "observe_tool_invocation", lambda **_: None)
    monkeypatch.setattr(metrics, "increment_tool_error", lambda **_: None)

    async def capture(event: dict) -> None:
        events.append(event)

    async with service.instrument(capture):
        await service.invoke("research.search", {"query": "telemetry"})

    assert events, "expected tool events to be emitted"
    first = events[0]
    assert first["tool"] == "research.search"
    assert first["status"] == "success"
    assert first["cached"] is False
    assert "latency" in first
    await service.aclose()


@pytest.mark.asyncio
async def test_enterprise_playbook_composite(monkeypatch: pytest.MonkeyPatch, tool_settings: MCPToolSettings) -> None:
    service = ToolService(tool_settings)

    async def fake_dispatch(self: ToolService, resolved_tool: str, payload: dict) -> dict:
        if resolved_tool == "enterprise/notion":
            assert payload == {"action": "search", "query": "GTM launch"}
            return {
                "results": [
                    {
                        "page_id": "play-1",
                        "title": "GTM Kickoff Checklist",
                        "snippet": "Align launch activities across sales, marketing, and success.",
                    }
                ]
            }
        if resolved_tool == "enterprise/policy_checker":
            return {
                "findings": [
                    {"policy": "restricted", "status": "pass", "details": "Keyword absent"},
                ],
                "compliant": True,
            }
        raise AssertionError(f"Unexpected resolved tool: {resolved_tool}")

    monkeypatch.setattr(ToolService, "_dispatch", fake_dispatch, raising=False)
    monkeypatch.setattr(service, "_fetch_catalog", AsyncMock(return_value=[]))
    monkeypatch.setattr(metrics, "observe_tool_invocation", lambda **_: None)
    monkeypatch.setattr(metrics, "increment_tool_error", lambda **_: None)

    payload = {
        "prompt": "Outline GTM enablement for the new platform launch",
        "metadata": {"topic": "GTM launch"},
        "prior_outputs": [],
    }

    result = await service.invoke("enterprise.playbook", payload)

    assert result.tool == "enterprise.playbook"
    assert result.resolved_tool == "enterprise/playbook"
    assert result.cached is False
    assert result.response["actions"], "expected composite wrapper to produce playbook actions"
    assert result.response["notion"]["results"][0]["page_id"] == "play-1"
    await service.aclose()