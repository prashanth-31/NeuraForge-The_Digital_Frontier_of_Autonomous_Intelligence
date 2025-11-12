import os
from http import HTTPStatus

import httpx
import pytest
from pytest_httpx import HTTPXMock

pytestmark = pytest.mark.asyncio


RESEARCH_SEARCH_ALIASES = {"search/tavily", "search/duckduckgo"}
RESEARCH_REQUIRED_TOOLS = {
    "research/arxiv",
    "research/wikipedia",
    "research/doc_loader",
    "research/qdrant",
    "research/summarizer",
}
EXPECTED_FINANCE_TOOLS = {
    "finance/alpha_vantage",
    "finance/yfinance",
    "finance/pandas",
    "finance/plot",
    "finance/coingecko_news",
    "finance/csv",
    "finance/finbert",
}
EXPECTED_CREATIVE_TOOLS = {
    "creative/stylizer",
    "creative/tone_checker",
    "creative/whisper_transcription",
    "creative/image_generator",
}
EXPECTED_ENTERPRISE_TOOLS = {
    "enterprise/notion",
    "enterprise/calendar",
    "enterprise/policy_checker",
    "enterprise/crm",
}

LIVE_BASE_URL = os.getenv("MCP_CATALOG_URL") or ""
LIVE_API_KEY = os.getenv("MCP_CATALOG_API_KEY") or ""


async def _fetch_live_catalog() -> dict:
    headers = {}
    if LIVE_API_KEY:
        headers["Authorization"] = f"Bearer {LIVE_API_KEY}"
    async with httpx.AsyncClient(timeout=10.0, verify=True) as client:
        response = await client.get(f"{LIVE_BASE_URL.rstrip('/')}/tools", headers=headers)
        response.raise_for_status()
        return response.json()


async def _assert_catalog_contains(payload: dict) -> None:
    entries = payload.get("tools") or []
    names = {item.get("name") for item in entries if isinstance(item, dict)}
    missing_research = RESEARCH_REQUIRED_TOOLS - names
    missing_finance = EXPECTED_FINANCE_TOOLS - names
    missing_creative = EXPECTED_CREATIVE_TOOLS - names
    missing_enterprise = EXPECTED_ENTERPRISE_TOOLS - names
    assert RESEARCH_SEARCH_ALIASES & names, (
        "Missing research search tool alias in catalog: expected one of "
        f"{sorted(RESEARCH_SEARCH_ALIASES)}"
    )
    assert not missing_research, f"Missing research tools in catalog: {sorted(missing_research)}"
    assert not missing_finance, f"Missing finance tools in catalog: {sorted(missing_finance)}"
    assert not missing_creative, f"Missing creative tools in catalog: {sorted(missing_creative)}"
    assert not missing_enterprise, f"Missing enterprise tools in catalog: {sorted(missing_enterprise)}"


async def test_catalog_includes_research_and_finance_tools(httpx_mock: HTTPXMock) -> None:
    if LIVE_BASE_URL:
        payload = await _fetch_live_catalog()
        await _assert_catalog_contains(payload)
        return

    expected_names = (
        RESEARCH_REQUIRED_TOOLS
        | RESEARCH_SEARCH_ALIASES
        | EXPECTED_FINANCE_TOOLS
        | EXPECTED_CREATIVE_TOOLS
        | EXPECTED_ENTERPRISE_TOOLS
    )
    catalog_payload = {
        "tools": [{"name": name, "description": "mock"} for name in expected_names]
    }
    for _ in range(2):
        httpx_mock.add_response(
            method="GET",
            url="http://localhost:6111/tools",
            status_code=HTTPStatus.OK,
            json=catalog_payload,
        )
    httpx_mock.add_response(
        method="GET",
        url="http://localhost:6111/health",
        status_code=HTTPStatus.OK,
        json={"status": "ok"},
    )

    from app.services.tools import ToolService
    from app.core.config import MCPToolSettings, ToolRateLimitSettings

    settings = MCPToolSettings(
        enabled=True,
        endpoint="http://localhost:6111",
        timeout_seconds=1.0,
        cache_ttl_seconds=0,
        healthcheck_path="/health",
        catalog_path="/tools",
        invoke_path_template="/tools/{tool}/invoke",
        catalog_refresh_seconds=0,
        verify_ssl=False,
        rate_limit=ToolRateLimitSettings(max_calls=10, period_seconds=60),
        max_retries=0,
        retry_backoff_seconds=0.01,
        retry_jitter_seconds=0.0,
        circuit_breaker_threshold=3,
        circuit_breaker_reset_seconds=1.0,
        api_key_header="Authorization",
        auth_scheme="Bearer",
        signing_header="X-MCP-Signature",
        signing_algorithm="hmac-sha256",
    )
    service = ToolService(settings)
    await service.initialize(validate=True)
    catalog = await service.refresh_catalog(force=True)
    for expected in expected_names:
        assert expected in catalog
    await service.aclose()
