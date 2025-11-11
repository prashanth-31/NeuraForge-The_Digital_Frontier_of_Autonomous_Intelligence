from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.mark.parametrize("search_alias", ["search/tavily", "search/duckduckgo"])
def test_mcp_diagnostics_enabled(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
    search_alias: str,
) -> None:
    diagnostics_payload = {
        "enabled": True,
        "endpoint": "http://mock",
        "catalog_size": 2,
        "aliases": {"research.search": search_alias},
        "last_health": {"status": "ok", "timestamp": "2025-10-13T00:00:00+00:00", "error": None},
        "last_catalog_refresh": "2025-10-13T00:00:05+00:00",
        "last_error": None,
        "last_invocation": {
            "tool": "research.search",
            "resolved": search_alias,
            "cached": False,
            "latency": 0.42,
            "timestamp": "2025-10-13T00:00:10+00:00",
        },
        "circuit": {"is_open": False, "seconds_until_close": 0.0, "failure_streak": 0},
    }

    mock_service = SimpleNamespace(get_diagnostics=lambda: diagnostics_payload)

    async def _fake_get_tool_service():
        return mock_service

    monkeypatch.setattr("app.api.routes.get_tool_service", _fake_get_tool_service)

    base_settings = get_settings().model_copy(deep=True)
    base_settings.tools.mcp.enabled = True
    app.dependency_overrides[get_settings] = lambda: base_settings

    response = client.get("/api/v1/diagnostics/mcp")
    assert response.status_code == 200
    assert response.json() == diagnostics_payload

    app.dependency_overrides.pop(get_settings, None)


def test_mcp_diagnostics_disabled(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    base_settings = get_settings().model_copy(deep=True)
    base_settings.tools.mcp.enabled = False
    app.dependency_overrides[get_settings] = lambda: base_settings

    response = client.get("/api/v1/diagnostics/mcp")
    assert response.status_code == 200
    assert response.json() == {"enabled": False}

    app.dependency_overrides.pop(get_settings, None)
