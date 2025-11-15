from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel
from prometheus_client.parser import text_string_to_metric_families

from app.main import app
from app.mcp.adapters.base import MCPToolAdapter as MCPAdapter
from app.tools.registry import tool_registry

mcp_router = importlib.import_module("app.mcp.router")


def test_tools_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/tools/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "registered_tools" in payload
    assert isinstance(payload["registered_tools"], list)
    assert "aliases" in payload


def test_tools_metrics_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/tools/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert response.text


class _MetricsInput(BaseModel):
    value: str | None = None


class _MetricsOutput(BaseModel):
    echo: str


class _MetricsAdapter(MCPAdapter):
    name = "tests/prometheus"
    description = "Prometheus metrics test adapter"
    labels: tuple[str, ...] = ()
    InputModel = _MetricsInput
    OutputModel = _MetricsOutput

    async def _invoke(self, payload_model: _MetricsInput) -> dict[str, str]:
        return {"echo": payload_model.value or "pong"}


def _reset_registry() -> None:
    tool_registry.clear()
    mcp_router.bootstrap_tool_registry.cache_clear()


@pytest.mark.asyncio
async def test_metrics_exposed(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_registry()
    monkeypatch.setattr(mcp_router, "ALL_ADAPTER_CLASSES", (_MetricsAdapter,))
    client = TestClient(app)

    try:
        invoke_response = client.post("/mcp/tools/tests/prometheus/invoke", json={"value": "ping"})
        assert invoke_response.status_code == 200

        metrics_response = client.get("/tools/metrics")
        assert metrics_response.status_code == 200

        metric_found = False
        for family in text_string_to_metric_families(metrics_response.text):
            if family.name not in {"mcp_tool_invocations_total", "mcp_tool_invocations"}:
                continue
            for sample in family.samples:
                if sample.labels.get("tool") == "tests/prometheus" and sample.labels.get("outcome") == "success":
                    assert sample.value >= 1
                    metric_found = True
                    break
            if metric_found:
                break

        assert metric_found, "Prometheus invocation counter for test adapter not found"
    finally:
        _reset_registry()
