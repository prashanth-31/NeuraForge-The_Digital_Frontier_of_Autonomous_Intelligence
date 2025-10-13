import asyncio
from collections import deque

import httpx
import pytest

from app.core import metrics
from app.services.mcp_client import CircuitOpenError, MCPClient, MCPClientConfig


@pytest.fixture
def noop_sleep(monkeypatch):
    calls = deque()

    async def _sleep(duration: float):
        calls.append(duration)

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    return calls


def _make_config(**overrides):
    config = MCPClientConfig(
        base_url="http://localhost",
        timeout_seconds=0.5,
        max_retries=2,
        retry_backoff_seconds=0.01,
        retry_jitter_seconds=0.0,
        verify_ssl=False,
        default_headers={"X-Test": "true"},
        circuit_breaker_threshold=3,
        circuit_breaker_reset_seconds=0.5,
        instrumentation_hooks=(),
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


@pytest.mark.asyncio
async def test_mcp_client_success_records_metrics(monkeypatch):
    observed = []

    def fake_observe(*, method: str, endpoint: str, status: int | None, success: bool, latency: float) -> None:
        observed.append((method, endpoint, status, success, latency))

    monkeypatch.setattr(metrics, "observe_mcp_request", fake_observe)
    monkeypatch.setattr(metrics, "increment_mcp_circuit_open", lambda **_: None)
    monkeypatch.setattr(metrics, "increment_mcp_circuit_trip", lambda **_: None)

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-Test"] == "true"
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as async_client:
        client = MCPClient(_make_config(), client=async_client)
        response = await client.request("GET", "/tools", trace_id="trace-123")
        assert response.status_code == 200
        assert response.json()["ok"] is True
    assert observed and observed[-1][:4] == ("GET", "/tools", 200, True)


@pytest.mark.asyncio
async def test_mcp_client_retries_on_transient_failure(monkeypatch, noop_sleep):
    observed = []

    def fake_observe(*, method: str, endpoint: str, status: int | None, success: bool, latency: float) -> None:
        observed.append((method, endpoint, status, success))

    trips = []

    monkeypatch.setattr(metrics, "observe_mcp_request", fake_observe)
    monkeypatch.setattr(metrics, "increment_mcp_circuit_open", lambda **_: None)
    monkeypatch.setattr(metrics, "increment_mcp_circuit_trip", lambda **_: trips.append(True))

    call_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(503, json={"detail": "temporary"})
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as async_client:
        client = MCPClient(_make_config(), client=async_client)
        response = await client.request("GET", "/invoke")
        assert response.status_code == 200
        assert call_count == 2

    assert observed[0] == ("GET", "/invoke", 503, False)
    assert observed[-1] == ("GET", "/invoke", 200, True)
    assert noop_sleep  # ensure retry awaited at least once
    assert not trips


@pytest.mark.asyncio
async def test_mcp_client_circuit_breaker_blocks_requests(monkeypatch):
    observed = []
    opens = []
    trips = []

    monkeypatch.setattr(metrics, "observe_mcp_request", lambda **kwargs: observed.append(kwargs))
    monkeypatch.setattr(metrics, "increment_mcp_circuit_open", lambda **kwargs: opens.append(True))
    monkeypatch.setattr(metrics, "increment_mcp_circuit_trip", lambda **kwargs: trips.append(True))

    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    transport = httpx.MockTransport(handler)
    config = _make_config(max_retries=0, circuit_breaker_threshold=2, circuit_breaker_reset_seconds=10.0)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as async_client:
        client = MCPClient(config, client=async_client)
        with pytest.raises(httpx.ConnectError):
            await client.request("GET", "/unstable")
        with pytest.raises(httpx.ConnectError):
            await client.request("GET", "/unstable")
        assert trips  # breaker tripped after second failure
        with pytest.raises(CircuitOpenError):
            await client.request("GET", "/unstable")

    assert opens  # third attempt blocked by circuit
    assert len(observed) == 2


@pytest.mark.asyncio
async def test_mcp_client_injects_dynamic_auth_headers(monkeypatch):
    monkeypatch.setattr(metrics, "observe_mcp_request", lambda **_: None)
    monkeypatch.setattr(metrics, "increment_mcp_circuit_open", lambda **_: None)
    monkeypatch.setattr(metrics, "increment_mcp_circuit_trip", lambda **_: None)

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer dynamic-token"
        assert request.headers["X-Test"] == "true"
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)

    async def auth_provider() -> dict[str, str]:
        return {"Authorization": "Bearer dynamic-token"}

    config = _make_config(auth_header_provider=auth_provider)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as async_client:
        client = MCPClient(config, client=async_client)
        response = await client.request("GET", "/secure")
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_mcp_client_applies_request_signature(monkeypatch):
    monkeypatch.setattr(metrics, "observe_mcp_request", lambda **_: None)
    monkeypatch.setattr(metrics, "increment_mcp_circuit_open", lambda **_: None)
    monkeypatch.setattr(metrics, "increment_mcp_circuit_trip", lambda **_: None)

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-Signature"] == "signed:POST:/sign/me"
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)

    def signer(method: str, endpoint: str, payload: dict | None, params: dict | None) -> dict[str, str]:
        assert method == "POST"
        assert endpoint == "/sign/me"
        return {"X-Signature": f"signed:{method}:{endpoint}"}

    config = _make_config(request_signer=signer)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as async_client:
        client = MCPClient(config, client=async_client)
        response = await client.request("POST", "sign/me", json={"foo": "bar"})
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_mcp_client_emits_instrumentation_events(monkeypatch):
    monkeypatch.setattr(metrics, "observe_mcp_request", lambda **_: None)
    monkeypatch.setattr(metrics, "increment_mcp_circuit_open", lambda **_: None)
    monkeypatch.setattr(metrics, "increment_mcp_circuit_trip", lambda **_: None)

    events = []

    def hook(payload: dict):
        events.append(payload)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)
    config = _make_config(instrumentation_hooks=(hook,))
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as async_client:
        client = MCPClient(config, client=async_client)
        await client.request("GET", "/instrument")

    event_names = {event["event"] for event in events}
    assert {"request.start", "request.success"}.issubset(event_names)