from __future__ import annotations

import pytest

from app.tools import base as tools_base
from app.tools.base import MCPToolAdapter, ToolInvocationError
from app.tools.registry import ToolRegistry


class NonSerializableAdapter(MCPToolAdapter):
    name = "tests/nonserial"

    async def _run(self, payload):
        return {"value": {1, 2}}


class FlakyAdapter(MCPToolAdapter):
    name = "tests/flaky"

    def __init__(self) -> None:
        self.calls = 0

    async def _run(self, payload):
        self.calls += 1
        if self.calls < 2:
            raise RuntimeError("boom")
        return {"ok": True}


@pytest.mark.asyncio
async def test_tool_returns_json_serializable(monkeypatch) -> None:
    registry = ToolRegistry()
    registry.configure_circuit(threshold=5, reset_seconds=1.0)
    monkeypatch.setattr(tools_base, "tool_registry", registry)

    adapter = NonSerializableAdapter()
    registry.register(adapter.name, adapter)
    adapter._backoff_base = 0.01

    result = await adapter.invoke({"value": "ignored"})
    assert result == {"value": [1, 2]}


@pytest.mark.asyncio
async def test_invoke_retries(monkeypatch) -> None:
    registry = ToolRegistry()
    registry.configure_circuit(threshold=3, reset_seconds=1.0)
    monkeypatch.setattr(tools_base, "tool_registry", registry)

    adapter = FlakyAdapter()
    registry.register(adapter.name, adapter)
    adapter._backoff_base = 0.01

    result = await adapter.invoke({})
    assert result == {"ok": True}
    assert adapter.calls == 2

    registry.record_failure(adapter.name)
    registry.record_failure(adapter.name)
    registry.record_failure(adapter.name)
    assert registry.is_circuit_open(adapter.name) is True
    with pytest.raises(ToolInvocationError):
        await adapter.invoke({})
