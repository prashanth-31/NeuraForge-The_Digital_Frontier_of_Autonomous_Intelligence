from __future__ import annotations

import asyncio

import pytest

from app.tools.registry import ToolRegistry


class DummyAdapter:
    async def invoke(self, payload):  # pragma: no cover - not used directly
        return payload


def test_registry_normalizes_names() -> None:
    registry = ToolRegistry()
    adapter = DummyAdapter()
    registry.register("Research/DuckDuckGo", adapter)

    assert registry.get("research/duckduckgo") is adapter
    assert registry.get("research.duckduckgo") is adapter
    assert registry.resolve("research.search") is None
    assert "research/duckduckgo" in registry.list()


def test_registry_alias_resolution() -> None:
    registry = ToolRegistry()
    adapter = DummyAdapter()
    registry.register("search/duckduckgo", adapter)
    registry.register_alias("research.search", "search/duckduckgo")

    assert registry.resolve("research.search") == "search/duckduckgo"
    assert registry.get("research.search") is adapter
    aliases = registry.aliases()
    assert "research.search" in aliases
    assert aliases["research.search"] == "search/duckduckgo"


@pytest.mark.asyncio
async def test_registry_circuit_breaker_recovers() -> None:
    registry = ToolRegistry()
    adapter = DummyAdapter()
    registry.register("search/duckduckgo", adapter)
    registry.configure_circuit(threshold=1, reset_seconds=0.01)

    assert registry.record_failure("search/duckduckgo") is True
    assert registry.is_circuit_open("search/duckduckgo") is True

    await asyncio.sleep(0.02)
    assert registry.is_circuit_open("search/duckduckgo") is False
    registry.record_success("search/duckduckgo")
    assert registry.failure_count("search/duckduckgo") == 0
