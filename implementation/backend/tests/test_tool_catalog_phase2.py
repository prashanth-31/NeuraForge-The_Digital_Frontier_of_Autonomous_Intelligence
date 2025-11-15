import asyncio
from typing import Any

import httpx
import pytest

from app.core.config import MCPToolSettings
from app.services.tool_onboarding import PlannedTool
from app.services.tool_reconciliation import ToolReconciliationJob
from app.services.tools import ToolService
from app.tools.catalog_store import ToolCatalogEntry, ToolCatalogStore, tool_catalog_store
from app.tools.registry import tool_registry


@pytest.mark.asyncio
async def test_tool_service_refresh_populates_catalog_store(monkeypatch: pytest.MonkeyPatch) -> None:
    tool_catalog_store.clear()

    persisted_snapshots: list[Any] = []

    def persist_stub(self: ToolCatalogStore, snapshot=None, *, settings=None) -> None:  # type: ignore[override]
        persisted_snapshots.append(snapshot or self._snapshot)

    monkeypatch.setattr(ToolCatalogStore, "persist", persist_stub)

    payload = {
        "tools": [
            {
                "name": "research/doc_loader",
                "description": "Load textual documents",
                "input_schema": {"title": "DocLoaderInput", "type": "object"},
                "output_schema": {"title": "DocLoaderOutput", "type": "object"},
                "labels": ["research"],
                "aliases": ["research.doc_loader"],
                "capabilities": ["ingest", "chunk"],
            }
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/tools":
            return httpx.Response(200, json=payload)
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="http://mcp-test")
    settings = MCPToolSettings(enabled=True, endpoint="http://mcp-test", use_local_router=False)
    service = ToolService(settings, http_client=client)

    try:
        await service.initialize(validate=True)
        snapshot = tool_catalog_store.snapshot()
        assert snapshot is not None
        assert snapshot.aliases["research.doc_loader"] == "research/doc_loader"
        entry = tool_catalog_store.entry_for("research/doc_loader")
        assert entry is not None
        assert entry.capabilities == ("ingest", "chunk")
        assert persisted_snapshots
    finally:
        await service.aclose()

    tool_catalog_store.clear()


@pytest.mark.asyncio
async def test_tool_reconciliation_job_detects_alignment(monkeypatch: pytest.MonkeyPatch) -> None:
    tool_catalog_store.clear()
    tool_registry.clear()

    entry = ToolCatalogEntry(
        name="research/doc_loader",
        description="",
        labels=("research",),
        aliases=("research.doc_loader",),
        capabilities=("ingest",),
        input_schema={},
        output_schema={},
    )

    def bootstrap_stub() -> bool:
        tool_catalog_store.clear()
        tool_registry.clear()
        tool_catalog_store.sync([entry], aliases={"research.doc_loader": entry.name}, source="test_bootstrap")
        class _Adapter:
            name = entry.name

        tool_registry.register(entry.name, _Adapter(), aliases=entry.aliases)
        return True

    monkeypatch.setattr("app.services.tool_reconciliation.bootstrap_tool_registry", bootstrap_stub)
    monkeypatch.setattr(
        "app.services.tool_reconciliation.all_planned_tools",
        lambda: [PlannedTool(alias="research.doc_loader", resolved="research/doc_loader", category="research", description="")],
    )

    result = await ToolReconciliationJob.run_once()
    assert result.missing_aliases == ()
    assert result.alias_mismatches == ()
    assert result.missing_catalog_entries == ()
    assert result.catalog_only_tools == ()

    tool_catalog_store.clear()
    tool_registry.clear()


@pytest.mark.asyncio
async def test_tool_reconciliation_job_flags_missing_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    tool_catalog_store.clear()
    tool_registry.clear()

    entry = ToolCatalogEntry(
        name="finance/yfinance",
        description="",
        labels=("finance",),
        aliases=(),
        capabilities=("snapshot",),
        input_schema={},
        output_schema={},
    )

    def bootstrap_stub() -> bool:
        tool_catalog_store.clear()
        tool_registry.clear()
        tool_catalog_store.sync([entry], aliases={}, source="test_bootstrap")
        class _Adapter:
            name = entry.name

        tool_registry.register(entry.name, _Adapter(), aliases=())
        return True

    monkeypatch.setattr("app.services.tool_reconciliation.bootstrap_tool_registry", bootstrap_stub)
    monkeypatch.setattr(
        "app.services.tool_reconciliation.all_planned_tools",
        lambda: [PlannedTool(alias="finance.snapshot", resolved="finance/yfinance", category="finance", description="")],
    )

    result = await ToolReconciliationJob.run_once()
    assert result.missing_aliases == ("finance.snapshot",)
    assert result.alias_mismatches == ()
    assert result.missing_catalog_entries == ()
    assert result.catalog_only_tools == ()

    tool_catalog_store.clear()
    tool_registry.clear()
