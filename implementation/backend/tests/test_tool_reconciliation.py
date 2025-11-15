from __future__ import annotations

import importlib

import pytest
from pydantic import BaseModel

from app.mcp.adapters.base import MCPToolAdapter
from app.services.tool_onboarding import PlannedTool
from app.services.tool_reconciliation import ToolReconciliationJob
from app.tools.catalog_store import tool_catalog_store
from app.tools.registry import tool_registry


class _InputModel(BaseModel):
    query: str


class _OutputModel(BaseModel):
    result: str


class _TestAdapter(MCPToolAdapter):
    name = "alpha/tool"
    description = "Alpha test tool"
    labels = ("research",)
    aliases = ("alpha.alias",)
    capabilities = ("search",)
    InputModel = _InputModel
    OutputModel = _OutputModel

    async def _invoke(self, payload_model: _InputModel) -> dict[str, str]:
        return {"result": payload_model.query}


@pytest.mark.asyncio
async def test_tool_reconciliation_detects_mismatches(monkeypatch: pytest.MonkeyPatch) -> None:
    tool_registry.clear()
    tool_catalog_store.clear()
    router = importlib.import_module("app.mcp.router")
    router.bootstrap_tool_registry.cache_clear()
    monkeypatch.setattr(router, "ALL_ADAPTER_CLASSES", (_TestAdapter,))
    monkeypatch.setattr(
        importlib.import_module("app.services.tools"),
        "DEFAULT_TOOL_ALIASES",
        {},
    )
    planned = [
        PlannedTool(alias="alpha.alias", resolved="alpha/tool", category="research", description="Alpha"),
        PlannedTool(alias="missing.alias", resolved="missing/tool", category="research", description="Missing"),
    ]
    monkeypatch.setattr(
        importlib.import_module("app.services.tool_reconciliation"),
        "all_planned_tools",
        lambda: planned,
    )

    result = await ToolReconciliationJob.run_once()

    assert "missing.alias" in result.missing_aliases
    assert "missing/tool" in result.missing_catalog_entries
    assert not result.alias_mismatches

    tool_registry.clear()
    tool_catalog_store.clear()
    router.bootstrap_tool_registry.cache_clear()
