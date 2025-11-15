from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.tools.catalog_store import ToolCatalogEntry, ToolCatalogStore
from app.tools.validation import ToolPayloadValidationError, ToolPayloadValidator


def test_tool_catalog_store_sync_and_persist(tmp_path) -> None:
    store = ToolCatalogStore()
    entries = [
        ToolCatalogEntry(
            name="alpha/tool",
            description="Alpha tool",
            labels=("research",),
            aliases=("alpha.alias",),
            capabilities=("search",),
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            output_schema={"type": "object"},
        )
    ]
    snapshot = store.sync(entries, aliases={"alpha.alias": "alpha/tool"}, source="test")
    assert snapshot.entries[0].name == "alpha/tool"
    settings = Settings()
    settings.tools.mcp.snapshot_path = str(tmp_path / "catalog.json")
    settings.tools.mcp.snapshot_history_dir = str(tmp_path / "history")
    settings.tools.mcp.snapshot_history_limit = 2
    store.persist(snapshot, settings=settings)
    snapshot_path = tmp_path / "catalog.json"
    assert snapshot_path.exists()
    payload = json.loads(snapshot_path.read_text())
    assert payload["summary"]["entries"] == 1
    history_dir = tmp_path / "history"
    assert list(history_dir.glob("catalog-*.json"))


def test_tool_payload_validator_enforces_schema() -> None:
    validator = ToolPayloadValidator()
    descriptor = SimpleNamespace(
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        }
    )
    try:
        validator.validate(
            resolved_tool="alpha/tool",
            payload={"query": "ok"},
            descriptor=descriptor,
            catalog_entry=None,
            max_string_length=32,
        )
    except ToolPayloadValidationError:  # pragma: no cover - defensive guard
        assert False, "Validator should accept valid payload"
    with pytest.raises(ToolPayloadValidationError):
        validator.validate(
            resolved_tool="alpha/tool",
            payload={},
            descriptor=descriptor,
            catalog_entry=None,
            max_string_length=32,
        )
