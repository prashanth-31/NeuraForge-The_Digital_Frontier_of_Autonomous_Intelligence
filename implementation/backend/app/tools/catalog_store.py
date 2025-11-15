from __future__ import annotations

import json
import contextlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.core import metrics


@dataclass(slots=True, frozen=True)
class ToolCatalogEntry:
    name: str
    description: str
    labels: tuple[str, ...]
    aliases: tuple[str, ...]
    capabilities: tuple[str, ...]
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any]


@dataclass(slots=True, frozen=True)
class ToolCatalogSnapshot:
    generated_at: datetime
    entries: tuple[ToolCatalogEntry, ...]
    aliases: dict[str, str]
    capabilities: dict[str, tuple[str, ...]]
    source: str


class ToolCatalogStore:
    """In-memory store tracking the MCP tool catalog metadata and snapshots."""

    def __init__(self) -> None:
        self._logger = get_logger(name=__name__)
        self._entries: dict[str, ToolCatalogEntry] = {}
        self._aliases: dict[str, str] = {}
        self._capabilities: dict[str, tuple[str, ...]] = {}
        self._snapshot: ToolCatalogSnapshot | None = None

    def clear(self) -> None:
        """Reset cached catalog data."""

        self._entries.clear()
        self._aliases.clear()
        self._capabilities.clear()
        self._snapshot = None
        metrics.observe_tool_catalog_entries(0)

    def sync(
        self,
        entries: Sequence[ToolCatalogEntry],
        *,
        aliases: Mapping[str, str] | None = None,
        source: str,
    ) -> ToolCatalogSnapshot:
        """Replace the in-memory catalog with the supplied entries and aliases."""

        normalized_aliases = {
            str(key): str(value)
            for key, value in (aliases or {}).items()
            if isinstance(key, str) and key.strip() and isinstance(value, str) and value.strip()
        }
        ordered_entries = tuple(sorted(entries, key=lambda entry: entry.name))
        capabilities = {
            entry.name: tuple(cap for cap in entry.capabilities if cap)
            for entry in ordered_entries
        }
        snapshot = ToolCatalogSnapshot(
            generated_at=datetime.now(timezone.utc),
            entries=ordered_entries,
            aliases=dict(sorted(normalized_aliases.items())),
            capabilities=capabilities,
            source=source,
        )
        self._entries = {entry.name: entry for entry in ordered_entries}
        self._aliases = dict(snapshot.aliases)
        self._capabilities = dict(snapshot.capabilities)
        self._snapshot = snapshot
        metrics.observe_tool_catalog_entries(len(ordered_entries))
        return snapshot

    def snapshot(self) -> ToolCatalogSnapshot | None:
        return self._snapshot

    def entries(self) -> list[ToolCatalogEntry]:
        return list(self._entries.values())

    def entry_for(self, name: str) -> ToolCatalogEntry | None:
        return self._entries.get(name)

    def alias_map(self) -> dict[str, str]:
        return dict(self._aliases)

    def capabilities_for(self, name: str) -> tuple[str, ...]:
        return self._capabilities.get(name, ())

    def persist(
        self,
        snapshot: ToolCatalogSnapshot | None = None,
        *,
        settings: Settings | None = None,
    ) -> None:
        """Persist the supplied snapshot (or the current one) to the configured locations."""

        materialized = snapshot or self._snapshot
        if materialized is None:
            return
        active_settings = settings or get_settings()
        target_path = active_settings.tools.mcp.snapshot_path
        history_dir = active_settings.tools.mcp.snapshot_history_dir
        if not target_path and not history_dir:
            return

        payload = self._serialize_snapshot(materialized)

        if target_path:
            path = Path(target_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            except Exception as exc:  # pragma: no cover - filesystem failure guard
                self._logger.warning("tool_catalog_snapshot_write_failed", path=str(path), error=str(exc))

        if history_dir:
            directory = Path(history_dir)
            directory.mkdir(parents=True, exist_ok=True)
            timestamp = materialized.generated_at.strftime("%Y%m%dT%H%M%S")
            history_path = directory / f"catalog-{timestamp}.json"
            try:
                history_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            except Exception as exc:  # pragma: no cover - filesystem failure guard
                self._logger.warning("tool_catalog_history_write_failed", path=str(history_path), error=str(exc))
            else:
                self._enforce_history_limit(directory, active_settings.tools.mcp.snapshot_history_limit)

    def _enforce_history_limit(self, directory: Path, limit: int) -> None:
        sanitized_limit = max(0, limit)
        if sanitized_limit == 0:
            return
        try:
            history_files = sorted(
                path for path in directory.iterdir() if path.is_file() and path.suffix == ".json"
            )
        except FileNotFoundError:  # pragma: no cover - directory concurrently removed
            return
        excess = len(history_files) - sanitized_limit
        for candidate in history_files[:excess]:
            with contextlib.suppress(Exception):
                candidate.unlink()

    def _serialize_snapshot(self, snapshot: ToolCatalogSnapshot) -> dict[str, Any]:
        entries_payload = []
        for entry in snapshot.entries:
            try:
                input_schema = json.loads(json.dumps(entry.input_schema, ensure_ascii=True, sort_keys=True))
            except (TypeError, ValueError):  # pragma: no cover - defensive guard
                input_schema = dict(entry.input_schema)
            try:
                output_schema = json.loads(json.dumps(entry.output_schema, ensure_ascii=True, sort_keys=True))
            except (TypeError, ValueError):  # pragma: no cover - defensive guard
                output_schema = dict(entry.output_schema)
            entries_payload.append(
                {
                    "name": entry.name,
                    "description": entry.description,
                    "labels": list(entry.labels),
                    "aliases": list(entry.aliases),
                    "capabilities": list(entry.capabilities),
                    "input_schema": input_schema,
                    "output_schema": output_schema,
                }
            )
        summary = {
            "entries": len(entries_payload),
            "aliases": len(snapshot.aliases),
            "capability_tags": sum(len(caps) for caps in snapshot.capabilities.values()),
        }
        return {
            "generated_at": snapshot.generated_at.isoformat(),
            "source": snapshot.source,
            "summary": summary,
            "aliases": dict(snapshot.aliases),
            "capabilities": {name: list(caps) for name, caps in snapshot.capabilities.items()},
            "entries": entries_payload,
        }


tool_catalog_store = ToolCatalogStore()

__all__ = [
    "ToolCatalogEntry",
    "ToolCatalogSnapshot",
    "ToolCatalogStore",
    "tool_catalog_store",
]
