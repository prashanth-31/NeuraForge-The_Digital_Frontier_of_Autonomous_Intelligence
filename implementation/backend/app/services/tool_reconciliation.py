from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from app.core import metrics
from app.core.logging import get_logger
from app.mcp.router import bootstrap_tool_registry
from app.services.tool_onboarding import PlannedTool, all_planned_tools
from app.tools.catalog_store import tool_catalog_store
from app.tools.registry import normalize_tool_name, tool_registry

logger = get_logger(name=__name__)


@dataclass(slots=True)
class ToolReconciliationResult:
    generated_at: datetime
    missing_aliases: tuple[str, ...]
    alias_mismatches: tuple[tuple[str, str, str], ...]
    missing_catalog_entries: tuple[str, ...]
    catalog_only_tools: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "missing_aliases": list(self.missing_aliases),
            "alias_mismatches": [list(item) for item in self.alias_mismatches],
            "missing_catalog_entries": list(self.missing_catalog_entries),
            "catalog_only_tools": list(self.catalog_only_tools),
        }


class ToolReconciliationJob:
    @classmethod
    async def run_once(cls) -> ToolReconciliationResult:
        bootstrap_tool_registry()
        planned = list(all_planned_tools())
        result = cls._compare(planned)
        cls._record_metrics(result)
        cls._log_outcome(result)
        return result

    @classmethod
    def _compare(cls, planned: Iterable[PlannedTool]) -> ToolReconciliationResult:
        catalog_entries = {entry.name for entry in tool_catalog_store.entries()}
        registry_aliases = tool_registry.aliases()
        registry_canonical = {normalize_tool_name(name) for name in tool_registry.list()}

        missing_aliases: set[str] = set()
        alias_mismatches: set[tuple[str, str, str]] = set()
        missing_catalog_entries: set[str] = set()

        for tool in planned:
            expected_canonical = normalize_tool_name(tool.resolved)
            resolved = tool_registry.resolve(tool.alias)
            if resolved is None:
                missing_aliases.add(tool.alias)
            else:
                if normalize_tool_name(resolved) != expected_canonical:
                    alias_mismatches.add((tool.alias, resolved, tool.resolved))
            if expected_canonical not in registry_canonical:
                missing_catalog_entries.add(tool.resolved)

        catalog_only_tools = sorted(tool for tool in catalog_entries if normalize_tool_name(tool) not in registry_canonical)

        timestamp = datetime.now(timezone.utc)
        return ToolReconciliationResult(
            generated_at=timestamp,
            missing_aliases=tuple(sorted(missing_aliases)),
            alias_mismatches=tuple(sorted(alias_mismatches)),
            missing_catalog_entries=tuple(sorted(missing_catalog_entries)),
            catalog_only_tools=tuple(catalog_only_tools),
        )

    @staticmethod
    def _record_metrics(result: ToolReconciliationResult) -> None:
        metrics.observe_tool_catalog_reconciliation(category="missing_aliases", count=len(result.missing_aliases))
        metrics.observe_tool_catalog_reconciliation(category="alias_mismatches", count=len(result.alias_mismatches))
        metrics.observe_tool_catalog_reconciliation(
            category="missing_catalog_entries",
            count=len(result.missing_catalog_entries),
        )
        metrics.observe_tool_catalog_reconciliation(
            category="catalog_only_tools",
            count=len(result.catalog_only_tools),
        )

    @staticmethod
    def _log_outcome(result: ToolReconciliationResult) -> None:
        if result.missing_aliases or result.alias_mismatches or result.missing_catalog_entries:
            logger.warning(
                "tool_reconciliation_detected_mismatches",
                report=result.as_dict(),
            )
        else:
            logger.info(
                "tool_reconciliation_clean",
                report=result.as_dict(),
            )


__all__ = [
    "ToolReconciliationJob",
    "ToolReconciliationResult",
]
