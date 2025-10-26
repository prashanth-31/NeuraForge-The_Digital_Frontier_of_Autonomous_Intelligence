from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
import inspect
from typing import Any, AsyncIterator, Dict, Mapping, MutableMapping, Optional
from uuid import UUID

try:
    import asyncpg
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    asyncpg = None  # type: ignore[assignment]

from ..core.config import Settings
from ..services.retrieval import ContextAssembler, ContextBundle
from ..utils.asyncpg_helpers import ensure_pool_ready
from ..utils.json_encoding import encode_jsonb


class ContextStage(str, Enum):
    INTAKE = "intake"
    NEGOTIATION = "negotiation"
    PLANNING = "planning"
    EXECUTION = "execution"
    CONSOLIDATION = "consolidation"


@dataclass(slots=True)
class ContextSnapshot:
    task_id: str
    stage: ContextStage
    payload: Dict[str, Any]
    agent: str | None = None
    run_id: UUID | None = None


class ContextAssemblyContract:
    def __init__(
        self,
        *,
        assembler: ContextAssembler,
        stage_overrides: Mapping[ContextStage, Mapping[str, Any]] | None = None,
    ) -> None:
        self._assembler = assembler
        self._overrides: MutableMapping[ContextStage, Mapping[str, Any]] = dict(stage_overrides or {})
        self._default_stage = ContextStage.INTAKE

    def set_default_stage(self, stage: ContextStage) -> None:
        self._default_stage = stage

    async def build(
        self,
        *,
        task: dict[str, Any],
        agent: str | None = None,
        stage: ContextStage | None = None,
    ) -> ContextBundle:
        bundle = await self._assembler.build(task=task, agent=agent)
        target_stage = stage or self._default_stage
        override = self._overrides.get(target_stage)
        if override:
            max_chars = int(override.get("max_chars", bundle.max_chars))
            snippet_limit = int(override.get("top_snippets", len(bundle.snippets)))
            snippets = bundle.snippets[: max(1, snippet_limit)]
            return ContextBundle(query=bundle.query, snippets=snippets, max_chars=max_chars)
        return bundle

    async def build_for_stage(
        self,
        stage: ContextStage,
        *,
        task: dict[str, Any],
        agent: str | None = None,
    ) -> ContextBundle:
        return await self.build(task=task, agent=agent, stage=stage)


class ContextSnapshotStore:
    _UPSERT = """
        INSERT INTO context_snapshots(task_id, run_id, stage, agent, payload)
        VALUES($1, $2, $3, $4, $5::jsonb)
    """

    def __init__(self, pool: Any | None) -> None:
        if asyncpg is None:
            raise RuntimeError("asyncpg is required for ContextSnapshotStore")
        self._pool_or_factory = pool
        self._pool: Any | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> "ContextSnapshotStore":
        if asyncpg is None:
            raise RuntimeError("asyncpg is not available")
        pool = asyncpg.create_pool(
            dsn=str(settings.postgres.dsn),
            min_size=settings.postgres.pool_min_size,
            max_size=settings.postgres.pool_max_size,
        )
        return cls(pool)

    async def record(self, snapshot: ContextSnapshot) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as connection:
            await connection.execute(
                self._UPSERT,
                snapshot.task_id,
                snapshot.run_id,
                snapshot.stage.value,
                snapshot.agent,
                encode_jsonb(snapshot.payload),
            )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @asynccontextmanager
    async def lifecycle(self) -> AsyncIterator["ContextSnapshotStore"]:
        try:
            await self._ensure_pool()
            yield self
        finally:
            await self.close()

    async def _ensure_pool(self) -> Any:
        if self._pool is not None:
            return self._pool
        candidate = self._pool_or_factory
        if candidate is None:
            raise RuntimeError("Invalid asyncpg pool supplied to ContextSnapshotStore")
        pool_type = getattr(asyncpg, "Pool", None)
        legacy_pool_type = getattr(getattr(asyncpg, "pool", None), "Pool", None)
        if inspect.isawaitable(candidate):
            candidate = await candidate
        elif callable(candidate) and not isinstance(candidate, (pool_type, legacy_pool_type)):
            maybe_pool = candidate()
            if inspect.isawaitable(maybe_pool):
                candidate = await maybe_pool
            else:
                candidate = maybe_pool
        if pool_type is not None and isinstance(candidate, pool_type):
            await ensure_pool_ready(candidate)
            self._pool = candidate
            return self._pool
        if legacy_pool_type is not None and isinstance(candidate, legacy_pool_type):  # type: ignore[arg-type]
            await ensure_pool_ready(candidate)
            self._pool = candidate
            return self._pool
        if hasattr(candidate, "acquire") and hasattr(candidate, "close"):
            await ensure_pool_ready(candidate)
            self._pool = candidate
            return self._pool
        raise RuntimeError("Invalid asyncpg pool supplied to ContextSnapshotStore")


class InMemoryContextSnapshotStore(ContextSnapshotStore):
    def __init__(self) -> None:  # pragma: no cover - lightweight fallback for tests
        self._records: list[ContextSnapshot] = []
        self._pool_or_factory = None
        self._pool = None

    async def record(self, snapshot: ContextSnapshot) -> None:
        self._records.append(snapshot)

    async def close(self) -> None:
        return

    async def _ensure_pool(self) -> Any:
        raise RuntimeError("In-memory store does not provide a database pool")

    @property
    def records(self) -> list[ContextSnapshot]:
        return list(self._records)

    @asynccontextmanager
    async def lifecycle(self) -> AsyncIterator["InMemoryContextSnapshotStore"]:
        try:
            yield self
        finally:
            return
