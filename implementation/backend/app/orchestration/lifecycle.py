from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator, Dict, Iterable, Optional
from uuid import UUID

try:
    import asyncpg
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    asyncpg = None  # type: ignore[assignment]

from ..core.config import Settings
from .enums import LifecycleStatus


@dataclass(slots=True)
class LifecycleEvent:
    task_id: str
    step_id: str
    event_type: str
    status: LifecycleStatus
    payload: Dict[str, Any]
    agent: str | None = None
    sequence: int = 0
    run_id: UUID | None = None
    attempt: int = 0
    eta: datetime | None = None
    deadline: datetime | None = None
    latency_ms: float | None = None
    created_at: datetime = datetime.now(timezone.utc)


class TaskLifecycleStore:
    _INSERT = """
        INSERT INTO task_lifecycle_events(
            task_id,
            run_id,
            step_id,
            sequence,
            event_type,
            status,
            agent,
            attempt,
            eta,
            deadline,
            latency_ms,
            payload
        )
        VALUES($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12::jsonb)
    """

    def __init__(self, pool: Any | None) -> None:
        if asyncpg is None:
            raise RuntimeError("asyncpg is required for TaskLifecycleStore")
        self._pool_or_factory = pool
        self._pool: Optional[asyncpg.Pool] = None  # type: ignore[name-defined]

    @classmethod
    def from_settings(cls, settings: Settings) -> "TaskLifecycleStore":
        if asyncpg is None:
            raise RuntimeError("asyncpg is not available")
        pool = asyncpg.create_pool(
            dsn=str(settings.postgres.dsn),
            min_size=settings.postgres.pool_min_size,
            max_size=settings.postgres.pool_max_size,
        )
        return cls(pool)

    async def record(self, event: LifecycleEvent) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as connection:
            await connection.execute(
                self._INSERT,
                event.task_id,
                event.run_id,
                event.step_id,
                event.sequence,
                event.event_type,
                event.status.value,
                event.agent,
                event.attempt,
                event.eta,
                event.deadline,
                event.latency_ms,
                event.payload,
            )

    async def record_plan(self, plan, *, run_id: UUID | None = None) -> None:
        sequence = 1
        for step in plan.steps:
            payload = {
                "description": step.description,
                "depends_on": step.depends_on,
                "metadata": step.metadata,
            }
            event = LifecycleEvent(
                task_id=plan.task_id,
                step_id=step.step_id,
                event_type="planned",
                status=LifecycleStatus.PLANNED,
                payload=payload,
                agent=step.agent,
                sequence=sequence,
                run_id=run_id,
                attempt=0,
                eta=_parse_iso(step.eta_iso),
                deadline=_parse_iso(step.deadline_iso),
            )
            await self.record(event)
            sequence += 1

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def _ensure_pool(self) -> asyncpg.Pool:  # type: ignore[name-defined]
        if self._pool is not None:
            return self._pool
        candidate = self._pool_or_factory
        if isinstance(candidate, asyncpg.Pool):
            self._pool = candidate
            return self._pool
        if hasattr(candidate, "__await__"):
            candidate = await candidate
        if isinstance(candidate, asyncpg.Pool):
            self._pool = candidate
            return self._pool
        raise RuntimeError("Invalid asyncpg pool supplied to TaskLifecycleStore")

    @asynccontextmanager
    async def lifecycle(self) -> AsyncIterator["TaskLifecycleStore"]:
        try:
            await self._ensure_pool()
            yield self
        finally:
            await self.close()


class InMemoryTaskLifecycleStore(TaskLifecycleStore):
    def __init__(self) -> None:  # pragma: no cover - fallback for unit tests
        self._events: list[LifecycleEvent] = []
        self._pool_or_factory = None
        self._pool = None

    async def record(self, event: LifecycleEvent) -> None:
        self._events.append(event)

    async def record_plan(self, plan, *, run_id: UUID | None = None) -> None:
        event_sequence = len(self._events) + 1
        for step in plan.steps:
            payload = {
                "description": step.description,
                "depends_on": step.depends_on,
                "metadata": step.metadata,
            }
            self._events.append(
                LifecycleEvent(
                    task_id=plan.task_id,
                    step_id=step.step_id,
                    event_type="planned",
                    status=LifecycleStatus.PLANNED,
                    payload=payload,
                    agent=step.agent,
                    sequence=event_sequence,
                    run_id=run_id,
                    attempt=0,
                    eta=_parse_iso(step.eta_iso),
                    deadline=_parse_iso(step.deadline_iso),
                )
            )
            event_sequence += 1

    async def close(self) -> None:
        return

    async def _ensure_pool(self) -> asyncpg.Pool:  # type: ignore[name-defined]
        raise RuntimeError("In-memory lifecycle store has no database pool")

    @property
    def events(self) -> Iterable[LifecycleEvent]:
        return list(self._events)

    @asynccontextmanager
    async def lifecycle(self) -> AsyncIterator["InMemoryTaskLifecycleStore"]:
        try:
            yield self
        finally:
            return


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    if value.startswith("+"):
        try:
            seconds = int(value.strip("+s"))
        except ValueError:
            return None
        return datetime.now(timezone.utc) + timedelta(seconds=seconds)
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
