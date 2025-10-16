from __future__ import annotations

import inspect
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable
from uuid import UUID

try:
    import asyncpg
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    asyncpg = None  # type: ignore[assignment]

from .state import OrchestratorEvent, OrchestratorRun, OrchestratorStatus, new_run

TimestampFactory = Callable[[], datetime]


class OrchestratorStateStore:
    _FETCH_RUN = """
        SELECT run_id, task_id, status, state, created_at, updated_at
        FROM orchestration_runs
        WHERE run_id = $1
    """

    _FETCH_EVENTS = """
        SELECT run_id, sequence, event_type, agent, payload, created_at
        FROM orchestration_events
        WHERE run_id = $1
        ORDER BY sequence ASC
    """

    def __init__(
        self,
        pool: Any,
        *,
        now: TimestampFactory | None = None,
    ) -> None:
        if asyncpg is None:
            raise RuntimeError("asyncpg is required for OrchestratorStateStore")
        self._pool_or_coroutine = pool
        self._pool: Any | None = None
        self._now: TimestampFactory = now or (lambda: datetime.now(timezone.utc))

    @classmethod
    def from_settings(cls, settings: Any) -> "OrchestratorStateStore":
        if asyncpg is None:
            raise RuntimeError("asyncpg is not available")
        pool = asyncpg.create_pool(
            dsn=str(settings.postgres.dsn),
            min_size=settings.postgres.pool_min_size,
            max_size=settings.postgres.pool_max_size,
        )
        return cls(pool)

    async def start_run(self, task_id: str, *, state: dict[str, Any]) -> OrchestratorRun:
        run = new_run(task_id, state=state)
        run.created_at = run.updated_at = self._now()
        payload = json.dumps(run.state)
        pool = await self._ensure_pool()
        async with pool.acquire() as connection:
            await connection.execute(
                """
                INSERT INTO orchestration_runs (run_id, task_id, status, state, created_at, updated_at)
                VALUES ($1, $2, $3, $4::jsonb, $5, $5)
                """,
                run.run_id,
                run.task_id,
                run.status.value,
                payload,
                run.created_at,
            )
        return run

    async def record_event(self, event: OrchestratorEvent) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as connection:
            await connection.execute(
                """
                INSERT INTO orchestration_events (run_id, sequence, event_type, agent, payload, created_at)
                VALUES ($1, $2, $3, $4, $5::jsonb, $6)
                """,
                event.run_id,
                event.sequence,
                event.event_type,
                event.agent,
                json.dumps(event.payload),
                event.created_at,
            )

    async def update_state(self, run_id: UUID, *, state: dict[str, Any], status: OrchestratorStatus | None = None) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as connection:
            await connection.execute(
                """
                UPDATE orchestration_runs
                SET state = $2::jsonb,
                    status = COALESCE($3, status),
                    updated_at = $4
                WHERE run_id = $1
                """,
                run_id,
                json.dumps(state),
                status.value if status is not None else None,
                self._now(),
            )

    async def finalize_run(
        self,
        run_id: UUID,
        *,
        state: dict[str, Any],
        status: OrchestratorStatus,
        error: str | None = None,
    ) -> None:
        payload = dict(state)
        if error is not None:
            payload = dict(payload)
            payload["error"] = error
        await self.update_state(run_id, state=payload, status=status)

    async def get_run(self, run_id: UUID) -> OrchestratorRun | None:
        pool = await self._ensure_pool()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(self._FETCH_RUN, run_id)
        if row is None:
            return None
        state_payload = row["state"]
        if isinstance(state_payload, str):
            try:
                state_payload = json.loads(state_payload)
            except json.JSONDecodeError:  # pragma: no cover - defensive guard
                state_payload = {}
        elif state_payload is None:
            state_payload = {}
        else:
            state_payload = dict(state_payload)
        return OrchestratorRun(
            run_id=row["run_id"],
            task_id=row["task_id"],
            status=OrchestratorStatus(row["status"]),
            state=state_payload,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    async def list_events(self, run_id: UUID) -> list[OrchestratorEvent]:
        pool = await self._ensure_pool()
        async with pool.acquire() as connection:
            rows = await connection.fetch(self._FETCH_EVENTS, run_id)
        events: list[OrchestratorEvent] = []
        for row in rows:
            payload = row["payload"]
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError:  # pragma: no cover - defensive guard
                    payload = {}
            elif payload is None:
                payload = {}
            else:
                payload = dict(payload)
            events.append(
                OrchestratorEvent(
                    run_id=row["run_id"],
                    sequence=row["sequence"],
                    event_type=row["event_type"],
                    agent=row["agent"],
                    payload=payload,
                    created_at=row["created_at"],
                )
            )
        return events

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @asynccontextmanager
    async def lifecycle(self) -> AsyncIterator["OrchestratorStateStore"]:
        try:
            await self._ensure_pool()
            yield self
        finally:
            await self.close()

    async def _ensure_pool(self) -> Any:
        if self._pool is not None:
            return self._pool
        candidate = self._pool_or_coroutine
        if inspect.isawaitable(candidate):
            candidate = await candidate
        if not isinstance(candidate, asyncpg.Pool):
            raise RuntimeError("Invalid asyncpg pool supplied to OrchestratorStateStore")
        self._pool = candidate
        return self._pool
