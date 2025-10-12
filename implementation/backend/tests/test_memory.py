from __future__ import annotations

import pytest

from app.services.memory import HybridMemoryService


class _FakeConnection:
    def __init__(self, store: dict[str, dict[str, object]]) -> None:
        self._store = store

    async def execute(self, query: str, task_id: str, payload: dict[str, object]) -> None:  # noqa: ARG002
        self._store[task_id] = payload

    async def fetchrow(self, query: str, task_id: str) -> dict[str, object] | None:  # noqa: ARG002
        if task_id not in self._store:
            return None
        return {"payload": self._store[task_id]}

    async def __aenter__(self) -> "_FakeConnection":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001, D401
        return False


class _FakeAcquireContext:
    def __init__(self, store: dict[str, dict[str, object]]) -> None:
        self._store = store

    async def __aenter__(self) -> _FakeConnection:
        return _FakeConnection(self._store)

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
        return False


class FakePool:
    def __init__(self) -> None:
        self._store: dict[str, dict[str, object]] = {}

    def acquire(self) -> _FakeAcquireContext:
        return _FakeAcquireContext(self._store)

    async def close(self) -> None:  # pragma: no cover - parity method
        self._store.clear()


@pytest.mark.asyncio
async def test_ephemeral_memory_persists_across_instances() -> None:
    pool = FakePool()
    memory = HybridMemoryService(pg_pool=pool)

    payload = {"result": {"status": "completed", "outputs": ["ok"]}}
    await memory.store_ephemeral_memory("task-123", payload)

    fresh_memory = HybridMemoryService(pg_pool=pool)
    fetched = await fresh_memory.fetch_ephemeral_memory("task-123")

    assert fetched == payload
