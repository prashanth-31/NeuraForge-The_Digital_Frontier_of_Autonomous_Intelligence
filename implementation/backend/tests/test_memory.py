from __future__ import annotations

import json
from typing import cast

import pytest

from app.services.memory import HybridMemoryService, MemoryServiceConfig, ReadPreference


class _FakeConnection:
    def __init__(self, pool: "FakePool") -> None:
        self._pool = pool

    async def execute(
        self,
        query: str,
        task_id: str,
        agent: str | None,
        payload: object,
    ) -> None:  # noqa: ARG002
        normalized = self._normalize_payload(payload)
        self._pool._upsert(task_id, agent, normalized)

    async def executemany(self, query: str, args: list[tuple[str, str | None, object]]) -> None:  # noqa: D401
        for task_id, agent, payload in args:
            await self.execute(query, task_id, agent, payload)

    async def fetchrow(self, query: str, task_id: str) -> dict[str, object] | None:  # noqa: ARG002
        record = self._pool._store.get(task_id)
        if record is None:
            return None
        payload = cast(dict[str, object], record["payload"]).copy()
        return {"task_id": task_id, "agent": record.get("agent"), "payload": payload}

    async def fetch(self, query: str, agent: str | None, limit: int) -> list[dict[str, object]]:  # noqa: ARG002
        return self._pool._fetch_recent(agent=agent, limit=limit)

    async def __aenter__(self) -> "_FakeConnection":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001, D401
        return False

    @staticmethod
    def _normalize_payload(payload: object) -> dict[str, object]:
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, str):
            try:
                decoded = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise AssertionError(f"Invalid JSON payload passed to FakePool: {payload}") from exc
            if isinstance(decoded, dict):
                return decoded
            raise AssertionError("Decoded JSON payload must be an object for FakePool")
        raise AssertionError(f"Unsupported payload type: {type(payload)!r}")


class _FakeAcquireContext:
    def __init__(self, pool: "FakePool") -> None:
        self._pool = pool

    async def __aenter__(self) -> _FakeConnection:
        return _FakeConnection(self._pool)

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
        return False


class FakePool:
    def __init__(self) -> None:
        self._store: dict[str, dict[str, object]] = {}
        self._sequence = 0

    def acquire(self) -> _FakeAcquireContext:
        return _FakeAcquireContext(self)

    async def close(self) -> None:  # pragma: no cover - parity method
        self._store.clear()

    def _upsert(self, task_id: str, agent: str | None, payload: dict[str, object]) -> None:
        self._sequence += 1
        stored_payload = payload.copy()
        metadata: dict[str, object] = {
            "payload": stored_payload,
            "agent": agent or stored_payload.get("agent"),
            "_order": self._sequence,
        }
        self._store[task_id] = metadata

    def _fetch_recent(self, *, agent: str | None, limit: int) -> list[dict[str, object]]:
        records = [
            {
                "task_id": task_id,
                "agent": data.get("agent"),
                "payload": cast(dict[str, object], data["payload"]).copy(),
                "_order": data["_order"],
            }
            for task_id, data in self._store.items()
        ]
        if agent is not None:
            records = [record for record in records if record.get("agent") == agent]
        records.sort(key=lambda record: record["_order"], reverse=True)
        for record in records:
            record.pop("_order", None)
        return records[:limit]


class FakeRedis:
    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    async def set(self, key: str, value: str, ex: int | None = None) -> None:  # noqa: ARG002
        self._store[key] = value

    async def get(self, key: str) -> str | None:
        return self._store.get(key)

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def close(self) -> None:  # pragma: no cover - compatibility hook
        self._store.clear()


def _make_service(
    *,
    pool: FakePool | None = None,
    redis: FakeRedis | None = None,
    read_pref: ReadPreference = "cache-first",
) -> HybridMemoryService:
    config = MemoryServiceConfig(read_preference=read_pref, redis_namespace="test-namespace")
    return HybridMemoryService(pg_pool=pool, redis_client=redis, config=config)


@pytest.mark.asyncio
async def test_ephemeral_memory_persists_across_instances() -> None:
    pool = FakePool()
    memory = _make_service(pool=pool)

    payload = {"result": {"status": "completed", "outputs": ["ok"]}, "agent": "planner"}
    await memory.store_ephemeral_memory("task-123", payload)

    fresh_memory = _make_service(pool=pool)
    fetched = await fresh_memory.fetch_ephemeral_memory("task-123")

    assert fetched == payload


@pytest.mark.asyncio
async def test_cache_first_prefers_redis_when_available() -> None:
    redis = FakeRedis()
    memory = _make_service(redis=redis, read_pref="cache-first")

    await memory.store_ephemeral_memory("task-cache", {"value": 1, "agent": "planner"})
    # Simulate loss of durable store
    memory._postgres_repo = None  # type: ignore[attr-defined]
    # ensure internal cache cleared to force redis path
    memory._ephemeral_cache.clear()

    fetched = await memory.fetch_ephemeral_memory("task-cache")
    assert fetched is not None
    assert fetched["value"] == 1


@pytest.mark.asyncio
async def test_fetch_recent_context_merges_store_and_cache() -> None:
    pool = FakePool()
    memory = _make_service(pool=pool)

    await memory.store_ephemeral_memory("task-a", {"value": "alpha", "agent": "planner"})
    await memory.store_ephemeral_memory("task-b", {"value": "beta", "agent": "planner"})
    memory._ephemeral_cache["task-c"] = {"value": "cached", "agent": "planner"}

    recent = await memory.fetch_recent_context(agent="planner", limit=3)

    values = {entry["value"] for entry in recent}
    assert values == {"alpha", "beta", "cached"}
