from __future__ import annotations

from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

try:
    from redis.asyncio import Redis
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    Redis = None  # type: ignore[misc,assignment]

try:
    import asyncpg
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    asyncpg = None  # type: ignore[assignment]

try:
    from qdrant_client import AsyncQdrantClient
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    AsyncQdrantClient = None  # type: ignore[misc,assignment]

from ..core.config import Settings
from ..core.logging import get_logger

logger = get_logger(name=__name__)


class HybridMemoryService:
    def __init__(
        self,
        *,
        redis_client: Redis | None = None,  # type: ignore[name-defined]
        pg_pool: Any | None = None,
        qdrant_client: Any | None = None,
    ) -> None:
        self._redis = redis_client
        self._pg_pool = pg_pool
        self._qdrant = qdrant_client
        self._exit_stack = AsyncExitStack()
        self._ephemeral_cache: dict[str, dict[str, Any]] = {}

    @classmethod
    def from_settings(cls, settings: Settings) -> "HybridMemoryService":
        redis_client = None
        if Redis is not None:
            redis_client = Redis.from_url(str(settings.redis.url))

        pg_pool = None
        if asyncpg is not None:
            pg_pool = asyncpg.create_pool(
                dsn=str(settings.postgres.dsn),
                min_size=settings.postgres.pool_min_size,
                max_size=settings.postgres.pool_max_size,
            )

        qdrant_client = None
        if AsyncQdrantClient is not None:
            qdrant_client = AsyncQdrantClient(
                url=settings.qdrant.url,
                api_key=settings.qdrant.api_key,
            )

        return cls(
            redis_client=redis_client,
            pg_pool=pg_pool,
            qdrant_client=qdrant_client,
        )

    @asynccontextmanager
    async def lifecycle(self) -> Any:
        try:
            if callable(self._pg_pool):
                self._pg_pool = await self._pg_pool  # type: ignore[func-returns-value]
            yield self
        finally:
            await self.close()

    async def close(self) -> None:
        await self._exit_stack.aclose()
        if self._redis is not None:
            await self._redis.close()
        if self._pg_pool is not None and hasattr(self._pg_pool, "close"):
            await self._pg_pool.close()  # type: ignore[func-returns-value]
        if self._qdrant is not None and hasattr(self._qdrant, "close"):
            await self._qdrant.close()

    async def store_working_memory(self, key: str, value: str, *, ttl: int = 600) -> None:
        if self._redis is None:
            logger.warning("redis_not_configured", key=key)
            return
        await self._redis.set(key, value, ex=ttl)

    async def store_ephemeral_memory(self, task_id: str, payload: dict[str, Any]) -> None:
        if self._pg_pool is None:
            logger.warning("postgres_not_configured", task_id=task_id)
        else:
            async with self._pg_pool.acquire() as connection:
                await connection.execute(
                    """
                    INSERT INTO agent_task_log(task_id, payload)
                    VALUES($1, $2::jsonb)
                    ON CONFLICT (task_id) DO UPDATE SET payload = EXCLUDED.payload
                    """,
                    task_id,
                    payload,
                )
        self._ephemeral_cache[task_id] = payload

    async def fetch_ephemeral_memory(self, task_id: str) -> dict[str, Any] | None:
        if self._pg_pool is not None:
            async with self._pg_pool.acquire() as connection:
                record = await connection.fetchrow(
                    """
                    SELECT payload
                    FROM agent_task_log
                    WHERE task_id = $1
                    """,
                    task_id,
                )
                if record is not None and "payload" in record:
                    payload = record["payload"]
                    self._ephemeral_cache[task_id] = payload
                    return payload
        return self._ephemeral_cache.get(task_id)

    async def store_semantic_memory(
        self,
        *,
        vector: list[float],
        payload: dict[str, Any],
        score: float = 1.0,
    ) -> None:
        if self._qdrant is None:
            logger.warning("qdrant_not_configured")
            return
        await self._qdrant.upsert(
            collection_name=payload.get("collection")
            or payload.get("collection_name")
            or "neura_tasks",
            wait=True,
            points=[
                {
                    "id": payload.get("id"),
                    "vector": vector,
                    "payload": payload,
                    "score": score,
                }
            ],
        )

    async def retrieve_context(self, *, query_vector: list[float], limit: int = 5) -> list[dict[str, Any]]:
        if self._qdrant is None:
            logger.warning("qdrant_not_configured")
            return []
        response = await self._qdrant.search(
            collection_name="neura_tasks",
            query_vector=query_vector,
            limit=limit,
        )
        return [hit.payload for hit in response]
