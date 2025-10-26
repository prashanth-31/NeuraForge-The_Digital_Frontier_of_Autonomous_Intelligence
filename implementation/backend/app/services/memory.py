from __future__ import annotations

from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
import inspect
import json
from typing import Any, AsyncIterator, Iterable, Literal, Sequence, TypeVar

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
from ..core.metrics import (
    increment_cache_hit,
    increment_cache_miss,
    increment_memory_ingest,
    increment_retrieval_results,
)
from ..utils.asyncpg_helpers import ensure_pool_ready
from ..utils.json_encoding import decode_jsonb, encode_jsonb

logger = get_logger(name=__name__)

ReadPreference = Literal["cache-first", "store-first", "cache-only", "store-only"]
T = TypeVar("T")


@dataclass(slots=True)
class MemoryServiceConfig:
    read_preference: ReadPreference = "cache-first"
    working_memory_ttl: int = 600
    ephemeral_ttl: int = 600
    batch_size: int = 50
    redis_namespace: str = "neuraforge"


@dataclass(slots=True)
class EpisodeRecord:
    task_id: str
    payload: dict[str, Any]
    agent: str | None = None

    def __post_init__(self) -> None:
        if self.agent is None:
            agent_value = self.payload.get("agent")
            if isinstance(agent_value, str):
                self.agent = agent_value


@dataclass(slots=True)
class SemanticVector:
    vector: list[float]
    payload: dict[str, Any]
    score: float = 1.0


class RedisRepository:
    def __init__(
        self,
        client: Any | None,
        *,
        namespace: str,
        default_ttl: int,
    ) -> None:
        self._client = client
        self._namespace = namespace
        self._default_ttl = default_ttl

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def _key(self, key: str) -> str:
        return f"{self._namespace}:{key}"

    async def store_working_memory(self, key: str, value: str, ttl: int | None = None) -> None:
        client = self._client
        if client is None:
            return
        await client.set(self._key(key), value, ex=ttl or self._default_ttl)
        increment_memory_ingest(store="redis", operation="set")

    async def store_working_batch(self, items: Sequence[tuple[str, str]], ttl: int | None = None) -> None:
        if not self.enabled or not items:
            return
        for key, value in items:
            await self.store_working_memory(key, value, ttl=ttl)

    async def cache_episode(self, record: EpisodeRecord, *, ttl: int | None = None) -> None:
        client = self._client
        if client is None:
            return
        cache_payload = record.payload.copy()
        if record.agent is not None:
            cache_payload.setdefault("agent", record.agent)
        await client.set(
            self._key(f"episode:{record.task_id}"),
            json.dumps(cache_payload),
            ex=ttl or self._default_ttl,
        )
        increment_memory_ingest(store="redis", operation="episode_cache")

    async def cache_episode_batch(self, records: Sequence[EpisodeRecord], *, ttl: int | None = None) -> None:
        if not self.enabled or not records:
            return
        for record in records:
            await self.cache_episode(record, ttl=ttl)

    async def fetch_cached_episode(self, task_id: str) -> dict[str, Any] | None:
        client = self._client
        if client is None:
            return None
        value = await client.get(self._key(f"episode:{task_id}"))
        if value is None:
            increment_cache_miss(layer="redis")
            return None
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        try:
            payload = json.loads(value)
            increment_cache_hit(layer="redis")
            return payload
        except json.JSONDecodeError:  # pragma: no cover - defensive guard
            logger.warning("redis_cache_payload_invalid", task_id=task_id)
            return None

    async def invalidate_episode(self, task_id: str) -> None:
        client = self._client
        if client is None:
            return
        await client.delete(self._key(f"episode:{task_id}"))
        increment_memory_ingest(store="redis", operation="invalidate")

    async def close(self) -> None:
        client = self._client
        if client is not None:
            await client.close()


class PostgresRepository:
    _UPSERT_EPISODE = """
        INSERT INTO episodic_memory(task_id, agent, payload)
        VALUES($1, $2, $3::jsonb)
        ON CONFLICT (task_id)
        DO UPDATE SET
            payload = EXCLUDED.payload,
            agent = EXCLUDED.agent,
            updated_at = NOW()
    """

    _FETCH_EPISODE = """
        SELECT task_id, agent, payload, updated_at
        FROM episodic_memory
        WHERE task_id = $1
    """

    _FETCH_RECENT = """
        SELECT task_id, agent, payload, updated_at
        FROM episodic_memory
        WHERE ($1::text IS NULL OR agent = $1)
        ORDER BY updated_at DESC
        LIMIT $2
    """

    def __init__(self, pool: Any | None) -> None:
        self._pool_or_factory = pool
        self._pool: Any | None = None

    @property
    def enabled(self) -> bool:
        return self._pool_or_factory is not None

    async def _ensure_pool(self) -> Any:
        if self._pool is not None:
            return self._pool
        candidate = self._pool_or_factory
        if candidate is None:
            return None
        if inspect.isawaitable(candidate):
            candidate = await candidate
        elif callable(candidate):
            maybe_pool = candidate()
            if inspect.isawaitable(maybe_pool):
                candidate = await maybe_pool
            else:
                candidate = maybe_pool
        if candidate is None:
            return None
        pool_type = getattr(asyncpg, "Pool", None) if asyncpg is not None else None
        legacy_pool_type = getattr(getattr(asyncpg, "pool", None), "Pool", None) if asyncpg is not None else None
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
        raise RuntimeError("Invalid asyncpg pool supplied to PostgresRepository")

    async def upsert_episode(self, record: EpisodeRecord) -> None:
        pool = await self._ensure_pool()
        if pool is None:
            return
        async with pool.acquire() as connection:
            agent = record.agent or record.payload.get("agent")
            payload = encode_jsonb(record.payload)
            await connection.execute(self._UPSERT_EPISODE, record.task_id, agent, payload)
        increment_memory_ingest(store="postgres", operation="upsert")

    async def upsert_batch(self, records: Sequence[EpisodeRecord]) -> None:
        pool = await self._ensure_pool()
        if pool is None or not records:
            return
        async with pool.acquire() as connection:
            args = [
                (
                    rec.task_id,
                    rec.agent or rec.payload.get("agent"),
                    encode_jsonb(rec.payload),
                )
                for rec in records
            ]
            executemany = getattr(connection, "executemany", None)
            if callable(executemany):
                result = executemany(self._UPSERT_EPISODE, args)
                if inspect.isawaitable(result):
                    await result
            else:
                for record_args in args:
                    await connection.execute(self._UPSERT_EPISODE, *record_args)
        increment_memory_ingest(store="postgres", operation="bulk_upsert")

    def _normalize_row(self, row: Any) -> EpisodeRecord:
        payload_raw = decode_jsonb(row["payload"])
        if isinstance(payload_raw, dict):
            payload = payload_raw
        else:
            payload = dict(payload_raw or {}) if payload_raw else {}
        agent = row["agent"]
        if agent and "agent" not in payload:
            payload["agent"] = agent
        return EpisodeRecord(task_id=row["task_id"], payload=payload, agent=agent)

    async def fetch_episode(self, task_id: str) -> EpisodeRecord | None:
        pool = await self._ensure_pool()
        if pool is None:
            return None
        async with pool.acquire() as connection:
            record = await connection.fetchrow(self._FETCH_EPISODE, task_id)
            if record is None:
                return None
            return self._normalize_row(record)

    async def fetch_recent(self, *, agent: str | None = None, limit: int = 5) -> list[EpisodeRecord]:
        pool = await self._ensure_pool()
        if pool is None:
            return []
        async with pool.acquire() as connection:
            rows = await connection.fetch(self._FETCH_RECENT, agent, limit)
        return [self._normalize_row(row) for row in rows]

    async def close(self) -> None:
        pool = self._pool or self._pool_or_factory
        if inspect.isawaitable(pool):
            pool = await pool
        if pool is not None and hasattr(pool, "close"):
            await pool.close()
        self._pool = None
        self._pool_or_factory = None


class QdrantRepository:
    def __init__(self, client: Any | None, *, collection_name: str) -> None:
        self._client = client
        self._collection_name = collection_name

    @property
    def enabled(self) -> bool:
        return self._client is not None

    async def upsert(self, vector: SemanticVector) -> None:
        client = self._client
        if client is None:
            return
        await client.upsert(
            collection_name=vector.payload.get("collection")
            or vector.payload.get("collection_name")
            or self._collection_name,
            wait=True,
            points=[
                {
                    "id": vector.payload.get("id"),
                    "vector": vector.vector,
                    "payload": vector.payload,
                    "score": vector.score,
                }
            ],
        )

    async def upsert_batch(self, vectors: Sequence[SemanticVector]) -> None:
        client = self._client
        if client is None or not vectors:
            return
        collection_name = self._collection_name
        points = [
            {
                "id": vec.payload.get("id"),
                "vector": vec.vector,
                "payload": vec.payload,
                "score": vec.score,
            }
            for vec in vectors
        ]
        await client.upsert(collection_name=collection_name, wait=True, points=points)

    async def search(self, *, query_vector: list[float], limit: int = 5) -> list[dict[str, Any]]:
        client = self._client
        if client is None:
            return []
        response = await client.search(
            collection_name=self._collection_name,
            query_vector=query_vector,
            limit=limit,
        )
        results: list[dict[str, Any]] = []
        for hit in response:
            payload = dict(getattr(hit, "payload", {}) or {})
            if "_score" not in payload:
                payload["_score"] = getattr(hit, "score", None)
            results.append(payload)
        return results

    async def close(self) -> None:
        client = self._client
        if client is not None and hasattr(client, "close"):
            await client.close()


def _chunk(items: Sequence[T], *, size: int) -> Iterable[Sequence[T]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


class HybridMemoryService:
    def __init__(
        self,
        *,
        redis_client: Redis | None = None,  # type: ignore[name-defined]
        pg_pool: Any | None = None,
        qdrant_client: Any | None = None,
        config: MemoryServiceConfig | None = None,
        redis_repository: RedisRepository | None = None,
        postgres_repository: PostgresRepository | None = None,
        qdrant_repository: QdrantRepository | None = None,
    ) -> None:
        self._redis = redis_client
        self._pg_pool = pg_pool
        self._qdrant = qdrant_client
        self.config = config or MemoryServiceConfig()
        self._exit_stack = AsyncExitStack()
        self._ephemeral_cache: dict[str, dict[str, Any]] = {}

        self._redis_repo = redis_repository or (
            RedisRepository(
                redis_client,
                namespace=self.config.redis_namespace,
                default_ttl=self.config.ephemeral_ttl,
            )
            if redis_client is not None
            else None
        )
        self._postgres_repo = postgres_repository
        if self._postgres_repo is None and pg_pool is not None and not inspect.isawaitable(pg_pool):
            self._postgres_repo = PostgresRepository(pg_pool)
        self._qdrant_repo = qdrant_repository or (
            QdrantRepository(qdrant_client, collection_name=self._default_collection_name)
            if qdrant_client is not None
            else None
        )

    @property
    def _default_collection_name(self) -> str:
        return "neura_tasks"

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
        memory_settings = getattr(settings, "memory", None)
        config = MemoryServiceConfig(
            read_preference=getattr(memory_settings, "read_preference", "cache-first"),
            working_memory_ttl=getattr(memory_settings, "working_memory_ttl", 600),
            ephemeral_ttl=getattr(memory_settings, "ephemeral_ttl", 600),
            batch_size=getattr(memory_settings, "batch_size", 50),
            redis_namespace=getattr(memory_settings, "redis_namespace", "neuraforge"),
        )

        qdrant_repo = None
        if qdrant_client is not None:
            collection_name = getattr(settings.qdrant, "collection_name", "neura_tasks")
            qdrant_repo = QdrantRepository(qdrant_client, collection_name=collection_name)

        return cls(
            redis_client=redis_client,
            pg_pool=pg_pool,
            qdrant_client=qdrant_client,
            config=config,
            qdrant_repository=qdrant_repo,
        )

    @asynccontextmanager
    async def lifecycle(self) -> AsyncIterator["HybridMemoryService"]:
        try:
            if inspect.isawaitable(self._pg_pool):
                resolved_pool = await self._pg_pool
                self._pg_pool = resolved_pool
            if self._postgres_repo is None and self._pg_pool is not None:
                self._postgres_repo = PostgresRepository(self._pg_pool)
            yield self
        finally:
            await self.close()

    async def close(self) -> None:
        await self._exit_stack.aclose()
        if self._redis_repo is not None:
            await self._redis_repo.close()
            self._redis_repo = None
        if self._postgres_repo is not None:
            await self._postgres_repo.close()
            self._postgres_repo = None
            self._pg_pool = None
        if self._qdrant_repo is not None:
            await self._qdrant_repo.close()
            self._qdrant_repo = None

    async def store_working_memory(self, key: str, value: str, *, ttl: int | None = None) -> None:
        if self._redis_repo is None:
            logger.warning("redis_not_configured", key=key)
            return
        await self._redis_repo.store_working_memory(key, value, ttl=ttl or self.config.working_memory_ttl)

    async def store_working_memory_batch(
        self,
        entries: Sequence[tuple[str, str]],
        *,
        ttl: int | None = None,
    ) -> None:
        if self._redis_repo is None:
            logger.warning("redis_not_configured_batch")
            return
        await self._redis_repo.store_working_batch(entries, ttl=ttl or self.config.working_memory_ttl)

    async def store_ephemeral_memory(
        self,
        task_id: str,
        payload: dict[str, Any],
        *,
        agent: str | None = None,
    ) -> None:
        record = EpisodeRecord(task_id=task_id, payload=payload, agent=agent)
        await self._persist_episode(record)

    async def store_ephemeral_batch(self, records: Sequence[EpisodeRecord]) -> None:
        if not records:
            return
        if self._postgres_repo is None:
            logger.warning("postgres_not_configured_batch", count=len(records))
        else:
            for chunk in _chunk(records, size=self.config.batch_size):
                await self._postgres_repo.upsert_batch(list(chunk))
        if self._redis_repo is not None:
            for chunk in _chunk(records, size=self.config.batch_size):
                await self._redis_repo.cache_episode_batch(list(chunk), ttl=self.config.ephemeral_ttl)
        for record in records:
            cache_payload = record.payload.copy()
            if record.agent is not None:
                cache_payload.setdefault("agent", record.agent)
            self._ephemeral_cache[record.task_id] = cache_payload

    async def _persist_episode(self, record: EpisodeRecord) -> None:
        if self._postgres_repo is None:
            logger.warning("postgres_not_configured", task_id=record.task_id)
        else:
            await self._postgres_repo.upsert_episode(record)

        if self._redis_repo is not None:
            await self._redis_repo.cache_episode(record, ttl=self.config.ephemeral_ttl)

        cache_payload = record.payload.copy()
        if record.agent is not None:
            cache_payload.setdefault("agent", record.agent)
        self._ephemeral_cache[record.task_id] = cache_payload

    async def fetch_ephemeral_memory(self, task_id: str) -> dict[str, Any] | None:
        preference = self.config.read_preference
        result: dict[str, Any] | None = None

        if preference in {"cache-first", "cache-only"} and self._redis_repo is not None:
            result = await self._redis_repo.fetch_cached_episode(task_id)
            if result is not None:
                self._ephemeral_cache[task_id] = result
                return result

        if preference != "cache-only" and self._postgres_repo is not None:
            record = await self._postgres_repo.fetch_episode(task_id)
            if record is not None:
                result = record.payload.copy()
                self._ephemeral_cache[task_id] = result
                if self._redis_repo is not None and preference == "cache-first":
                    await self._redis_repo.cache_episode(record, ttl=self.config.ephemeral_ttl)
                return result

        if preference == "store-first" and result is None and self._redis_repo is not None:
            result = await self._redis_repo.fetch_cached_episode(task_id)
            if result is not None:
                self._ephemeral_cache[task_id] = result
                return result

        cached = self._ephemeral_cache.get(task_id)
        if cached is not None:
            increment_cache_hit(layer="local")
        else:
            increment_cache_miss(layer="local")
        return cached

    async def fetch_recent_context(
        self,
        *,
        agent: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        aggregated: list[tuple[str | None, dict[str, Any]]] = []

        if self._postgres_repo is not None and self.config.read_preference != "cache-only":
            store_records = await self._postgres_repo.fetch_recent(agent=agent, limit=limit)
            for record in store_records:
                aggregated.append((record.task_id, record.payload.copy()))

        existing_ids: set[str] = {task_id for task_id, _ in aggregated if task_id is not None}

        if len(aggregated) < limit and self._ephemeral_cache:
            for task_id, payload in reversed(list(self._ephemeral_cache.items())):
                if agent is not None and payload.get("agent") != agent:
                    continue
                if task_id in existing_ids:
                    continue
                aggregated.append((task_id, payload.copy()))
                existing_ids.add(task_id)
                if len(aggregated) >= limit:
                    break

        episodic_results = [payload for _, payload in aggregated[:limit]]
        increment_retrieval_results(source="episodic", count=len(episodic_results))
        return episodic_results

    async def enumerate_recent_task_ids(
        self,
        *,
        agent: str | None = None,
        limit: int = 50,
    ) -> list[str]:
        candidates: list[str] = []

        if self._postgres_repo is not None and self.config.read_preference != "cache-only":
            records = await self._postgres_repo.fetch_recent(agent=agent, limit=limit)
            for record in records:
                if record.task_id not in candidates:
                    candidates.append(record.task_id)
                    if len(candidates) >= limit:
                        return candidates

        if self._ephemeral_cache:
            for task_id, payload in reversed(list(self._ephemeral_cache.items())):
                if agent is not None and isinstance(payload, dict):
                    payload_agent = payload.get("agent") or payload.get("result", {}).get("agent")
                    if payload_agent is not None and payload_agent != agent:
                        continue
                if task_id not in candidates:
                    candidates.append(task_id)
                if len(candidates) >= limit:
                    break

        return candidates[:limit]

    async def store_semantic_memory(
        self,
        *,
        vector: list[float],
        payload: dict[str, Any],
        score: float = 1.0,
    ) -> None:
        if self._qdrant_repo is None:
            logger.warning("qdrant_not_configured")
            return
        await self._qdrant_repo.upsert(SemanticVector(vector=vector, payload=payload, score=score))

    async def store_semantic_batch(self, vectors: Sequence[SemanticVector]) -> None:
        if self._qdrant_repo is None:
            logger.warning("qdrant_not_configured_batch", count=len(vectors))
            return
        for chunk in _chunk(vectors, size=self.config.batch_size):
            await self._qdrant_repo.upsert_batch(list(chunk))

    async def retrieve_context(self, *, query_vector: list[float], limit: int = 5) -> list[dict[str, Any]]:
        if self._qdrant_repo is None:
            logger.warning("qdrant_not_configured")
            return []
        results = await self._qdrant_repo.search(query_vector=query_vector, limit=limit)
        increment_retrieval_results(source="semantic", count=len(results))
        return results

    async def delete_ephemeral_memory(self, task_id: str) -> None:
        self._ephemeral_cache.pop(task_id, None)
        if self._redis_repo is not None:
            await self._redis_repo.invalidate_episode(task_id)


__all__ = [
    "HybridMemoryService",
    "EpisodeRecord",
    "SemanticVector",
    "MemoryServiceConfig",
    "ReadPreference",
]
