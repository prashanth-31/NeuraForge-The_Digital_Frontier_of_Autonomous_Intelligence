from __future__ import annotations

import asyncio
import json
import math
from collections import Counter
from datetime import UTC, datetime
from typing import Any, Iterable, Mapping

from pydantic import BaseModel, Field, model_validator

from .base import MCPToolAdapter


class _MemoryStore:
    _entries: dict[str, list[dict[str, Any]]] = {}
    _lock: asyncio.Lock = asyncio.Lock()

    @classmethod
    async def append(cls, namespace: str, entry: Mapping[str, Any]) -> None:
        async with cls._lock:
            cls._entries.setdefault(namespace, []).append(dict(entry))

    @classmethod
    async def get(cls, namespace: str) -> list[dict[str, Any]]:
        async with cls._lock:
            entries = cls._entries.get(namespace, [])
            return [dict(item) for item in entries]

    @classmethod
    async def clear(cls, namespace: str | None = None) -> int:
        async with cls._lock:
            if namespace is None:
                count = sum(len(items) for items in cls._entries.values())
                cls._entries.clear()
                return count
            entries = cls._entries.pop(namespace, [])
            return len(entries)


def _serialise_payload(payload: Any) -> str:
    try:
        return json.dumps(payload, sort_keys=True)
    except TypeError:
        return str(payload)


class MemoryStoreInput(BaseModel):
    namespace: str = Field(..., min_length=1, max_length=64)
    key: str = Field(..., min_length=1, max_length=128)
    value: Any
    tags: list[str] = Field(default_factory=list)
    metadata: Mapping[str, Any] | None = Field(default=None)

    model_config = {"extra": "forbid"}


class MemoryStoreOutput(BaseModel):
    namespace: str
    key: str
    stored_at: datetime
    index: int


class MemoryStoreAdapter(MCPToolAdapter):
    name = "memory/store"
    description = "Stores lightweight facts in an in-memory timeline suitable for simulations."
    labels = ("memory", "store")
    aliases = ("memory.store",)
    capabilities = ("memory", "write")
    InputModel = MemoryStoreInput
    OutputModel = MemoryStoreOutput

    async def _invoke(self, payload_model: MemoryStoreInput) -> Mapping[str, Any]:
        now = datetime.now(UTC)
        entry = {
            "namespace": payload_model.namespace,
            "key": payload_model.key,
            "value": payload_model.value,
            "tags": list(payload_model.tags),
            "metadata": dict(payload_model.metadata or {}),
            "stored_at": now,
            "text": _serialise_payload(payload_model.value).lower(),
        }
        await _MemoryStore.append(payload_model.namespace, entry)
        entries = await _MemoryStore.get(payload_model.namespace)
        index = len(entries) - 1
        return {
            "namespace": payload_model.namespace,
            "key": payload_model.key,
            "stored_at": now,
            "index": index,
        }


class MemoryRetrieveInput(BaseModel):
    namespace: str = Field(..., min_length=1, max_length=64)
    key: str | None = Field(default=None)
    limit: int = Field(10, ge=1, le=100)

    model_config = {"extra": "forbid"}


class MemoryRetrieveOutput(BaseModel):
    items: list[Mapping[str, Any]]


FALLBACK_TIME = datetime.min.replace(tzinfo=UTC)


class MemoryRetrieveAdapter(MCPToolAdapter):
    name = "memory/retrieve"
    description = "Retrieves stored memory entries using namespace and optional key filters."
    labels = ("memory", "read")
    aliases = ("memory.retrieve",)
    capabilities = ("memory", "read")
    InputModel = MemoryRetrieveInput
    OutputModel = MemoryRetrieveOutput

    async def _invoke(self, payload_model: MemoryRetrieveInput) -> Mapping[str, Any]:
        entries = await _MemoryStore.get(payload_model.namespace)
        if payload_model.key:
            entries = [entry for entry in entries if entry.get("key") == payload_model.key]
        entries.sort(key=lambda item: item.get("stored_at") or FALLBACK_TIME, reverse=True)
        return {"items": entries[: payload_model.limit]}


class MemoryTimelineInput(BaseModel):
    namespace: str = Field(..., min_length=1, max_length=64)
    limit: int = Field(20, ge=1, le=200)

    model_config = {"extra": "forbid"}


class MemoryTimelineOutput(BaseModel):
    events: list[Mapping[str, Any]]


class MemoryTimelineAdapter(MCPToolAdapter):
    name = "memory/timeline"
    description = "Returns the chronological timeline for a namespace."
    labels = ("memory", "timeline")
    aliases = ("memory.timeline",)
    capabilities = ("memory", "timeline")
    InputModel = MemoryTimelineInput
    OutputModel = MemoryTimelineOutput

    async def _invoke(self, payload_model: MemoryTimelineInput) -> Mapping[str, Any]:
        entries = await _MemoryStore.get(payload_model.namespace)
        entries.sort(key=lambda item: item.get("stored_at") or FALLBACK_TIME)
        return {"events": entries[-payload_model.limit :]}


class MemorySearchRecentInput(BaseModel):
    namespace: str = Field(..., min_length=1, max_length=64)
    query: str = Field(..., min_length=2, max_length=256)
    limit: int = Field(5, ge=1, le=50)

    model_config = {"extra": "forbid"}


class MemorySearchRecentOutput(BaseModel):
    matches: list[Mapping[str, Any]]


class MemorySearchRecentAdapter(MCPToolAdapter):
    name = "memory/search_recent"
    description = "Performs keyword lookup against the latest entries for a namespace."
    labels = ("memory", "search")
    aliases = ("memory.search_recent",)
    capabilities = ("memory", "search")
    InputModel = MemorySearchRecentInput
    OutputModel = MemorySearchRecentOutput

    async def _invoke(self, payload_model: MemorySearchRecentInput) -> Mapping[str, Any]:
        entries = await _MemoryStore.get(payload_model.namespace)
        entries.sort(key=lambda item: item.get("stored_at") or FALLBACK_TIME, reverse=True)
        tokens = set(payload_model.query.lower().split())
        matches: list[dict[str, Any]] = []
        for entry in entries:
            text = str(entry.get("text") or "")
            if tokens & set(text.split()):
                matches.append(entry)
            if len(matches) >= payload_model.limit:
                break
        return {"matches": matches}


class MemoryVectorSearchInput(BaseModel):
    namespace: str = Field(..., min_length=1, max_length=64)
    query: str = Field(..., min_length=2, max_length=512)
    limit: int = Field(5, ge=1, le=20)

    model_config = {"extra": "forbid"}


class MemoryVectorSearchMatch(BaseModel):
    key: str
    score: float
    value: Any
    stored_at: datetime


class MemoryVectorSearchOutput(BaseModel):
    matches: list[MemoryVectorSearchMatch]


class MemoryVectorSearchAdapter(MCPToolAdapter):
    name = "memory/vector_search"
    description = "Executes an in-memory cosine similarity search over stored entries."
    labels = ("memory", "vector")
    aliases = ("memory.vector_search",)
    capabilities = ("memory", "vector")
    InputModel = MemoryVectorSearchInput
    OutputModel = MemoryVectorSearchOutput

    async def _invoke(self, payload_model: MemoryVectorSearchInput) -> Mapping[str, Any]:
        entries = await _MemoryStore.get(payload_model.namespace)
        query_vector = Counter(payload_model.query.lower().split())
        scored: list[MemoryVectorSearchMatch] = []
        for entry in entries:
            text = str(entry.get("text") or "")
            candidate_vector = Counter(text.split())
            score = self._cosine_similarity(query_vector, candidate_vector)
            if score <= 0.0:
                continue
            scored.append(
                MemoryVectorSearchMatch(
                    key=str(entry.get("key")),
                    score=round(score, 4),
                    value=entry.get("value"),
                    stored_at=entry.get("stored_at", datetime.now(UTC)),
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return {"matches": [match.model_dump() for match in scored[: payload_model.limit]]}

    @staticmethod
    def _cosine_similarity(a: Counter[str], b: Counter[str]) -> float:
        if not a or not b:
            return 0.0
        numerator = sum(a[token] * b[token] for token in a.keys() & b.keys())
        if numerator == 0:
            return 0.0
        a_magnitude = math.sqrt(sum(value * value for value in a.values()))
        b_magnitude = math.sqrt(sum(value * value for value in b.values()))
        if a_magnitude == 0 or b_magnitude == 0:
            return 0.0
        return numerator / (a_magnitude * b_magnitude)


class MemoryClearInput(BaseModel):
    namespace: str | None = Field(default=None)

    model_config = {"extra": "forbid"}


class MemoryClearOutput(BaseModel):
    cleared: int


class MemoryClearAdapter(MCPToolAdapter):
    name = "memory/clear"
    description = "Clears memory entries across a namespace or the full store for isolated testing."
    labels = ("memory", "admin")
    aliases = ("memory.clear",)
    capabilities = ("memory", "admin")
    InputModel = MemoryClearInput
    OutputModel = MemoryClearOutput

    async def _invoke(self, payload_model: MemoryClearInput) -> Mapping[str, Any]:
        cleared = await _MemoryStore.clear(payload_model.namespace)
        return {"cleared": cleared}


MEMORY_ADAPTER_CLASSES: tuple[type[MCPToolAdapter], ...] = (
    MemoryStoreAdapter,
    MemoryRetrieveAdapter,
    MemoryTimelineAdapter,
    MemorySearchRecentAdapter,
    MemoryVectorSearchAdapter,
    MemoryClearAdapter,
)


__all__ = [
    "MemoryStoreAdapter",
    "MemoryRetrieveAdapter",
    "MemoryTimelineAdapter",
    "MemorySearchRecentAdapter",
    "MemoryVectorSearchAdapter",
    "MemoryClearAdapter",
    "MEMORY_ADAPTER_CLASSES",
]
