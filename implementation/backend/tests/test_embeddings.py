from __future__ import annotations

from typing import Any, Sequence, cast

import pytest

import app.utils.embeddings as emb_utils
from app.services.embedding import (
	EmbeddingCache,
	EmbeddingModelInfo,
	EmbeddingService,
	EmbeddingServiceConfig,
)
from app.services.memory import HybridMemoryService
from app.utils.embeddings import get_embedding_model, warm_embedding_model


class DummyBackend:
	def __init__(self, name: str, provider: str, dimension: int = 3) -> None:
		self.info = EmbeddingModelInfo(
			provider=provider,
			model_name=name,
			model_version="1",
			vector_dimension=dimension,
		)
		self.calls: list[list[str]] = []

	async def embed(self, texts: Sequence[str]) -> list[list[float]]:
		batch = list(texts)
		self.calls.append(batch)
		base = float(len(self.calls))
		return [[base + index for _ in range(self.info.vector_dimension or 1)] for index in range(len(batch))]

	async def close(self) -> None:
		return None


class FailingBackend:
	def __init__(self) -> None:
		self.info = EmbeddingModelInfo(
			provider="primary",
			model_name="unavailable",
			model_version="1",
			vector_dimension=3,
		)

	async def embed(self, texts: Sequence[str]) -> list[list[float]]:  # noqa: D401
		raise RuntimeError("backend down")

	async def close(self) -> None:
		return None


class RecordingMemory:
	def __init__(self) -> None:
		self.calls: list[dict[str, Any]] = []

	async def store_semantic_memory(
		self,
		*,
		vector: list[float],
		payload: dict[str, Any],
		score: float = 1.0,
	) -> None:
		self.calls.append({"vector": vector, "payload": payload, "score": score})


@pytest.mark.asyncio
async def test_embedding_service_uses_cache() -> None:
	backend = DummyBackend("primary-model", "sentence", dimension=3)
	cache = EmbeddingCache(None, namespace="test-cache", ttl=60)
	service = EmbeddingService(
		EmbeddingServiceConfig(model_name="primary-model", collection_name="test"),
		cache=cache,
		primary_backend=backend,
	)

	record_first = await service.embed_text("hello world")
	record_second = await service.embed_text("hello world")

	assert len(backend.calls) == 1
	assert record_first.vector == record_second.vector
	assert record_second.metadata["cached"] is True

	await service.aclose()


@pytest.mark.asyncio
async def test_embedding_service_falls_back_when_primary_fails() -> None:
	fallback = DummyBackend("fallback-model", "fallback", dimension=3)
	cache = EmbeddingCache(None, namespace="fallback-cache", ttl=60)
	service = EmbeddingService(
		EmbeddingServiceConfig(model_name="primary-model", collection_name="test"),
		cache=cache,
		primary_backend=FailingBackend(),
		fallback_backends=[fallback],
	)

	record = await service.embed_text("needs assistance")

	assert fallback.calls, "Fallback backend should have been invoked"
	assert record.metadata["model_name"] == "fallback-model"

	await service.aclose()


@pytest.mark.asyncio
async def test_embedding_service_stores_metadata_with_memory() -> None:
	backend = DummyBackend("primary-model", "sentence", dimension=3)
	cache = EmbeddingCache(None, namespace="store-cache", ttl=60)
	memory = RecordingMemory()
	service = EmbeddingService(
		EmbeddingServiceConfig(model_name="primary-model", collection_name="semantic"),
		cache=cache,
		primary_backend=backend,
		memory_service=cast(HybridMemoryService, memory),
	)

	record = await service.embed_text(
		"store this",
		metadata={"doc_id": "123"},
		store=True,
		vector_id="doc-123",
		collection="custom",
		score=0.5,
	)

	assert record.metadata["model_name"] == "primary-model"
	assert memory.calls, "Memory service should have been called"
	stored = cast(dict[str, Any], memory.calls[0])
	assert stored["payload"]["collection"] == "custom"
	assert stored["payload"]["metadata"] == {"doc_id": "123"}
	assert stored["payload"]["model"] == "primary-model"
	assert stored["score"] == 0.5

	await service.aclose()


@pytest.mark.asyncio
async def test_warm_embedding_model_preloads(monkeypatch: pytest.MonkeyPatch) -> None:
	# Ensure the cache is clear before testing.
	get_embedding_model.cache_clear()  # type: ignore[attr-defined]

	load_calls: list[str] = []

	class DummySentenceTransformer:
		def __init__(self, name: str) -> None:
			load_calls.append(name)

	monkeypatch.setattr(emb_utils, "SentenceTransformer", DummySentenceTransformer)

	loaded = await warm_embedding_model("test-model")

	assert loaded is True
	assert load_calls == ["test-model"]
	assert get_embedding_model("test-model") is not None
