from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from typing import TYPE_CHECKING, Any, Protocol, Sequence

try:
	from redis.asyncio import Redis
except ModuleNotFoundError:  # pragma: no cover - optional dependency
	Redis = None  # type: ignore[misc,assignment]

try:
	from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:  # pragma: no cover - optional heavy dependency
	SentenceTransformer = None  # type: ignore[misc,assignment]

try:  # pragma: no cover - optional dependency
	import httpx
except ModuleNotFoundError:  # pragma: no cover
	httpx = None  # type: ignore[misc,assignment]

from ..core.config import EmbeddingSettings, Settings
from ..core.logging import get_logger
from ..utils.embeddings import get_embedding_model
from .memory import HybridMemoryService

logger = get_logger(name=__name__)

if TYPE_CHECKING:  # pragma: no cover - typing aids
	from redis.asyncio import Redis as RedisType
	from sentence_transformers import SentenceTransformer as SentenceTransformerType
else:  # pragma: no cover - runtime fallback
	RedisType = Any
	SentenceTransformerType = Any


@dataclass(slots=True)
class EmbeddingServiceConfig:
	model_name: str = "all-MiniLM-L6-v2"
	fallback_model: str | None = "nomic-embed-text"
	cache_namespace: str = "neuraforge:embedding"
	cache_ttl_seconds: int = 86_400
	preferred_dimension: int | None = None
	cache_enabled: bool = True
	collection_name: str = "neura_tasks"


@dataclass(slots=True)
class EmbeddingModelInfo:
	provider: str
	model_name: str
	model_version: str | None
	vector_dimension: int | None


@dataclass(slots=True)
class EmbeddingRecord:
	vector: list[float]
	metadata: dict[str, Any]
	cache_key: str


class EmbeddingBackend(Protocol):
	info: EmbeddingModelInfo

	async def embed(self, texts: Sequence[str]) -> list[list[float]]:
		...

	async def close(self) -> None:  # pragma: no cover - optional cleanup hook
		...


class EmbeddingCache:
	def __init__(
		self,
	client: RedisType | None,
		*,
		namespace: str,
		ttl: int,
		enabled: bool = True,
	) -> None:
		self._client = client
		self._namespace = namespace
		self._ttl = ttl
		self._enabled = enabled
		self._local: dict[str, tuple[list[float], dict[str, Any]]] = {}

	def _key(self, cache_key: str) -> str:
		return f"{self._namespace}:{cache_key}"

	async def get(self, cache_key: str) -> tuple[list[float], dict[str, Any]] | None:
		if not self._enabled:
			return None
		namespaced = self._key(cache_key)
		if self._client is not None:
			try:
				value = await self._client.get(namespaced)
			except Exception as exc:  # pragma: no cover - network failures
				logger.warning("embedding_cache_get_failed", key=namespaced, error=str(exc))
				value = None
			if value is not None:
				if isinstance(value, bytes):
					value = value.decode("utf-8")
				try:
					decoded = json.loads(value)
					vector = [float(v) for v in decoded.get("vector", [])]
					metadata = dict(decoded.get("metadata", {}))
					self._local[cache_key] = (vector, metadata)
					return vector, metadata
				except (json.JSONDecodeError, TypeError) as exc:  # pragma: no cover
					logger.warning("embedding_cache_decode_failed", key=namespaced, error=str(exc))
		return self._local.get(cache_key)

	async def set(self, cache_key: str, vector: Sequence[float], metadata: dict[str, Any]) -> None:
		if not self._enabled:
			return
		payload = json.dumps({"vector": list(vector), "metadata": metadata})
		self._local[cache_key] = (list(vector), metadata.copy())
		if self._client is not None:
			try:
				await self._client.set(self._key(cache_key), payload, ex=self._ttl)
			except Exception as exc:  # pragma: no cover - network failures
				logger.warning("embedding_cache_set_failed", key=cache_key, error=str(exc))

	async def close(self) -> None:
		if self._client is not None:
			close = getattr(self._client, "aclose", None)
			if callable(close):
				result = close()
				if asyncio.iscoroutine(result):
					await result
			else:
				close = getattr(self._client, "close", None)
				if callable(close):
					result = close()
					if asyncio.iscoroutine(result):
						await result
		self._local.clear()


class SentenceTransformerBackend:
	def __init__(self, *, model_name: str, normalize: bool = True) -> None:
		if SentenceTransformer is None:
			raise RuntimeError("sentence_transformers_not_available")
		self._model_name = model_name
		self._normalize = normalize
		self._model: SentenceTransformerType | None = None
		self.info = EmbeddingModelInfo(
			provider="sentence-transformers",
			model_name=model_name,
			model_version=getattr(SentenceTransformer, "__version__", None),
			vector_dimension=None,
		)

	def _load_model(self) -> SentenceTransformerType:
		if self._model is None:
			model = get_embedding_model(self._model_name)
			if model is None:
				raise RuntimeError("sentence_transformers_not_available")
			self._model = model
			try:
				getter = getattr(self._model, "get_sentence_embedding_dimension", None)
				if callable(getter):
					dimension_value = getter()
					if isinstance(dimension_value, int):
						dimension = dimension_value
					else:
						dimension = None
				else:
					dimension = None
			except Exception:  # pragma: no cover - fallback when API changes
				dimension = None
			self.info = EmbeddingModelInfo(
				provider=self.info.provider,
				model_name=self._model_name,
				model_version=getattr(self._model, "__version__", None),
				vector_dimension=dimension,
			)
		return self._model

	async def embed(self, texts: Sequence[str]) -> list[list[float]]:
		model = self._load_model()

		def _encode() -> list[list[float]]:
			embeddings = model.encode(list(texts), normalize_embeddings=self._normalize)
			return [list(map(float, embedding)) for embedding in embeddings]

		loop = asyncio.get_running_loop()
		return await loop.run_in_executor(None, _encode)

	async def close(self) -> None:  # pragma: no cover - compatibility hook
		return None


class OllamaEmbeddingBackend:
	def __init__(self, *, host: str, port: int, model_name: str, timeout: float = 30.0) -> None:
		if httpx is None:
			raise RuntimeError("httpx_not_available")
		base = host.rstrip("/")
		if ":" not in base.rsplit("/", maxsplit=1)[-1]:
			base = f"{base}:{port}"
		self._endpoint = f"{base}/api/embeddings"
		self._model_name = model_name
		self._client = httpx.AsyncClient(timeout=timeout)
		self.info = EmbeddingModelInfo(
			provider="ollama",
			model_name=model_name,
			model_version=None,
			vector_dimension=None,
		)

	async def embed(self, texts: Sequence[str]) -> list[list[float]]:
		vectors: list[list[float]] = []
		for text in texts:
			response = await self._client.post(self._endpoint, json={"model": self._model_name, "prompt": text})
			response.raise_for_status()
			payload = response.json()
			embedding = payload.get("embedding")
			if not isinstance(embedding, list):
				raise RuntimeError("ollama_embedding_invalid_response")
			vector = [float(value) for value in embedding]
			vectors.append(vector)
		if vectors and self.info.vector_dimension is None:
			self.info = EmbeddingModelInfo(
				provider=self.info.provider,
				model_name=self.info.model_name,
				model_version=self.info.model_version,
				vector_dimension=len(vectors[0]),
			)
		return vectors

	async def close(self) -> None:
		await self._client.aclose()


class DeterministicFallbackBackend:
	def __init__(self, *, dimension: int = 384) -> None:
		self._dimension = max(1, dimension)
		self.info = EmbeddingModelInfo(
			provider="deterministic",
			model_name="sha256-fallback",
			model_version="1",
			vector_dimension=self._dimension,
		)

	async def embed(self, texts: Sequence[str]) -> list[list[float]]:
		return [self._hash_to_vector(text) for text in texts]

	def _hash_to_vector(self, text: str) -> list[float]:
		digest = hashlib.sha256(text.encode("utf-8")).digest()
		repeated = (digest * ((self._dimension * 4 // len(digest)) + 1))[: self._dimension * 4]
		slices = [repeated[i : i + 4] for i in range(0, len(repeated), 4)]
		integers = [int.from_bytes(chunk, "big") for chunk in slices[: self._dimension]]
		max_value = max(integers) or 1
		return [value / max_value for value in integers[: self._dimension]]

	async def close(self) -> None:  # pragma: no cover - compatibility hook
		return None


class EmbeddingService:
	def __init__(
		self,
		config: EmbeddingServiceConfig,
		*,
		cache: EmbeddingCache,
		primary_backend: EmbeddingBackend | None = None,
		fallback_backends: Sequence[EmbeddingBackend] | None = None,
		memory_service: HybridMemoryService | None = None,
		settings: Settings | None = None,
	) -> None:
		self.config = config
		self._cache = cache
		self._memory = memory_service
		self._settings = settings
		self._primary = primary_backend
		self._fallbacks = list(fallback_backends or [])

		if self._primary is None:
			try:
				self._primary = SentenceTransformerBackend(model_name=config.model_name)
			except Exception as exc:  # pragma: no cover - handled by fallbacks
				logger.warning(
					"embedding_primary_unavailable",
					model=config.model_name,
					error=str(exc),
				)
				self._primary = None

		if not self._fallbacks:
			self._fallbacks = self._build_default_fallbacks()

	@classmethod
	def from_settings(
		cls,
		settings: Settings,
		*,
		memory_service: HybridMemoryService | None = None,
	redis_client: RedisType | None = None,
		primary_backend: EmbeddingBackend | None = None,
		fallback_backends: Sequence[EmbeddingBackend] | None = None,
	) -> "EmbeddingService":
		embedding_settings: EmbeddingSettings = settings.embedding
		cache_client = redis_client
		if embedding_settings.cache_enabled and cache_client is None and Redis is not None:
			try:
				cache_client = Redis.from_url(str(settings.redis.url))
			except Exception as exc:  # pragma: no cover - connection issues
				logger.warning("embedding_cache_redis_unavailable", error=str(exc))
				cache_client = None

		config = EmbeddingServiceConfig(
			model_name=embedding_settings.default_model,
			fallback_model=embedding_settings.fallback_model,
			cache_namespace=embedding_settings.cache_namespace,
			cache_ttl_seconds=embedding_settings.cache_ttl_seconds,
			preferred_dimension=embedding_settings.preferred_dimension,
			cache_enabled=embedding_settings.cache_enabled,
			collection_name=settings.qdrant.collection_name,
		)

		cache = EmbeddingCache(
			cache_client,
			namespace=config.cache_namespace,
			ttl=config.cache_ttl_seconds,
			enabled=config.cache_enabled,
		)

		return cls(
			config=config,
			cache=cache,
			primary_backend=primary_backend,
			fallback_backends=fallback_backends,
			memory_service=memory_service,
			settings=settings,
		)

	async def embed_text(
		self,
		text: str,
		*,
		metadata: dict[str, Any] | None = None,
		store: bool = False,
		vector_id: str | None = None,
		collection: str | None = None,
		score: float = 1.0,
	) -> EmbeddingRecord:
		records = await self.embed_documents(
			[text],
			metadatas=[metadata] if metadata else None,
			store=store,
			ids=[vector_id] if vector_id else None,
			collection=collection,
			score=score,
		)
		return records[0]

	async def embed_documents(
		self,
		documents: Sequence[str],
		*,
		metadatas: Sequence[dict[str, Any]] | None = None,
		store: bool = False,
		ids: Sequence[str | None] | None = None,
		collection: str | None = None,
		score: float = 1.0,
	) -> list[EmbeddingRecord]:
		if not documents:
			return []

		results: list[EmbeddingRecord | None] = [None] * len(documents)
		misses: list[str] = []
		miss_indices: list[int] = []
		miss_keys: list[str] = []

		for index, doc in enumerate(documents):
			normalized = doc.strip()
			cache_key = self._make_cache_key(normalized)
			cached = await self._cache.get(cache_key)
			user_meta = metadatas[index] if metadatas and index < len(metadatas) else None
			if cached is not None:
				vector, cached_metadata = cached
				metadata = cached_metadata.copy()
				metadata["cached"] = True
				if user_meta:
					metadata["user_metadata"] = user_meta
				results[index] = EmbeddingRecord(vector=vector, metadata=metadata, cache_key=cache_key)
			else:
				misses.append(normalized)
				miss_indices.append(index)
				miss_keys.append(cache_key)

		backend = None
		if misses:
			backend, vectors, info = await self._embed_with_backends(misses)
			for offset, vector in enumerate(vectors):
				idx = miss_indices[offset]
				key = miss_keys[offset]
				user_meta = metadatas[idx] if metadatas and idx < len(metadatas) else None
				base_metadata = self._build_base_metadata(
					info=info,
					text=misses[offset],
					cache_key=key,
					vector=vector,
				)
				cache_metadata = base_metadata.copy()
				await self._cache.set(key, vector, cache_metadata)
				record_metadata = base_metadata.copy()
				record_metadata["cached"] = False
				if user_meta:
					record_metadata["user_metadata"] = user_meta
				results[idx] = EmbeddingRecord(vector=list(vector), metadata=record_metadata, cache_key=key)

		records: list[EmbeddingRecord] = [record for record in results if record is not None]
		for idx, record in enumerate(results):
			if record is None:
				raise RuntimeError("embedding_result_unresolved")

		final_records = [record for record in results if record is not None]

		if store and self._memory is not None:
			await self._store_records(
				documents=documents,
				records=final_records,
				ids=ids,
				collection=collection,
				score=score,
			)

		return final_records

	async def _store_records(
		self,
		*,
		documents: Sequence[str],
		records: Sequence[EmbeddingRecord],
		ids: Sequence[str | None] | None,
		collection: str | None,
		score: float,
	) -> None:
		if self._memory is None:
			return

		for index, record in enumerate(records):
			user_metadata = record.metadata.get("user_metadata")
			payload: dict[str, Any] = {
				"id": ids[index] if ids and index < len(ids) and ids[index] else record.cache_key,
				"collection": collection or self.config.collection_name,
				"model": record.metadata.get("model_name"),
				"model_version": record.metadata.get("model_version"),
				"model_provider": record.metadata.get("model_provider"),
				"vector_dimension": record.metadata.get("vector_dimension"),
				"source_hash": record.metadata.get("source_hash"),
				"text": documents[index],
				"metadata": user_metadata or {},
			}
			try:
				await self._memory.store_semantic_memory(
					vector=record.vector,
					payload=payload,
					score=score,
				)
			except Exception as exc:  # pragma: no cover - persistence issues
				logger.exception("embedding_store_semantic_failed", error=str(exc), payload_id=payload.get("id"))

	async def _embed_with_backends(
		self, texts: Sequence[str]
	) -> tuple[EmbeddingBackend, list[list[float]], EmbeddingModelInfo]:
		candidates: list[EmbeddingBackend] = [backend for backend in [self._primary, *self._fallbacks] if backend is not None]
		if not candidates:
			candidates = [DeterministicFallbackBackend(dimension=self._fallback_dimension())]

		last_error: Exception | None = None
		for backend in candidates:
			try:
				vectors = await backend.embed(texts)
				info = backend.info
				return backend, vectors, info
			except Exception as exc:
				last_error = exc
				logger.exception(
					"embedding_backend_failed",
					backend=backend.info.provider,
					error=str(exc),
				)
		raise RuntimeError("all_embedding_backends_failed") from last_error

	def _build_base_metadata(
		self,
		*,
		info: EmbeddingModelInfo,
		text: str,
		cache_key: str,
		vector: Sequence[float],
	) -> dict[str, Any]:
		dimension = info.vector_dimension or len(vector)
		source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
		return {
			"model_name": info.model_name,
			"model_version": info.model_version,
			"model_provider": info.provider,
			"vector_dimension": dimension,
			"generated_at": datetime.now(timezone.utc).isoformat(),
			"cache_key": cache_key,
			"source_hash": source_hash,
		}

	def _make_cache_key(self, text: str) -> str:
		digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
		target_model = self.config.model_name or "default"
		return f"{target_model}:{digest}"

	def _build_default_fallbacks(self) -> list[EmbeddingBackend]:
		fallbacks: list[EmbeddingBackend] = []

		if self._settings is not None and self.config.fallback_model:
			try:
				backend = OllamaEmbeddingBackend(
					host=self._settings.ollama.host,
					port=self._settings.ollama.port,
					model_name=self.config.fallback_model,
				)
				fallbacks.append(backend)
			except Exception as exc:  # pragma: no cover - optional dependency missing
				logger.warning(
					"embedding_ollama_fallback_unavailable",
					model=self.config.fallback_model,
					error=str(exc),
				)

		fallbacks.append(DeterministicFallbackBackend(dimension=self._fallback_dimension()))
		return fallbacks

	def _fallback_dimension(self) -> int:
		if self.config.preferred_dimension:
			return self.config.preferred_dimension
		if self._primary is not None and self._primary.info.vector_dimension:
			return self._primary.info.vector_dimension  # type: ignore[return-value]
		return 384

	async def aclose(self) -> None:
		await self._cache.close()
		for backend in filter(None, [self._primary, *self._fallbacks]):
			close = getattr(backend, "close", None)
			if close is None:
				continue
			result = close()
			if asyncio.iscoroutine(result):
				await result


__all__ = [
	"EmbeddingBackend",
	"EmbeddingCache",
	"EmbeddingModelInfo",
	"EmbeddingRecord",
	"EmbeddingService",
	"EmbeddingServiceConfig",
]
