from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.config import RetrievalSettings, Settings
from ..core.logging import get_logger
from ..core.metrics import increment_retrieval_results, observe_context_chars
from .embedding import EmbeddingRecord, EmbeddingService
from .memory import HybridMemoryService

logger = get_logger(name=__name__)


@dataclass(slots=True)
class RetrievalConfig:
    semantic_limit: int = 5
    episodic_limit: int = 5
    max_context_chars: int = 2_000
    relevance_threshold: float = 0.0
    document_chunk_limit: int = 0

    @classmethod
    def from_settings(cls, settings: Settings) -> "RetrievalConfig":
        options: RetrievalSettings = settings.retrieval
        document_settings = getattr(settings, "documents", None)
        chunk_limit = 0
        if document_settings is not None and getattr(document_settings, "enabled", True):
            chunk_limit = max(0, int(getattr(document_settings, "max_context_chunks", 0)))
        return cls(
            semantic_limit=options.semantic_limit,
            episodic_limit=options.episodic_limit,
            max_context_chars=options.max_context_chars,
            relevance_threshold=options.relevance_threshold,
            document_chunk_limit=chunk_limit,
        )


@dataclass(slots=True)
class RetrievalResult:
    query: str
    query_vector: list[float]
    semantic_hits: list[dict[str, Any]] = field(default_factory=list)
    episodic_hits: list[dict[str, Any]] = field(default_factory=list)
    document_hits: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class ContextSnippet:
    source: str
    content: str
    metadata: dict[str, Any]
    score: float | None = None


@dataclass(slots=True)
class ContextBundle:
    query: str
    snippets: list[ContextSnippet]
    max_chars: int

    def as_prompt_section(self) -> str:
        if not self.snippets:
            return "(no retrieved context)"
        lines: list[str] = []
        remaining = self.max_chars
        for snippet in self.snippets:
            line = self._format_snippet(snippet)
            if len(line) > remaining:
                break
            lines.append(line)
            remaining -= len(line)
        return "\n".join(lines) if lines else "(no retrieved context)"

    def _format_snippet(self, snippet: ContextSnippet) -> str:
        score_part = f" score={snippet.score:.3f}" if snippet.score is not None else ""
        meta_label = snippet.metadata.get("label") or snippet.metadata.get("agent") or snippet.source
        return f"[{meta_label}{score_part}] {snippet.content.strip()}"


class RetrievalService:
    def __init__(
        self,
        *,
        memory: HybridMemoryService,
        embedder: EmbeddingService,
        config: RetrievalConfig,
        document_chunk_limit: int | None = None,
    ) -> None:
        self._memory = memory
        self._embedder = embedder
        self._config = config
        limit = document_chunk_limit if document_chunk_limit is not None else config.document_chunk_limit
        self._document_chunk_limit = max(0, int(limit))

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        memory: HybridMemoryService,
        embedder: EmbeddingService,
    ) -> "RetrievalService":
        config = RetrievalConfig.from_settings(settings)
        document_settings = getattr(settings, "documents", None)
        chunk_limit = config.document_chunk_limit
        if document_settings is not None and not getattr(document_settings, "enabled", True):
            chunk_limit = 0
        elif document_settings is not None:
            chunk_limit = max(chunk_limit, int(getattr(document_settings, "max_context_chunks", chunk_limit)))
        return cls(memory=memory, embedder=embedder, config=config, document_chunk_limit=chunk_limit)

    async def gather(
        self,
        *,
        query: str,
        agent: str | None = None,
        task_id: str | None = None,
        documents: list[str] | tuple[str, ...] | None = None,
    ) -> RetrievalResult:
        vector_record: EmbeddingRecord | None = None
        try:
            vector_record = await self._embedder.embed_text(
                query,
                metadata={"purpose": "retrieval", "task_id": task_id or ""},
                store=False,
            )
        except Exception as exc:  # pragma: no cover - defensive instrumentation
            logger.exception("embedding_failed_for_retrieval", error=str(exc))
            vector_record = None

        semantic_hits: list[dict[str, Any]] = []
        if vector_record is not None:
            try:
                semantic_hits = await self._memory.retrieve_context(
                    query_vector=vector_record.vector,
                    limit=self._config.semantic_limit,
                )
            except Exception as exc:  # pragma: no cover - downstream store failures
                logger.exception("semantic_retrieval_failed", error=str(exc))
                semantic_hits = []

        episodic_hits: list[dict[str, Any]] = []
        try:
            episodic_hits = await self._memory.fetch_recent_context(agent=agent, limit=self._config.episodic_limit)
        except Exception as exc:  # pragma: no cover - downstream store failures
            logger.exception("episodic_retrieval_failed", error=str(exc))
            episodic_hits = []

        document_hits: list[dict[str, Any]] = []
        try:
            if documents:
                document_hits = await self._gather_documents(documents)
        except Exception as exc:  # pragma: no cover - defensive guardrail
            logger.exception("document_context_gather_failed", error=str(exc))
            document_hits = []

        increment_retrieval_results(source="assembled", count=len(semantic_hits) + len(episodic_hits) + len(document_hits))
        return RetrievalResult(
            query=query,
            query_vector=vector_record.vector if vector_record else [],
            semantic_hits=semantic_hits,
            episodic_hits=episodic_hits,
            document_hits=document_hits,
        )

    async def _gather_documents(self, document_ids: Any) -> list[dict[str, Any]]:
        if self._document_chunk_limit <= 0:
            return []
        if not isinstance(document_ids, (list, tuple, set)):
            return []
        results: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw_id in document_ids:
            if not isinstance(raw_id, str):
                continue
            document_id = raw_id.strip()
            if not document_id or document_id in seen:
                continue
            seen.add(document_id)
            metadata = await self._memory.fetch_ephemeral_memory(document_id)
            if not isinstance(metadata, dict):
                continue
            chunk_keys = metadata.get("chunk_keys")
            if not isinstance(chunk_keys, list):
                continue
            chunk_limit = min(self._document_chunk_limit, len(chunk_keys))
            collected = 0
            for key in chunk_keys:
                if collected >= chunk_limit:
                    break
                if not isinstance(key, str) or not key:
                    continue
                chunk_text = await self._memory.fetch_working_memory(key)
                if not chunk_text:
                    continue
                chunk_index = self._extract_chunk_index(key)
                results.append(
                    {
                        "text": chunk_text,
                        "document_id": document_id,
                        "chunk_index": chunk_index,
                        "chunk_count": metadata.get("chunk_count"),
                        "filename": metadata.get("filename"),
                        "content_type": metadata.get("content_type"),
                        "ingested_at": metadata.get("ingested_at"),
                        "_score": 1.0,
                        "label": metadata.get("filename") or f"document:{document_id[:8]}",
                    }
                )
                collected += 1
        return results

    @staticmethod
    def _extract_chunk_index(key: str) -> int | None:
        try:
            return int(key.rsplit(":", 1)[-1])
        except (ValueError, TypeError):
            return None


class ContextAssembler:
    def __init__(self, *, retrieval: RetrievalService, max_chars: int | None = None) -> None:
        self._retrieval = retrieval
        self._max_chars = max_chars or retrieval._config.max_context_chars

    async def build(
        self,
        *,
        task: dict[str, Any],
        agent: str | None = None,
    ) -> ContextBundle:
        query = task.get("prompt") or ""
        task_id = task.get("id") or None
        document_ids = self._extract_document_ids(task)
        result = await self._retrieval.gather(query=query, agent=agent, task_id=task_id, documents=document_ids)
        snippets = self._convert_to_snippets(result)
        observe_context_chars(agent=agent or "unknown", length=sum(len(snippet.content) for snippet in snippets))
        return ContextBundle(query=result.query, snippets=snippets, max_chars=self._max_chars)

    def _convert_to_snippets(self, result: RetrievalResult) -> list[ContextSnippet]:
        snippets: list[ContextSnippet] = []
        seen_hashes: set[str] = set()
        threshold = self._retrieval._config.relevance_threshold
        for source, hits in ("semantic", result.semantic_hits), ("episodic", result.episodic_hits), ("document", result.document_hits):
            for item in hits:
                text = self._extract_text(item)
                if not text:
                    continue
                unique_key = f"{source}:{hash(text)}"
                if unique_key in seen_hashes:
                    continue
                seen_hashes.add(unique_key)
                score = self._extract_score(item)
                if source == "semantic" and score is not None and score < threshold:
                    continue
                snippets.append(
                    ContextSnippet(
                        source=source,
                        content=text,
                        metadata={k: v for k, v in item.items() if k not in {"text", "content", "_score"}},
                        score=score,
                    )
                )
        return snippets

    @staticmethod
    def _extract_text(item: dict[str, Any]) -> str:
        if "text" in item and isinstance(item["text"], str):
            return item["text"]
        if "content" in item and isinstance(item["content"], str):
            return item["content"]
        if "result" in item and isinstance(item["result"], dict):
            return ContextAssembler._extract_text(item["result"])
        return ""

    @staticmethod
    def _extract_score(item: dict[str, Any]) -> float | None:
        raw = item.get("_score") or item.get("score")
        try:
            return float(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_document_ids(task: dict[str, Any]) -> list[str]:
        metadata = task.get("metadata") if isinstance(task, dict) else None
        if not isinstance(metadata, dict):
            return []
        documents = metadata.get("documents")
        if not isinstance(documents, (list, tuple, set)):
            return []
        seen: set[str] = set()
        ordered: list[str] = []
        for value in documents:
            if not isinstance(value, str):
                continue
            normalized = value.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
        return ordered
