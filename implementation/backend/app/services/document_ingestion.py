from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

from ..core.config import DocumentSettings
from ..core.logging import get_logger
from .document_parser import DocumentParseResult
from .embedding import EmbeddingService
from .memory import HybridMemoryService

logger = get_logger(name=__name__)


@dataclass(slots=True)
class IngestedDocument:
    document_id: str
    metadata: dict[str, Any]
    chunk_count: int
    preview: str
    truncated: bool


class DocumentIngestionService:
    def __init__(
        self,
        *,
        memory: HybridMemoryService,
        embedder: EmbeddingService | None,
        config: DocumentSettings,
    ) -> None:
        self._memory = memory
        self._embedder = embedder
        self._config = config

    async def ingest(self, parsed: DocumentParseResult) -> IngestedDocument:
        document_id = str(uuid.uuid4())
        text = parsed.text.strip()
        chunks, truncated = _chunk_text(
            text,
            chunk_size=self._config.chunk_size,
            overlap=self._config.chunk_overlap,
            max_chunks=self._config.max_chunks,
        )
        if not chunks:
            chunks = [text[: self._config.chunk_size]]

        chunk_keys: list[str] = []
        for index, chunk in enumerate(chunks):
            key = f"document:{document_id}:chunk:{index}"
            chunk_keys.append(key)
            await self._memory.store_working_memory(key, chunk, ttl=self._config.working_ttl_seconds)

        await self._memory.store_working_memory(
            f"document:{document_id}:full",
            text,
            ttl=self._config.working_ttl_seconds,
        )

        document_metadata = {
            "document_id": document_id,
            "filename": parsed.metadata.get("filename"),
            "content_type": parsed.metadata.get("content_type"),
            "extension": parsed.metadata.get("extension"),
            "filesize_bytes": parsed.metadata.get("filesize_bytes"),
            "line_count": parsed.metadata.get("line_count"),
            "character_count": parsed.metadata.get("character_count"),
            "chunk_count": len(chunk_keys),
            "chunk_keys": chunk_keys,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }

        await self._memory.store_ephemeral_memory(document_id, document_metadata)

        if self._embedder is not None:
            try:
                await self._embed_chunks(document_id, chunks, parsed.metadata)
            except Exception as exc:  # pragma: no cover - embedding failures should not block ingestion
                logger.warning("document_embedding_failed", document_id=document_id, error=str(exc))

        preview = text[: self._config.preview_chars].strip()
        return IngestedDocument(
            document_id=document_id,
            metadata={k: v for k, v in document_metadata.items() if k != "chunk_keys"},
            chunk_count=len(chunk_keys),
            preview=preview,
            truncated=truncated or (len(text) > self._config.preview_chars),
        )

    async def _embed_chunks(
        self,
        document_id: str,
        chunks: Sequence[str],
        metadata: dict[str, Any],
    ) -> None:
        if not chunks or self._embedder is None:
            return
        metadatas: list[dict[str, Any]] = []
        for index, _chunk in enumerate(chunks):
            metadatas.append(
                {
                    "document_id": document_id,
                    "chunk_index": index,
                    "chunk_count": len(chunks),
                    "filename": metadata.get("filename"),
                    "content_type": metadata.get("content_type"),
                }
            )
        ids = [f"{document_id}:{index}" for index in range(len(chunks))]
        await self._embedder.embed_documents(
            chunks,
            metadatas=metadatas,
            store=True,
            ids=ids,
            collection=self._config.embedding_collection,
            score=1.0,
        )


def _chunk_text(
    text: str,
    *,
    chunk_size: int,
    overlap: int,
    max_chunks: int,
) -> tuple[list[str], bool]:
    normalized = text.strip()
    if not normalized:
        return [""], False
    chunk_size = max(1, chunk_size)
    overlap = max(0, min(overlap, chunk_size // 2))
    step = chunk_size - overlap
    if step <= 0:
        step = max(1, math.ceil(chunk_size * 0.5))
    chunks: list[str] = []
    start = 0
    length = len(normalized)
    while start < length and len(chunks) < max_chunks:
        end = min(length, start + chunk_size)
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    truncated = (start < length) or (len(chunks) >= max_chunks and end < length)
    return chunks or [normalized], truncated


__all__ = ["DocumentIngestionService", "IngestedDocument"]
