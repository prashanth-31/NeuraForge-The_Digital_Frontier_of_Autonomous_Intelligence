from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from ..core.config import ConsolidationSettings, Settings
from ..core.logging import get_logger
from ..core.metrics import observe_consolidation_run
from .embedding import EmbeddingService
from .memory import HybridMemoryService

logger = get_logger(name=__name__)


@dataclass(slots=True)
class ConsolidationConfig:
    batch_size: int
    max_tasks: int

    @classmethod
    def from_settings(cls, settings: Settings) -> "ConsolidationConfig":
        options: ConsolidationSettings = settings.consolidation
        return cls(batch_size=options.batch_size, max_tasks=options.max_tasks)


@dataclass(slots=True)
class ConsolidationResult:
    processed: int
    embedded: int
    skipped: int
    duration_seconds: float


class ConsolidationJob:
    def __init__(
        self,
        *,
        memory: HybridMemoryService,
        embedder: EmbeddingService,
        config: ConsolidationConfig,
    ) -> None:
        self._memory = memory
        self._embedder = embedder
        self._config = config

    @classmethod
    async def run_once_from_settings(cls, settings: Settings) -> ConsolidationResult:
        memory_service = HybridMemoryService.from_settings(settings)
        async with memory_service.lifecycle() as memory:
            try:
                embedder = EmbeddingService.from_settings(settings, memory_service=memory)
            except Exception as exc:  # pragma: no cover - configuration errors surfaced in logs
                logger.exception("embedding_service_unavailable", error=str(exc))
                return ConsolidationResult(processed=0, embedded=0, skipped=0, duration_seconds=0.0)

            try:
                job = cls(memory=memory, embedder=embedder, config=ConsolidationConfig.from_settings(settings))
                return await job.run_once()
            finally:
                await embedder.aclose()

    async def run_once(self) -> ConsolidationResult:
        start = time.perf_counter()
        processed = embedded = skipped = 0

        candidates = await self._memory.enumerate_recent_task_ids(limit=self._config.max_tasks)
        if not candidates:
            duration = time.perf_counter() - start
            observe_consolidation_run(status="empty", duration=duration, processed=0, embedded=0, skipped=0)
            return ConsolidationResult(processed=0, embedded=0, skipped=0, duration_seconds=duration)

        for task_id in candidates[: self._config.batch_size]:
            processed += 1
            payload = await self._memory.fetch_ephemeral_memory(task_id)
            if not isinstance(payload, dict):
                skipped += 1
                continue
            summary_text = self._build_summary(task_id, payload)
            if not summary_text:
                skipped += 1
                continue
            try:
                await self._embedder.embed_text(
                    summary_text,
                    metadata={
                        "task_id": task_id,
                        "source": "consolidation",
                        "role": payload.get("result", {}).get("status"),
                    },
                    store=True,
                    vector_id=f"{task_id}-summary",
                    collection=payload.get("collection") or None,
                    score=1.0,
                )
                embedded += 1
            except Exception as exc:  # pragma: no cover - downstream store or embedding failure
                skipped += 1
                logger.exception("consolidation_embedding_failed", task_id=task_id, error=str(exc))

        duration = time.perf_counter() - start
        observe_consolidation_run(
            status="completed",
            duration=duration,
            processed=processed,
            embedded=embedded,
            skipped=skipped,
        )
        return ConsolidationResult(processed=processed, embedded=embedded, skipped=skipped, duration_seconds=duration)

    def _build_summary(self, task_id: str, payload: dict[str, Any]) -> str:
        result = payload.get("result")
        if isinstance(result, dict):
            outputs = result.get("outputs")
            if isinstance(outputs, list) and outputs:
                fragments: list[str] = []
                for entry in outputs[-3:]:
                    agent = entry.get("agent", "unknown") if isinstance(entry, dict) else "unknown"
                    content = self._normalize_content(entry.get("content") if isinstance(entry, dict) else entry)
                    fragments.append(f"{agent}: {content}")
                prompt = result.get("prompt") or payload.get("prompt") or ""
                return f"Task {task_id}\nPrompt: {prompt}\n" + "\n".join(fragments)
        return self._normalize_content(result or payload)

    @staticmethod
    def _normalize_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        try:
            return json.dumps(content, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            return str(content)
