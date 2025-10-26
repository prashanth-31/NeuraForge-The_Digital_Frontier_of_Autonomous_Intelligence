from __future__ import annotations

from typing import Any, cast

import pytest

from app.services.consolidation import ConsolidationConfig, ConsolidationJob


class StubMemory:
    def __init__(self, *, tasks: list[str], payloads: dict[str, dict[str, Any]]) -> None:
        self._tasks = tasks
        self._payloads = payloads

    async def enumerate_recent_task_ids(self, *, limit: int) -> list[str]:  # noqa: ARG002
        return list(self._tasks)

    async def fetch_ephemeral_memory(self, task_id: str) -> dict[str, Any] | None:
        return self._payloads.get(task_id)


class StubEmbedder:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def embed_text(
        self,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
        store: bool,
        vector_id: str | None,
        collection: str | None,
        score: float,
    ) -> None:
        self.calls.append(
            {
                "text": text,
                "metadata": metadata,
                "store": store,
                "vector_id": vector_id,
                "collection": collection,
                "score": score,
            }
        )


@pytest.mark.asyncio
async def test_consolidation_embeds_recent_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    metrics_calls: list[dict[str, Any]] = []

    def record_metrics(**kwargs: Any) -> None:
        metrics_calls.append(kwargs)

    monkeypatch.setattr("app.services.consolidation.observe_consolidation_run", record_metrics)

    memory = StubMemory(
        tasks=["task-1"],
        payloads={
            "task-1": {
                "result": {
                    "prompt": "Plan expansion",
                    "status": "complete",
                    "outputs": [
                        {"agent": "research", "content": "market data"},
                        {"agent": "finance", "content": "budget"},
                    ],
                }
            }
        },
    )
    embedder = StubEmbedder()
    job = ConsolidationJob(
        memory=cast(Any, memory),
        embedder=cast(Any, embedder),
        config=ConsolidationConfig(batch_size=5, max_tasks=5),
    )

    outcome = await job.run_once()

    assert outcome.processed == 1
    assert outcome.embedded == 1
    assert outcome.skipped == 0
    assert embedder.calls, "expected consolidation to embed summary"
    summary = embedder.calls[0]["text"]
    assert "Task task-1" in summary
    assert "research: market data" in summary
    assert metrics_calls and metrics_calls[0]["status"] == "completed"
    assert metrics_calls[0]["embedded"] == 1


@pytest.mark.asyncio
async def test_consolidation_records_empty_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    metrics_calls: list[dict[str, Any]] = []

    def record_metrics(**kwargs: Any) -> None:
        metrics_calls.append(kwargs)

    monkeypatch.setattr("app.services.consolidation.observe_consolidation_run", record_metrics)

    memory = StubMemory(tasks=[], payloads={})
    embedder = StubEmbedder()
    job = ConsolidationJob(
        memory=cast(Any, memory),
        embedder=cast(Any, embedder),
        config=ConsolidationConfig(batch_size=2, max_tasks=5),
    )

    outcome = await job.run_once()

    assert outcome.processed == 0
    assert outcome.embedded == 0
    assert not embedder.calls
    assert metrics_calls and metrics_calls[0]["status"] == "empty"