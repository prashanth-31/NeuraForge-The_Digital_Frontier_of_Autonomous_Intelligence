from __future__ import annotations

import json
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Iterable, Sequence

from app.orchestration.guardrails import GuardrailDecision
from app.orchestration.state import OrchestratorEvent, OrchestratorRun, OrchestratorStatus, new_run
from app.services.embedding import EmbeddingRecord, EmbeddingServiceConfig


class StubMemoryService:
    """In-memory stand-in for HybridMemoryService used in regression tests."""

    def __init__(self) -> None:
        self.ephemeral: dict[str, dict[str, Any]] = {}
        self.working: dict[str, list[str]] = {}
        self.semantic_index: list[tuple[list[float], dict[str, Any]]] = []

    @asynccontextmanager
    async def lifecycle(self) -> Any:
        yield self

    async def store_working_memory(self, key: str, value: str, *, ttl: int = 600) -> None:  # noqa: ARG002
        self.working.setdefault(key, []).append(value)

    async def store_ephemeral_memory(self, task_id: str, payload: dict[str, Any]) -> None:
        self.ephemeral[task_id] = payload

    async def fetch_ephemeral_memory(self, task_id: str) -> dict[str, Any] | None:
        return self.ephemeral.get(task_id)

    async def fetch_recent_context(self, *, agent: str | None = None, limit: int = 5) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for task_id, payload in reversed(self.ephemeral.items()):
            candidate = payload.get("result") if isinstance(payload, dict) else payload
            if not isinstance(candidate, dict):
                continue
            if agent is not None and candidate.get("agent") not in {agent, None}:
                continue
            items.append(candidate)
            if len(items) >= limit:
                break
        return items

    async def enumerate_recent_task_ids(self, *, agent: str | None = None, limit: int = 50) -> list[str]:
        task_ids: deque[str] = deque(maxlen=limit)
        for task_id, payload in reversed(self.ephemeral.items()):
            if agent is not None and isinstance(payload, dict):
                agent_value = payload.get("result", {}).get("agent") or payload.get("agent")
                if agent_value not in {agent, None}:
                    continue
            if task_id not in task_ids:
                task_ids.appendleft(task_id)
        return list(task_ids)

    async def retrieve_context(self, *, query_vector: Iterable[float], limit: int = 5) -> list[dict[str, Any]]:  # noqa: ARG002
        return [payload for _, payload in self.semantic_index][:limit]

    async def store_semantic_memory(
        self,
        *,
        vector: list[float],
        payload: dict[str, Any],
        score: float = 1.0,
    ) -> None:  # noqa: ARG002
        self.semantic_index.append((vector, payload))

    async def store_semantic_batch(self, vectors: Iterable[Any]) -> None:
        for item in vectors:
            if isinstance(item, tuple) and len(item) == 3:
                vector, payload, _score = item
            else:
                vector = getattr(item, "vector", [])
                payload = getattr(item, "payload", {})
            if isinstance(vector, list) and isinstance(payload, dict):
                self.semantic_index.append((vector, payload))

    async def delete_ephemeral_memory(self, task_id: str) -> None:
        self.ephemeral.pop(task_id, None)


class StubLLMService:
    """Deterministic planner + response generator for orchestrator flows."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.plan_calls: list[dict[str, Any]] = []

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,  # noqa: ARG002
    ) -> str:
        if "Planning Instructions:" in prompt or "planner" in prompt.lower():
            plan = {
                "steps": [
                    {
                        "agent": "general_agent",
                        "tools": [],
                        "fallback_tools": [],
                        "reason": "initial triage",
                        "confidence": 0.9,
                    },
                    {
                        "agent": "research_agent",
                        "tools": ["research.search", "research.summarizer"],
                        "fallback_tools": ["research.doc_loader"],
                        "reason": "gather supporting insights",
                        "confidence": 0.85,
                    },
                    {
                        "agent": "finance_agent",
                        "tools": ["finance.snapshot"],
                        "fallback_tools": [
                            "finance.snapshot.alpha",
                            "finance.snapshot.cached",
                            "finance.news",
                        ],
                        "reason": "cover financial considerations",
                        "confidence": 0.82,
                    },
                    {
                        "agent": "creative_agent",
                        "tools": ["creative.tonecheck"],
                        "fallback_tools": ["creative.image"],
                        "reason": "shape messaging and narrative",
                        "confidence": 0.8,
                    },
                ],
                "metadata": {
                    "handoff_strategy": "sequential",
                    "notes": "Planner stub returning deterministic multi-agent plan",
                },
                "confidence": 0.84,
            }
            response = json.dumps(plan)
            self.plan_calls.append({"prompt": prompt, "system_prompt": system_prompt})
        else:
            response = f"stubbed-response-{len(self.calls) + 1}"
            self.calls.append({"prompt": prompt, "system_prompt": system_prompt})
        return response


class ImmediateQueue:
    """Queue manager that executes work immediately on enqueue."""

    def __init__(self) -> None:
        self.enqueued = 0

    async def enqueue(self, job: Callable[[], Awaitable[Any]]) -> None:
        self.enqueued += 1
        await job()


class StubToolInvocationResult:
    def __init__(self, tool: str, payload: dict[str, Any]) -> None:
        self.tool = tool
        self.resolved_tool = tool
        self.payload = payload
        self.response = {"status": "ok", "tool": tool}
        self.cached = False
        self.latency = 0.0


class StubToolService:
    """Captures tool invocations for assertions while returning canned payloads."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None

    async def invoke(self, tool: str, payload: dict[str, Any]) -> StubToolInvocationResult:
        self.calls.append((tool, payload))
        if self._callback is not None:
            await self._callback({"tool": tool, "payload": payload})
        return StubToolInvocationResult(tool, payload)

    @asynccontextmanager
    async def instrument(self, callback: Callable[[dict[str, Any]], Awaitable[None]]):
        self._callback = callback
        try:
            yield self
        finally:
            self._callback = None

    def get_diagnostics(self) -> dict[str, str]:
        return {"status": "stubbed"}


class StubOrchestratorStateStore:
    """Lightweight in-memory run/event store for orchestrator regression tests."""

    def __init__(self) -> None:
        self.runs: dict[str, OrchestratorRun] = {}
        self.events: dict[str, list[OrchestratorEvent]] = {}

    @classmethod
    def from_settings(cls, _settings: Any) -> "StubOrchestratorStateStore":
        return cls()

    @asynccontextmanager
    async def lifecycle(self) -> Any:
        yield self

    async def start_run(self, task_id: str, *, state: dict[str, Any]) -> OrchestratorRun:
        run = new_run(task_id, state=state)
        run.created_at = run.updated_at = datetime.now(timezone.utc)
        self.runs[str(run.run_id)] = run
        self.events.setdefault(str(run.run_id), [])
        return run

    async def record_event(self, event: OrchestratorEvent) -> None:
        key = str(event.run_id)
        event.created_at = event.created_at or datetime.now(timezone.utc)
        self.events.setdefault(key, []).append(event)

    async def update_state(
        self,
        run_id,
        *,
        state: dict[str, Any],
        status: OrchestratorStatus | None = None,
    ) -> None:
        key = str(run_id)
        run = self.runs.get(key)
        if run is None:
            return
        run.state = dict(state)
        if status is not None:
            run.status = status
        run.updated_at = datetime.now(timezone.utc)
        self.runs[key] = run

    async def finalize_run(
        self,
        run_id,
        *,
        state: dict[str, Any],
        status: OrchestratorStatus,
        error: str | None = None,
    ) -> None:
        payload = dict(state)
        if error:
            payload["error"] = error
        await self.update_state(run_id, state=payload, status=status)

    async def get_run(self, run_id) -> OrchestratorRun | None:
        return self.runs.get(str(run_id))

    async def get_latest_run_for_task(self, task_id: str) -> OrchestratorRun | None:
        candidates = [run for run in self.runs.values() if run.task_id == task_id]
        if not candidates:
            return None
        return max(candidates, key=lambda run: run.created_at)

    async def list_events(self, run_id, *, limit: int | None = None) -> list[OrchestratorEvent]:
        items = self.events.get(str(run_id), [])
        if limit is None:
            return list(items)
        return list(items)[:limit]

    async def close(self) -> None:  # noqa: D401 - no-op close for parity with production store
        return


class StubTaskLifecycleStore:
    """In-memory lifecycle recorder to bypass Postgres dependencies."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    @classmethod
    def from_settings(cls, _settings: Any) -> "StubTaskLifecycleStore":
        return cls()

    @asynccontextmanager
    async def lifecycle(self) -> Any:
        yield self

    async def record(self, event: Any) -> None:
        self.events.append(event)

    async def record_plan(self, plan: Any, *, run_id: Any | None = None) -> None:  # noqa: ANN401
        self.events.append(("plan", plan, run_id))

    async def close(self) -> None:
        return


class StubContextSnapshotStore:
    """Collects snapshots locally instead of writing to persistence."""

    def __init__(self) -> None:
        self.snapshots: list[Any] = []

    @classmethod
    def from_settings(cls, _settings: Any) -> "StubContextSnapshotStore":
        return cls()

    @asynccontextmanager
    async def lifecycle(self) -> Any:
        yield self

    async def record(self, snapshot: Any) -> None:
        self.snapshots.append(snapshot)

    async def close(self) -> None:
        return


class StubGuardrailStore:
    """Keeps guardrail audit events in memory for validation."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    @classmethod
    def from_settings(cls, _settings: Any) -> "StubGuardrailStore":
        return cls()

    @asynccontextmanager
    async def lifecycle(self) -> Any:
        yield self

    async def record(
        self,
        *,
        task_id: str,
        run_id: Any | None,
        decision: GuardrailDecision,
        agent: str | None,
        payload: dict[str, Any],
    ) -> None:
        self.records.append(
            {
                "task_id": task_id,
                "run_id": run_id,
                "decision": decision,
                "agent": agent,
                "payload": payload,
            }
        )

    async def close(self) -> None:
        return


class StubEmbeddingService:
    """Deterministic embedding provider that never reaches external systems."""

    def __init__(self) -> None:
        self.config = EmbeddingServiceConfig()
        self._counter = 0
        self.requests: list[str] = []

    @classmethod
    def from_settings(
        cls,
        _settings: Any,
        *,
        memory_service: Any | None = None,
        redis_client: Any | None = None,
        primary_backend: Any | None = None,
        fallback_backends: Sequence[Any] | None = None,
    ) -> "StubEmbeddingService":  # noqa: ARG003
        return cls()

    async def embed_text(
        self,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
        store: bool = False,  # noqa: ARG002
        vector_id: str | None = None,  # noqa: ARG002
        collection: str | None = None,  # noqa: ARG002
        score: float = 1.0,  # noqa: ARG002
    ) -> EmbeddingRecord:
        records = await self.embed_documents([text], metadatas=[metadata or {}])
        return records[0]

    async def embed_documents(
        self,
        documents: Sequence[str],
        *,
        metadatas: Sequence[dict[str, Any]] | None = None,
        store: bool = False,  # noqa: ARG002
        ids: Sequence[str | None] | None = None,  # noqa: ARG002
        collection: str | None = None,  # noqa: ARG002
        score: float = 1.0,  # noqa: ARG002
    ) -> list[EmbeddingRecord]:
        records: list[EmbeddingRecord] = []
        for index, text in enumerate(documents):
            self.requests.append(text)
            self._counter += 1
            seed = sum(ord(ch) for ch in text) + self._counter
            vector = [((seed + offset) % 17) / 17 for offset in range(4)]
            metadata = {"cached": False}
            if metadatas and index < len(metadatas):
                metadata.update({"user_metadata": metadatas[index]})
            records.append(EmbeddingRecord(vector=vector, metadata=metadata, cache_key=f"stub:{seed}"))
        return records

    async def aclose(self) -> None:
        return
