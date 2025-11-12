from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any, Awaitable, Callable

import pytest
from fastapi.testclient import TestClient

from app.api import routes as routes_module
from app.dependencies import get_hybrid_memory, get_task_queue
from app.main import app


class StubMemoryService:
    def __init__(self) -> None:
        self.ephemeral: dict[str, dict[str, Any]] = {}
        self.working: dict[str, list[str]] = {}

    @asynccontextmanager
    async def lifecycle(self) -> Any:
        yield self

    async def store_working_memory(self, key: str, value: str, *, ttl: int = 600) -> None:  # noqa: ARG002
        self.working.setdefault(key, []).append(value)

    async def store_ephemeral_memory(self, task_id: str, payload: dict[str, Any]) -> None:
        self.ephemeral[task_id] = payload

    async def fetch_ephemeral_memory(self, task_id: str) -> dict[str, Any] | None:
        return self.ephemeral.get(task_id)


class StubLLMService:
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
        if "Planning Instructions:" in prompt:
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
                        "fallback_tools": ["finance.news"],
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


@pytest.fixture()
def submit_client(monkeypatch: pytest.MonkeyPatch):
    memory = StubMemoryService()
    llm = StubLLMService()
    queue = ImmediateQueue()
    tool_service = StubToolService()

    monkeypatch.setattr(routes_module.HybridMemoryService, "from_settings", lambda settings: memory)
    monkeypatch.setattr(
        routes_module.LLMService,
        "from_settings",
        lambda settings, *, model=None, client=None: llm,
    )

    async def _mock_tool_service():
        return tool_service

    monkeypatch.setattr(routes_module, "get_tool_service", _mock_tool_service)

    async def override_queue():
        yield queue

    async def override_memory():
        yield memory

    app.dependency_overrides[get_task_queue] = override_queue
    app.dependency_overrides[get_hybrid_memory] = override_memory

    client = TestClient(app)
    try:
        yield client, memory, llm, queue
    finally:
        app.dependency_overrides.clear()



def test_submit_task_executes_agents_and_persists_results(submit_client):
    client, memory, llm, queue = submit_client

    response = client.post(
        "/api/v1/submit_task",
        json={"prompt": "Draft a launch plan", "metadata": {"priority": "high"}},
    )

    assert response.status_code == 200
    body = response.json()
    task_id = body["task_id"]
    assert body["status"] == "queued"
    assert queue.enqueued == 1

    assert len(llm.calls) == 4  # Four domain agents participate
    assert task_id in memory.ephemeral
    stored = memory.ephemeral[task_id]["result"]
    assert stored["status"] == "completed"
    assert "report" in stored
    assert stored["report"].get("headline")

    history = client.get(f"/api/v1/history/{task_id}")
    assert history.status_code == 200
    results = history.json()
    assert len(results) == 4
    assert all(entry["task_id"] == task_id for entry in results)
    assert all("text" in entry["content"] for entry in results)


def test_submit_task_records_failure_when_llm_unavailable(monkeypatch: pytest.MonkeyPatch):
    memory = StubMemoryService()
    queue = ImmediateQueue()
    tool_service = StubToolService()

    monkeypatch.setattr(routes_module.HybridMemoryService, "from_settings", lambda settings: memory)
    def _raise_llm(settings, *, model=None, client=None):  # noqa: ANN001, ARG001
        raise RuntimeError("llm missing")

    monkeypatch.setattr(routes_module.LLMService, "from_settings", _raise_llm)

    async def _mock_tool_service():
        return tool_service

    monkeypatch.setattr(routes_module, "get_tool_service", _mock_tool_service)

    async def override_queue():
        yield queue

    async def override_memory():
        yield memory

    app.dependency_overrides[get_task_queue] = override_queue
    app.dependency_overrides[get_hybrid_memory] = override_memory

    client = TestClient(app)
    try:
        response = client.post(
            "/api/v1/submit_task",
            json={"prompt": "Collect insights", "metadata": {}},
        )

        assert response.status_code == 200
        task_id = response.json()["task_id"]
        failure = memory.ephemeral[task_id]["result"]
        assert failure["status"] == "failed"
        assert "error" in failure

        history = client.get(f"/api/v1/history/{task_id}")
        assert history.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_submit_task_stream_emits_agent_events(monkeypatch: pytest.MonkeyPatch):
    memory = StubMemoryService()
    llm = StubLLMService()
    tool_service = StubToolService()

    monkeypatch.setattr(routes_module.HybridMemoryService, "from_settings", lambda settings: memory)
    monkeypatch.setattr(
        routes_module.LLMService,
        "from_settings",
        lambda settings, *, model=None, client=None: llm,
    )

    async def _mock_tool_service():
        return tool_service

    monkeypatch.setattr(routes_module, "get_tool_service", _mock_tool_service)

    client = TestClient(app)
    events: list[tuple[str, dict[str, Any]]] = []

    with client.stream(
        "POST",
        "/api/v1/submit_task/stream",
        json={"prompt": "Stream insights", "metadata": {"audience": "executive"}},
    ) as stream:
        current_event = ""
        for line in stream.iter_lines():
            if not line:
                continue
            if line.startswith("event: "):
                current_event = line.split("event: ", maxsplit=1)[1]
            elif line.startswith("data: ") and current_event:
                payload = json.loads(line.split("data: ", maxsplit=1)[1])
                events.append((current_event, payload))
                if current_event == "task_completed":
                    break

    agent_completed = [event for event in events if event[0] == "agent_completed"]
    assert len(agent_completed) == 4
    assert all("payload" in payload and isinstance(payload["payload"], dict) for _, payload in agent_completed)
    assert any(event[0] == "task_started" for event in events)
    assert any(event[0] == "task_completed" for event in events)
    assert len(llm.calls) == 4
    assert memory.ephemeral
    completed_envelope = next(payload for event, payload in events if event == "task_completed")
    final_payload = completed_envelope["payload"]
    assert "report" in final_payload
    assert final_payload["report"].get("headline")
    task_id = completed_envelope.get("task_id")
    assert isinstance(task_id, str) and task_id in memory.ephemeral
    stored_result = memory.ephemeral[task_id]["result"]
    assert "report" in stored_result
    assert stored_result["report"].get("headline")