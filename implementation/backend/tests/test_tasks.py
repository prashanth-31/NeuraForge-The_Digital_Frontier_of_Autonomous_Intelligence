from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api import routes as routes_module
from app.dependencies import get_hybrid_memory, get_task_queue
from app.main import app

from tests.helpers.stubs import ImmediateQueue, StubLLMService, StubMemoryService, StubToolService


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