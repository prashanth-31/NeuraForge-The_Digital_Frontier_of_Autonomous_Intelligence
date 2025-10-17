from __future__ import annotations

import json
from collections.abc import Iterator


class _StubOrchestrator:
    async def route_task(self, state, *, context, progress_cb=None):  # noqa: ANN001, D401 - simple stub
        if progress_cb is not None:
            await progress_cb({"event": "agent_completed", "agent": "stub", "summary": "completed"})
        result = {
            **state,
            "status": "completed",
            "outputs": [
                {
                    "agent": "stub",
                    "summary": "Contract validation",
                    "confidence": 0.99,
                }
            ],
        }
        return result

from fastapi.testclient import TestClient

from app.api import routes as routes_module
from app.main import app
from tests.test_tasks import StubLLMService, StubMemoryService


def _iter_sse_events(stream: Iterator[str]):
    current_event = ""
    for line in stream:
        if not line:
            continue
        if line.startswith("event: "):
            current_event = line.split("event: ", maxsplit=1)[1]
            continue
        if line.startswith("data: ") and current_event:
            payload = json.loads(line.split("data: ", maxsplit=1)[1])
            yield current_event, payload


def test_submit_task_stream_contract(monkeypatch):
    """Ensure the streaming endpoint emits start, agent, and completion events."""
    memory = StubMemoryService()
    llm = StubLLMService()

    monkeypatch.setattr(routes_module.HybridMemoryService, "from_settings", lambda settings: memory)
    monkeypatch.setattr(routes_module.LLMService, "from_settings", lambda settings: llm)

    async def _build_stub_pipeline(**kwargs):  # noqa: ANN003 - signature mirrors original helper
        return _StubOrchestrator(), None, None

    monkeypatch.setattr(routes_module, "_build_orchestration_pipeline", _build_stub_pipeline)

    async def _noop_rate_limit():  # noqa: ANN202 - test helper
        return None

    def _state_store_disabled(cls, settings):  # noqa: ANN001, ARG001
        raise RuntimeError("disabled in tests")

    monkeypatch.setattr(routes_module, "rate_limit_task_submission", _noop_rate_limit)
    monkeypatch.setattr(routes_module.OrchestratorStateStore, "from_settings", classmethod(_state_store_disabled))

    client = TestClient(app)
    events: list[tuple[str, dict[str, object]]] = []

    with client.stream(
        "POST",
        "/api/v1/submit_task/stream",
        json={"prompt": "Contract test", "metadata": {"source": "pytest"}},
    ) as stream:
        for event_name, payload in _iter_sse_events(stream.iter_lines()):
            events.append((event_name, payload))
            if event_name == "task_completed":
                break

    assert events, "no SSE events captured"
    event_names = [name for name, _ in events]
    assert event_names[0] == "task_started"
    assert any(name == "agent_completed" for name in event_names)
    assert events[-1][0] == "task_completed"

    task_started_payload = events[0][1]
    task_id = task_started_payload.get("task_id")
    assert isinstance(task_id, str) and task_id
    assert task_id in memory.ephemeral
    assert memory.ephemeral[task_id]["result"]["status"] == "completed"
    completed_payload = events[-1][1]
    assert completed_payload.get("status") == "completed"
