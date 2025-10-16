from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.dependencies import get_hybrid_memory, get_orchestrator_state_store
from app.orchestration.state import OrchestratorEvent, OrchestratorRun, OrchestratorStatus


class _FakeMemory:
    def __init__(self, snapshot: dict[str, object]) -> None:
        self._snapshot = snapshot

    async def fetch_ephemeral_memory(self, task_id: str) -> dict[str, object] | None:
        return self._snapshot


class _FakeStore:
    def __init__(self, run: OrchestratorRun, events: list[OrchestratorEvent]) -> None:
        self._run = run
        self._events = events

    async def get_run(self, run_id: uuid.UUID) -> OrchestratorRun | None:
        if run_id == self._run.run_id:
            return self._run
        return None

    async def list_events(self, run_id: uuid.UUID) -> list[OrchestratorEvent]:
        if run_id == self._run.run_id:
            return list(self._events)
        return []


@pytest.fixture
def task_status_client() -> Generator[TestClient, None, None]:
    run_id = uuid.uuid4()
    now = datetime.now(timezone.utc)
    run = OrchestratorRun(
        run_id=run_id,
        task_id="task-123",
        status=OrchestratorStatus.COMPLETED,
        state={"id": "task-123"},
        created_at=now,
        updated_at=now,
    )
    events = [
        OrchestratorEvent(
            run_id=run_id,
            sequence=1,
            event_type="agent_completed",
            agent="ResearchAgent",
            payload={"latency": 1.2},
            created_at=now,
        )
    ]
    memory_payload = {
        "result": {
            "id": "task-123",
            "status": "completed",
            "prompt": "Synthesize findings",
            "metadata": {"source": "test"},
            "outputs": [{"agent": "ResearchAgent", "summary": "done"}],
            "run_id": str(run_id),
            "guardrails": {"decisions": [{"policy_id": "test", "decision": "allow"}]},
        }
    }
    memory = _FakeMemory(memory_payload)
    store = _FakeStore(run, events)

    async def _override_memory():
        yield memory

    async def _override_store():
        yield store

    app.dependency_overrides[get_hybrid_memory] = _override_memory
    app.dependency_overrides[get_orchestrator_state_store] = _override_store

    client = TestClient(app)
    try:
        yield client
    finally:
        app.dependency_overrides.pop(get_hybrid_memory, None)
        app.dependency_overrides.pop(get_orchestrator_state_store, None)
        client.close()


def test_task_status_endpoint(task_status_client: TestClient) -> None:
    response = task_status_client.get("/api/v1/tasks/task-123")
    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == "task-123"
    assert data["status"] == "completed"
    assert data["run_id"]
    assert data["metrics"]["agents_completed"] == 1
    assert data["metrics"]["guardrail_events"] == 1
    assert data["events"]
    assert data["events"][0]["event_type"] == "agent_completed"
