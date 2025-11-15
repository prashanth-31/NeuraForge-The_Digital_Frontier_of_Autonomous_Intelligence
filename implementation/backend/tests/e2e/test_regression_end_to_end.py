from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any

import httpx
import pytest
import pytest_asyncio

from app.api import routes as routes_module
from app.dependencies import get_hybrid_memory, get_orchestrator_state_store, get_task_queue
from app.main import app

from tests.helpers.stubs import (
    ImmediateQueue,
    StubLLMService,
    StubMemoryService,
    StubOrchestratorStateStore,
    StubTaskLifecycleStore,
    StubContextSnapshotStore,
    StubGuardrailStore,
    StubEmbeddingService,
    StubToolService,
)


@pytest_asyncio.fixture()
async def e2e_client(monkeypatch: pytest.MonkeyPatch) -> AsyncGenerator[tuple[httpx.AsyncClient, StubMemoryService, StubLLMService, ImmediateQueue, StubToolService], None]:
    memory = StubMemoryService()
    llm = StubLLMService()
    queue = ImmediateQueue()
    tool_service = StubToolService()
    orchestrator_store = StubOrchestratorStateStore()
    settings = routes_module.get_settings()
    original_flags = {
        "environment": settings.environment,
        "planning_enabled": settings.planning.enabled,
        "tools_enabled": settings.tools.mcp.enabled,
        "rate_limit_enabled": settings.rate_limit.enabled,
        "snapshots_enabled": settings.snapshots.enabled,
        "guardrails_enabled": settings.guardrails.enabled,
        "meta_agent_enabled": settings.meta_agent.enabled,
    }

    settings.environment = "test"
    settings.planning.enabled = False
    settings.tools.mcp.enabled = True
    settings.rate_limit.enabled = False
    settings.snapshots.enabled = False
    settings.guardrails.enabled = False
    settings.meta_agent.enabled = False

    monkeypatch.setattr(routes_module.HybridMemoryService, "from_settings", lambda settings: memory)
    monkeypatch.setattr(
        routes_module.LLMService,
        "from_settings",
        lambda settings, *, model=None, client=None: llm,
    )
    monkeypatch.setattr(routes_module, "OrchestratorStateStore", StubOrchestratorStateStore)
    monkeypatch.setattr(routes_module, "TaskLifecycleStore", StubTaskLifecycleStore)
    monkeypatch.setattr(routes_module, "ContextSnapshotStore", StubContextSnapshotStore)
    monkeypatch.setattr(routes_module, "GuardrailStore", StubGuardrailStore)
    monkeypatch.setattr(routes_module, "EmbeddingService", StubEmbeddingService)

    async def _mock_tool_service() -> StubToolService:
        return tool_service

    monkeypatch.setattr(routes_module, "get_tool_service", _mock_tool_service)

    async def override_queue() -> AsyncGenerator[ImmediateQueue, None]:
        yield queue

    async def override_memory() -> AsyncGenerator[StubMemoryService, None]:
        yield memory

    async def override_state_store() -> AsyncGenerator[StubOrchestratorStateStore, None]:
        yield orchestrator_store

    app.dependency_overrides[get_task_queue] = override_queue
    app.dependency_overrides[get_hybrid_memory] = override_memory
    app.dependency_overrides[get_orchestrator_state_store] = override_state_store

    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client, memory, llm, queue, tool_service
    finally:
        app.dependency_overrides.clear()
        await transport.aclose()
        settings.environment = original_flags["environment"]
        settings.planning.enabled = original_flags["planning_enabled"]
        settings.tools.mcp.enabled = original_flags["tools_enabled"]
        settings.rate_limit.enabled = original_flags["rate_limit_enabled"]
        settings.snapshots.enabled = original_flags["snapshots_enabled"]
        settings.guardrails.enabled = original_flags["guardrails_enabled"]
        settings.meta_agent.enabled = original_flags["meta_agent_enabled"]


@pytest.mark.asyncio
async def test_submit_status_and_history_flow(e2e_client) -> None:
    client, memory, llm, queue, tool_service = e2e_client

    response = await client.post(
        "/api/v1/submit_task",
        json={"prompt": "Compile launch rundown", "metadata": {"priority": "p1"}},
    )

    assert response.status_code == 200
    payload = response.json()
    task_id = payload["task_id"]
    assert queue.enqueued == 1
    assert task_id in memory.ephemeral

    status = await client.get(f"/api/v1/tasks/{task_id}")
    assert status.status_code == 200
    status_body = status.json()
    assert status_body["status"] == "completed"
    assert status_body["plan"]["status"] in {"planned", "scheduled"}
    assert len(status_body["outputs"]) == 4
    assert status_body["metrics"]["agents_completed"] >= 4
    assert status_body["report"]["headline"]

    history = await client.get(f"/api/v1/history/{task_id}")
    assert history.status_code == 200
    history_entries = history.json()
    assert len(history_entries) == 4
    assert {entry["agent"] for entry in history_entries} == {
        "general_agent",
        "research_agent",
        "finance_agent",
        "creative_agent",
    }

    assert any(call[0] == "finance.snapshot" for call in tool_service.calls)
    assert len(llm.calls) == 4


@pytest.mark.asyncio
async def test_stream_endpoint_emits_ordered_events(e2e_client) -> None:
    client, memory, llm, queue, tool_service = e2e_client

    async with client.stream(
        "POST",
        "/api/v1/submit_task/stream",
        json={"prompt": "Stream regression validation", "metadata": {"trace": "sse"}},
    ) as response:
        assert response.status_code == 200
        events: list[tuple[str, dict[str, Any]]] = []
        current_event = ""
        async for line in response.aiter_lines():
            if not line:
                continue
            if line.startswith("event: "):
                current_event = line.split("event: ", maxsplit=1)[1]
            elif line.startswith("data: ") and current_event:
                data = json.loads(line.split("data: ", maxsplit=1)[1])
                events.append((current_event, data))
                if current_event == "task_completed":
                    break

    assert any(name == "task_started" for name, _ in events)
    completed = [data for name, data in events if name == "task_completed"][0]
    assert completed["payload"]["status"] == "completed"
    assert completed["payload"]["report"]["headline"]
    assert completed.get("task_id") in memory.ephemeral
    assert len([name for name, _ in events if name == "agent_completed"]) == 4
    assert len(llm.calls) == 4
    assert queue.enqueued == 0
    assert tool_service.calls


@pytest.mark.asyncio
async def test_multiple_tasks_do_not_clobber_state(e2e_client) -> None:
    client, memory, llm, queue, tool_service = e2e_client

    prompts = [
        "Draft investor update",
        "Create engineering summary",
        "Summarize customer feedback",
    ]

    responses = await asyncio.gather(
        *[
            client.post(
                "/api/v1/submit_task",
                json={"prompt": prompt, "metadata": {"batch": "phase6"}},
            )
            for prompt in prompts
        ]
    )

    task_ids = [response.json()["task_id"] for response in responses]
    assert queue.enqueued == len(prompts)
    assert len({memory.ephemeral[task_id]["result"]["report"]["headline"] for task_id in task_ids}) == len(prompts)
    assert sum(1 for call in tool_service.calls if call[0].startswith("research")) >= len(prompts)
    assert len(llm.calls) == 4 * len(prompts)