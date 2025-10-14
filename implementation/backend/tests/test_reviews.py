from __future__ import annotations

import asyncio
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app import dependencies
from app.core.config import EscalationSettings, get_settings
from app.core.security import create_access_token
from app.main import app
from app.orchestration.graph import Orchestrator
from app.orchestration.meta import MetaResolution
from app.orchestration.review import InMemoryReviewStore, ReviewManager, ReviewStatus, build_review_store
from app.services.notifications import ReviewEvent


class _StubMetaAgent:
    def __init__(self, resolution: MetaResolution) -> None:
        self._resolution = resolution

    async def synthesize(self, *, task, outputs, negotiation):  # noqa: D401 - simple stub
        return self._resolution


@pytest.mark.asyncio
async def test_orchestrator_creates_review_ticket_on_escalation() -> None:
    settings = EscalationSettings(enabled=True, require_auth=False, auto_assign_reviewer="reviewer-1", audit_log_enabled=False)
    manager = ReviewManager(store=InMemoryReviewStore(), settings=settings)
    resolution = MetaResolution(
        summary="Needs human oversight",
        confidence=0.42,
        mode="llm",
        evidence=[],
        dispute=None,
        validation=[],
        should_escalate=True,
    )
    orchestrator = Orchestrator(
        agents=[],
        meta_agent=_StubMetaAgent(resolution),
        review_manager=manager,
    )
    state: dict[str, object] = {
        "id": "task-escalate",
        "prompt": "Test",
        "outputs": [
            {
                "agent": "demo",
                "summary": "Result",
                "confidence": 0.8,
            }
        ],
    }

    await orchestrator._synthesize_meta_resolution(state, tracker=None)

    escalation = state.get("escalation")
    assert isinstance(escalation, dict)
    assert escalation.get("ticket_id") is not None

    ticket = await manager.get_ticket(UUID(str(escalation.get("ticket_id"))))
    assert ticket is not None
    assert ticket.status == ReviewStatus.OPEN
    assert ticket.assigned_to == "reviewer-1"


def test_build_review_store_uses_in_memory_for_test_env() -> None:
    settings = get_settings({"environment": "test"})
    store = build_review_store(settings)
    assert isinstance(store, InMemoryReviewStore)


def test_review_api_flow() -> None:
    dependencies._review_manager_singleton = None  # reset singleton for test isolation
    dependencies._review_notification_service = None
    settings = get_settings({"environment": "test"})
    manager = dependencies.get_review_manager_singleton(settings)
    events: list[ReviewEvent] = []

    notification_service = dependencies._review_notification_service
    assert notification_service is not None

    async def capture(event: ReviewEvent) -> None:
        events.append(event)

    notification_service.subscribe(capture)

    ticket = asyncio.run(
        manager.ensure_ticket(
            task_state={
                "id": "task-api",
                "prompt": "Investigate anomaly",
                "outputs": [],
                "meta": {},
                "negotiation": {},
                "guardrails": {},
            },
            resolution_summary="Manual confirmation required",
            sources=["meta_agent"],
        )
    )

    client = TestClient(app)
    token = create_access_token("human-1", extra_claims={"roles": ["reviewer"]})
    client.headers.update({"Authorization": f"Bearer {token}"})
    response = client.get("/api/v1/reviews")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list) and data
    ticket_id = data[0]["ticket_id"]

    assign_response = client.post(f"/api/v1/reviews/{ticket_id}/assign", json={})
    assert assign_response.status_code == 200
    assert assign_response.json()["assigned_to"] == "human-1"

    unassign_response = client.patch(f"/api/v1/reviews/{ticket_id}/unassign")
    assert unassign_response.status_code == 200
    assert unassign_response.json()["assigned_to"] is None
    assert unassign_response.json()["status"] == "open"

    note_response = client.post(
        f"/api/v1/reviews/{ticket_id}/notes",
        json={"content": "Triaged and awaiting evidence."},
    )
    assert note_response.status_code == 200
    assert len(note_response.json()["notes"]) == 1

    resolve_response = client.post(
        f"/api/v1/reviews/{ticket_id}/resolve",
        json={"status": "resolved", "summary": "Verified OK"},
    )
    assert resolve_response.status_code == 200
    assert resolve_response.json()["status"] == "resolved"

    detail_response = client.get(f"/api/v1/reviews/{ticket_id}")
    assert detail_response.status_code == 200
    assert detail_response.json()["summary"] == "Verified OK"

    event_types = [event.event for event in events]
    assert "review.ticket.created" in event_types
    assert "review.ticket.assigned" in event_types
    assert "review.ticket.unassigned" in event_types
    assert "review.ticket.note_added" in event_types
    assert "review.ticket.resolved" in event_types

    notification_service.unsubscribe(capture)