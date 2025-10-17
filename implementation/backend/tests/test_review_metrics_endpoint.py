from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from fastapi.testclient import TestClient

from app import dependencies
from app.core.config import EscalationSettings
from app.core.security import create_access_token
from app.dependencies import get_review_manager
from app.main import app
from app.orchestration.review import InMemoryReviewStore, ReviewManager, ReviewStatus, ReviewTicket


def test_review_metrics_requires_auth() -> None:
    dependencies._review_manager_singleton = None
    dependencies._review_notification_service = None

    client = TestClient(app)
    response = client.get("/api/v1/reviews/metrics")

    assert response.status_code == 401


def test_review_metrics_endpoint_exposes_backlog_fields() -> None:
    dependencies._review_manager_singleton = None
    dependencies._review_notification_service = None

    store = InMemoryReviewStore()
    manager = ReviewManager(store=store, settings=EscalationSettings())

    now = datetime.now(timezone.utc)

    tickets = (
        ReviewTicket(
            ticket_id=uuid4(),
            task_id="task-open-unassigned",
            status=ReviewStatus.OPEN,
            summary=None,
            created_at=now - timedelta(hours=3),
            updated_at=now - timedelta(hours=3),
            assigned_to=None,
            sources=("guardrail",),
            escalation_payload={},
        ),
        ReviewTicket(
            ticket_id=uuid4(),
            task_id="task-open-assigned",
            status=ReviewStatus.OPEN,
            summary=None,
            created_at=now - timedelta(hours=1, minutes=30),
            updated_at=now - timedelta(hours=1, minutes=30),
            assigned_to="reviewer-a",
            sources=("agent",),
            escalation_payload={},
        ),
        ReviewTicket(
            ticket_id=uuid4(),
            task_id="task-in-review",
            status=ReviewStatus.IN_REVIEW,
            summary=None,
            created_at=now - timedelta(hours=2),
            updated_at=now - timedelta(hours=1, minutes=45),
            assigned_to="reviewer-b",
            sources=("manual",),
            escalation_payload={},
        ),
        ReviewTicket(
            ticket_id=uuid4(),
            task_id="task-resolved",
            status=ReviewStatus.RESOLVED,
            summary="cleared",
            created_at=now - timedelta(hours=4),
            updated_at=now - timedelta(hours=1),
            assigned_to="reviewer-b",
            sources=("manual",),
            escalation_payload={},
        ),
    )

    for ticket in tickets:
        store._tickets[ticket.ticket_id] = ticket  # type: ignore[attr-defined]
        store._task_index[ticket.task_id] = ticket.ticket_id  # type: ignore[attr-defined]

    async def override_review_manager():
        yield manager

    app.dependency_overrides[get_review_manager] = override_review_manager

    token = create_access_token("reviewer-test", extra_claims={"roles": ["reviewer"]})
    client = TestClient(app)
    client.headers.update({"Authorization": f"Bearer {token}"})

    try:
        response = client.get("/api/v1/reviews/metrics")
    finally:
        app.dependency_overrides.pop(get_review_manager, None)
        dependencies._review_manager_singleton = None
        dependencies._review_notification_service = None

    assert response.status_code == 200
    data = response.json()

    assert data["totals"]["open"] == 2
    assert data["totals"]["in_review"] == 1
    assert data["assignment"]["unassigned_open"] == 1
    assert data["assignment"]["by_reviewer"]["reviewer-b"] == 1
    assert data["aging"]["open_oldest_minutes"] >= 170
    assert data["resolution"]["completed_last_24h"] == 1