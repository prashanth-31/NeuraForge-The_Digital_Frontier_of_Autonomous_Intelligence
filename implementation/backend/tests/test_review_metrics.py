from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from app.core.config import EscalationSettings
from app.orchestration.review import (
    InMemoryReviewStore,
    ReviewManager,
    ReviewStatus,
    ReviewTicket,
)


@pytest.mark.asyncio
async def test_review_metrics_snapshot() -> None:
    store = InMemoryReviewStore()
    manager = ReviewManager(store=store, settings=EscalationSettings())
    now = datetime.now(timezone.utc)

    open_ticket = ReviewTicket(
        ticket_id=uuid4(),
        task_id="task-open",
        status=ReviewStatus.OPEN,
        summary=None,
        created_at=now - timedelta(hours=3),
        updated_at=now - timedelta(hours=3),
        assigned_to=None,
        sources=("guardrail",),
        escalation_payload={},
    )
    in_review_ticket = ReviewTicket(
        ticket_id=uuid4(),
        task_id="task-review",
        status=ReviewStatus.IN_REVIEW,
        summary=None,
        created_at=now - timedelta(hours=2),
        updated_at=now - timedelta(hours=1, minutes=30),
        assigned_to="alice",
        sources=("manual",),
        escalation_payload={},
    )
    resolved_ticket = ReviewTicket(
        ticket_id=uuid4(),
        task_id="task-resolved",
        status=ReviewStatus.RESOLVED,
        summary="done",
        created_at=now - timedelta(hours=2),
        updated_at=now - timedelta(hours=1),
        assigned_to="bob",
        sources=("meta",),
        escalation_payload={},
    )
    dismissed_ticket = ReviewTicket(
        ticket_id=uuid4(),
        task_id="task-dismissed",
        status=ReviewStatus.DISMISSED,
        summary="dismissed",
        created_at=now - timedelta(hours=5),
        updated_at=now - timedelta(hours=4, minutes=30),
        assigned_to="charlie",
        sources=("meta",),
        escalation_payload={},
    )

    for ticket in (open_ticket, in_review_ticket, resolved_ticket, dismissed_ticket):
        store._tickets[ticket.ticket_id] = ticket  # type: ignore[attr-defined]
        store._task_index[ticket.task_id] = ticket.ticket_id  # type: ignore[attr-defined]

    metrics = await manager.get_metrics()

    assert metrics["totals"]["open"] == 1
    assert metrics["totals"]["in_review"] == 1
    assert metrics["assignment"]["by_reviewer"]["alice"] == 1
    assert metrics["assignment"]["unassigned_open"] == 1
    assert metrics["aging"]["open_average_minutes"] > 0
    assert metrics["resolution"]["completed_last_24h"] == 2
    assert metrics["resolution"]["average_minutes"] == pytest.approx(45.0, rel=1e-3)
    assert metrics["resolution"]["median_minutes"] == pytest.approx(45.0, rel=1e-3)
