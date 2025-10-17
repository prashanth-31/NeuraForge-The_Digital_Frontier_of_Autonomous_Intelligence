from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from collections import defaultdict
from statistics import median
from typing import Any, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

try:  # pragma: no cover - optional dependency
    import asyncpg
except ModuleNotFoundError:  # pragma: no cover - asyncpg optional
    asyncpg = None  # type: ignore[assignment]

from ..core.config import EscalationSettings, Settings
from ..core.logging import get_logger
from ..core.metrics import record_review_oldest_ticket_age, record_review_ticket_counts
from ..services.notifications import ReviewNotificationService, ReviewEvent

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..services.notifications import ReviewEventType

logger = get_logger(name=__name__)


class ReviewStatus(str, Enum):
    OPEN = "open"
    IN_REVIEW = "in_review"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


@dataclass(slots=True)
class ReviewNote:
    note_id: UUID
    author: str
    content: str
    created_at: datetime


@dataclass(slots=True)
class ReviewTicket:
    ticket_id: UUID
    task_id: str
    status: ReviewStatus
    summary: str | None
    created_at: datetime
    updated_at: datetime
    assigned_to: str | None
    sources: tuple[str, ...]
    escalation_payload: dict[str, Any]
    notes: list[ReviewNote] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticket_id": str(self.ticket_id),
            "task_id": self.task_id,
            "status": self.status.value,
            "summary": self.summary,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "assigned_to": self.assigned_to,
            "sources": list(self.sources),
            "escalation_payload": self.escalation_payload,
            "notes": [
                {
                    "note_id": str(note.note_id),
                    "author": note.author,
                    "content": note.content,
                    "created_at": note.created_at.isoformat(),
                }
                for note in self.notes
            ],
        }


class ReviewStore:
    async def ensure_ticket(
        self,
        *,
        task_id: str,
        sources: Sequence[str],
        summary: str | None,
        payload: Mapping[str, Any],
        assigned_to: str | None,
    ) -> ReviewTicket:
        existing = await self.get_by_task(task_id)
        if existing is not None and existing.status in {ReviewStatus.OPEN, ReviewStatus.IN_REVIEW}:
            await self._refresh_metrics()
            return existing
        now = datetime.now(timezone.utc)
        ticket = ReviewTicket(
            ticket_id=uuid4(),
            task_id=task_id,
            status=ReviewStatus.OPEN,
            summary=summary,
            created_at=now,
            updated_at=now,
            assigned_to=assigned_to,
            sources=tuple(sources),
            escalation_payload=dict(payload),
        )
        return await self._create_ticket(ticket)

    async def add_note(self, ticket_id: UUID, *, author: str, content: str) -> ReviewNote:
        note = ReviewNote(note_id=uuid4(), author=author, content=content.strip(), created_at=datetime.now(timezone.utc))
        await self._append_note(ticket_id, note)
        return note

    async def assign(self, ticket_id: UUID, *, reviewer: str) -> ReviewTicket:
        ticket = await self.get(ticket_id)
        if ticket is None:
            raise KeyError("ticket_not_found")
        ticket.assigned_to = reviewer
        ticket.status = ReviewStatus.IN_REVIEW
        ticket.updated_at = datetime.now(timezone.utc)
        await self._replace_ticket(ticket)
        return ticket

    async def unassign(self, ticket_id: UUID) -> ReviewTicket:
        ticket = await self.get(ticket_id)
        if ticket is None:
            raise KeyError("ticket_not_found")
        ticket.assigned_to = None
        ticket.status = ReviewStatus.OPEN
        ticket.updated_at = datetime.now(timezone.utc)
        await self._replace_ticket(ticket)
        return ticket

    async def update_status(
        self,
        ticket_id: UUID,
        *,
        status: ReviewStatus,
        reviewer: str | None,
        summary: str | None = None,
    ) -> ReviewTicket:
        ticket = await self.get(ticket_id)
        if ticket is None:
            raise KeyError("ticket_not_found")
        ticket.status = status
        ticket.updated_at = datetime.now(timezone.utc)
        if reviewer:
            ticket.assigned_to = reviewer
        if summary:
            ticket.summary = summary
        await self._replace_ticket(ticket)
        return ticket

    async def list(self, *, status: ReviewStatus | None = None) -> list[ReviewTicket]:
        tickets = await self._list_all()
        if status is None:
            return tickets
        return [ticket for ticket in tickets if ticket.status == status]

    async def get(self, ticket_id: UUID) -> ReviewTicket | None:
        return await self._get_ticket(ticket_id)

    async def get_by_task(self, task_id: str) -> ReviewTicket | None:
        return await self._get_by_task(task_id)

    # Abstract hooks -----------------------------------------------------------------

    async def _create_ticket(self, ticket: ReviewTicket) -> ReviewTicket:
        raise NotImplementedError

    async def _append_note(self, ticket_id: UUID, note: ReviewNote) -> None:
        raise NotImplementedError

    async def _replace_ticket(self, ticket: ReviewTicket) -> None:
        raise NotImplementedError

    async def _list_all(self) -> list[ReviewTicket]:
        raise NotImplementedError

    async def _get_ticket(self, ticket_id: UUID) -> ReviewTicket | None:
        raise NotImplementedError

    async def _get_by_task(self, task_id: str) -> ReviewTicket | None:
        raise NotImplementedError


class InMemoryReviewStore(ReviewStore):  # pragma: no cover - exercised via higher level tests
    def __init__(self) -> None:
        self._tickets: dict[UUID, ReviewTicket] = {}
        self._task_index: dict[str, UUID] = {}
        self._lock = asyncio.Lock()

    async def _create_ticket(self, ticket: ReviewTicket) -> ReviewTicket:
        async with self._lock:
            self._tickets[ticket.ticket_id] = ticket
            self._task_index[ticket.task_id] = ticket.ticket_id
        return self._clone(ticket)

    async def _append_note(self, ticket_id: UUID, note: ReviewNote) -> None:
        async with self._lock:
            ticket = self._tickets.get(ticket_id)
            if ticket is None:
                raise KeyError("ticket_not_found")
            ticket.notes.append(note)
            ticket.updated_at = note.created_at

    async def _replace_ticket(self, ticket: ReviewTicket) -> None:
        async with self._lock:
            if ticket.ticket_id not in self._tickets:
                raise KeyError("ticket_not_found")
            self._tickets[ticket.ticket_id] = ticket
            self._task_index[ticket.task_id] = ticket.ticket_id
            if ticket.assigned_to is None:
                # Ensure open tickets without assignee remain indexed by task id
                self._task_index[ticket.task_id] = ticket.ticket_id

    async def _list_all(self) -> list[ReviewTicket]:
        async with self._lock:
            return [self._clone(ticket) for ticket in self._tickets.values()]

    async def _get_ticket(self, ticket_id: UUID) -> ReviewTicket | None:
        async with self._lock:
            ticket = self._tickets.get(ticket_id)
            return None if ticket is None else self._clone(ticket)

    async def _get_by_task(self, task_id: str) -> ReviewTicket | None:
        async with self._lock:
            ticket_id = self._task_index.get(task_id)
            if ticket_id is None:
                return None
            ticket = self._tickets.get(ticket_id)
            return None if ticket is None else self._clone(ticket)

    @staticmethod
    def _clone(ticket: ReviewTicket) -> ReviewTicket:
        return ReviewTicket(
            ticket_id=ticket.ticket_id,
            task_id=ticket.task_id,
            status=ticket.status,
            summary=ticket.summary,
            created_at=ticket.created_at,
            updated_at=ticket.updated_at,
            assigned_to=ticket.assigned_to,
            sources=tuple(ticket.sources),
            escalation_payload=dict(ticket.escalation_payload),
            notes=list(ticket.notes),
        )


class ReviewManager:
    def __init__(
        self,
        *,
        store: ReviewStore,
        settings: EscalationSettings,
        notifications: ReviewNotificationService | None = None,
    ) -> None:
        self._store = store
        self._settings = settings
        self._notifications = notifications

    async def ensure_ticket(
        self,
        *,
        task_state: Mapping[str, Any],
        resolution_summary: str | None,
        sources: Iterable[str],
    ) -> ReviewTicket:
        if not self._settings.enabled:
            raise RuntimeError("escalations_disabled")
        assigned = self._settings.auto_assign_reviewer
        task_id = str(task_state.get("id") or task_state.get("task_id") or "")
        existing = await self._store.get_by_task(task_id)
        ticket = await self._store.ensure_ticket(
            task_id=task_id,
            sources=list(dict.fromkeys([src for src in sources if src])),
            summary=resolution_summary,
            payload={
                "meta": task_state.get("meta", {}),
                "negotiation": task_state.get("negotiation", {}),
                "guardrails": task_state.get("guardrails", {}),
                "prompt": task_state.get("prompt"),
                "outputs": task_state.get("outputs", []),
            },
            assigned_to=assigned,
        )
        await self._refresh_metrics()
        logger.info(
            "review_ticket_issued",
            ticket_id=str(ticket.ticket_id),
            task_id=ticket.task_id,
            assigned_to=ticket.assigned_to,
        )
        if existing is None or existing.ticket_id != ticket.ticket_id:
            await self._publish_event("review.ticket.created", ticket, actor=assigned)
        return ticket

    async def list_tickets(self, status: ReviewStatus | None = None) -> list[ReviewTicket]:
        return await self._store.list(status=status)

    async def get_ticket(self, ticket_id: UUID) -> ReviewTicket | None:
        return await self._store.get(ticket_id)

    async def get_metrics(self) -> dict[str, Any]:
        tickets = await self._store.list()
        now = datetime.now(timezone.utc)
        totals: dict[str, int] = defaultdict(int)
        assignment_counts: defaultdict[str, int] = defaultdict(int)
        open_unassigned = 0
        open_ages: list[float] = []
        in_review_ages: list[float] = []
        resolution_durations: list[float] = []
        completed_last_24h = 0
        resolved_last_7d = 0
        dismissed_last_7d = 0
        resolution_durations_7d: list[float] = []
        escalations_pending = 0
        sla_breaches = 0
        seven_days_ago = now - timedelta(days=7)

        for ticket in tickets:
            totals[ticket.status.value] += 1
            age_minutes = (now - ticket.created_at).total_seconds() / 60.0
            if ticket.status is ReviewStatus.OPEN:
                open_ages.append(age_minutes)
                if ticket.assigned_to:
                    assignment_counts[ticket.assigned_to] += 1
                else:
                    open_unassigned += 1
            elif ticket.status is ReviewStatus.IN_REVIEW:
                in_review_ages.append(age_minutes)
                if ticket.assigned_to:
                    assignment_counts[ticket.assigned_to] += 1
            elif ticket.status in {ReviewStatus.RESOLVED, ReviewStatus.DISMISSED}:
                duration_minutes = (ticket.updated_at - ticket.created_at).total_seconds() / 60.0
                resolution_durations.append(duration_minutes)
                if now - ticket.updated_at <= timedelta(hours=24):
                    completed_last_24h += 1
                if ticket.updated_at >= seven_days_ago:
                    resolution_durations_7d.append(duration_minutes)
                    if ticket.status is ReviewStatus.RESOLVED:
                        resolved_last_7d += 1
                    else:
                        dismissed_last_7d += 1

            payload = ticket.escalation_payload or {}
            if isinstance(payload, dict) and payload:
                sla_info = payload.get("sla")
                if isinstance(sla_info, dict):
                    breach_flag = sla_info.get("breach") or sla_info.get("breached")
                    if isinstance(breach_flag, bool) and breach_flag:
                        sla_breaches += 1
                if ticket.status in {ReviewStatus.OPEN, ReviewStatus.IN_REVIEW}:
                    escalations_pending += 1

        def _average(values: list[float]) -> float:
            return round(sum(values) / len(values), 2) if values else 0.0

        resolution_average = _average(resolution_durations) if resolution_durations else None
        resolution_median = round(median(resolution_durations), 2) if resolution_durations else None
        resolution_median_7d = round(median(resolution_durations_7d), 2) if resolution_durations_7d else None

        for status in ReviewStatus:
            totals.setdefault(status.value, 0)

        active_reviewers = len(assignment_counts)
        backlog_denominator = max(1, active_reviewers or 0)
        backlog_pressure = (
            (totals[ReviewStatus.OPEN.value] + totals[ReviewStatus.IN_REVIEW.value]) / backlog_denominator
            if backlog_denominator
            else float(totals[ReviewStatus.OPEN.value] + totals[ReviewStatus.IN_REVIEW.value])
        )

        return {
            "generated_at": now,
            "totals": dict(totals),
            "assignment": {
                "by_reviewer": dict(assignment_counts),
                "unassigned_open": open_unassigned,
            },
            "aging": {
                "open_average_minutes": _average(open_ages),
                "open_oldest_minutes": round(max(open_ages), 2) if open_ages else 0.0,
                "in_review_average_minutes": _average(in_review_ages),
            },
            "resolution": {
                "average_minutes": resolution_average,
                "median_minutes": resolution_median,
                "completed_last_24h": completed_last_24h,
            },
            "queue_health": {
                "backlog_pressure": round(backlog_pressure, 2),
                "sla_breaches": sla_breaches,
                "escalations_pending": escalations_pending,
            },
            "trends": {
                "resolved_last_7d": resolved_last_7d,
                "dismissed_last_7d": dismissed_last_7d,
                "median_resolution_minutes_7d": resolution_median_7d,
            },
        }

    async def add_note(self, ticket_id: UUID, *, author: str, content: str) -> ReviewNote:
        if not content.strip():
            raise ValueError("empty_note")
        note = await self._store.add_note(ticket_id, author=author, content=content)
        ticket = await self._store.get(ticket_id)
        if ticket is not None:
            await self._publish_event("review.ticket.note_added", ticket, note=note, actor=author)
        return note

    async def assign(self, ticket_id: UUID, reviewer: str) -> ReviewTicket:
        before = await self._store.get(ticket_id)
        ticket = await self._store.assign(ticket_id, reviewer=reviewer)
        await self._refresh_metrics()
        event = "review.ticket.assigned"
        if before is not None and before.assigned_to and before.assigned_to != reviewer:
            event = "review.ticket.reassigned"
        await self._publish_event(event, ticket, actor=reviewer)
        return ticket

    async def resolve(
        self,
        ticket_id: UUID,
        *,
        status: ReviewStatus,
        reviewer: str,
        summary: str | None,
    ) -> ReviewTicket:
        if status not in {ReviewStatus.RESOLVED, ReviewStatus.DISMISSED}:
            raise ValueError("invalid_resolution_status")
        ticket = await self._store.update_status(ticket_id, status=status, reviewer=reviewer, summary=summary)
        await self._refresh_metrics()
        event = "review.ticket.resolved" if status == ReviewStatus.RESOLVED else "review.ticket.dismissed"
        await self._publish_event(event, ticket, actor=reviewer)
        return ticket

    async def unassign(self, ticket_id: UUID) -> ReviewTicket:
        ticket = await self._store.unassign(ticket_id)
        await self._refresh_metrics()
        await self._publish_event("review.ticket.unassigned", ticket)
        return ticket

    async def _refresh_metrics(self) -> None:
        tickets = await self._store.list()
        now = datetime.now(timezone.utc)

        open_tickets = [ticket for ticket in tickets if ticket.status is ReviewStatus.OPEN]
        in_review_tickets = [ticket for ticket in tickets if ticket.status is ReviewStatus.IN_REVIEW]
        resolved_tickets = [ticket for ticket in tickets if ticket.status is ReviewStatus.RESOLVED]
        dismissed_tickets = [ticket for ticket in tickets if ticket.status is ReviewStatus.DISMISSED]

        open_unassigned = sum(1 for ticket in open_tickets if not ticket.assigned_to)
        oldest_open_seconds = (
            max((now - ticket.created_at).total_seconds() for ticket in open_tickets)
            if open_tickets
            else 0.0
        )

        record_review_ticket_counts(
            open_count=len(open_tickets),
            in_review=len(in_review_tickets),
            resolved=len(resolved_tickets),
            dismissed=len(dismissed_tickets),
            unassigned_open=open_unassigned,
        )
        record_review_oldest_ticket_age(seconds=oldest_open_seconds)

    async def _publish_event(
        self,
        event: "ReviewEventType",
        ticket: ReviewTicket,
        *,
        note: ReviewNote | None = None,
        actor: str | None = None,
    ) -> None:
        if self._notifications is None:
            return
        try:
            await self._notifications.publish(ReviewEvent(event=event, ticket=ticket, note=note, actor=actor))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("review_notification_failed", error=str(exc), event=event)


class PostgresReviewStore(ReviewStore):
    _INSERT_TICKET = """
        INSERT INTO review_tickets(
            ticket_id,
            task_id,
            status,
            summary,
            assigned_to,
            sources,
            escalation_payload,
            created_at,
            updated_at
        ) VALUES($1, $2, $3, $4, $5, $6::text[], $7::jsonb, $8, $9)
        RETURNING ticket_id,
                  task_id,
                  status,
                  summary,
                  created_at,
                  updated_at,
                  assigned_to,
                  sources,
                  escalation_payload
    """

    _UPDATE_METADATA = """
        UPDATE review_tickets
        SET assigned_to = $2,
            status = $3,
            summary = $4,
            escalation_payload = $5::jsonb,
            sources = $6::text[],
            updated_at = $7
        WHERE ticket_id = $1
    """

    _TOUCH_TICKET = """
        UPDATE review_tickets
        SET updated_at = $2
        WHERE ticket_id = $1
    """

    _INSERT_NOTE = """
        INSERT INTO review_notes(
            note_id,
            ticket_id,
            author,
            content,
            created_at
        ) VALUES($1, $2, $3, $4, $5)
    """

    _LIST_TICKETS = """
        SELECT
            ticket_id,
            task_id,
            status,
            summary,
            created_at,
            updated_at,
            assigned_to,
            sources,
            escalation_payload
        FROM review_tickets
        ORDER BY created_at DESC
    """

    _FETCH_TICKET = """
        SELECT
            ticket_id,
            task_id,
            status,
            summary,
            created_at,
            updated_at,
            assigned_to,
            sources,
            escalation_payload
        FROM review_tickets
        WHERE ticket_id = $1
    """

    _FETCH_TASK = """
        SELECT
            ticket_id,
            task_id,
            status,
            summary,
            created_at,
            updated_at,
            assigned_to,
            sources,
            escalation_payload
        FROM review_tickets
        WHERE task_id = $1
        ORDER BY created_at DESC
        LIMIT 1
    """

    _FETCH_NOTES_FOR_TICKETS = """
        SELECT
            ticket_id,
            note_id,
            author,
            content,
            created_at
        FROM review_notes
        WHERE ticket_id = ANY($1::uuid[])
        ORDER BY created_at ASC
    """

    _FETCH_NOTES_SINGLE = """
        SELECT
            ticket_id,
            note_id,
            author,
            content,
            created_at
        FROM review_notes
        WHERE ticket_id = $1
        ORDER BY created_at ASC
    """

    def __init__(self, pool: Any) -> None:
        if asyncpg is None:  # pragma: no cover - guarded by build_review_store
            raise RuntimeError("asyncpg is required for PostgresReviewStore")
        self._pool_or_factory = pool
        self._pool: asyncpg.Pool | None = None  # type: ignore[name-defined]

    @classmethod
    def from_settings(cls, settings: Settings) -> "PostgresReviewStore":
        if asyncpg is None:
            raise RuntimeError("asyncpg is not available")
        pool = asyncpg.create_pool(
            dsn=str(settings.postgres.dsn),
            min_size=settings.postgres.pool_min_size,
            max_size=settings.postgres.pool_max_size,
        )
        return cls(pool)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def _ensure_pool(self) -> asyncpg.Pool:  # type: ignore[name-defined]
        if self._pool is not None:
            return self._pool
        candidate = self._pool_or_factory
        if isinstance(candidate, asyncpg.Pool):
            self._pool = candidate
            return self._pool
        if hasattr(candidate, "__await__"):
            candidate = await candidate
        if isinstance(candidate, asyncpg.Pool):
            self._pool = candidate
            return self._pool
        raise RuntimeError("Invalid asyncpg pool for PostgresReviewStore")

    async def _create_ticket(self, ticket: ReviewTicket) -> ReviewTicket:
        pool = await self._ensure_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                row = await connection.fetchrow(
                    self._INSERT_TICKET,
                    ticket.ticket_id,
                    ticket.task_id,
                    ticket.status.value,
                    ticket.summary,
                    ticket.assigned_to,
                    list(ticket.sources),
                    ticket.escalation_payload,
                    ticket.created_at,
                    ticket.updated_at,
                )
                notes = await connection.fetch(self._FETCH_NOTES_SINGLE, ticket.ticket_id)
        return self._row_to_ticket(row, notes)

    async def _append_note(self, ticket_id: UUID, note: ReviewNote) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    self._INSERT_NOTE,
                    note.note_id,
                    ticket_id,
                    note.author,
                    note.content,
                    note.created_at,
                )
                await connection.execute(self._TOUCH_TICKET, ticket_id, note.created_at)

    async def _replace_ticket(self, ticket: ReviewTicket) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as connection:
            await connection.execute(
                self._UPDATE_METADATA,
                ticket.ticket_id,
                ticket.assigned_to,
                ticket.status.value,
                ticket.summary,
                ticket.escalation_payload,
                list(ticket.sources),
                ticket.updated_at,
            )

    async def _list_all(self) -> list[ReviewTicket]:
        pool = await self._ensure_pool()
        async with pool.acquire() as connection:
            rows = await connection.fetch(self._LIST_TICKETS)
            if not rows:
                return []
            ticket_ids = [row["ticket_id"] for row in rows]
            notes = await connection.fetch(self._FETCH_NOTES_FOR_TICKETS, ticket_ids)
        notes_map = self._group_notes(notes)
        return [self._row_to_ticket(row, notes_map.get(row["ticket_id"], [])) for row in rows]

    async def _get_ticket(self, ticket_id: UUID) -> ReviewTicket | None:
        pool = await self._ensure_pool()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(self._FETCH_TICKET, ticket_id)
            if row is None:
                return None
            notes = await connection.fetch(self._FETCH_NOTES_SINGLE, ticket_id)
        return self._row_to_ticket(row, notes)

    async def _get_by_task(self, task_id: str) -> ReviewTicket | None:
        pool = await self._ensure_pool()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(self._FETCH_TASK, task_id)
            if row is None:
                return None
            notes = await connection.fetch(self._FETCH_NOTES_SINGLE, row["ticket_id"])
        return self._row_to_ticket(row, notes)

    def _group_notes(self, rows: Sequence[Any]) -> dict[UUID, list[Any]]:
        grouped: dict[UUID, list[Any]] = defaultdict(list)
        for record in rows:
            grouped[record["ticket_id"]].append(record)
        return grouped

    def _row_to_ticket(self, row: Any, notes: Sequence[Any]) -> ReviewTicket:
        ticket = ReviewTicket(
            ticket_id=row["ticket_id"],
            task_id=row["task_id"],
            status=ReviewStatus(row["status"]),
            summary=row["summary"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            assigned_to=row["assigned_to"],
            sources=tuple(row["sources"] or ()),
            escalation_payload=dict(row["escalation_payload"] or {}),
            notes=[
                ReviewNote(
                    note_id=note["note_id"],
                    author=note["author"],
                    content=note["content"],
                    created_at=note["created_at"],
                )
                for note in notes
            ],
        )
        return ticket


__all__ = [
    "InMemoryReviewStore",
    "PostgresReviewStore",
    "ReviewManager",
    "ReviewNote",
    "ReviewStatus",
    "ReviewTicket",
    "build_review_store",
]


def build_review_store(settings: Settings) -> ReviewStore:
    if not settings.escalation.enabled:
        return InMemoryReviewStore()
    if settings.environment == "test":
        logger.info("review_store_in_memory", reason="test_environment")
        return InMemoryReviewStore()
    if asyncpg is None:
        logger.warning("asyncpg_not_available_review_store")
        return InMemoryReviewStore()
    try:
        store = PostgresReviewStore.from_settings(settings)
    except Exception as exc:  # pragma: no cover - connection issues
        logger.warning("review_store_pool_failed", error=str(exc))
        return InMemoryReviewStore()
    logger.info("review_store_postgres_enabled", environment=settings.environment)
    return store
