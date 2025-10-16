from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from ..orchestration.review import ReviewNote, ReviewStatus, ReviewTicket

ReviewStatusLiteral = Literal["open", "in_review", "resolved", "dismissed"]


class ReviewNoteModel(BaseModel):
    note_id: UUID
    author: str
    content: str
    created_at: datetime

    @classmethod
    def from_domain(cls, note: ReviewNote) -> "ReviewNoteModel":
        return cls(
            note_id=note.note_id,
            author=note.author,
            content=note.content,
            created_at=note.created_at,
        )


class ReviewTicketModel(BaseModel):
    ticket_id: UUID
    task_id: str
    status: ReviewStatusLiteral
    summary: str | None
    created_at: datetime
    updated_at: datetime
    assigned_to: str | None
    sources: list[str]
    escalation_payload: dict[str, Any] = Field(default_factory=dict)
    notes: list[ReviewNoteModel] = Field(default_factory=list)

    @classmethod
    def from_domain(cls, ticket: ReviewTicket) -> "ReviewTicketModel":
        return cls(
            ticket_id=ticket.ticket_id,
            task_id=ticket.task_id,
            status=ticket.status.value,
            summary=ticket.summary,
            created_at=ticket.created_at,
            updated_at=ticket.updated_at,
            assigned_to=ticket.assigned_to,
            sources=list(ticket.sources),
            escalation_payload=dict(ticket.escalation_payload),
            notes=[ReviewNoteModel.from_domain(note) for note in ticket.notes],
        )


class ReviewAssignmentRequest(BaseModel):
    reviewer_id: str | None = Field(default=None, min_length=1)


class ReviewNoteCreate(BaseModel):
    author: str | None = Field(default=None, min_length=1)
    content: str = Field(min_length=1)


class ReviewResolutionRequest(BaseModel):
    status: ReviewStatusLiteral
    reviewer_id: str | None = Field(default=None, min_length=1)
    summary: str | None = Field(default=None)

    def to_status(self) -> ReviewStatus:
        return ReviewStatus(self.status)


class ReviewAssignmentBreakdown(BaseModel):
    by_reviewer: dict[str, int] = Field(default_factory=dict)
    unassigned_open: int = 0


class ReviewAgingStats(BaseModel):
    open_average_minutes: float = 0.0
    open_oldest_minutes: float = 0.0
    in_review_average_minutes: float = 0.0


class ReviewResolutionStats(BaseModel):
    average_minutes: float | None = None
    median_minutes: float | None = None
    completed_last_24h: int = 0


class ReviewMetricsResponse(BaseModel):
    generated_at: datetime
    totals: dict[str, int] = Field(default_factory=dict)
    assignment: ReviewAssignmentBreakdown = Field(default_factory=ReviewAssignmentBreakdown)
    aging: ReviewAgingStats = Field(default_factory=ReviewAgingStats)
    resolution: ReviewResolutionStats = Field(default_factory=ReviewResolutionStats)