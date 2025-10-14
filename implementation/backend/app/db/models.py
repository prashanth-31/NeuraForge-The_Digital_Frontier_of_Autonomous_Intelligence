from __future__ import annotations

import uuid

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID

from ..orchestration.enums import GuardrailDecisionType, LifecycleStatus

metadata = MetaData()

episodic_memory = Table(
    "episodic_memory",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("task_id", String(length=64), nullable=False, unique=True),
    Column("agent", String(length=64), nullable=True),
    Column("payload", JSONB(astext_type=Text()), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column(
        "updated_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    ),
)
Index("ix_episodic_memory_task_id", episodic_memory.c.task_id)

negotiation_transcripts = Table(
    "negotiation_transcripts",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("task_id", String(length=64), nullable=False),
    Column("agent", String(length=64), nullable=False),
    Column("message", Text(), nullable=False),
    Column("confidence", Float(), nullable=True),
    Column("metadata", JSONB(astext_type=Text()), nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)
Index("ix_negotiation_transcripts_task_id", negotiation_transcripts.c.task_id)

memory_consolidation_runs = Table(
    "memory_consolidation_runs",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column(
        "run_id",
        UUID(as_uuid=True),
        nullable=False,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    ),
    Column("status", String(length=32), nullable=False, default="pending"),
    Column("started_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("completed_at", DateTime(timezone=True), nullable=True),
    Column("summary", Text(), nullable=True),
    Column("stats", JSONB(astext_type=Text()), nullable=True),
    Column("error", Text(), nullable=True),
)
Index("ix_memory_consolidation_runs_run_id", memory_consolidation_runs.c.run_id, unique=True)

orchestration_runs = Table(
    "orchestration_runs",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("run_id", UUID(as_uuid=True), nullable=False, server_default=text("gen_random_uuid()")),
    Column("task_id", String(length=128), nullable=False),
    Column("status", String(length=32), nullable=False, default="pending"),
    Column("state", JSONB(astext_type=Text()), nullable=False, server_default=text("'{}'::jsonb")),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column(
        "updated_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    ),
    UniqueConstraint("run_id", name="uq_orchestration_runs_run_id"),
)
Index("ix_orchestration_runs_task_id", orchestration_runs.c.task_id)

orchestration_events = Table(
    "orchestration_events",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("run_id", UUID(as_uuid=True), ForeignKey("orchestration_runs.run_id", ondelete="CASCADE"), nullable=False),
    Column("sequence", BigInteger, nullable=False),
    Column("event_type", String(length=64), nullable=False),
    Column("agent", String(length=64), nullable=True),
    Column("payload", JSONB(astext_type=Text()), nullable=False, server_default=text("'{}'::jsonb")),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)
Index("ix_orchestration_events_run_id", orchestration_events.c.run_id)
Index("ix_orchestration_events_sequence", orchestration_events.c.sequence)

lifecycle_status_enum = Enum(LifecycleStatus, name="lifecycle_status")
guardrail_decision_enum = Enum(GuardrailDecisionType, name="guardrail_decision")

review_status_enum = Enum("open", "in_review", "resolved", "dismissed", name="review_status")

task_lifecycle_events = Table(
    "task_lifecycle_events",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("task_id", String(length=128), nullable=False),
    Column("run_id", UUID(as_uuid=True), nullable=True),
    Column("step_id", String(length=128), nullable=False),
    Column("sequence", BigInteger, nullable=False, server_default=text("0")),
    Column("event_type", String(length=64), nullable=False),
    Column("status", lifecycle_status_enum, nullable=False, server_default=LifecycleStatus.QUEUED.value),
    Column("agent", String(length=64), nullable=True),
    Column("attempt", BigInteger, nullable=False, server_default=text("0")),
    Column("eta", DateTime(timezone=True), nullable=True),
    Column("deadline", DateTime(timezone=True), nullable=True),
    Column("latency_ms", Float(), nullable=True),
    Column("payload", JSONB(astext_type=Text()), nullable=False, server_default=text("'{}'::jsonb")),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column(
        "updated_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    ),
)
Index("ix_task_lifecycle_task", task_lifecycle_events.c.task_id, task_lifecycle_events.c.step_id)
Index("ix_task_lifecycle_run", task_lifecycle_events.c.run_id)

context_snapshots = Table(
    "context_snapshots",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("task_id", String(length=128), nullable=False),
    Column("run_id", UUID(as_uuid=True), ForeignKey("orchestration_runs.run_id", ondelete="CASCADE"), nullable=True),
    Column("stage", String(length=64), nullable=False),
    Column("agent", String(length=64), nullable=True),
    Column("payload", JSONB(astext_type=Text()), nullable=False, server_default=text("'{}'::jsonb")),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)
Index("ix_context_snapshots_task", context_snapshots.c.task_id)
Index("ix_context_snapshots_stage", context_snapshots.c.stage)

guardrail_events = Table(
    "guardrail_events",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("task_id", String(length=128), nullable=False),
    Column("run_id", UUID(as_uuid=True), ForeignKey("orchestration_runs.run_id", ondelete="CASCADE"), nullable=True),
    Column("decision", guardrail_decision_enum, nullable=False),
    Column("reason", Text(), nullable=True),
    Column("risk_score", Float(), nullable=True),
    Column("policy_id", String(length=64), nullable=True),
    Column("agent", String(length=64), nullable=True),
    Column("payload", JSONB(astext_type=Text()), nullable=False, server_default=text("'{}'::jsonb")),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)
Index("ix_guardrail_events_task", guardrail_events.c.task_id)
Index("ix_guardrail_events_decision", guardrail_events.c.decision)

review_tickets = Table(
    "review_tickets",
    metadata,
    Column("ticket_id", UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")),
    Column("task_id", String(length=128), nullable=False, unique=True),
    Column("status", review_status_enum, nullable=False, server_default="open"),
    Column("summary", Text(), nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()),
    Column("assigned_to", String(length=128), nullable=True),
    Column("sources", ARRAY(String(length=128)), nullable=False, server_default=text("'{}'::text[]")),
    Column("escalation_payload", JSONB(astext_type=Text()), nullable=False, server_default=text("'{}'::jsonb")),
)
Index("ix_review_tickets_status", review_tickets.c.status)
Index("ix_review_tickets_assigned", review_tickets.c.assigned_to)

review_notes = Table(
    "review_notes",
    metadata,
    Column("note_id", UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")),
    Column(
        "ticket_id",
        UUID(as_uuid=True),
        ForeignKey("review_tickets.ticket_id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("author", String(length=128), nullable=False),
    Column("content", Text(), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)
Index("ix_review_notes_ticket", review_notes.c.ticket_id)

__all__ = [
    "metadata",
    "episodic_memory",
    "negotiation_transcripts",
    "memory_consolidation_runs",
    "orchestration_runs",
    "orchestration_events",
    "task_lifecycle_events",
    "context_snapshots",
    "guardrail_events",
    "review_tickets",
    "review_notes",
]
