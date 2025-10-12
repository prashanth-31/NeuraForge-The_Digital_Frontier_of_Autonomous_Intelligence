from __future__ import annotations

import uuid

from sqlalchemy import BigInteger, Column, DateTime, Float, Index, MetaData, String, Table, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID

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

__all__ = [
    "metadata",
    "episodic_memory",
    "negotiation_transcripts",
    "memory_consolidation_runs",
]
