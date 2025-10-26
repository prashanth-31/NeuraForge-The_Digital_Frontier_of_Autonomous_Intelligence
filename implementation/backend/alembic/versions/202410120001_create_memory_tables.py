"""create memory tables

Revision ID: 202410120001
Revises: 
Create Date: 2025-10-12 00:00:00
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "202410120001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    op.create_table(
        "episodic_memory",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("task_id", sa.String(length=64), nullable=False),
        sa.Column("agent", sa.String(length=64), nullable=True),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("task_id", name="uq_episodic_memory_task_id"),
    )
    op.create_index("ix_episodic_memory_task_id", "episodic_memory", ["task_id"], unique=False)

    op.create_table(
        "negotiation_transcripts",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("task_id", sa.String(length=64), nullable=False),
        sa.Column("agent", sa.String(length=64), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_negotiation_transcripts_task_id", "negotiation_transcripts", ["task_id"], unique=False)

    op.create_table(
        "memory_consolidation_runs",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.text("gen_random_uuid()")),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="pending"),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("stats", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
    )
    op.create_index(
        "ix_memory_consolidation_runs_run_id",
        "memory_consolidation_runs",
        ["run_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ix_memory_consolidation_runs_run_id", table_name="memory_consolidation_runs")
    op.drop_table("memory_consolidation_runs")
    op.drop_index("ix_negotiation_transcripts_task_id", table_name="negotiation_transcripts")
    op.drop_table("negotiation_transcripts")
    op.drop_index("ix_episodic_memory_task_id", table_name="episodic_memory")
    op.drop_table("episodic_memory")
