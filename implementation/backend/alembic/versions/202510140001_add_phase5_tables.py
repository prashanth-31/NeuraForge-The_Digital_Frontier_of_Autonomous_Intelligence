"""Add Phase 5 orchestration tables

Revision ID: 202510140001
Revises: 202410130001
Create Date: 2025-10-14
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "202510140001"
down_revision = "202410130001"
branch_labels = None
depends_on = None


lifecycle_status_enum = sa.Enum(
    "queued",
    "planned",
    "scheduled",
    "in_progress",
    "completed",
    "failed",
    "cancelled",
    name="lifecycle_status",
)

guardrail_decision_enum = sa.Enum(
    "allow",
    "deny",
    "escalate",
    "review",
    name="guardrail_decision",
)


def upgrade() -> None:
    bind = op.get_bind()
    lifecycle_status_enum.create(bind, checkfirst=True)
    guardrail_decision_enum.create(bind, checkfirst=True)

    op.create_table(
        "task_lifecycle_events",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("task_id", sa.String(length=128), nullable=False),
        sa.Column("run_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("step_id", sa.String(length=128), nullable=False),
        sa.Column("sequence", sa.BigInteger(), nullable=False, server_default=sa.text("0")),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("status", lifecycle_status_enum, nullable=False, server_default="queued"),
        sa.Column("agent", sa.String(length=64), nullable=True),
        sa.Column("attempt", sa.BigInteger(), nullable=False, server_default=sa.text("0")),
        sa.Column("eta", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deadline", sa.DateTime(timezone=True), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("payload", sa.dialects.postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index("ix_task_lifecycle_task", "task_lifecycle_events", ["task_id", "step_id"])
    op.create_index("ix_task_lifecycle_run", "task_lifecycle_events", ["run_id"])

    op.create_table(
        "context_snapshots",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("task_id", sa.String(length=128), nullable=False),
        sa.Column(
            "run_id",
            sa.dialects.postgresql.UUID(as_uuid=True),
            sa.ForeignKey("orchestration_runs.run_id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("stage", sa.String(length=64), nullable=False),
        sa.Column("agent", sa.String(length=64), nullable=True),
        sa.Column("payload", sa.dialects.postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_context_snapshots_task", "context_snapshots", ["task_id"])
    op.create_index("ix_context_snapshots_stage", "context_snapshots", ["stage"])

    op.create_table(
        "guardrail_events",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("task_id", sa.String(length=128), nullable=False),
        sa.Column(
            "run_id",
            sa.dialects.postgresql.UUID(as_uuid=True),
            sa.ForeignKey("orchestration_runs.run_id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("decision", guardrail_decision_enum, nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("risk_score", sa.Float(), nullable=True),
        sa.Column("policy_id", sa.String(length=64), nullable=True),
        sa.Column("agent", sa.String(length=64), nullable=True),
        sa.Column("payload", sa.dialects.postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_guardrail_events_task", "guardrail_events", ["task_id"])
    op.create_index("ix_guardrail_events_decision", "guardrail_events", ["decision"])


def downgrade() -> None:
    op.drop_index("ix_guardrail_events_decision", table_name="guardrail_events")
    op.drop_index("ix_guardrail_events_task", table_name="guardrail_events")
    op.drop_table("guardrail_events")

    op.drop_index("ix_context_snapshots_stage", table_name="context_snapshots")
    op.drop_index("ix_context_snapshots_task", table_name="context_snapshots")
    op.drop_table("context_snapshots")

    op.drop_index("ix_task_lifecycle_run", table_name="task_lifecycle_events")
    op.drop_index("ix_task_lifecycle_task", table_name="task_lifecycle_events")
    op.drop_table("task_lifecycle_events")

    guardrail_decision_enum.drop(op.get_bind(), checkfirst=True)
    lifecycle_status_enum.drop(op.get_bind(), checkfirst=True)
