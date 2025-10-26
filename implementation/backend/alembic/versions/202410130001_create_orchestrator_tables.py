"""create orchestrator tables

Revision ID: 202410130001
Revises: 202410120001
Create Date: 2025-10-13 19:20:00
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "202410130001"
down_revision = "202410120001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "orchestration_runs",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.text("gen_random_uuid()")),
        sa.Column("task_id", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="pending"),
        sa.Column("state", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("run_id", name="uq_orchestration_runs_run_id"),
    )
    op.create_index("ix_orchestration_runs_task_id", "orchestration_runs", ["task_id"], unique=False)

    op.create_table(
        "orchestration_events",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("sequence", sa.BigInteger(), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("agent", sa.String(length=64), nullable=True),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["run_id"], ["orchestration_runs.run_id"], ondelete="CASCADE"),
    )
    op.create_index("ix_orchestration_events_run_id", "orchestration_events", ["run_id"], unique=False)
    op.create_index("ix_orchestration_events_sequence", "orchestration_events", ["sequence"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_orchestration_events_sequence", table_name="orchestration_events")
    op.drop_index("ix_orchestration_events_run_id", table_name="orchestration_events")
    op.drop_table("orchestration_events")
    op.drop_index("ix_orchestration_runs_task_id", table_name="orchestration_runs")
    op.drop_table("orchestration_runs")
