"""Add review ticket tables

Revision ID: 202510140002
Revises: 202510140001
Create Date: 2025-10-14
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "202510140002"
down_revision = "202510140001"
branch_labels = None
depends_on = None


review_status_enum = sa.Enum(
    "open",
    "in_review",
    "resolved",
    "dismissed",
    name="review_status",
)


def upgrade() -> None:
    bind = op.get_bind()
    review_status_enum.create(bind, checkfirst=True)

    op.create_table(
        "review_tickets",
        sa.Column("ticket_id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("task_id", sa.String(length=128), nullable=False, unique=True),
        sa.Column("status", review_status_enum, nullable=False, server_default="open"),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column("assigned_to", sa.String(length=128), nullable=True),
        sa.Column("sources", sa.dialects.postgresql.ARRAY(sa.String(length=128)), nullable=False, server_default=sa.text("'{}'::text[]")),
        sa.Column("escalation_payload", sa.dialects.postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
    )
    op.create_index("ix_review_tickets_status", "review_tickets", ["status"])
    op.create_index("ix_review_tickets_assigned", "review_tickets", ["assigned_to"])

    op.create_table(
        "review_notes",
        sa.Column("note_id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column(
            "ticket_id",
            sa.dialects.postgresql.UUID(as_uuid=True),
            sa.ForeignKey("review_tickets.ticket_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("author", sa.String(length=128), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_review_notes_ticket", "review_notes", ["ticket_id"])


def downgrade() -> None:
    op.drop_index("ix_review_notes_ticket", table_name="review_notes")
    op.drop_table("review_notes")

    op.drop_index("ix_review_tickets_assigned", table_name="review_tickets")
    op.drop_index("ix_review_tickets_status", table_name="review_tickets")
    op.drop_table("review_tickets")

    review_status_enum.drop(op.get_bind(), checkfirst=True)
