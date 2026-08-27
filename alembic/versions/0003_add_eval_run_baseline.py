"""add is_baseline to eval_runs, enforce single baseline

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-27

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "eval_runs",
        sa.Column("is_baseline", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    # Partial unique index: at most one row can have is_baseline = true at a
    # time. POST /eval/runs/{id}/mark_baseline unsets the previous baseline
    # in the same transaction as setting the new one -- this index is the
    # DB-level guarantee that the "exactly one baseline" invariant holds even
    # if that application-level logic ever has a bug.
    op.create_index(
        "ix_eval_runs_single_baseline",
        "eval_runs",
        ["is_baseline"],
        unique=True,
        postgresql_where=sa.text("is_baseline = true"),
    )


def downgrade() -> None:
    op.drop_index("ix_eval_runs_single_baseline", table_name="eval_runs")
    op.drop_column("eval_runs", "is_baseline")
