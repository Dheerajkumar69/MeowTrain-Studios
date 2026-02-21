"""Add worker_pid to training_runs.

Revision ID: f1a2b3c4d5e6
Revises: e5f6a7b8c9d0
Create Date: 2026-02-21
"""

from alembic import op
import sqlalchemy as sa

revision = "f1a2b3c4d5e6"
down_revision = "e5f6a7b8c9d0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("training_runs", sa.Column("worker_pid", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("training_runs", "worker_pid")
