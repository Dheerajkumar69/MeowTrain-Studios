"""add email verification and oauth fields

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2025-01-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "d4e5f6a7b8c9"
down_revision: Union[str, None] = "c3d4e5f6a7b8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Email verification columns
    with op.batch_alter_table("users") as batch_op:
        batch_op.add_column(sa.Column("email_verified", sa.Boolean(), nullable=True, server_default="0"))
        batch_op.add_column(sa.Column("email_verification_token", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("oauth_provider", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("oauth_id", sa.String(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("users") as batch_op:
        batch_op.drop_column("oauth_id")
        batch_op.drop_column("oauth_provider")
        batch_op.drop_column("email_verification_token")
        batch_op.drop_column("email_verified")
