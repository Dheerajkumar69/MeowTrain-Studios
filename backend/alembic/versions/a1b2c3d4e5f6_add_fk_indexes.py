"""add FK indexes and training_run status index

Revision ID: a1b2c3d4e5f6
Revises: 7822084ddf4c
Create Date: 2026-02-16 12:00:00.000000
"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = '7822084ddf4c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Foreign key indexes — critical for query performance on list/filter ops.
    # SQLite does NOT auto-create indexes for foreign keys.
    with op.batch_alter_table('projects', schema=None) as batch_op:
        batch_op.create_index('ix_projects_user_id', ['user_id'], unique=False)

    with op.batch_alter_table('datasets', schema=None) as batch_op:
        batch_op.create_index('ix_datasets_project_id', ['project_id'], unique=False)

    with op.batch_alter_table('model_configs', schema=None) as batch_op:
        batch_op.create_index('ix_model_configs_project_id', ['project_id'], unique=False)

    with op.batch_alter_table('training_runs', schema=None) as batch_op:
        batch_op.create_index('ix_training_runs_project_id', ['project_id'], unique=False)
        batch_op.create_index('ix_training_runs_model_config_id', ['model_config_id'], unique=False)
        batch_op.create_index('ix_training_runs_status', ['status'], unique=False)

    with op.batch_alter_table('prompt_templates', schema=None) as batch_op:
        batch_op.create_index('ix_prompt_templates_project_id', ['project_id'], unique=False)


def downgrade() -> None:
    with op.batch_alter_table('prompt_templates', schema=None) as batch_op:
        batch_op.drop_index('ix_prompt_templates_project_id')

    with op.batch_alter_table('training_runs', schema=None) as batch_op:
        batch_op.drop_index('ix_training_runs_status')
        batch_op.drop_index('ix_training_runs_model_config_id')
        batch_op.drop_index('ix_training_runs_project_id')

    with op.batch_alter_table('model_configs', schema=None) as batch_op:
        batch_op.drop_index('ix_model_configs_project_id')

    with op.batch_alter_table('datasets', schema=None) as batch_op:
        batch_op.drop_index('ix_datasets_project_id')

    with op.batch_alter_table('projects', schema=None) as batch_op:
        batch_op.drop_index('ix_projects_user_id')
