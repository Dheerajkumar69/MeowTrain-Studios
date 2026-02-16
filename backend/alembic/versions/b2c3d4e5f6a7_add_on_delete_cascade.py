"""add ON DELETE CASCADE to all foreign keys

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-02-17 12:00:00.000000

SQLite doesn't support ALTER CONSTRAINT, so we use batch_alter_table
to recreate each table with the correct ON DELETE CASCADE.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic
revision: str = 'b2c3d4e5f6a7'
down_revision: Union[str, None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── projects.user_id → users.id CASCADE ──
    with op.batch_alter_table('projects', schema=None) as batch_op:
        batch_op.drop_constraint('fk_projects_user_id', type_='foreignkey')
        batch_op.create_foreign_key(
            'fk_projects_user_id', 'users',
            ['user_id'], ['id'], ondelete='CASCADE',
        )

    # ── datasets.project_id → projects.id CASCADE ──
    with op.batch_alter_table('datasets', schema=None) as batch_op:
        batch_op.drop_constraint('fk_datasets_project_id', type_='foreignkey')
        batch_op.create_foreign_key(
            'fk_datasets_project_id', 'projects',
            ['project_id'], ['id'], ondelete='CASCADE',
        )

    # ── model_configs.project_id → projects.id CASCADE ──
    with op.batch_alter_table('model_configs', schema=None) as batch_op:
        batch_op.drop_constraint('fk_model_configs_project_id', type_='foreignkey')
        batch_op.create_foreign_key(
            'fk_model_configs_project_id', 'projects',
            ['project_id'], ['id'], ondelete='CASCADE',
        )

    # ── training_runs.project_id → projects.id CASCADE ──
    # ── training_runs.model_config_id → model_configs.id CASCADE ──
    with op.batch_alter_table('training_runs', schema=None) as batch_op:
        batch_op.drop_constraint('fk_training_runs_project_id', type_='foreignkey')
        batch_op.create_foreign_key(
            'fk_training_runs_project_id', 'projects',
            ['project_id'], ['id'], ondelete='CASCADE',
        )
        batch_op.drop_constraint('fk_training_runs_model_config_id', type_='foreignkey')
        batch_op.create_foreign_key(
            'fk_training_runs_model_config_id', 'model_configs',
            ['model_config_id'], ['id'], ondelete='CASCADE',
        )

    # ── prompt_templates.project_id → projects.id CASCADE ──
    with op.batch_alter_table('prompt_templates', schema=None) as batch_op:
        batch_op.drop_constraint('fk_prompt_templates_project_id', type_='foreignkey')
        batch_op.create_foreign_key(
            'fk_prompt_templates_project_id', 'projects',
            ['project_id'], ['id'], ondelete='CASCADE',
        )


def downgrade() -> None:
    # Revert to plain FKs without ON DELETE CASCADE
    with op.batch_alter_table('prompt_templates', schema=None) as batch_op:
        batch_op.drop_constraint('fk_prompt_templates_project_id', type_='foreignkey')
        batch_op.create_foreign_key(
            'fk_prompt_templates_project_id', 'projects',
            ['project_id'], ['id'],
        )

    with op.batch_alter_table('training_runs', schema=None) as batch_op:
        batch_op.drop_constraint('fk_training_runs_model_config_id', type_='foreignkey')
        batch_op.create_foreign_key(
            'fk_training_runs_model_config_id', 'model_configs',
            ['model_config_id'], ['id'],
        )
        batch_op.drop_constraint('fk_training_runs_project_id', type_='foreignkey')
        batch_op.create_foreign_key(
            'fk_training_runs_project_id', 'projects',
            ['project_id'], ['id'],
        )

    with op.batch_alter_table('model_configs', schema=None) as batch_op:
        batch_op.drop_constraint('fk_model_configs_project_id', type_='foreignkey')
        batch_op.create_foreign_key(
            'fk_model_configs_project_id', 'projects',
            ['project_id'], ['id'],
        )

    with op.batch_alter_table('datasets', schema=None) as batch_op:
        batch_op.drop_constraint('fk_datasets_project_id', type_='foreignkey')
        batch_op.create_foreign_key(
            'fk_datasets_project_id', 'projects',
            ['project_id'], ['id'],
        )

    with op.batch_alter_table('projects', schema=None) as batch_op:
        batch_op.drop_constraint('fk_projects_user_id', type_='foreignkey')
        batch_op.create_foreign_key(
            'fk_projects_user_id', 'users',
            ['user_id'], ['id'],
        )
