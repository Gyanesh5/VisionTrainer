"""deployments/model_releases 失效标注列（票 126，ADR 0015 Decision 2 后果）

删除冲突治理：已部署/已发布的 Run 放行删除时，部署与发布记录保留并标注失效
（invalidated_at），冗余 run_uid 让 Run 行删除后记录仍可对账——账目不静默消失。
历史行 run_uid 由 runs 表回填（孤儿行留 NULL，如实缺失）。

Revision ID: b0004_delete_invalidation
Revises: b0003_model_releases
Create Date: 2026-11-18

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision: str = 'b0004_delete_invalidation'
down_revision: str | None = 'b0003_model_releases'
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column('deployments', sa.Column('run_uid', sa.String(length=16), nullable=True))
    op.add_column('deployments', sa.Column('invalidated_at', sa.DateTime(), nullable=True))
    op.add_column('model_releases', sa.Column('run_uid', sa.String(length=16), nullable=True))
    op.add_column('model_releases', sa.Column('invalidated_at', sa.DateTime(), nullable=True))
    # 历史行回填：部署/发布发生时 Run 在账（孤儿行无源可回，留 NULL）
    op.execute("UPDATE deployments SET run_uid = (SELECT uid FROM runs WHERE runs.id = deployments.run_id)")
    op.execute("UPDATE model_releases SET run_uid = (SELECT uid FROM runs WHERE runs.id = model_releases.run_id)")


def downgrade() -> None:
    op.drop_column('model_releases', 'invalidated_at')
    op.drop_column('model_releases', 'run_uid')
    op.drop_column('deployments', 'invalidated_at')
    op.drop_column('deployments', 'run_uid')
