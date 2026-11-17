"""model_releases 目录已清标注列（review 修复：发布目录平台清理入口）

票 126 派生治理补全：发布目录此前只能手工删盘（无账目入口）。平台清理动作
（DELETE /release-library/{name}）只删派生物并写 cleared_at 标注，账目本体
（时间/指纹/run 标识）不动——发布记录只增不改。

Revision ID: b0005_release_dir_cleanup
Revises: b0004_delete_invalidation
Create Date: 2026-11-18

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision: str = 'b0005_release_dir_cleanup'
down_revision: str | None = 'b0004_delete_invalidation'
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column('model_releases', sa.Column('cleared_at', sa.DateTime(), nullable=True))


def downgrade() -> None:
    op.drop_column('model_releases', 'cleared_at')
