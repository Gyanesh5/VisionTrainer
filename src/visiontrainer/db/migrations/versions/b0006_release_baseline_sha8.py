"""model_releases 血缘列 baseline_sha8（票 147，ADR 0018）

契约 v1.1 §5.4 发布面血缘声明：该 Run 起训权重内容指纹（快照 weight 段冻结口径，
sha256[:8]）。本地文件起训 → 指纹；官方名/无快照 → NULL（null=开新族候选，合法值；
ADR 0018 Decision 3 宁缺勿错，历史行不从 Task.base_model 现算回填）。
同票 sample_models 列（数据血缘：实效集命中批次 models 并集）wave 2 另行加列，本迁移不加。

Revision ID: b0006_release_baseline_sha8
Revises: b0005_release_dir_cleanup
Create Date: 2026-11-23

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision: str = 'b0006_release_baseline_sha8'
down_revision: str | None = 'b0005_release_dir_cleanup'
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column('model_releases', sa.Column('baseline_sha8', sa.String(8), nullable=True))


def downgrade() -> None:
    op.drop_column('model_releases', 'baseline_sha8')
