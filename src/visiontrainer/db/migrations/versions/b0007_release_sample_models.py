"""model_releases 血缘列 sample_models（票 147，ADR 0018 Decision 4）

契约 v1.1 §5.4 发布面数据血缘：该 Run 实效集命中的各导出批次 manifest.models 并集
（``[{model_id, model_sha8}]`` 去重保序，形态与导出包 manifest.models 一致，对端
对账零翻译）。取数与 §6.2 trained 归因同构（消费 #146 共享归因查询
``datasets.producer_writeback.trained_attribution``）；数据集无联动导入 → 空数组。
历史行 NULL 不回填（宁缺勿错，同 b0006 baseline_sha8 口径）。

Revision ID: b0007_release_sample_models
Revises: b0006_release_baseline_sha8
Create Date: 2026-11-23

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision: str = 'b0007_release_sample_models'
down_revision: str | None = 'b0006_release_baseline_sha8'
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column('model_releases', sa.Column('sample_models', sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column('model_releases', 'sample_models')
