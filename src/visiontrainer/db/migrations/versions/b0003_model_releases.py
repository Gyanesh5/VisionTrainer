"""model_releases：模型发布记录表（票 123，ADR 0015 Decision 1）

只增不改账目：时间、指纹、run 标识、发布目录与拉取包路径、限时签名 URL 过期时刻、
打包元数据快照（payload）。既有库 stamp 接入后 upgrade head 平滑补建（同 b0002 先例）。

Revision ID: b0003_model_releases
Revises: b0002_runtime_settings
Create Date: 2026-11-18

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision: str = 'b0003_model_releases'
down_revision: str | None = 'b0002_runtime_settings'
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_table('model_releases',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('run_id', sa.Integer(), nullable=False),
    sa.Column('fingerprint', sa.String(length=64), nullable=False),
    sa.Column('release_dir', sa.String(length=1024), nullable=False),
    sa.Column('bundle_path', sa.String(length=1024), nullable=False),
    sa.Column('url_expires_at', sa.DateTime(), nullable=False),
    sa.Column('payload', sa.JSON(), nullable=False),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.ForeignKeyConstraint(['run_id'], ['runs.id']),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_model_releases_run_id'), 'model_releases', ['run_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_model_releases_run_id'), table_name='model_releases')
    op.drop_table('model_releases')
