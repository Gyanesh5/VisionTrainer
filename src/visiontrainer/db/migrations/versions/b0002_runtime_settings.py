"""runtime_settings：运行期可变设置键值表（票 122）

Web 可改、改后即生效的设置落这里，config.toml 同名键只作兜底默认。
首个键 producer.base_url（推理端连接地址；#129 改名，旧键读兼容见 core.producer）。

Revision ID: b0002_runtime_settings
Revises: b0001_baseline
Create Date: 2026-11-18

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision: str = 'b0002_runtime_settings'
down_revision: str | None = 'b0001_baseline'
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_table('runtime_settings',
    sa.Column('key', sa.String(length=100), nullable=False),
    sa.Column('value', sa.String(length=500), nullable=False),
    sa.Column('updated_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
    sa.PrimaryKeyConstraint('key')
    )


def downgrade() -> None:
    op.drop_table('runtime_settings')
