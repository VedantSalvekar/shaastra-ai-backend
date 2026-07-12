"""add extracted_text to documents

Revision ID: b7f2c9d4e1a3
Revises: 5a84c41b8d37
Create Date: 2026-07-12 20:55:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "b7f2c9d4e1a3"
down_revision: Union[str, Sequence[str], None] = "5a84c41b8d37"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("extracted_text", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("documents", "extracted_text")
