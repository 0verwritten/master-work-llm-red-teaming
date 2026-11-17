from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "research_inputs",
        sa.Column("stable_id", sa.String(length=300), nullable=False),
        sa.Column("role", sa.String(length=100), nullable=False),
        sa.Column("source_name", sa.String(length=100), nullable=False),
        sa.Column("source_version", sa.String(length=100), nullable=False),
        sa.Column("source_item_id", sa.String(length=300), nullable=False),
        sa.Column("source_sha256", sa.String(length=64), nullable=False),
        sa.Column("language", sa.String(length=20), nullable=False),
        sa.Column("category", sa.String(length=100), nullable=False),
        sa.Column("subcategory", sa.String(length=100), nullable=False),
        sa.Column("content_text", sa.Text(), nullable=False),
        sa.Column("content_sha256", sa.String(length=64), nullable=False),
        sa.Column("source_metadata_json", sa.JSON(), nullable=False),
        sa.Column("release_status", sa.String(length=50), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("stable_id", name="uq_research_input_stable_id"),
    )


def downgrade() -> None:
    op.drop_table("research_inputs")
