from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "benchmark_versions",
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("version", sa.String(length=100), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("source_manifest_json", sa.JSON(), nullable=False),
        sa.Column("frozen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", "version", name="uq_benchmark_name_version"),
    )
    op.create_table(
        "benchmark_items",
        sa.Column("benchmark_version_id", sa.Uuid(), nullable=False),
        sa.Column("stable_id", sa.String(length=300), nullable=False),
        sa.Column("source_name", sa.String(length=100), nullable=False),
        sa.Column("source_version", sa.String(length=100), nullable=False),
        sa.Column("source_item_id", sa.String(length=300), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("license_id", sa.String(length=100), nullable=False),
        sa.Column("license_url", sa.Text(), nullable=False),
        sa.Column("provenance", sa.String(length=50), nullable=False),
        sa.Column("split", sa.String(length=50), nullable=False),
        sa.Column("language", sa.String(length=20), nullable=False),
        sa.Column("modality", sa.String(length=50), nullable=False),
        sa.Column("category", sa.String(length=100), nullable=False),
        sa.Column("functional_category", sa.String(length=100), nullable=False),
        sa.Column("prompt_text", sa.Text(), nullable=False),
        sa.Column("context_text", sa.Text(), nullable=True),
        sa.Column("rubric_text", sa.Text(), nullable=False),
        sa.Column("expected_behavior", sa.String(length=100), nullable=False),
        sa.Column("pair_id", sa.String(length=200), nullable=True),
        sa.Column("translation_provenance_json", sa.JSON(), nullable=False),
        sa.Column("reviewer_status", sa.String(length=50), nullable=False),
        sa.Column("release_status", sa.String(length=50), nullable=False),
        sa.Column("content_sha256", sa.String(length=64), nullable=False),
        sa.Column("source_metadata_json", sa.JSON(), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["benchmark_version_id"], ["benchmark_versions.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "benchmark_version_id",
            "stable_id",
            name="uq_benchmark_item_stable_id",
        ),
    )
    op.create_index(
        "ix_benchmark_item_pair",
        "benchmark_items",
        ["benchmark_version_id", "pair_id"],
        unique=False,
    )
    op.create_index(
        "ix_benchmark_item_split",
        "benchmark_items",
        ["benchmark_version_id", "split"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_benchmark_item_split", table_name="benchmark_items")
    op.drop_index("ix_benchmark_item_pair", table_name="benchmark_items")
    op.drop_table("benchmark_items")
    op.drop_table("benchmark_versions")
