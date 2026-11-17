from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "legacy_import_batches",
        sa.Column("provenance", sa.String(length=32), nullable=False),
        sa.Column("source_filename", sa.String(length=300), nullable=False),
        sa.Column("source_sha256", sa.String(length=64), nullable=False),
        sa.Column("source_size_bytes", sa.Integer(), nullable=False),
        sa.Column("expected_record_count", sa.Integer(), nullable=False),
        sa.Column("importer_version", sa.String(length=32), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("provenance = 'legacy'", name="ck_legacy_import_provenance"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source_sha256", name="uq_legacy_import_source_sha256"),
    )
    op.create_table(
        "legacy_records",
        sa.Column("batch_id", sa.Uuid(), nullable=False),
        sa.Column("source_row_number", sa.Integer(), nullable=False),
        sa.Column("record_sha256", sa.String(length=64), nullable=False),
        sa.Column("benchmark_item_id", sa.String(length=200), nullable=False),
        sa.Column("source_collection", sa.String(length=100), nullable=False),
        sa.Column("category", sa.String(length=100), nullable=False),
        sa.Column("claimed_model", sa.String(length=200), nullable=False),
        sa.Column("repetition", sa.Integer(), nullable=False),
        sa.Column("temperature", sa.Float(), nullable=False),
        sa.Column("recorded_timestamp", sa.String(length=40), nullable=False),
        sa.Column("text_refused", sa.Boolean(), nullable=False),
        sa.Column("image_refused", sa.Boolean(), nullable=False),
        sa.Column("text_judge_verdict", sa.String(length=32), nullable=False),
        sa.Column("image_judge_verdict", sa.String(length=32), nullable=False),
        sa.Column("payload_json", sa.JSON(), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(["batch_id"], ["legacy_import_batches.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("batch_id", "source_row_number", name="uq_legacy_record_source_row"),
    )


def downgrade() -> None:
    op.drop_table("legacy_records")
    op.drop_table("legacy_import_batches")
