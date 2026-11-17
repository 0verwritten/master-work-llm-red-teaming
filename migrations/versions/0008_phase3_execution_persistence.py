from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "phase3_execution_batches",
        sa.Column("manifest_sha256", sa.String(length=64), nullable=False),
        sa.Column("plan_sha256", sa.String(length=64), nullable=False),
        sa.Column("benchmark_sha256", sa.String(length=64), nullable=False),
        sa.Column("roster_sha256", sa.String(length=64), nullable=False),
        sa.Column("execution_envelope_json", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("stop_reason", sa.String(length=100), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status IN ('materialized', 'running', 'completed', 'aborted', 'indeterminate')",
            name="ck_phase3_execution_batch_status",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("manifest_sha256", name="uq_phase3_execution_manifest"),
    )
    op.create_table(
        "phase3_execution_cells",
        sa.Column("batch_id", sa.Uuid(), nullable=False),
        sa.Column("cell_id", sa.String(length=64), nullable=False),
        sa.Column("stage", sa.String(length=50), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("cell_sha256", sa.String(length=64), nullable=False),
        sa.Column("cell_spec_json", sa.JSON(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("terminal_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("stop_reason", sa.String(length=100), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("ordinal >= 0", name="ck_phase3_execution_cell_ordinal"),
        sa.CheckConstraint(
            "status IN ('queued', 'claimed', 'terminal', 'indeterminate')",
            name="ck_phase3_execution_cell_status",
        ),
        sa.ForeignKeyConstraint(
            ["batch_id"],
            ["phase3_execution_batches.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("batch_id", "cell_id", name="uq_phase3_execution_cell"),
        sa.UniqueConstraint(
            "batch_id",
            "stage",
            "ordinal",
            name="uq_phase3_execution_stage_order",
        ),
        sa.UniqueConstraint("run_id", name="uq_phase3_execution_cell_run"),
    )
    op.create_index(
        "ix_phase3_execution_claim",
        "phase3_execution_cells",
        ["batch_id", "stage", "status", "ordinal"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_phase3_execution_claim", table_name="phase3_execution_cells")
    op.drop_table("phase3_execution_cells")
    op.drop_table("phase3_execution_batches")
