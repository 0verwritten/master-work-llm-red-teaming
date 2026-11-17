from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "phase4_instrument_batches",
        sa.Column("manifest_sha256", sa.String(length=64), nullable=False),
        sa.Column("source_phase3_manifest_sha256", sa.String(length=64), nullable=False),
        sa.Column("response_set_sha256", sa.String(length=64), nullable=False),
        sa.Column("target_endpoint_profile", sa.String(length=300), nullable=False),
        sa.Column("protocol_json", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("stop_reason", sa.String(length=100), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status IN ('materialized', 'running', 'completed', 'aborted', 'indeterminate')",
            name="ck_phase4_instrument_batch_status",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("manifest_sha256", name="uq_phase4_instrument_manifest"),
    )
    op.create_table(
        "phase4_instrument_items",
        sa.Column("batch_id", sa.Uuid(), nullable=False),
        sa.Column("source_round_id", sa.Uuid(), nullable=False),
        sa.Column("source_evaluation_id", sa.Uuid(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("source_language", sa.String(length=20), nullable=False),
        sa.Column("source_response_sha256", sa.String(length=64), nullable=False),
        sa.Column("source_evaluation_sha256", sa.String(length=64), nullable=False),
        sa.Column("input_bundle_sha256", sa.String(length=64), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("terminal_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("stop_reason", sa.String(length=100), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("ordinal >= 0", name="ck_phase4_instrument_ordinal"),
        sa.CheckConstraint(
            "status IN ('queued', 'claimed', 'terminal', 'indeterminate')",
            name="ck_phase4_instrument_item_status",
        ),
        sa.ForeignKeyConstraint(["batch_id"], ["phase4_instrument_batches.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"]),
        sa.ForeignKeyConstraint(["source_round_id"], ["adversarial_rounds.id"]),
        sa.ForeignKeyConstraint(["source_evaluation_id"], ["evaluations.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("batch_id", "ordinal", name="uq_phase4_instrument_ordinal"),
        sa.UniqueConstraint("batch_id", "source_round_id", name="uq_phase4_instrument_round"),
        sa.UniqueConstraint("run_id", name="uq_phase4_instrument_run"),
    )
    op.create_index(
        "ix_phase4_instrument_claim",
        "phase4_instrument_items",
        ["batch_id", "status", "ordinal"],
        unique=False,
    )
    op.create_table(
        "phase4_self_judge_evaluations",
        sa.Column("item_id", sa.Uuid(), nullable=False),
        sa.Column("llm_call_id", sa.Uuid(), nullable=False),
        sa.Column("label", sa.String(length=32), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("parser_version", sa.String(length=100), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "label IN ('refused', 'partial', 'complied')",
            name="ck_phase4_self_judge_label",
        ),
        sa.ForeignKeyConstraint(["item_id"], ["phase4_instrument_items.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["llm_call_id"], ["llm_calls.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("item_id", name="uq_phase4_self_judge_item"),
        sa.UniqueConstraint("llm_call_id", name="uq_phase4_self_judge_llm_call"),
    )


def downgrade() -> None:
    op.drop_table("phase4_self_judge_evaluations")
    op.drop_index("ix_phase4_instrument_claim", table_name="phase4_instrument_items")
    op.drop_table("phase4_instrument_items")
    op.drop_table("phase4_instrument_batches")
