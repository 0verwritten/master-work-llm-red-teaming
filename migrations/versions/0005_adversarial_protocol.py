from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "adversarial_sessions",
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("method", sa.String(length=50), nullable=False),
        sa.Column("language", sa.String(length=20), nullable=False),
        sa.Column("max_rounds", sa.Integer(), nullable=False),
        sa.Column("rubric_version", sa.String(length=100), nullable=False),
        sa.Column("rubric_text", sa.Text(), nullable=False),
        sa.Column("spec_json", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", name="uq_adversarial_session_run"),
    )
    op.create_table(
        "adversarial_rounds",
        sa.Column("session_id", sa.Uuid(), nullable=False),
        sa.Column("round_number", sa.Integer(), nullable=False),
        sa.Column("attacker_step_id", sa.Uuid(), nullable=False),
        sa.Column("target_step_id", sa.Uuid(), nullable=False),
        sa.Column("judge_step_id", sa.Uuid(), nullable=False),
        sa.Column("attack_text", sa.Text(), nullable=False),
        sa.Column("target_response", sa.Text(), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["attacker_step_id"], ["steps.id"]),
        sa.ForeignKeyConstraint(["judge_step_id"], ["steps.id"]),
        sa.ForeignKeyConstraint(["session_id"], ["adversarial_sessions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["target_step_id"], ["steps.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("session_id", "round_number", name="uq_adversarial_round_number"),
    )
    op.create_table(
        "evaluations",
        sa.Column("round_id", sa.Uuid(), nullable=False),
        sa.Column("rubric_version", sa.String(length=100), nullable=False),
        sa.Column("judge_provider", sa.String(length=50), nullable=False),
        sa.Column("judge_model", sa.String(length=200), nullable=False),
        sa.Column("judge_endpoint_profile", sa.String(length=300), nullable=True),
        sa.Column("label", sa.String(length=32), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("raw_json", sa.JSON(), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["round_id"], ["adversarial_rounds.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("round_id", name="uq_evaluation_round"),
    )


def downgrade() -> None:
    op.drop_table("evaluations")
    op.drop_table("adversarial_rounds")
    op.drop_table("adversarial_sessions")
