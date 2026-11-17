from collections.abc import Sequence

from alembic import op

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_check_constraint(
        "ck_phase4_instrument_language",
        "phase4_instrument_items",
        "source_language IN ('en', 'uk')",
    )
    for name, column in (
        ("ck_phase4_instrument_response_sha256", "source_response_sha256"),
        ("ck_phase4_instrument_evaluation_sha256", "source_evaluation_sha256"),
        ("ck_phase4_instrument_bundle_sha256", "input_bundle_sha256"),
    ):
        op.create_check_constraint(
            name, "phase4_instrument_items", f"{column} ~ '^[0-9a-f]{{64}}$'"
        )
    op.create_check_constraint(
        "ck_phase4_self_judge_confidence",
        "phase4_self_judge_evaluations",
        "confidence >= 0 AND confidence <= 1",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_phase4_self_judge_confidence",
        "phase4_self_judge_evaluations",
        type_="check",
    )
    for name in (
        "ck_phase4_instrument_bundle_sha256",
        "ck_phase4_instrument_evaluation_sha256",
        "ck_phase4_instrument_response_sha256",
        "ck_phase4_instrument_language",
    ):
        op.drop_constraint(name, "phase4_instrument_items", type_="check")
