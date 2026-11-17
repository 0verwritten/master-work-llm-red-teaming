from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column("adversarial_rounds", "attacker_step_id", nullable=True)
    op.add_column(
        "adversarial_rounds",
        sa.Column(
            "candidate_metadata_json",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
    )
    op.alter_column("adversarial_rounds", "candidate_metadata_json", server_default=None)


def downgrade() -> None:
    op.drop_column("adversarial_rounds", "candidate_metadata_json")
    op.alter_column("adversarial_rounds", "attacker_step_id", nullable=False)
