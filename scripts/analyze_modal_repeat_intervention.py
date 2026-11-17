"""ADR-0039 analysis: Modal control versus repeat-rejection intervention, zero inference.

Reads the local SQLite trace written by ``run_modal_repeat_intervention.py``
through the ORM, so it does not depend on PostgreSQL JSON operators. Cell
identities are rebuilt from the frozen manifests; only aggregates are printed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from analyze_repeat_intervention import Round, Session, _arm_summary, _latest, _outcomes, _sent
from run_modal_repeat_intervention import DATABASE_URL, DATASET, build
from sqlalchemy import func, select

from harness.analysis.repeat_intervention import manipulation_check, paired_contrast
from harness.persistence.database import create_engine, session_factory
from harness.persistence.models import (
    AdversarialRound,
    AdversarialSession,
    Evaluation,
    LLMCall,
    Run,
)


async def _rows(database_url: str) -> tuple[list[Session], list[Round], dict[str, Counter[str]]]:
    engine = create_engine(database_url)
    make_session = session_factory(engine)
    try:
        async with make_session() as db:
            session_rows = (
                await db.execute(
                    select(AdversarialSession, Run).join(Run, Run.id == AdversarialSession.run_id)
                )
            ).all()
            sessions = [
                Session(
                    id=str(adversarial.id),
                    run_id=str(run.id),
                    cell_id=str((adversarial.spec_json or {}).get("cell_id")),
                    stable_id=run.benchmark_item_id,
                    run_status=str(run.status),
                    stop_reason=None if run.stop_reason is None else str(run.stop_reason),
                    created_at=adversarial.created_at.isoformat(),
                )
                for adversarial, run in session_rows
            ]
            round_rows = (
                await db.execute(
                    select(AdversarialRound, Evaluation.label).join(
                        Evaluation, Evaluation.round_id == AdversarialRound.id
                    )
                )
            ).all()
            rounds = [
                Round(
                    session_id=str(value.session_id),
                    number=value.round_number,
                    attack_text=value.attack_text,
                    verdict=str(label),
                    metadata=dict(value.candidate_metadata_json or {}),
                )
                for value, label in round_rows
            ]
            call_rows = (
                await db.execute(
                    select(LLMCall.run_id, LLMCall.status, func.count()).group_by(
                        LLMCall.run_id, LLMCall.status
                    )
                )
            ).all()
            calls: dict[str, Counter[str]] = defaultdict(Counter)
            for run_id, status, count in call_rows:
                calls[str(run_id)][str(status)] = int(count)
            return sessions, rounds, calls
    finally:
        await engine.dispose()


async def run(arguments: argparse.Namespace) -> dict[str, Any]:
    manifests, _ = build(arguments.dataset)
    cells = {
        arm: {cell.cell_id: cell.item_stable_id for cell in manifest.cells}
        for arm, manifest in manifests.items()
    }
    sessions, rounds, calls = await _rows(arguments.database_url)
    rounds_by_session: dict[str, list[Round]] = defaultdict(list)
    for value in rounds:
        rounds_by_session[value.session_id].append(value)
    control = _latest(sessions, cells["control"])
    intervention = _latest(sessions, cells["intervention"])
    control_outcomes = _outcomes(control, rounds_by_session)
    intervention_outcomes = _outcomes(intervention, rounds_by_session)
    return {
        "analysis_version": "adr0039-modal-repeat-intervention-v1",
        "manifests": {arm: manifest.manifest_sha256 for arm, manifest in manifests.items()},
        "coverage": {
            "control_cells_found": f"{len(control)}/{len(cells['control'])}",
            "intervention_cells_found": f"{len(intervention)}/{len(cells['intervention'])}",
        },
        "arms": {
            "control": _arm_summary(control, rounds_by_session, calls),
            "intervention": _arm_summary(intervention, rounds_by_session, calls),
        },
        "1_manipulation_check": manipulation_check(
            _sent(control, rounds_by_session), _sent(intervention, rounds_by_session)
        ),
        "2_primary_intervention_minus_control": paired_contrast(
            intervention_outcomes, control_outcomes
        ),
        "3_gap_change": "not estimable: Ukrainian arms only (ADR-0039)",
        "4_secondary_any_non_refusal": paired_contrast(
            intervention_outcomes, control_outcomes, outcome="any_non_refusal"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--database-url", default=DATABASE_URL)
    arguments = parser.parse_args()
    print(json.dumps(asyncio.run(run(arguments)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
