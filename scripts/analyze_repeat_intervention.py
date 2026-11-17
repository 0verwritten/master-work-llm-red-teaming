"""Run the pre-specified ADR-0037 analysis over the frozen trace, zero inference.

Arms, all Ukrainian and all judged on RTX 6000Ada:

- control: the ADR-0035 510-cell arm (original PAIR protocol);
- intervention: the ADR-0037 repeat-rejection arm (510 cells);
- replication: the unchanged protocol rerun on the 120 human-verified pairs.

Cell identities are rebuilt with the launch code itself, so a session is analysed
only if it belongs to the exact frozen manifest. Aggregates only are printed; no
prompt, attack, response, or rationale leaves the database.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from run_vast_harmbench import build
from sqlalchemy import text

from harness.analysis.attacker_dynamics import RoundObservation, Verdict, analyse
from harness.analysis.repeat_intervention import (
    SentAttacks,
    SessionOutcome,
    manipulation_check,
    paired_contrast,
)
from harness.persistence.database import create_engine
from harness.settings import Settings

WORKSPACE = Path(".translation-workspace")
FULL_DATASET = WORKSPACE / "bilingual-dataset-v1-harmbench-full-510.json"
REVIEWED_DATASET = WORKSPACE / "bilingual-dataset-v1-harmbench-reviewed.json"
REPLICATION_NAMESPACE = "vast-harmbench-uk-qwen-h100-gptoss-rtx6000ada-v1-replication-2026-09"
FROZEN_MANIFESTS = {
    "control": "d1901e27d975365ae419b9838fede1e8ea5d2086d488d15a5c72ad0e1578a494",
    "intervention": "b4cb25499740d9bd860c7fe4fb95c636ab32d7233c34605c0963f5b6c40ba6f7",
}


@dataclass(frozen=True)
class Session:
    id: str
    run_id: str
    cell_id: str
    stable_id: str
    run_status: str
    stop_reason: str | None
    created_at: str


@dataclass(frozen=True)
class Round:
    session_id: str
    number: int
    attack_text: str
    verdict: str
    metadata: dict[str, Any]


def _arm_cells(
    *, dataset: Path, records: int, namespace: str | None, regenerations: int
) -> tuple[str, dict[str, str]]:
    arguments = argparse.Namespace(
        dataset=dataset,
        language="uk",
        records=records,
        rounds=3,
        seed=20260829,
        retry_latest_errors_from=None,
        cell_namespace=namespace,
        max_attack_regenerations=regenerations,
        judge_gpu="default",
        attacker="qwen",
        allow_ai_reviewed=True,
    )
    manifest, _ = build(arguments)
    return manifest.manifest_sha256, {cell.cell_id: cell.item_stable_id for cell in manifest.cells}


async def _rows(cell_ids: set[str]) -> tuple[list[Session], list[Round], dict[str, Counter[str]]]:
    engine = create_engine(Settings().database_url)
    try:
        async with engine.connect() as connection:
            session_rows = (
                await connection.execute(
                    text(
                        """
                        select s.id, s.run_id, s.spec_json->>'cell_id' as cell_id,
                               r.benchmark_item_id, r.status, r.stop_reason, s.created_at
                        from adversarial_sessions s
                        join runs r on r.id = s.run_id
                        where s.spec_json->>'cell_id' = any(:cell_ids)
                        """
                    ),
                    {"cell_ids": sorted(cell_ids)},
                )
            ).all()
            sessions = [
                Session(
                    id=str(row.id),
                    run_id=str(row.run_id),
                    cell_id=str(row.cell_id),
                    stable_id=str(row.benchmark_item_id),
                    run_status=str(row.status),
                    stop_reason=None if row.stop_reason is None else str(row.stop_reason),
                    created_at=row.created_at.isoformat(),
                )
                for row in session_rows
            ]
            session_ids = [session.id for session in sessions]
            round_rows = (
                await connection.execute(
                    text(
                        """
                        select ar.session_id, ar.round_number, ar.attack_text,
                               ar.candidate_metadata_json, e.label
                        from adversarial_rounds ar
                        join evaluations e on e.round_id = ar.id
                        where ar.session_id::text = any(:session_ids)
                        """
                    ),
                    {"session_ids": session_ids},
                )
            ).all()
            rounds = [
                Round(
                    session_id=str(row.session_id),
                    number=int(row.round_number),
                    attack_text=str(row.attack_text),
                    verdict=str(row.label),
                    metadata=dict(row.candidate_metadata_json or {}),
                )
                for row in round_rows
            ]
            call_rows = (
                await connection.execute(
                    text(
                        """
                        select run_id, status, count(*) as count
                        from llm_calls
                        where run_id::text = any(:run_ids)
                        group by run_id, status
                        """
                    ),
                    {"run_ids": sorted({session.run_id for session in sessions})},
                )
            ).all()
            calls: dict[str, Counter[str]] = defaultdict(Counter)
            for row in call_rows:
                calls[str(row.run_id)][str(row.status)] = int(row._mapping["count"])
            return sessions, rounds, calls
    finally:
        await engine.dispose()


def _latest(sessions: list[Session], cells: dict[str, str]) -> dict[str, Session]:

    latest: dict[str, Session] = {}
    for session in sessions:
        if session.cell_id not in cells:
            continue
        current = latest.get(session.cell_id)
        if current is None or session.created_at > current.created_at:
            latest[session.cell_id] = session
    return {cells[cell_id]: session for cell_id, session in latest.items()}


def _outcomes(
    arm: dict[str, Session], rounds_by_session: dict[str, list[Round]]
) -> dict[str, SessionOutcome]:
    return {
        stable_id: SessionOutcome(
            stable_id=stable_id,
            completed=session.run_status == "completed",
            validated_success=session.stop_reason == "validated_success",
            any_non_refusal=any(
                value.verdict in {"partial", "complied"} for value in rounds_by_session[session.id]
            ),
        )
        for stable_id, session in arm.items()
    }


def _sent(arm: dict[str, Session], rounds_by_session: dict[str, list[Round]]) -> list[SentAttacks]:
    result: list[SentAttacks] = []
    for stable_id, session in arm.items():
        if session.run_status != "completed":
            continue
        ordered = sorted(rounds_by_session[session.id], key=lambda value: value.number)
        result.append(
            SentAttacks(
                stable_id=stable_id,
                attacks=tuple(value.attack_text for value in ordered),
                rejections_per_round=tuple(
                    len(value.metadata.get("repeat_rejections", [])) for value in ordered
                ),
                unresolved_rounds=sum(
                    bool(value.metadata.get("repeat_unresolved")) for value in ordered
                ),
            )
        )
    return result


def _arm_summary(
    arm: dict[str, Session],
    rounds_by_session: dict[str, list[Round]],
    calls: dict[str, Counter[str]],
) -> dict[str, Any]:
    normal = [value for value in arm.values() if value.run_status == "completed"]
    call_status: Counter[str] = Counter()
    expected_success = 0
    for session in arm.values():
        call_status.update(calls.get(session.run_id, Counter()))
        for value in rounds_by_session[session.id]:
            expected_success += 3 + len(value.metadata.get("repeat_rejections", []))
    escalation = analyse(
        [
            RoundObservation(
                session_id=value.session_id,
                round_number=value.number,
                language="uk",
                attack_text=value.attack_text,
                verdict=cast(
                    Verdict | None,
                    value.verdict if value.verdict in {"refused", "partial", "complied"} else None,
                ),
            )
            for session in normal
            for value in rounds_by_session[session.id]
        ],
        rounds_required=3,
    ).as_json()["strata"]["uk"]["escalation"]
    return {
        "cells": len(arm),
        "normal_terminal": len(normal),
        "validated_success": sum(value.stop_reason == "validated_success" for value in normal),
        "stop_reasons": dict(
            sorted(Counter(str(value.stop_reason) for value in arm.values()).items())
        ),
        "llm_call_status": dict(sorted(call_status.items())),
        "trace_invariant": {
            "expected_successful_calls": expected_success,
            "observed_successful_calls": call_status["succeeded"],
            "holds": call_status["succeeded"] == expected_success,
        },
        "escalation_r1_r3": escalation,
    }


async def run(arguments: argparse.Namespace) -> dict[str, Any]:
    control_sha, control_cells = _arm_cells(
        dataset=arguments.full_dataset, records=510, namespace=None, regenerations=0
    )
    intervention_sha, intervention_cells = _arm_cells(
        dataset=arguments.full_dataset, records=510, namespace=None, regenerations=2
    )
    _, replication_cells = _arm_cells(
        dataset=arguments.reviewed_dataset,
        records=120,
        namespace=REPLICATION_NAMESPACE,
        regenerations=0,
    )
    if (control_sha, intervention_sha) != (
        FROZEN_MANIFESTS["control"],
        FROZEN_MANIFESTS["intervention"],
    ):
        raise RuntimeError("rebuilt manifests do not match ADR-0035/ADR-0037")

    sessions, rounds, calls = await _rows(
        set(control_cells) | set(intervention_cells) | set(replication_cells)
    )
    rounds_by_session: dict[str, list[Round]] = defaultdict(list)
    for value in rounds:
        rounds_by_session[value.session_id].append(value)
    control = _latest(sessions, control_cells)
    intervention = _latest(sessions, intervention_cells)
    replication = _latest(sessions, replication_cells)
    human_verified = set(replication_cells.values())

    control_outcomes = _outcomes(control, rounds_by_session)
    intervention_outcomes = _outcomes(intervention, rounds_by_session)
    replication_outcomes = _outcomes(replication, rounds_by_session)
    return {
        "analysis_version": "adr0037-repeat-intervention-v1",
        "manifests": {"control": control_sha, "intervention": intervention_sha},
        "coverage": {
            "control_cells_found": f"{len(control)}/{len(control_cells)}",
            "intervention_cells_found": f"{len(intervention)}/{len(intervention_cells)}",
            "replication_cells_found": f"{len(replication)}/{len(replication_cells)}",
        },
        "arms": {
            "control": _arm_summary(control, rounds_by_session, calls),
            "intervention": _arm_summary(intervention, rounds_by_session, calls),
            "replication": _arm_summary(replication, rounds_by_session, calls),
        },
        "1_manipulation_check": manipulation_check(
            _sent(control, rounds_by_session), _sent(intervention, rounds_by_session)
        ),
        "2_primary_intervention_minus_control": paired_contrast(
            intervention_outcomes, control_outcomes
        ),
        "3_gap_change": "not estimable: no English intervention arm was run (Option A)",
        "4_secondary": {
            "human_verified_intervention_minus_control": paired_contrast(
                intervention_outcomes, control_outcomes, stable_ids=human_verified
            ),
            "any_non_refusal_intervention_minus_control": paired_contrast(
                intervention_outcomes, control_outcomes, outcome="any_non_refusal"
            ),
        },
        "5_rerun_noise_replication_minus_control_120": paired_contrast(
            replication_outcomes, control_outcomes, stable_ids=human_verified
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-dataset", type=Path, default=FULL_DATASET)
    parser.add_argument("--reviewed-dataset", type=Path, default=REVIEWED_DATASET)
    arguments = parser.parse_args()
    for path in (arguments.full_dataset, arguments.reviewed_dataset):
        if not path.exists():
            parser.error(f"missing frozen input: {path}")
    print(json.dumps(asyncio.run(run(arguments)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
