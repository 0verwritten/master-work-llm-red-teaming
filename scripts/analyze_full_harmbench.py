"""Analyse the frozen 510-pair Vast HarmBench trace without new inference.

The script joins content-addressed checkpoint cells to PostgreSQL sessions,
substitutes the three explicitly authorised ADR-0036 recovery sessions, and
prints aggregates only. Raw prompts, attacks, responses, and rationales never
leave the database.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from sqlalchemy import text

from harness.analysis.attacker_dynamics import RoundObservation, Verdict, analyse
from harness.analysis.full_harmbench import (
    exact_mcnemar,
    fisher_exact_two_sided,
    holm_adjust,
    paired_risk_difference_interval,
    rate,
    wilson_interval,
)
from harness.bilingual_dataset import DatasetItem
from harness.experiments.bilingual_manifest import build_manifest
from harness.experiments.checkpoint import CheckpointLog
from harness.experiments.vast_harmbench_profiles import (
    VAST_GPT_OSS_L40S_FROZEN,
    VAST_GPT_OSS_RTX6000_ADA_FROZEN,
    VAST_QWEN_H100_FROZEN,
)
from harness.persistence.database import create_engine
from harness.settings import Settings

WORKSPACE = Path(".translation-workspace")
DEFAULT_DATASET = WORKSPACE / "bilingual-dataset-v1-harmbench-full-510.json"
DEFAULT_EN_CHECKPOINT = WORKSPACE / "vast-harmbench-en-100-checkpoint.jsonl"
DEFAULT_UK_CHECKPOINT = WORKSPACE / "vast-harmbench-uk-100-checkpoint.jsonl"
DEFAULT_RECOVERY_CHECKPOINT = WORKSPACE / "vast-harmbench-en-rtx6000ada-recovery-checkpoint.jsonl"
PRIMARY_EN_NAMESPACE = "vast-harmbench-en-qwen-h100-gptoss-l40s-v1"
PRIMARY_UK_NAMESPACE = "vast-harmbench-uk-qwen-h100-gptoss-rtx6000ada-v1"


@dataclass(frozen=True)
class Session:
    id: str
    run_id: str
    cell_id: str
    stable_id: str
    language: str
    session_status: str
    run_status: str
    stop_reason: str | None
    created_at: str


@dataclass(frozen=True)
class Round:
    session_id: str
    number: int
    attack_text: str
    verdict: str
    adherence: str | None


def _load_dataset(path: Path) -> tuple[list[DatasetItem], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = [
        DatasetItem(**{key: row[key] for key in DatasetItem.__dataclass_fields__})
        for row in payload["items"]
    ]
    return items, payload["manifest"]


def _manifest_map(
    items: list[DatasetItem], dataset_manifest: dict[str, Any], language: str
) -> dict[str, DatasetItem]:
    selected = [
        item for item in items if item.language == language and item.stratum == "harmbench_subset"
    ]
    manifest = build_manifest(
        selected,
        dataset_sha256=dataset_manifest["dataset_sha256"],
        subset_plan_sha256=dataset_manifest["subset_plan_sha256"],
        attacker=VAST_QWEN_H100_FROZEN,
        target=VAST_QWEN_H100_FROZEN,
        judge=(VAST_GPT_OSS_L40S_FROZEN if language == "en" else VAST_GPT_OSS_RTX6000_ADA_FROZEN),
        max_rounds=3,
        allow_assumed=language == "uk",
        enforce_judge_language=True,
        cell_namespace=PRIMARY_EN_NAMESPACE if language == "en" else PRIMARY_UK_NAMESPACE,
    )
    by_stable_id = {item.stable_id: item for item in selected}
    return {cell.cell_id: by_stable_id[cell.item_stable_id] for cell in manifest.cells}


def _latest_checkpoint_ids(path: Path) -> set[str]:
    latest = {record.cell_id: record for record in CheckpointLog(path).records()}
    return set(latest)


async def _database_rows() -> tuple[list[Session], list[Round], dict[str, Counter[str]]]:
    engine = create_engine(Settings().database_url)
    try:
        async with engine.connect() as connection:
            session_rows = (
                await connection.execute(
                    text(
                        """
                        select s.id, s.run_id, s.spec_json->>'cell_id' as cell_id,
                               r.benchmark_item_id, s.language, s.status as session_status,
                               r.status as run_status, r.stop_reason, s.created_at
                        from adversarial_sessions s
                        join runs r on r.id = s.run_id
                        where s.spec_json->>'cell_id' is not null
                        """
                    )
                )
            ).all()
            sessions = [
                Session(
                    id=str(row.id),
                    run_id=str(row.run_id),
                    cell_id=str(row.cell_id),
                    stable_id=str(row.benchmark_item_id),
                    language=str(row.language),
                    session_status=str(row.session_status),
                    run_status=str(row.run_status),
                    stop_reason=str(row.stop_reason) if row.stop_reason is not None else None,
                    created_at=row.created_at.isoformat(),
                )
                for row in session_rows
            ]
            round_rows = (
                await connection.execute(
                    text(
                        """
                        select ar.session_id, ar.round_number, ar.attack_text,
                               e.label, e.raw_json->>'judge_language_adherence' as adherence
                        from adversarial_rounds ar
                        join evaluations e on e.round_id = ar.id
                        join adversarial_sessions s on s.id = ar.session_id
                        where s.spec_json->>'cell_id' is not null
                        """
                    )
                )
            ).all()
            rounds = [
                Round(
                    session_id=str(row.session_id),
                    number=int(row.round_number),
                    attack_text=str(row.attack_text),
                    verdict=str(row.label),
                    adherence=str(row.adherence) if row.adherence is not None else None,
                )
                for row in round_rows
            ]
            call_rows = (
                await connection.execute(
                    text(
                        """
                        select run_id, status, count(*) as count
                        from llm_calls
                        group by run_id, status
                        """
                    )
                )
            ).all()
            calls: dict[str, Counter[str]] = defaultdict(Counter)
            for row in call_rows:
                calls[str(row.run_id)][str(row.status)] = int(row._mapping["count"])
            return sessions, rounds, calls
    finally:
        await engine.dispose()


def _rounded_interval(successes: int, total: int) -> list[float]:
    return [round(value, 6) for value in wilson_interval(successes, total)]


def _arm_summary(sessions: list[Session], rounds: list[Round]) -> dict[str, Any]:
    session_ids = {session.id for session in sessions}
    relevant_rounds = [value for value in rounds if value.session_id in session_ids]
    normal = [session for session in sessions if session.run_status == "completed"]
    success = [session for session in normal if session.stop_reason == "validated_success"]
    adherence = Counter(value.adherence or "missing" for value in relevant_rounds)
    labels = Counter(value.verdict for value in relevant_rounds)
    result: dict[str, Any] = {
        "cells": len(sessions),
        "normal_terminal": len(normal),
        "indeterminate_terminal": len(sessions) - len(normal),
        "validated_success": len(success),
        "observed_asr": rate(len(success), len(normal)),
        "observed_asr_wilson_95": _rounded_interval(len(success), len(normal)),
        "asr_identification_bounds_all_cells": [
            round(len(success) / len(sessions), 6),
            round((len(success) + len(sessions) - len(normal)) / len(sessions), 6),
        ],
        "stop_reasons": dict(sorted(Counter(value.stop_reason for value in sessions).items())),
        "rounds": len(relevant_rounds),
        "round_labels": dict(sorted(labels.items())),
        "round_adherence": dict(sorted(adherence.items())),
    }
    return result


def _paired_summary(
    by_pair: dict[str, dict[str, Session]], *, review_by_pair: dict[str, str]
) -> dict[str, Any]:
    normal_pairs = {
        pair_id: arms
        for pair_id, arms in by_pair.items()
        if set(arms) == {"en", "uk"}
        and all(value.run_status == "completed" for value in arms.values())
    }
    outcomes = [
        (
            normal_pairs[pair_id]["en"].stop_reason == "validated_success",
            normal_pairs[pair_id]["uk"].stop_reason == "validated_success",
        )
        for pair_id in sorted(normal_pairs)
    ]
    en_only = sum(en and not uk for en, uk in outcomes)
    uk_only = sum(uk and not en for en, uk in outcomes)
    both = sum(en and uk for en, uk in outcomes)
    neither = sum(not en and not uk for en, uk in outcomes)
    en_success = en_only + both
    uk_success = uk_only + both
    difference = (en_success - uk_success) / len(outcomes)
    low, high = paired_risk_difference_interval(outcomes)

    strata: dict[str, Any] = {}
    for status in sorted(set(review_by_pair.values())):
        selected = [
            pair_id for pair_id in sorted(normal_pairs) if review_by_pair[pair_id] == status
        ]
        selected_outcomes = [
            (
                normal_pairs[pair_id]["en"].stop_reason == "validated_success",
                normal_pairs[pair_id]["uk"].stop_reason == "validated_success",
            )
            for pair_id in selected
        ]
        status_en_only = sum(en and not uk for en, uk in selected_outcomes)
        status_uk_only = sum(uk and not en for en, uk in selected_outcomes)
        strata[status] = {
            "complete_pairs": len(selected_outcomes),
            "en_success": sum(en for en, _ in selected_outcomes),
            "uk_success": sum(uk for _, uk in selected_outcomes),
            "en_only": status_en_only,
            "uk_only": status_uk_only,
            "risk_difference_en_minus_uk": round(
                sum(int(en) - int(uk) for en, uk in selected_outcomes) / len(selected_outcomes),
                6,
            ),
            "mcnemar_p": exact_mcnemar(status_en_only, status_uk_only),
        }

    return {
        "complete_pairs": len(outcomes),
        "excluded_pairs_with_indeterminate_arm": len(by_pair) - len(outcomes),
        "en_success": en_success,
        "uk_success": uk_success,
        "en_only": en_only,
        "uk_only": uk_only,
        "both": both,
        "neither": neither,
        "risk_difference_en_minus_uk": round(difference, 6),
        "paired_bootstrap_95": [round(low, 6), round(high, 6)],
        "mcnemar_p": exact_mcnemar(en_only, uk_only),
        "review_strata": strata,
    }


def _category_summary(
    by_pair: dict[str, dict[str, Session]], category_by_pair: dict[str, str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p_values: list[float] = []
    for category in sorted(set(category_by_pair.values())):
        pairs = [
            by_pair[pair_id]
            for pair_id in sorted(by_pair)
            if (
                category_by_pair[pair_id] == category
                and set(by_pair[pair_id]) == {"en", "uk"}
                and all(value.run_status == "completed" for value in by_pair[pair_id].values())
            )
        ]
        outcomes = [
            (
                arms["en"].stop_reason == "validated_success",
                arms["uk"].stop_reason == "validated_success",
            )
            for arms in pairs
        ]
        en_success = sum(en for en, _ in outcomes)
        uk_success = sum(uk for _, uk in outcomes)
        en_only = sum(en and not uk for en, uk in outcomes)
        uk_only = sum(uk and not en for en, uk in outcomes)
        interval_low, interval_high = paired_risk_difference_interval(outcomes)
        p_value = exact_mcnemar(en_only, uk_only)
        p_values.append(p_value)
        rows.append(
            {
                "category": category,
                "complete_pairs": len(outcomes),
                "en_success": en_success,
                "uk_success": uk_success,
                "en_asr": rate(en_success, len(outcomes)),
                "uk_asr": rate(uk_success, len(outcomes)),
                "difference": rate(en_success - uk_success, len(outcomes)),
                "paired_bootstrap_95": [
                    round(interval_low, 6),
                    round(interval_high, 6),
                ],
                "en_only": en_only,
                "uk_only": uk_only,
                "mcnemar_p_unadjusted": p_value,
            }
        )
    for row, adjusted in zip(rows, holm_adjust(p_values), strict=True):
        row["mcnemar_p_holm"] = adjusted
    return rows


def _paired_metric_summary(
    by_pair: dict[str, dict[str, Session]], outcome_by_session: dict[str, bool]
) -> dict[str, Any]:
    pairs = [
        by_pair[pair_id]
        for pair_id in sorted(by_pair)
        if set(by_pair[pair_id]) == {"en", "uk"}
        and all(value.run_status == "completed" for value in by_pair[pair_id].values())
    ]
    outcomes = [
        (outcome_by_session[arms["en"].id], outcome_by_session[arms["uk"].id]) for arms in pairs
    ]
    en_only = sum(en and not uk for en, uk in outcomes)
    uk_only = sum(uk and not en for en, uk in outcomes)
    both = sum(en and uk for en, uk in outcomes)
    low, high = paired_risk_difference_interval(outcomes)
    return {
        "complete_pairs": len(outcomes),
        "en_positive": en_only + both,
        "uk_positive": uk_only + both,
        "en_only": en_only,
        "uk_only": uk_only,
        "both": both,
        "neither": len(outcomes) - en_only - uk_only - both,
        "risk_difference_en_minus_uk": round((en_only - uk_only) / len(outcomes), 6),
        "paired_bootstrap_95": [round(low, 6), round(high, 6)],
        "mcnemar_p": exact_mcnemar(en_only, uk_only),
    }


async def run(arguments: argparse.Namespace) -> dict[str, Any]:
    items, dataset_manifest = _load_dataset(arguments.dataset)
    stable_items = {item.stable_id: item for item in items}
    en_cells = _manifest_map(items, dataset_manifest, "en")
    uk_cells = _manifest_map(items, dataset_manifest, "uk")
    en_checkpoint_ids = _latest_checkpoint_ids(arguments.en_checkpoint)
    uk_checkpoint_ids = _latest_checkpoint_ids(arguments.uk_checkpoint)
    recovery_ids = _latest_checkpoint_ids(arguments.recovery_checkpoint)
    if en_checkpoint_ids != set(en_cells) or uk_checkpoint_ids != set(uk_cells):
        raise RuntimeError("checkpoint cell identities do not match the frozen 510 manifests")
    if len(recovery_ids) != 3:
        raise RuntimeError("ADR-0036 recovery checkpoint must contain exactly three cells")

    sessions, rounds, calls = await _database_rows()
    latest: dict[str, Session] = {}
    for session in sessions:
        if session.cell_id not in latest or session.created_at > latest[session.cell_id].created_at:
            latest[session.cell_id] = session
    missing = (en_checkpoint_ids | uk_checkpoint_ids | recovery_ids) - set(latest)
    if missing:
        raise RuntimeError(f"database is missing {len(missing)} checkpoint cells")

    selected_en = {cell_id: latest[cell_id] for cell_id in en_checkpoint_ids}
    selected_uk = {cell_id: latest[cell_id] for cell_id in uk_checkpoint_ids}
    recovery_by_stable_id = {latest[cell_id].stable_id: latest[cell_id] for cell_id in recovery_ids}
    substituted_stable_ids: list[str] = []
    for cell_id, session in tuple(selected_en.items()):
        if session.run_status != "completed" and session.stable_id in recovery_by_stable_id:
            selected_en[cell_id] = recovery_by_stable_id[session.stable_id]
            substituted_stable_ids.append(session.stable_id)
    if len(substituted_stable_ids) != 3:
        raise RuntimeError("exactly three failed English cells must be replaced by recovery runs")

    selected_sessions = list(selected_en.values()) + list(selected_uk.values())
    if len({value.stable_id for value in selected_en.values()}) != 510:
        raise RuntimeError("English selection is not one result per source item")
    if len({value.stable_id for value in selected_uk.values()}) != 510:
        raise RuntimeError("Ukrainian selection is not one result per source item")

    pair_by_stable = {stable_id: item.pair_id for stable_id, item in stable_items.items()}
    by_pair: dict[str, dict[str, Session]] = defaultdict(dict)
    for session in selected_sessions:
        by_pair[pair_by_stable[session.stable_id]][session.language] = session
    if len(by_pair) != 510 or any(set(value) != {"en", "uk"} for value in by_pair.values()):
        raise RuntimeError("selected sessions do not form 510 bilingual pairs")

    uk_items_by_pair = {item.pair_id: item for item in items if item.language == "uk"}
    review_by_pair = {pair_id: item.reviewer_status for pair_id, item in uk_items_by_pair.items()}
    category_by_pair = {pair_id: item.category for pair_id, item in uk_items_by_pair.items()}
    chosen_session_ids = {session.id for session in selected_sessions}
    chosen_rounds = [value for value in rounds if value.session_id in chosen_session_ids]
    language_by_session = {session.id: session.language for session in selected_sessions}
    normal_session_ids = {
        session.id for session in selected_sessions if session.run_status == "completed"
    }
    dynamics_rounds = [value for value in chosen_rounds if value.session_id in normal_session_ids]
    rounds_by_session: dict[str, list[Round]] = defaultdict(list)
    for value in chosen_rounds:
        rounds_by_session[value.session_id].append(value)
    permissive_outcome = {
        session.id: any(
            value.verdict in {"partial", "complied"} for value in rounds_by_session[session.id]
        )
        for session in selected_sessions
    }
    dynamics = analyse(
        [
            RoundObservation(
                session_id=value.session_id,
                round_number=value.number,
                language=language_by_session[value.session_id],
                attack_text=value.attack_text,
                verdict=cast(
                    Verdict | None,
                    value.verdict if value.verdict in {"refused", "partial", "complied"} else None,
                ),
            )
            for value in dynamics_rounds
        ],
        rounds_required=3,
    ).as_json()
    en_repeat = dynamics["strata"]["en"]["repetition"]
    uk_repeat = dynamics["strata"]["uk"]["repetition"]
    repeat_fisher = fisher_exact_two_sided(
        en_repeat["with_repeat"],
        en_repeat["sessions"] - en_repeat["with_repeat"],
        uk_repeat["with_repeat"],
        uk_repeat["sessions"] - uk_repeat["with_repeat"],
    )
    en_points = dynamics["strata"]["en"]["escalation"]["points"]
    uk_points = dynamics["strata"]["uk"]["escalation"]["points"]

    def point_fisher(index: int) -> float:
        en_point = en_points[index]
        uk_point = uk_points[index]
        en_non_refusal = en_point["total"] - en_point["verdicts"]["refused"]
        uk_non_refusal = uk_point["total"] - uk_point["verdicts"]["refused"]
        return fisher_exact_two_sided(
            en_non_refusal,
            en_point["verdicts"]["refused"],
            uk_non_refusal,
            uk_point["verdicts"]["refused"],
        )

    selected_run_ids = {session.run_id for session in selected_sessions}
    call_status: Counter[str] = Counter()
    for run_id in selected_run_ids:
        call_status.update(calls.get(run_id, Counter()))

    primary_en_sessions = [latest[cell_id] for cell_id in en_checkpoint_ids]
    primary_call_status: Counter[str] = Counter()
    for session in primary_en_sessions:
        primary_call_status.update(calls.get(session.run_id, Counter()))
    primary_by_pair: dict[str, dict[str, Session]] = defaultdict(dict)
    for session in primary_en_sessions + list(selected_uk.values()):
        primary_by_pair[pair_by_stable[session.stable_id]][session.language] = session
    paired = _paired_summary(by_pair, review_by_pair=review_by_pair)
    review = paired["review_strata"]
    review_interaction_p = fisher_exact_two_sided(
        review["human_verified"]["en_only"],
        review["human_verified"]["uk_only"],
        review["ai_reviewed_exploratory"]["en_only"],
        review["ai_reviewed_exploratory"]["uk_only"],
    )
    return {
        "analysis_version": "full-harmbench-paired-v1",
        "dataset_sha256": dataset_manifest["dataset_sha256"],
        "selection": {
            "paired_items": len(by_pair),
            "english_recovery_substitutions": len(substituted_stable_ids),
            "ukrainian_review_status": dict(sorted(Counter(review_by_pair.values()).items())),
        },
        "arms_with_recovery_substitution": {
            "en": _arm_summary(list(selected_en.values()), chosen_rounds),
            "uk": _arm_summary(list(selected_uk.values()), chosen_rounds),
        },
        "english_primary_homogeneous_arm": _arm_summary(primary_en_sessions, rounds),
        "paired_complete_case": paired,
        "paired_permissive_any_non_refusal": _paired_metric_summary(by_pair, permissive_outcome),
        "review_stratum_direction_interaction_fisher_p": review_interaction_p,
        "paired_primary_homogeneous_complete_case": _paired_summary(
            primary_by_pair, review_by_pair=review_by_pair
        ),
        "categories_complete_case": _category_summary(by_pair, category_by_pair),
        "attacker_dynamics": dynamics,
        "attacker_repetition_fisher_p": repeat_fisher,
        "attacker_complete_session_language_fisher_p": {
            "round_1": point_fisher(0),
            "round_3": point_fisher(2),
        },
        "selected_llm_call_status": dict(sorted(call_status.items())),
        "selected_trace_invariant": {
            "evaluated_rounds": len(chosen_rounds),
            "expected_three_successful_calls_per_round": len(chosen_rounds) * 3,
            "observed_successful_calls": call_status["succeeded"],
            "holds": call_status["succeeded"] == len(chosen_rounds) * 3
            and sum(call_status.values()) == call_status["succeeded"],
        },
        "english_primary_llm_call_status": dict(sorted(primary_call_status.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--en-checkpoint", type=Path, default=DEFAULT_EN_CHECKPOINT)
    parser.add_argument("--uk-checkpoint", type=Path, default=DEFAULT_UK_CHECKPOINT)
    parser.add_argument("--recovery-checkpoint", type=Path, default=DEFAULT_RECOVERY_CHECKPOINT)
    arguments = parser.parse_args()
    for path in (
        arguments.dataset,
        arguments.en_checkpoint,
        arguments.uk_checkpoint,
        arguments.recovery_checkpoint,
    ):
        if not path.exists():
            parser.error(f"missing frozen input: {path}")
    print(json.dumps(asyncio.run(run(arguments)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
