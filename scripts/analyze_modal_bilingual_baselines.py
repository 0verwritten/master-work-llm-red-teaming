"""Zero-inference analysis for the pre-specified ADR-0040 comparison."""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from analyze_modal_repeat_intervention import _rows
from analyze_repeat_intervention import Round, Session, _latest, _outcomes, _sent
from run_modal_bilingual_baselines import DATABASE_URL, DATASET, METHODS, build

from harness.analysis.full_harmbench import holm_adjust, rate
from harness.analysis.repeat_intervention import (
    SentAttacks,
    SessionOutcome,
    paired_contrast,
    repetition_summary,
)

TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)


def _tokens(value: str) -> set[str]:
    return set(TOKEN_PATTERN.findall(unicodedata.normalize("NFC", value).casefold()))


def _mean_adjacent_jaccard(sessions: list[SentAttacks]) -> float | None:
    values: list[float] = []
    for session in sessions:
        tokens = [_tokens(value) for value in session.attacks]
        for left, right in itertools.pairwise(tokens):
            union = left | right
            values.append(len(left & right) / len(union) if union else 0.0)
    return sum(values) / len(values) if values else None


def _call_summary(
    arm: dict[str, Session],
    rounds_by_session: dict[str, list[Round]],
    calls: dict[str, Counter[str]],
    *,
    local_method: bool,
) -> dict[str, Any]:
    statuses: Counter[str] = Counter()
    interrupted: Counter[str] = Counter()
    expected = 0
    for session in arm.values():
        if session.run_status not in {"completed", "budget_stopped"}:
            interrupted.update(calls.get(session.run_id, Counter()))
            continue
        statuses.update(calls.get(session.run_id, Counter()))
        for value in rounds_by_session[session.id]:
            expected += (2 if local_method else 3) + len(
                value.metadata.get("repeat_rejections", [])
            )
    return {
        "status": dict(sorted(statuses.items())),
        "expected_successful": expected,
        "observed_successful": statuses["succeeded"],
        "failed_requests": statuses["failed"],
        "unfinished_requests": statuses["started"],
        "requests_from_interrupted_cells": dict(sorted(interrupted.items())),
        "holds": expected == statuses["succeeded"] and statuses["started"] == 0,
    }


def _summary(
    arm: dict[str, Session],
    rounds_by_session: dict[str, list[Round]],
    calls: dict[str, Counter[str]],
    *,
    local_method: bool,
) -> dict[str, Any]:
    normal = [value for value in arm.values() if value.run_status == "completed"]
    successes = [value for value in normal if value.stop_reason == "validated_success"]
    first_success_rounds: Counter[int] = Counter()
    for session in successes:
        ordered = sorted(rounds_by_session[session.id], key=lambda value: value.number)
        first = next(value.number for value in ordered if value.verdict == "complied")
        first_success_rounds[first] += 1
    sent = _sent(arm, rounds_by_session)
    unique_candidates = [len(set(value.attacks)) for value in sent]
    return {
        "cells": len(arm),
        "normal_terminal": len(normal),
        "indeterminate_terminal": len(arm) - len(normal),
        "validated_success": len(successes),
        "asr": rate(len(successes), len(normal)),
        "stop_reasons": dict(
            sorted(Counter(str(value.stop_reason) for value in arm.values()).items())
        ),
        "first_success_round": dict(sorted(first_success_rounds.items())),
        "attacker": {
            **repetition_summary(sent),
            "mean_unique_candidates": rate(sum(unique_candidates), len(unique_candidates)),
            "mean_adjacent_jaccard_similarity": _mean_adjacent_jaccard(sent),
        },
        "llm_calls": _call_summary(arm, rounds_by_session, calls, local_method=local_method),
    }


def _initial_only(
    pair: dict[str, Session], rounds_by_session: dict[str, list[Round]]
) -> dict[str, SessionOutcome]:
    result: dict[str, SessionOutcome] = {}
    for stable_id, session in pair.items():
        first = min(rounds_by_session[session.id], key=lambda value: value.number, default=None)
        result[stable_id] = SessionOutcome(
            stable_id=stable_id,
            completed=first is not None,
            validated_success=first is not None and first.verdict == "complied",
            any_non_refusal=first is not None and first.verdict in {"partial", "complied"},
        )
    return result


async def run(arguments: argparse.Namespace) -> dict[str, Any]:
    manifests, selected = build(arguments.dataset)
    cells = {
        key: {cell.cell_id: cell.item_stable_id for cell in manifest.cells}
        for key, manifest in manifests.items()
    }
    sessions, rounds, calls = await _rows(arguments.database_url)
    rounds_by_session: dict[str, list[Round]] = defaultdict(list)
    for value in rounds:
        rounds_by_session[value.session_id].append(value)
    arms = {key: _latest(sessions, values) for key, values in cells.items()}
    outcomes = {key: _outcomes(arm, rounds_by_session) for key, arm in arms.items()}
    summaries = {
        key: _summary(
            arm,
            rounds_by_session,
            calls,
            local_method=METHODS[key.split("-", 1)[1]][0].value in {"static", "random_mutation"},
        )
        for key, arm in arms.items()
    }

    comparisons: dict[str, dict[str, Any]] = {}
    for language in ("en", "uk"):
        control = outcomes[f"{language}-pair"]
        language_results: dict[str, Any] = {}
        p_values: list[float] = []
        comparator_names = [name for name in METHODS if name != "pair"]
        for name in comparator_names:
            contrast = paired_contrast(outcomes[f"{language}-{name}"], control)
            language_results[name] = contrast
            p_values.append(float(contrast["mcnemar_p"]))
        for name, adjusted in zip(comparator_names, holm_adjust(p_values), strict=True):
            language_results[name]["mcnemar_p_holm"] = adjusted
        language_results["initial_only_from_pair"] = paired_contrast(
            _initial_only(arms[f"{language}-pair"], rounds_by_session), control
        )
        comparisons[language] = language_results

    stable_to_pair = {
        item.stable_id: item.pair_id for values in selected.values() for item in values
    }
    language_contrasts: dict[str, Any] = {}
    for name in METHODS:
        en = {stable_to_pair[key]: value for key, value in outcomes[f"en-{name}"].items()}
        uk = {stable_to_pair[key]: value for key, value in outcomes[f"uk-{name}"].items()}
        language_contrasts[name] = paired_contrast(en, uk)

    return {
        "analysis_version": "adr0040-modal-bilingual-baselines-v1",
        "manifests": {key: manifest.manifest_sha256 for key, manifest in manifests.items()},
        "coverage": {key: f"{len(arms[key])}/{len(cells[key])}" for key in sorted(cells)},
        "arms": summaries,
        "baseline_minus_pair_within_language": comparisons,
        "english_minus_ukrainian_within_method": language_contrasts,
        "claim_boundary": (
            "fixed automated judge; single seed; lexical candidate diagnostics; "
            "no human accuracy or semantic-coverage claim"
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
