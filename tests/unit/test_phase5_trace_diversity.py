from __future__ import annotations

import json
import uuid

import pytest

from harness.analysis.phase5_trace_diversity import (
    TraceObservation,
    aggregate_trace_diversity,
)


def _run(
    *,
    language: str,
    attacker: str,
    category: str,
    attacks: tuple[str, str, str],
    responses: tuple[str, str, str],
    labels: tuple[str, str, str] = ("refused", "refused", "refused"),
) -> tuple[TraceObservation, ...]:
    run_id = uuid.uuid4()
    return tuple(
        TraceObservation(
            run_id=run_id,
            round_number=index,
            language=language,  # type: ignore[arg-type]
            attacker=attacker,
            category=category,
            method="pair",
            attack_text=attacks[index - 1],
            target_response=responses[index - 1],
            fixed_judge_label=labels[index - 1],  # type: ignore[arg-type]
        )
        for index in (1, 2, 3)
    )


def _observations() -> tuple[TraceObservation, ...]:
    return (
        *_run(
            language="en",
            attacker="qwen",
            category="a",
            attacks=("alpha beta", "alpha gamma", "alpha gamma"),
            responses=("safe one", "safe two", "safe two"),
        ),
        *_run(
            language="en",
            attacker="gemma",
            category="a",
            attacks=("delta", "delta epsilon", "zeta"),
            responses=("same", "same", "same"),
        ),
        *_run(
            language="en",
            attacker="qwen",
            category="b",
            attacks=("one", "two", "three"),
            responses=("r1", "r2", "r3"),
        ),
        *_run(
            language="uk",
            attacker="gemma",
            category="a",
            attacks=("один два", "один три", "чотири"),
            responses=("відмова", "відмова", "відмова"),
        ),
        *_run(
            language="uk",
            attacker="qwen",
            category="b",
            attacks=("п'ять", "шість", "сім"),
            responses=("відповідь один", "відповідь два", "відповідь три"),
            labels=("refused", "partial", "complied"),
        ),
        *_run(
            language="uk",
            attacker="qwen",
            category="b",
            attacks=("eight", "nine", "ten"),
            responses=("x", "y", "z"),
        ),
        *_run(
            language="uk",
            attacker="qwen",
            category="c",
            attacks=("red", "blue", "green"),
            responses=("p", "q", "r"),
        ),
    )


def test_trace_diversity_is_deterministic_aggregate_only_and_tracks_coverage() -> None:
    report = aggregate_trace_diversity(_observations())
    overall = report.strata["overall"]

    assert report.run_count == 7
    assert report.round_count == 21
    assert report.strata["en"].round_count == 9
    assert report.strata["uk"].round_count == 12
    assert set(report.strata) == {
        "overall",
        "en",
        "uk",
        "attacker:gemma",
        "attacker:qwen",
    }
    assert overall.attacks.observation_count == 21
    assert overall.attacks.exact_unique_count == 20
    assert overall.attacks.exact_duplicate_rate.numerator == 1
    assert overall.responses.exact_unique_count == 16
    assert overall.responses.exact_duplicate_rate.numerator == 5
    assert overall.attacks.total_tokens > 0
    assert 0 <= (overall.attacks.distinct_1 or 0) <= 1
    assert 0 <= (overall.attacks.distinct_2 or 0) <= 1
    assert 0 <= (overall.attacks.mean_pairwise_token_jaccard_distance or 0) <= 1

    assert [point.exact_new_within_run.numerator for point in overall.refinement] == [
        7,
        7,
        6,
    ]
    assert overall.exact_collapse.no_exact_collapse == 6
    assert overall.exact_collapse.partial_exact_collapse == 1
    assert overall.exact_collapse.complete_exact_collapse == 0
    assert [
        point.adjacent_exact_repeat.numerator if point.adjacent_exact_repeat else None
        for point in overall.refinement
    ] == [None, 0, 1]
    assert [point.mean_cumulative_unique_attacks for point in overall.refinement] == [
        1.0,
        2.0,
        pytest.approx(20 / 7),
    ]
    assert [point.observations for point in overall.coverage_curve] == [7, 14, 21]
    assert [point.fixed_judge_positive_rounds for point in overall.coverage_curve] == [
        0,
        0,
        1,
    ]
    final = overall.coverage_curve[-1]
    assert final.fixed_judge_positive_runs == 1
    assert final.unique_fixed_judge_positive_responses == 1
    assert final.fixed_judge_positive_categories == 1

    assert report.semantic_vulnerability_coverage_estimable is False
    assert report.vendi_score_estimable is False
    assert report.inferential_outputs_estimable is False
    assert report.comparative_method_outputs_estimable is False

    payload = json.dumps(report.model_dump(mode="json"), ensure_ascii=False)
    for private_text in ("alpha beta", "один два", "safe one", "відмова"):
        assert private_text not in payload


def test_trace_diversity_rejects_incomplete_or_comparative_inputs() -> None:
    with pytest.raises(ValueError, match="exactly 21 rounds"):
        aggregate_trace_diversity(_observations()[:-1])
    observations = list(_observations())
    final_run_id = observations[-1].run_id
    observations = [
        TraceObservation(**{**value.__dict__, "method": "static"})
        if value.run_id == final_run_id
        else value
        for value in observations
    ]
    with pytest.raises(ValueError, match="only PAIR"):
        aggregate_trace_diversity(tuple(observations))
