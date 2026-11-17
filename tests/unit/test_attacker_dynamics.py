from __future__ import annotations

import pytest

from harness.analysis.attacker_dynamics import (
    RoundObservation,
    analyse,
    exact_mcnemar,
)


def observation(
    session: str,
    round_number: int,
    *,
    language: str = "en",
    attack: str = "attack text",
    verdict: str | None = "refused",
) -> RoundObservation:
    return RoundObservation(
        session_id=session,
        round_number=round_number,
        language=language,
        attack_text=attack,
        verdict=verdict,  # type: ignore[arg-type]
    )


def test_round_numbers_are_one_based() -> None:
    with pytest.raises(ValueError, match="one-based"):
        observation("s", 0)


def test_unknown_verdict_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown verdict"):
        observation("s", 1, verdict="abstained")


def test_a_single_round_protocol_has_no_dynamics_to_measure() -> None:
    with pytest.raises(ValueError, match="at least two rounds"):
        analyse([observation("s", 1)], rounds_required=1)


def test_identical_attacks_score_zero_novelty_and_count_as_repeats() -> None:
    report = analyse(
        [observation("s", n, attack="same words every time") for n in (1, 2, 3)],
        rounds_required=3,
    )
    novelty = report.strata["en"].novelty
    assert novelty.transitions == 2
    assert novelty.mean_similarity == 1.0
    assert novelty.exact_repeat_transitions == 2
    assert report.strata["en"].repetition.with_repeat == 1


def test_disjoint_attacks_score_full_novelty() -> None:
    report = analyse(
        [
            observation("s", 1, attack="alpha beta"),
            observation("s", 2, attack="gamma delta"),
            observation("s", 3, attack="epsilon zeta"),
        ],
        rounds_required=3,
    )
    novelty = report.strata["en"].novelty
    assert novelty.mean_similarity == 0.0
    assert novelty.exact_repeat_transitions == 0
    assert report.strata["en"].repetition.with_repeat == 0


def test_incomplete_sessions_are_excluded_from_escalation() -> None:

    rows = [
        observation("halted", 1, attack="a", verdict="complied"),
        *[observation("full", n, attack=f"a{n}") for n in (1, 2, 3)],
    ]
    escalation = analyse(rows, rounds_required=3).strata["en"].escalation
    assert escalation.complete_sessions == 1
    assert escalation.points[0].total == 1
    assert escalation.points[0].non_refusal_rate == 0.0


def test_escalation_is_measured_as_a_paired_contrast() -> None:
    rows = []
    for index in range(10):
        rows.append(observation(f"up{index}", 1, attack=f"x{index}", verdict="refused"))
        rows.append(observation(f"up{index}", 2, attack=f"y{index}", verdict="refused"))
        rows.append(observation(f"up{index}", 3, attack=f"z{index}", verdict="complied"))
    escalation = analyse(rows, rounds_required=3).strata["en"].escalation
    assert escalation.last_round_only == 10
    assert escalation.first_round_only == 0
    assert escalation.mcnemar_p == pytest.approx(2 / 1024)


def test_partial_and_complied_are_pooled_as_non_refusal() -> None:

    rows = [
        observation("s", 1, attack="a", verdict="refused"),
        observation("s", 2, attack="b", verdict="refused"),
        observation("s", 3, attack="c", verdict="partial"),
    ]
    escalation = analyse(rows, rounds_required=3).strata["en"].escalation
    assert escalation.last_round_only == 1
    assert escalation.points[2].non_refusal_rate == 1.0


def test_languages_are_stratified_independently() -> None:
    rows = [
        *[observation("e", n, language="en", attack=f"e{n}") for n in (1, 2, 3)],
        *[observation("u", n, language="uk", attack="same") for n in (1, 2, 3)],
    ]
    report = analyse(rows, rounds_required=3)
    assert set(report.strata) == {"en", "uk"}
    assert report.strata["en"].repetition.with_repeat == 0
    assert report.strata["uk"].repetition.with_repeat == 1


def test_exact_mcnemar_is_symmetric_and_bounded() -> None:
    assert exact_mcnemar(0, 0) == 1.0
    assert exact_mcnemar(5, 5) == 1.0
    assert exact_mcnemar(2, 17) == exact_mcnemar(17, 2)
    assert 0.0 < exact_mcnemar(2, 17) < 0.01


def test_report_json_carries_no_text() -> None:

    secret = "a very distinctive attack string"
    session_id = "session-identifier-that-must-not-leak"
    rows = [observation(session_id, n, attack=f"{secret} {n}") for n in (1, 2, 3)]
    payload = repr(analyse(rows, rounds_required=3).as_json())
    assert secret not in payload
    assert session_id not in payload
