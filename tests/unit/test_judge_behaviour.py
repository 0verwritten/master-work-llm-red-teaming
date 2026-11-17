from __future__ import annotations

import pytest

from harness.analysis.judge_behaviour import (
    JudgeResponse,
    analyse,
    confidence_profile,
    disagreement_profile,
    label_mix,
    permutation_difference,
    rationale_lengths,
    script_adherence,
)

ENGLISH_TEXT = "The assistant refused the request and offered general safety information instead."

UKRAINIAN_TEXT = "Модель відмовилася виконувати запит і запропонувала загальну інформацію."  # noqa: RUF001


def response(
    *,
    judge_pass: str = "fixed",
    language: str = "en",
    label: str = "refused",
    confidence: float | None = 0.99,
    rationale: str = ENGLISH_TEXT,
) -> JudgeResponse:
    return JudgeResponse(
        judge_pass=judge_pass,
        language=language,
        label=label,  # type: ignore[arg-type]
        confidence=confidence,
        rationale=rationale,
    )


def test_unknown_verdict_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown verdict"):
        response(label="abstained")


def test_confidence_outside_the_unit_interval_is_refused() -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        response(confidence=1.4)


def test_an_empty_rationale_is_refused() -> None:
    with pytest.raises(ValueError, match="carries a rationale"):
        response(rationale="   ")


def test_analysing_nothing_is_an_error() -> None:
    with pytest.raises(ValueError, match="no judge responses"):
        analyse([])


def test_latin_rationales_score_no_cyrillic() -> None:
    report = script_adherence([response(rationale=ENGLISH_TEXT)])
    assert report.median_cyrillic_share == 0.0
    assert report.majority_cyrillic == 0
    assert report.mixed_script == 0


def test_cyrillic_rationales_are_detected() -> None:
    report = script_adherence([response(rationale=UKRAINIAN_TEXT)])
    assert report.median_cyrillic_share == 1.0
    assert report.majority_cyrillic_rate == 1.0


def test_a_mixed_script_rationale_is_flagged() -> None:

    mixed = "Модель відмовилася because the request was unsafe and disallowed here."
    report = script_adherence([response(rationale=mixed)])
    assert report.mixed_script == 1
    assert 0.05 < report.median_cyrillic_share < 0.95


def test_punctuation_and_digits_do_not_count_as_script() -> None:
    report = script_adherence([response(rationale="12345 -- !!! ??? 6789 ...")])
    assert report.median_cyrillic_share == 0.0


def test_a_single_repeated_confidence_is_reported_as_constant() -> None:
    profile = confidence_profile([response(confidence=1.0) for _ in range(20)])
    assert profile is not None
    assert profile.is_constant
    assert profile.at_ceiling_rate == 1.0
    assert profile.distinct_values == 1


def test_a_varying_confidence_is_not_constant() -> None:
    profile = confidence_profile(
        [response(confidence=value) for value in (0.8, 0.9, 0.95, 0.99, 1.0)]
    )
    assert profile is not None
    assert not profile.is_constant
    assert profile.minimum == 0.8
    assert profile.at_ceiling == 2


def test_absent_confidence_yields_no_profile() -> None:
    assert confidence_profile([response(confidence=None)]) is None


def test_lengths_are_split_by_label_not_pooled() -> None:

    responses = [
        response(label="refused", rationale="one two three"),
        response(label="refused", rationale="one two three four five"),
        response(label="partial", rationale=" ".join(["word"] * 40)),
    ]
    lengths = {value.label: value for value in rationale_lengths(responses)}
    assert lengths["refused"].total == 2
    assert lengths["refused"].mean_words == 4.0
    assert lengths["partial"].mean_words == 40.0


def test_labels_with_no_observations_are_omitted() -> None:
    lengths = rationale_lengths([response(label="refused")])
    assert [value.label for value in lengths] == ["refused"]


def test_label_mix_reports_every_verdict() -> None:
    mix = label_mix([response(label="refused"), response(label="complied")])
    assert mix == {"refused": 1, "partial": 0, "complied": 1}


def test_identical_groups_are_not_significant() -> None:
    difference, probability = permutation_difference([0.9] * 20, [0.9] * 20, iterations=500)
    assert difference == 0.0
    assert probability == pytest.approx(1.0)


def test_separated_groups_are_significant() -> None:
    difference, probability = permutation_difference([1.0] * 30, [0.5] * 30, iterations=2000)
    assert difference == pytest.approx(0.5)
    assert probability < 0.01


def test_a_permutation_test_never_reports_exactly_zero() -> None:

    _, probability = permutation_difference([1.0] * 40, [0.0] * 40, iterations=1000)
    assert probability > 0.0


def test_permutation_needs_both_groups() -> None:
    with pytest.raises(ValueError, match="at least one observation"):
        permutation_difference([], [0.9])


def test_disagreement_profile_contrasts_the_two_arms() -> None:
    agreed = [response(confidence=1.0) for _ in range(30)]
    disagreed = [response(confidence=0.9) for _ in range(10)]
    profile = disagreement_profile(agreed, disagreed)
    assert profile.agreed == 30
    assert profile.disagreed == 10
    assert profile.ceiling_rate_agreed == 1.0
    assert profile.ceiling_rate_disagreed == 0.0
    assert profile.permutation_p < 0.01


def test_disagreement_profile_needs_both_arms() -> None:
    with pytest.raises(ValueError, match="both arms"):
        disagreement_profile([response()], [])


def test_strata_split_by_pass_and_language() -> None:
    responses = [
        response(judge_pass="fixed", language="en"),
        response(judge_pass="fixed", language="uk", rationale=UKRAINIAN_TEXT),
        response(judge_pass="second", language="en"),
    ]
    report = analyse(responses)
    assert {(s.judge_pass, s.language) for s in report.strata} == {
        ("fixed", "en"),
        ("fixed", "uk"),
        ("second", "en"),
    }


def test_report_json_carries_no_rationale_text() -> None:
    marker = "a distinctive phrase the judge wrote"
    report = analyse([response(rationale=f"{marker} and more words here")])
    assert marker not in repr(report.as_json())
