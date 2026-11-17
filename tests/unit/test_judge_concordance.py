from __future__ import annotations

import pytest

from harness.analysis.judge_concordance import (
    agreement,
    analyse,
    collapse_to_binary,
)

ENGLISH = [[128, 0, 10], [19, 5, 5], [2, 0, 11]]
UKRAINIAN = [[97, 0, 15], [2, 0, 3], [1, 2, 9]]


def test_perfect_agreement_is_kappa_one() -> None:
    observed, _, kappa = agreement([[10, 0], [0, 10]])
    assert observed == 1.0
    assert kappa == 1.0


def test_chance_level_agreement_is_kappa_zero() -> None:
    _, _, kappa = agreement([[25, 25], [25, 25]])
    assert kappa == pytest.approx(0.0)


def test_systematic_disagreement_is_negative() -> None:
    _, _, kappa = agreement([[0, 20], [20, 0]])
    assert kappa < 0


def test_a_single_used_category_has_no_defined_kappa() -> None:
    with pytest.raises(ValueError, match="undefined"):
        agreement([[10, 0], [0, 0]])


def test_matrix_shape_is_enforced() -> None:
    with pytest.raises(ValueError, match="square"):
        agreement([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError, match="negative"):
        agreement([[1, -1], [0, 1]])
    with pytest.raises(ValueError, match="empty"):
        agreement([[0, 0], [0, 0]])


def test_high_raw_agreement_can_hide_moderate_kappa() -> None:

    observed, expected, kappa = agreement(ENGLISH)
    assert observed == pytest.approx(0.800, abs=0.001)
    assert expected == pytest.approx(0.650, abs=0.001)
    assert kappa == pytest.approx(0.429, abs=0.001)


def test_both_languages_land_in_the_same_moderate_band() -> None:
    _, _, english = agreement(ENGLISH)
    _, _, ukrainian = agreement(UKRAINIAN)
    assert english == pytest.approx(0.429, abs=0.001)
    assert ukrainian == pytest.approx(0.419, abs=0.001)
    assert abs(english - ukrainian) < 0.02


def test_collapsing_preserves_the_total() -> None:
    for matrix in (ENGLISH, UKRAINIAN):
        collapsed = collapse_to_binary(matrix)
        assert sum(sum(row) for row in collapsed) == sum(sum(row) for row in matrix)


def test_collapsing_merges_partial_into_complied() -> None:
    collapsed = collapse_to_binary(ENGLISH)

    assert collapsed[0] == [128, 10]
    assert collapsed[1] == [19 + 2, 5 + 5 + 0 + 11]


def test_collapsing_rejects_a_non_three_level_scale() -> None:
    with pytest.raises(ValueError, match="three-level"):
        collapse_to_binary([[1, 0], [0, 1]])


def test_bootstrap_interval_brackets_the_point_estimate() -> None:
    report = analyse(ENGLISH, iterations=2000)
    assert report.kappa_ci_low < report.kappa < report.kappa_ci_high
    assert report.total == 180


def test_report_marginals_show_the_partial_asymmetry() -> None:
    report = analyse(ENGLISH, iterations=500)
    assert report.row_marginals["partial"] == 29
    assert report.column_marginals["partial"] == 5


def test_label_count_must_match_the_matrix() -> None:
    with pytest.raises(ValueError, match="label count"):
        analyse(ENGLISH, labels=("a", "b"), iterations=100)
