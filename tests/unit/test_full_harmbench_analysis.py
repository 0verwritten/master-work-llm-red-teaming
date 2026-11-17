from __future__ import annotations

import pytest

from harness.analysis.full_harmbench import (
    exact_mcnemar,
    fisher_exact_two_sided,
    holm_adjust,
    paired_risk_difference_interval,
    rate,
    wilson_interval,
)


def test_rate_keeps_empty_denominator_explicit() -> None:
    assert rate(1, 4) == 0.25
    assert rate(0, 0) is None


def test_wilson_interval_contains_observed_rate() -> None:
    low, high = wilson_interval(15, 119)
    assert low < 15 / 119 < high
    assert (low, high) == pytest.approx((0.078, 0.199), abs=0.002)


def test_exact_mcnemar_is_symmetric() -> None:
    assert exact_mcnemar(2, 17) == exact_mcnemar(17, 2)
    assert exact_mcnemar(0, 0) == 1.0


def test_paired_bootstrap_preserves_pairs_and_is_deterministic() -> None:
    pairs = [(True, False)] * 8 + [(False, True)] * 2
    first = paired_risk_difference_interval(pairs, resamples=1_000, seed=7)
    second = paired_risk_difference_interval(pairs, resamples=1_000, seed=7)
    assert first == second
    assert first[0] <= 0.6 <= first[1]


def test_fisher_exact_matches_known_extreme_table() -> None:
    assert fisher_exact_two_sided(1, 9, 11, 3) == pytest.approx(0.002759456, rel=1e-5)
    assert fisher_exact_two_sided(0, 0, 0, 0) == 1.0


def test_holm_adjust_is_monotone_in_rank() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        ((88, 412, 225, 259), 1.3314896931877186e-22),
        ((58, 59, 9, 107), 5.297776341981544e-13),
    ],
)
def test_fisher_preserves_very_small_tail_probabilities(
    table: tuple[int, int, int, int], expected: float
) -> None:

    a, b, c, d = table
    for equivalent in ((a, b, c, d), (c, d, a, b), (a, c, b, d), (b, a, d, c)):
        assert fisher_exact_two_sided(*equivalent) == pytest.approx(expected, rel=1e-12, abs=0)
