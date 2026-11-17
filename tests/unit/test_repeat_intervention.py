from __future__ import annotations

import pytest

from harness.analysis.repeat_intervention import (
    SentAttacks,
    SessionOutcome,
    manipulation_check,
    paired_contrast,
    repetition_summary,
)


def _outcome(stable_id: str, *, success: bool, completed: bool = True) -> SessionOutcome:
    return SessionOutcome(
        stable_id=stable_id,
        completed=completed,
        validated_success=success,
        any_non_refusal=success,
    )


def test_paired_contrast_pairs_by_item_and_drops_indeterminate_arms() -> None:
    left = {
        "a": _outcome("a", success=True),
        "b": _outcome("b", success=True),
        "c": _outcome("c", success=False),
        "d": _outcome("d", success=True, completed=False),
        "only-left": _outcome("only-left", success=True),
    }
    right = {
        "a": _outcome("a", success=False),
        "b": _outcome("b", success=True),
        "c": _outcome("c", success=False),
        "d": _outcome("d", success=False),
    }
    result = paired_contrast(left, right)
    assert result["complete_pairs"] == 3
    assert result["excluded_items"] == 1
    assert (result["left_only"], result["right_only"], result["both"], result["neither"]) == (
        1,
        0,
        1,
        1,
    )
    assert result["risk_difference_left_minus_right"] == pytest.approx(1 / 3, abs=1e-6)
    subset = paired_contrast(left, right, stable_ids={"b", "c"})
    assert subset["complete_pairs"] == 2
    assert subset["risk_difference_left_minus_right"] == 0


def test_paired_contrast_rejects_unknown_outcome_and_handles_empty() -> None:
    with pytest.raises(ValueError, match="unknown outcome"):
        paired_contrast({}, {}, outcome="asr")
    empty = paired_contrast({}, {})
    assert empty["complete_pairs"] == 0
    assert empty["paired_bootstrap_95"] is None


def test_repetition_summary_distinguishes_byte_and_canonical_repeats() -> None:
    sessions = [
        SentAttacks("byte", ("x", "x", "y")),
        SentAttacks("case-only", ("Attack", " attack ", "z")),
        SentAttacks("clean", ("p", "q", "r"), rejections_per_round=(0, 2, 1), unresolved_rounds=1),
        SentAttacks("single", ("only",)),
    ]
    summary = repetition_summary(sessions)
    assert summary["eligible_sessions"] == 3
    assert summary["sent_byte_repeat_sessions"] == 1
    assert summary["sent_canonical_repeat_sessions"] == 2
    assert summary["total_rejections"] == 3
    assert summary["sessions_with_rejection"] == 1
    assert summary["sessions_with_unresolved_round"] == 1


def test_manipulation_check_detects_removed_repeats() -> None:
    control = [SentAttacks(str(index), ("a", "a", "b")) for index in range(20)]
    intervention = [
        SentAttacks(str(index), ("a", "c", "b"), rejections_per_round=(0, 1, 0))
        for index in range(20)
    ]
    result = manipulation_check(control, intervention)
    assert result["control"]["sent_byte_repeat_rate"] == 1.0
    assert result["intervention"]["sent_byte_repeat_rate"] == 0.0
    assert result["byte_repeat_fisher_p"] < 0.001
