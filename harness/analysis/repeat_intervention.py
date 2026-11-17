from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from harness.agents.native.adversarial import canonical_attack
from harness.analysis.full_harmbench import (
    exact_mcnemar,
    fisher_exact_two_sided,
    paired_risk_difference_interval,
    rate,
)


@dataclass(frozen=True)
class SessionOutcome:
    stable_id: str
    completed: bool
    validated_success: bool
    any_non_refusal: bool


@dataclass(frozen=True)
class SentAttacks:
    stable_id: str
    attacks: tuple[str, ...]
    rejections_per_round: tuple[int, ...] = ()
    unresolved_rounds: int = 0


def paired_contrast(
    left: dict[str, SessionOutcome],
    right: dict[str, SessionOutcome],
    *,
    outcome: str = "validated_success",
    stable_ids: set[str] | None = None,
) -> dict[str, Any]:

    if outcome not in {"validated_success", "any_non_refusal"}:
        raise ValueError(f"unknown outcome {outcome!r}")
    candidates = set(left) & set(right)
    if stable_ids is not None:
        candidates &= stable_ids
    ids = sorted(
        stable_id
        for stable_id in candidates
        if left[stable_id].completed and right[stable_id].completed
    )
    pairs = [
        (bool(getattr(left[value], outcome)), bool(getattr(right[value], outcome))) for value in ids
    ]
    left_only = sum(a and not b for a, b in pairs)
    right_only = sum(b and not a for a, b in pairs)
    both = sum(a and b for a, b in pairs)
    result: dict[str, Any] = {
        "outcome": outcome,
        "complete_pairs": len(pairs),
        "excluded_items": len(candidates) - len(pairs),
        "left_positive": left_only + both,
        "right_positive": right_only + both,
        "left_rate": rate(left_only + both, len(pairs)),
        "right_rate": rate(right_only + both, len(pairs)),
        "left_only": left_only,
        "right_only": right_only,
        "both": both,
        "neither": len(pairs) - left_only - right_only - both,
        "mcnemar_p": exact_mcnemar(left_only, right_only),
    }
    if pairs:
        low, high = paired_risk_difference_interval(pairs)
        result["risk_difference_left_minus_right"] = round((left_only - right_only) / len(pairs), 6)
        result["paired_bootstrap_95"] = [round(low, 6), round(high, 6)]
    else:
        result["risk_difference_left_minus_right"] = None
        result["paired_bootstrap_95"] = None
    return result


def repetition_summary(sessions: list[SentAttacks]) -> dict[str, Any]:

    eligible = [session for session in sessions if len(session.attacks) >= 2]
    byte_repeat = sum(len(set(value.attacks)) < len(value.attacks) for value in eligible)
    canonical_repeat = sum(
        len({canonical_attack(text) for text in value.attacks}) < len(value.attacks)
        for value in eligible
    )
    rejections = [sum(value.rejections_per_round) for value in sessions]
    return {
        "sessions": len(sessions),
        "eligible_sessions": len(eligible),
        "sent_byte_repeat_sessions": byte_repeat,
        "sent_byte_repeat_rate": rate(byte_repeat, len(eligible)),
        "sent_canonical_repeat_sessions": canonical_repeat,
        "sent_canonical_repeat_rate": rate(canonical_repeat, len(eligible)),
        "sessions_with_rejection": sum(value > 0 for value in rejections),
        "total_rejections": sum(rejections),
        "mean_rejections_per_session": rate(sum(rejections), len(sessions)),
        "unresolved_rounds": sum(value.unresolved_rounds for value in sessions),
        "sessions_with_unresolved_round": sum(value.unresolved_rounds > 0 for value in sessions),
    }


def manipulation_check(
    control: list[SentAttacks], intervention: list[SentAttacks]
) -> dict[str, Any]:

    control_summary = repetition_summary(control)
    intervention_summary = repetition_summary(intervention)
    fisher = fisher_exact_two_sided(
        control_summary["sent_byte_repeat_sessions"],
        control_summary["eligible_sessions"] - control_summary["sent_byte_repeat_sessions"],
        intervention_summary["sent_byte_repeat_sessions"],
        intervention_summary["eligible_sessions"]
        - intervention_summary["sent_byte_repeat_sessions"],
    )
    return {
        "control": control_summary,
        "intervention": intervention_summary,
        "byte_repeat_fisher_p": fisher,
    }
