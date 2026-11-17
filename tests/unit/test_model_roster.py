from __future__ import annotations

from dataclasses import replace

import pytest

from harness.experiments.model_roster import (
    PHASE2_MODEL_ROSTER_V1,
    FrozenModelRoster,
)


def test_phase2_roster_freezes_three_matrix_profiles_and_independent_judge() -> None:
    roster = PHASE2_MODEL_ROSTER_V1

    assert {profile.key for profile in roster.matrix_profiles} == {
        "gemma4-31b-it",
        "mistral-small-3.2-24b",
        "qwen36-27b-fp8",
    }
    assert roster.judge_profile.key == "gpt-oss-20b-judge"
    assert all(
        profile.endpoint_profile and "@" in profile.endpoint_profile
        for profile in (*roster.matrix_profiles, roster.judge_profile)
    )
    assert len({profile.revision for profile in roster.matrix_profiles}) == 3
    assert roster.roster_sha256 == (
        "4d5ef7acdcc16a32b346b6ac47a817ba8bb5003399241ea5e5983e33b98a645b"
    )
    assert roster.manifest_json()["roster_sha256"] == roster.roster_sha256


def test_roster_rejects_duplicate_matrix_or_non_independent_judge() -> None:
    roster = PHASE2_MODEL_ROSTER_V1
    first = roster.matrix_profiles[0]

    with pytest.raises(ValueError, match="keys must be unique"):
        FrozenModelRoster(
            version="invalid",
            matrix_profiles=(first, first, roster.matrix_profiles[2]),
            judge_profile=roster.judge_profile,
        )
    with pytest.raises(ValueError, match="independent"):
        replace(roster, judge_profile=first)
