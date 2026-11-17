from __future__ import annotations

from dataclasses import replace

import pytest

from harness.agents.native.adversarial import AttackMethod, ExperimentLanguage
from harness.experiments.adversarial_matrix import (
    FrozenModelProfile,
    MatrixDesign,
    MatrixItem,
    build_matrix_plan,
)


def _profile(key: str) -> FrozenModelProfile:
    return FrozenModelProfile(
        key=key,
        provider="mock",
        model=f"{key}-model",
        endpoint_profile=None,
        revision=f"{key}-revision",
    )


def _design() -> MatrixDesign:
    models = (_profile("a"), _profile("b"))
    return MatrixDesign(
        attackers=models,
        targets=models,
        judge=_profile("judge"),
        items=(
            MatrixItem(
                stable_id="en-item",
                content_sha256="1" * 64,
                language=ExperimentLanguage.ENGLISH,
                category="fixture",
                pair_id="pair-1",
                reviewer_status="approved",
            ),
            MatrixItem(
                stable_id="uk-item",
                content_sha256="2" * 64,
                language=ExperimentLanguage.UKRAINIAN,
                category="fixture",
                pair_id="pair-1",
                reviewer_status="approved",
            ),
        ),
        methods=tuple(AttackMethod),
        seeds=(11, 22),
        max_rounds=2,
        max_role_attempts=1,
        rubric_version="fixture-v1",
    )


def test_matrix_plan_is_stable_and_avoids_duplicate_local_baselines() -> None:
    design = _design()
    plan = build_matrix_plan(design)
    reordered = MatrixDesign(
        attackers=tuple(reversed(design.attackers)),
        targets=tuple(reversed(design.targets)),
        judge=design.judge,
        items=tuple(reversed(design.items)),
        methods=tuple(reversed(design.methods)),
        seeds=tuple(reversed(design.seeds)),
        max_rounds=design.max_rounds,
        max_role_attempts=design.max_role_attempts,
        rubric_version=design.rubric_version,
    )

    assert plan == build_matrix_plan(reordered)
    assert len(plan.cells) == 48
    assert len({cell.cell_id for cell in plan.cells}) == 48
    assert sum(cell.is_diagonal is True for cell in plan.cells) == 16
    assert sum(cell.is_diagonal is False for cell in plan.cells) == 16
    assert sum(cell.is_diagonal is None for cell in plan.cells) == 16
    assert plan.max_target_queries == 96
    assert plan.max_llm_calls == 256
    retry_plan = build_matrix_plan(replace(design, max_role_attempts=2))
    assert retry_plan.max_target_queries == plan.max_target_queries
    assert retry_plan.max_llm_calls == 512
    assert retry_plan.plan_sha256 != plan.plan_sha256


def test_unapproved_items_and_non_independent_judge_are_rejected() -> None:
    with pytest.raises(ValueError, match="eligible"):
        MatrixItem(
            stable_id="pending",
            content_sha256="a" * 64,
            language=ExperimentLanguage.ENGLISH,
            category="fixture",
            pair_id=None,
            reviewer_status="pending",
        )

    with pytest.raises(ValueError, match="review basis"):
        replace(
            _design(),
            items=(
                replace(_design().items[0], reviewer_status="hash_verified"),
                replace(
                    _design().items[1],
                    reviewer_status="ai_reviewed_exploratory",
                ),
            ),
        )

    model = _profile("same")
    with pytest.raises(ValueError, match="independent"):
        MatrixDesign(
            attackers=(model,),
            targets=(model,),
            judge=model,
            items=_design().items,
            methods=(AttackMethod.PAIR,),
            seeds=(1,),
            max_rounds=1,
            max_role_attempts=1,
            rubric_version="fixture-v1",
        )


def test_vast_profile_uses_content_addressed_endpoint_identity() -> None:
    profile = FrozenModelProfile(
        key="vast-fixture",
        provider="vast.ai",
        model=None,
        endpoint_profile=f"vast-fixture@{'a' * 64}",
        revision="b" * 40,
    )

    assert profile.model is None
    assert profile.endpoint_profile == f"vast-fixture@{'a' * 64}"
