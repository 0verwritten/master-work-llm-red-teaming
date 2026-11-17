from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pytest

from harness.agents.native.adversarial import ExperimentLanguage
from harness.experiments.adversarial_matrix import FrozenModelProfile, MatrixItem
from harness.experiments.bilingual_pilot import (
    ReviewedPair,
    build_bilingual_pilot_plan,
    estimate_pilot_cost,
)


def _profile(key: str) -> FrozenModelProfile:
    return FrozenModelProfile(
        key=key,
        provider="mock",
        model=f"{key}-model",
        endpoint_profile=None,
        revision=f"{key}-revision",
    )


def _pair(index: int) -> ReviewedPair:
    pair_id = f"pair-{index:02d}"
    category = f"category-{index % 12:02d}"
    return ReviewedPair(
        pair_id=pair_id,
        english=MatrixItem(
            stable_id=f"en-{index:02d}",
            content_sha256=f"{index:064x}",
            language=ExperimentLanguage.ENGLISH,
            category=category,
            pair_id=pair_id,
            reviewer_status="approved",
        ),
        ukrainian=MatrixItem(
            stable_id=f"uk-{index:02d}",
            content_sha256=f"{index + 100:064x}",
            language=ExperimentLanguage.UKRAINIAN,
            category=category,
            pair_id=pair_id,
            reviewer_status="approved",
        ),
        expected_behavior=("comply_safely" if index % 7 == 0 else "refuse_or_safe_redirect"),
    )


def _exploratory_pair(index: int) -> ReviewedPair:
    pair = _pair(index)
    return replace(
        pair,
        english=replace(pair.english, reviewer_status="hash_verified"),
        ukrainian=replace(
            pair.ukrainian,
            reviewer_status="ai_reviewed_exploratory",
        ),
    )


def test_pilot_selection_is_stable_stratified_and_budget_counted() -> None:
    pairs = tuple(_pair(index) for index in range(1, 38))
    target = _profile("target")
    plan = build_bilingual_pilot_plan(
        pairs=pairs,
        target=target,
        cross_attacker=_profile("cross"),
        judge=_profile("judge"),
        selection_seed=20260728,
        run_seed=17,
        rubric_version="pilot-v1",
    )
    reordered = build_bilingual_pilot_plan(
        pairs=tuple(reversed(pairs)),
        target=target,
        cross_attacker=_profile("cross"),
        judge=_profile("judge"),
        selection_seed=20260728,
        run_seed=17,
        rubric_version="pilot-v1",
    )

    assert plan == reordered
    assert len(plan.selected_pairs) == 10
    assert len({pair.english.category for pair in plan.selected_pairs}) == 10
    assert {pair.expected_behavior for pair in plan.selected_pairs} == {
        "comply_safely",
        "refuse_or_safe_redirect",
    }
    assert len(plan.matrix.cells) == 40
    assert sum(cell.is_diagonal is True for cell in plan.matrix.cells) == 20
    assert sum(cell.is_diagonal is False for cell in plan.matrix.cells) == 20
    assert plan.matrix.max_target_queries == 120
    assert plan.matrix.max_llm_calls == 360
    assert plan.manifest_json()["plan_sha256"] == plan.plan_sha256


def test_pilot_rejects_pair_and_profile_mismatches() -> None:
    pair = _pair(1)
    with pytest.raises(ValueError, match="same category"):
        replace(
            pair,
            ukrainian=replace(pair.ukrainian, category="different"),
        )
    target = _profile("target")
    with pytest.raises(ValueError, match="cross attacker"):
        build_bilingual_pilot_plan(
            pairs=(pair,),
            target=target,
            cross_attacker=target,
            judge=_profile("judge"),
            selection_seed=1,
            run_seed=1,
            sample_size=1,
            rubric_version="pilot-v1",
        )


def test_pilot_accepts_only_explicit_exploratory_review_basis() -> None:
    pairs = tuple(_exploratory_pair(index) for index in range(1, 11))
    kwargs = {
        "pairs": pairs,
        "target": _profile("target"),
        "cross_attacker": _profile("cross"),
        "judge": _profile("judge"),
        "selection_seed": 1,
        "run_seed": 1,
        "sample_size": 10,
        "rubric_version": "pilot-v1",
    }
    with pytest.raises(ValueError, match="review basis"):
        build_bilingual_pilot_plan(**kwargs)

    plan = build_bilingual_pilot_plan(
        **kwargs,
        review_basis="ai_reviewed_exploratory",
    )

    assert len(plan.matrix.cells) == 40
    assert {cell.review_basis for cell in plan.matrix.cells} == {"ai_reviewed_exploratory"}


def test_pilot_rejects_duplicate_or_insufficient_pairs() -> None:
    pair = _pair(1)
    kwargs = {
        "target": _profile("target"),
        "cross_attacker": _profile("cross"),
        "judge": _profile("judge"),
        "selection_seed": 1,
        "run_seed": 1,
        "rubric_version": "pilot-v1",
    }
    with pytest.raises(ValueError, match="not enough"):
        build_bilingual_pilot_plan(pairs=(pair,), sample_size=2, **kwargs)
    with pytest.raises(ValueError, match="unique"):
        build_bilingual_pilot_plan(pairs=(pair, pair), sample_size=2, **kwargs)


def test_pilot_cost_envelope_preserves_provider_reserve() -> None:
    plan = build_bilingual_pilot_plan(
        pairs=tuple(_pair(index) for index in range(1, 38)),
        target=_profile("target"),
        cross_attacker=_profile("cross"),
        judge=_profile("judge"),
        selection_seed=20260728,
        run_seed=17,
        rubric_version="pilot-v1",
    )

    envelope = estimate_pilot_cost(
        plan,
        startup_seconds_per_profile=300,
        request_seconds_per_call=30,
        active_cost_usd_per_second=Decimal("0.00150"),
        noncompute_allowance_usd=Decimal("0.75"),
        observed_window_use_usd=Decimal("5.05167558"),
        provider_cap_usd=Decimal("30.00"),
        required_reserve_usd=Decimal("6.00"),
    )

    assert envelope.unique_profile_count == 3
    assert envelope.estimated_incremental_usd == Decimal("18.30000")
    assert envelope.projected_window_use_usd == Decimal("23.35167558")
    assert envelope.projected_reserve_usd == Decimal("6.64832442")
    assert envelope.manifest_json()["projected_reserve_usd"] == "6.64832442"


def test_pilot_cost_envelope_rejects_plan_that_consumes_reserve() -> None:
    plan = build_bilingual_pilot_plan(
        pairs=tuple(_pair(index) for index in range(1, 38)),
        target=_profile("target"),
        cross_attacker=_profile("cross"),
        judge=_profile("judge"),
        selection_seed=20260728,
        run_seed=17,
        rubric_version="pilot-v1",
    )

    with pytest.raises(ValueError, match="preserve"):
        estimate_pilot_cost(
            plan,
            startup_seconds_per_profile=300,
            request_seconds_per_call=32,
            active_cost_usd_per_second=Decimal("0.00150"),
            noncompute_allowance_usd=Decimal("0.75"),
            observed_window_use_usd=Decimal("5.05167558"),
            provider_cap_usd=Decimal("30.00"),
            required_reserve_usd=Decimal("6.00"),
        )
