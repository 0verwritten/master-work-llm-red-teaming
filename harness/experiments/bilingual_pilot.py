from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from harness.agents.native.adversarial import AttackMethod, ExperimentLanguage
from harness.experiments.adversarial_matrix import (
    FrozenModelProfile,
    MatrixDesign,
    MatrixItem,
    MatrixPlan,
    ReviewBasis,
    build_matrix_plan,
)


@dataclass(frozen=True)
class ReviewedPair:
    pair_id: str
    english: MatrixItem
    ukrainian: MatrixItem
    expected_behavior: str

    def __post_init__(self) -> None:
        if not self.pair_id or not self.expected_behavior:
            raise ValueError("pair_id and expected_behavior are required")
        if self.english.language != ExperimentLanguage.ENGLISH:
            raise ValueError("english item must have language=en")
        if self.ukrainian.language != ExperimentLanguage.UKRAINIAN:
            raise ValueError("ukrainian item must have language=uk")
        if self.english.pair_id != self.pair_id or self.ukrainian.pair_id != self.pair_id:
            raise ValueError("both items must match pair_id")
        if self.english.category != self.ukrainian.category:
            raise ValueError("paired items must have the same category")

    def as_json(self) -> dict[str, Any]:
        return {
            "category": self.english.category,
            "english": self.english.as_json(),
            "expected_behavior": self.expected_behavior,
            "pair_id": self.pair_id,
            "ukrainian": self.ukrainian.as_json(),
        }


@dataclass(frozen=True)
class BilingualPilotPlan:
    selected_pairs: tuple[ReviewedPair, ...]
    selection_seed: int
    matrix: MatrixPlan
    plan_sha256: str

    def manifest_json(self) -> dict[str, Any]:
        return {
            "matrix": self.matrix.manifest_json(),
            "plan_sha256": self.plan_sha256,
            "selected_pairs": [pair.as_json() for pair in self.selected_pairs],
            "selection_seed": self.selection_seed,
        }


@dataclass(frozen=True)
class PilotCostEnvelope:
    max_llm_calls: int
    unique_profile_count: int
    startup_seconds_per_profile: int
    request_seconds_per_call: int
    active_cost_usd_per_second: Decimal
    noncompute_allowance_usd: Decimal
    observed_window_use_usd: Decimal
    provider_cap_usd: Decimal
    required_reserve_usd: Decimal
    estimated_incremental_usd: Decimal
    projected_window_use_usd: Decimal
    projected_reserve_usd: Decimal

    def manifest_json(self) -> dict[str, Any]:
        return {
            "active_cost_usd_per_second": str(self.active_cost_usd_per_second),
            "estimated_incremental_usd": str(self.estimated_incremental_usd),
            "max_llm_calls": self.max_llm_calls,
            "noncompute_allowance_usd": str(self.noncompute_allowance_usd),
            "observed_window_use_usd": str(self.observed_window_use_usd),
            "projected_reserve_usd": str(self.projected_reserve_usd),
            "projected_window_use_usd": str(self.projected_window_use_usd),
            "provider_cap_usd": str(self.provider_cap_usd),
            "request_seconds_per_call": self.request_seconds_per_call,
            "required_reserve_usd": str(self.required_reserve_usd),
            "startup_seconds_per_profile": self.startup_seconds_per_profile,
            "unique_profile_count": self.unique_profile_count,
        }


def estimate_pilot_cost(
    plan: BilingualPilotPlan,
    *,
    startup_seconds_per_profile: int,
    request_seconds_per_call: int,
    active_cost_usd_per_second: Decimal,
    noncompute_allowance_usd: Decimal,
    observed_window_use_usd: Decimal,
    provider_cap_usd: Decimal,
    required_reserve_usd: Decimal,
) -> PilotCostEnvelope:
    if startup_seconds_per_profile < 0 or request_seconds_per_call < 1:
        raise ValueError("startup and request seconds must be non-negative and positive")
    monetary_values = (
        active_cost_usd_per_second,
        noncompute_allowance_usd,
        observed_window_use_usd,
        provider_cap_usd,
        required_reserve_usd,
    )
    if any(value < 0 for value in monetary_values) or active_cost_usd_per_second == 0:
        raise ValueError("cost values must be non-negative and active rate must be positive")
    identities = {
        profile.identity_sha256
        for cell in plan.matrix.cells
        for profile in (cell.attacker, cell.target, cell.judge)
        if profile is not None
    }
    startup_seconds = len(identities) * startup_seconds_per_profile
    request_seconds = plan.matrix.max_llm_calls * request_seconds_per_call
    estimated_incremental = (
        Decimal(startup_seconds + request_seconds) * active_cost_usd_per_second
        + noncompute_allowance_usd
    )
    projected_window_use = observed_window_use_usd + estimated_incremental
    projected_reserve = provider_cap_usd - projected_window_use
    if projected_reserve < required_reserve_usd:
        raise ValueError("pilot estimate does not preserve the required provider reserve")
    return PilotCostEnvelope(
        max_llm_calls=plan.matrix.max_llm_calls,
        unique_profile_count=len(identities),
        startup_seconds_per_profile=startup_seconds_per_profile,
        request_seconds_per_call=request_seconds_per_call,
        active_cost_usd_per_second=active_cost_usd_per_second,
        noncompute_allowance_usd=noncompute_allowance_usd,
        observed_window_use_usd=observed_window_use_usd,
        provider_cap_usd=provider_cap_usd,
        required_reserve_usd=required_reserve_usd,
        estimated_incremental_usd=estimated_incremental,
        projected_window_use_usd=projected_window_use,
        projected_reserve_usd=projected_reserve,
    )


def build_bilingual_pilot_plan(
    *,
    pairs: tuple[ReviewedPair, ...],
    target: FrozenModelProfile,
    cross_attacker: FrozenModelProfile,
    judge: FrozenModelProfile,
    selection_seed: int,
    run_seed: int,
    sample_size: int = 10,
    max_rounds: int = 3,
    max_role_attempts: int = 1,
    rubric_version: str,
    review_basis: ReviewBasis = "human_approved",
) -> BilingualPilotPlan:
    selected = _select_pairs(pairs, sample_size=sample_size, seed=selection_seed)
    if target.identity_sha256 == cross_attacker.identity_sha256:
        raise ValueError("cross attacker must differ from target")
    matrix = build_matrix_plan(
        MatrixDesign(
            attackers=(target, cross_attacker),
            targets=(target,),
            judge=judge,
            items=tuple(
                item
                for pair in selected
                for item in (
                    pair.english,
                    pair.ukrainian,
                )
            ),
            methods=(AttackMethod.PAIR,),
            seeds=(run_seed,),
            max_rounds=max_rounds,
            max_role_attempts=max_role_attempts,
            rubric_version=rubric_version,
            review_basis=review_basis,
        )
    )
    payload = {
        "matrix_plan_sha256": matrix.plan_sha256,
        "selected_pairs": [pair.as_json() for pair in selected],
        "selection_seed": selection_seed,
    }
    return BilingualPilotPlan(
        selected_pairs=selected,
        selection_seed=selection_seed,
        matrix=matrix,
        plan_sha256=_json_sha256(payload),
    )


def _select_pairs(
    pairs: tuple[ReviewedPair, ...],
    *,
    sample_size: int,
    seed: int,
) -> tuple[ReviewedPair, ...]:
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    if len(pairs) < sample_size:
        raise ValueError("not enough reviewed pairs for requested sample")
    if len({pair.pair_id for pair in pairs}) != len(pairs):
        raise ValueError("pair IDs must be unique")

    remaining = list(pairs)
    selected: list[ReviewedPair] = []
    seen_behaviors: set[str] = set()
    seen_categories: set[str] = set()
    while len(selected) < sample_size:
        chosen = min(
            remaining,
            key=lambda pair: (
                pair.expected_behavior in seen_behaviors,
                pair.english.category in seen_categories,
                _selection_digest(seed=seed, pair_id=pair.pair_id),
                pair.pair_id,
            ),
        )
        selected.append(chosen)
        remaining.remove(chosen)
        seen_behaviors.add(chosen.expected_behavior)
        seen_categories.add(chosen.english.category)
    return tuple(sorted(selected, key=lambda pair: pair.pair_id))


def _selection_digest(*, seed: int, pair_id: str) -> str:
    return hashlib.sha256(f"{seed}:{pair_id}".encode()).hexdigest()


def _json_sha256(value: Any) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()
