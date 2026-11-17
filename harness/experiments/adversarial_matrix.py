from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

from harness.agents.native.adversarial import AttackMethod, ExperimentLanguage

ReviewBasis = Literal["human_approved", "ai_reviewed_exploratory"]

GENERATIVE_METHODS = frozenset({AttackMethod.PAIR, AttackMethod.SINGLE_SHOT})
LOCAL_METHODS = frozenset({AttackMethod.STATIC, AttackMethod.RANDOM_MUTATION})


@dataclass(frozen=True)
class FrozenModelProfile:
    key: str
    provider: str
    model: str | None
    endpoint_profile: str | None
    revision: str

    def __post_init__(self) -> None:
        if not self.key or not self.provider or not self.revision:
            raise ValueError("model profile key, provider, and revision are required")
        if self.provider in {"modal", "vast.ai"}:
            if not self.endpoint_profile or self.model is not None:
                raise ValueError("profiled serving providers require endpoint_profile and no model")
        elif not self.model or self.endpoint_profile is not None:
            raise ValueError("non-profiled providers require model and no endpoint_profile")

    def as_json(self) -> dict[str, Any]:
        return {
            "endpoint_profile": self.endpoint_profile,
            "key": self.key,
            "model": self.model,
            "provider": self.provider,
            "revision": self.revision,
        }

    @property
    def identity_sha256(self) -> str:
        return _json_sha256(self.as_json())


@dataclass(frozen=True)
class MatrixItem:
    stable_id: str
    content_sha256: str
    language: ExperimentLanguage
    category: str
    pair_id: str | None
    reviewer_status: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "language", ExperimentLanguage(self.language))
        if not self.stable_id or not self.category:
            raise ValueError("item stable_id and category are required")
        if len(self.content_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.content_sha256
        ):
            raise ValueError("item content_sha256 must be a lowercase SHA-256 digest")
        if self.reviewer_status not in {
            "approved",
            "hash_verified",
            "ai_reviewed_exploratory",
        }:
            raise ValueError("matrix item reviewer status is not eligible for planning")
        if self.language == ExperimentLanguage.UKRAINIAN and not self.pair_id:
            raise ValueError("Ukrainian matrix items require pair_id")

    def as_json(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "content_sha256": self.content_sha256,
            "language": self.language.value,
            "pair_id": self.pair_id,
            "reviewer_status": self.reviewer_status,
            "stable_id": self.stable_id,
        }


@dataclass(frozen=True)
class MatrixDesign:
    attackers: tuple[FrozenModelProfile, ...]
    targets: tuple[FrozenModelProfile, ...]
    judge: FrozenModelProfile
    items: tuple[MatrixItem, ...]
    methods: tuple[AttackMethod, ...]
    seeds: tuple[int, ...]
    max_rounds: int
    max_role_attempts: int
    rubric_version: str
    review_basis: ReviewBasis = "human_approved"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "methods",
            tuple(AttackMethod(method) for method in self.methods),
        )
        if not self.attackers or not self.targets or not self.items:
            raise ValueError("attackers, targets, and items are required")
        if not self.methods or not self.seeds:
            raise ValueError("methods and seeds are required")
        if len(set(self.methods)) != len(self.methods):
            raise ValueError("methods must be unique")
        if set(self.methods) - GENERATIVE_METHODS - LOCAL_METHODS:
            raise ValueError("unknown adversarial method")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if self.max_rounds < 1 or self.max_role_attempts < 1 or not self.rubric_version:
            raise ValueError("rounds, role attempts, and rubric_version are required")
        self._require_unique_keys(self.attackers, role="attacker")
        self._require_unique_keys(self.targets, role="target")
        if len({item.stable_id for item in self.items}) != len(self.items):
            raise ValueError("matrix item stable IDs must be unique")
        if self.review_basis not in {"human_approved", "ai_reviewed_exploratory"}:
            raise ValueError("unsupported matrix review basis")
        eligible_statuses = (
            {
                ExperimentLanguage.ENGLISH: {"approved", "hash_verified"},
                ExperimentLanguage.UKRAINIAN: {"approved"},
            }
            if self.review_basis == "human_approved"
            else {
                ExperimentLanguage.ENGLISH: {"hash_verified"},
                ExperimentLanguage.UKRAINIAN: {"ai_reviewed_exploratory"},
            }
        )
        if any(item.reviewer_status not in eligible_statuses[item.language] for item in self.items):
            raise ValueError(f"matrix items do not satisfy review basis {self.review_basis}")
        experimental_identities = {
            profile.identity_sha256 for profile in (*self.attackers, *self.targets)
        }
        if self.judge.identity_sha256 in experimental_identities:
            raise ValueError("fixed judge must be independent of attacker and target profiles")

    @staticmethod
    def _require_unique_keys(
        profiles: tuple[FrozenModelProfile, ...],
        *,
        role: str,
    ) -> None:
        if len({profile.key for profile in profiles}) != len(profiles):
            raise ValueError(f"{role} profile keys must be unique")


@dataclass(frozen=True)
class MatrixCell:
    cell_id: str
    item: MatrixItem
    method: AttackMethod
    attacker: FrozenModelProfile | None
    target: FrozenModelProfile
    judge: FrozenModelProfile
    seed: int
    max_rounds: int
    max_role_attempts: int
    rubric_version: str
    review_basis: ReviewBasis

    @property
    def max_target_queries(self) -> int:
        return self.max_rounds

    @property
    def max_llm_calls(self) -> int:
        per_round = 3 if self.method in GENERATIVE_METHODS else 2
        return per_round * self.max_rounds * self.max_role_attempts

    @property
    def is_diagonal(self) -> bool | None:
        if self.attacker is None:
            return None
        return self.attacker.identity_sha256 == self.target.identity_sha256

    def as_json(self) -> dict[str, Any]:
        return {
            "attacker": self.attacker.as_json() if self.attacker else None,
            "cell_id": self.cell_id,
            "item": self.item.as_json(),
            "judge": self.judge.as_json(),
            "max_rounds": self.max_rounds,
            "max_role_attempts": self.max_role_attempts,
            "method": self.method.value,
            "rubric_version": self.rubric_version,
            "review_basis": self.review_basis,
            "seed": self.seed,
            "target": self.target.as_json(),
        }


@dataclass(frozen=True)
class MatrixPlan:
    cells: tuple[MatrixCell, ...]
    plan_sha256: str
    max_target_queries: int
    max_llm_calls: int

    def manifest_json(self) -> dict[str, Any]:
        return {
            "cells": [cell.as_json() for cell in self.cells],
            "max_llm_calls": self.max_llm_calls,
            "max_target_queries": self.max_target_queries,
            "plan_sha256": self.plan_sha256,
        }


def build_matrix_plan(design: MatrixDesign) -> MatrixPlan:
    cells: list[MatrixCell] = []
    sorted_attackers = tuple(sorted(design.attackers, key=lambda value: value.key))
    for item in sorted(design.items, key=lambda value: value.stable_id):
        for target in sorted(design.targets, key=lambda value: value.key):
            for method in sorted(design.methods, key=lambda value: value.value):
                attackers: tuple[FrozenModelProfile | None, ...] = (
                    sorted_attackers if method in GENERATIVE_METHODS else (None,)
                )
                for attacker in attackers:
                    for seed in sorted(design.seeds):
                        payload = {
                            "attacker_identity": (attacker.identity_sha256 if attacker else None),
                            "item_content_sha256": item.content_sha256,
                            "judge_identity": design.judge.identity_sha256,
                            "language": item.language.value,
                            "max_rounds": design.max_rounds,
                            "max_role_attempts": design.max_role_attempts,
                            "method": method.value,
                            "rubric_version": design.rubric_version,
                            "review_basis": design.review_basis,
                            "seed": seed,
                            "target_identity": target.identity_sha256,
                        }
                        cells.append(
                            MatrixCell(
                                cell_id=_json_sha256(payload),
                                item=item,
                                method=method,
                                attacker=attacker,
                                target=target,
                                judge=design.judge,
                                seed=seed,
                                max_rounds=design.max_rounds,
                                max_role_attempts=design.max_role_attempts,
                                rubric_version=design.rubric_version,
                                review_basis=design.review_basis,
                            )
                        )
    ordered = tuple(sorted(cells, key=lambda cell: cell.cell_id))
    manifest = [cell.as_json() for cell in ordered]
    return MatrixPlan(
        cells=ordered,
        plan_sha256=_json_sha256(manifest),
        max_target_queries=sum(cell.max_target_queries for cell in ordered),
        max_llm_calls=sum(cell.max_llm_calls for cell in ordered),
    )


def _json_sha256(value: Any) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()
