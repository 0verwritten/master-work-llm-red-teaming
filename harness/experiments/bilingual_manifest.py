from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from harness.bilingual_dataset import ASSUMED_VALID, DatasetItem, require_verified
from harness.experiments.adversarial_matrix import FrozenModelProfile

COST_PER_CALL_LOW = 0.04503
COST_PER_CALL_HIGH = 0.05083

MANIFEST_VERSION = "bilingual-study-v1"
CALLS_PER_ROUND = 3
ATTACK_METHODS = frozenset({"pair", "single_shot", "static", "random_mutation"})


@dataclass(frozen=True)
class StudyCell:
    cell_id: str
    pair_id: str
    language: str
    stratum: str
    category: str
    item_stable_id: str
    item_content_sha256: str
    expected_behavior: str
    reviewer_status: str
    max_rounds: int

    def as_json(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "pair_id": self.pair_id,
            "language": self.language,
            "stratum": self.stratum,
            "category": self.category,
            "item_stable_id": self.item_stable_id,
            "item_content_sha256": self.item_content_sha256,
            "expected_behavior": self.expected_behavior,
            "reviewer_status": self.reviewer_status,
            "max_rounds": self.max_rounds,
        }


@dataclass(frozen=True)
class CostEnvelope:
    cells: int
    max_rounds: int
    worst_case_calls: int
    worst_case_usd_low: float
    worst_case_usd_high: float

    def as_json(self) -> dict[str, Any]:
        return {
            "cells": self.cells,
            "max_rounds": self.max_rounds,
            "worst_case_calls": self.worst_case_calls,
            "worst_case_usd_low": round(self.worst_case_usd_low, 5),
            "worst_case_usd_high": round(self.worst_case_usd_high, 5),
        }


@dataclass(frozen=True)
class BilingualStudyManifest:
    manifest_version: str
    dataset_sha256: str
    subset_plan_sha256: str
    assumed_validation: bool
    attacker: FrozenModelProfile
    target: FrozenModelProfile
    judge: FrozenModelProfile
    cells: tuple[StudyCell, ...]
    envelope: CostEnvelope
    enforce_judge_language: bool
    cell_namespace: str | None

    max_attack_regenerations: int = 0
    attack_method: str = "pair"

    @property
    def calls_per_round(self) -> int:
        return calls_per_round(self.max_attack_regenerations, self.attack_method)

    @property
    def manifest_sha256(self) -> str:
        payload = json.dumps(
            self.as_json(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def as_json(self) -> dict[str, Any]:
        def profile(value: FrozenModelProfile) -> dict[str, Any]:
            return {
                "key": value.key,
                "provider": value.provider,
                "model": value.model,
                "endpoint_profile": value.endpoint_profile,
                "revision": value.revision,
            }

        payload = {
            "manifest_version": self.manifest_version,
            "dataset_sha256": self.dataset_sha256,
            "subset_plan_sha256": self.subset_plan_sha256,
            "assumed_validation": self.assumed_validation,
            "enforce_judge_language": self.enforce_judge_language,
            "roles": {
                "attacker": profile(self.attacker),
                "target": profile(self.target),
                "judge": profile(self.judge),
            },
            "envelope": self.envelope.as_json(),
            "cells": [cell.as_json() for cell in self.cells],
        }
        if self.cell_namespace is not None:
            payload["cell_namespace"] = self.cell_namespace

        if self.max_attack_regenerations:
            payload["attacker_protocol"] = _attacker_protocol(self.max_attack_regenerations)
        if self.attack_method != "pair":
            payload["attack_method"] = self.attack_method
        return payload

    def cells_by_category(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for cell in self.cells:
            counts[cell.category] = counts.get(cell.category, 0) + 1
        return dict(sorted(counts.items()))


def calls_per_round(max_attack_regenerations: int, attack_method: str = "pair") -> int:

    base = 2 if attack_method in {"static", "random_mutation"} else CALLS_PER_ROUND
    return base + max_attack_regenerations


def _attacker_protocol(max_attack_regenerations: int) -> dict[str, Any]:
    return {
        "max_attack_regenerations": max_attack_regenerations,
        "reject_repeated_attacks": True,
    }


def _cell_id(
    item: DatasetItem,
    max_rounds: int,
    *,
    namespace: str | None,
    max_attack_regenerations: int = 0,
    attack_method: str = "pair",
) -> str:
    identity: dict[str, Any] = {
        "content": item.content_sha256,
        "language": item.language,
        "max_rounds": max_rounds,
        "pair_id": item.pair_id,
    }
    if namespace is not None:
        identity["namespace"] = namespace
    if max_attack_regenerations:
        identity["attacker_protocol"] = _attacker_protocol(max_attack_regenerations)
    if attack_method != "pair":
        identity["attack_method"] = attack_method
    payload = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def build_manifest(
    items: list[DatasetItem],
    *,
    dataset_sha256: str,
    subset_plan_sha256: str,
    attacker: FrozenModelProfile,
    target: FrozenModelProfile,
    judge: FrozenModelProfile,
    max_rounds: int = 3,
    allow_assumed: bool = False,
    enforce_judge_language: bool = True,
    cell_namespace: str | None = None,
    max_attack_regenerations: int = 0,
    attack_method: str = "pair",
) -> BilingualStudyManifest:

    if max_rounds < 1:
        raise ValueError("max_rounds must be positive")
    if not items:
        raise ValueError("a manifest needs at least one item")
    if cell_namespace is not None and not cell_namespace.strip():
        raise ValueError("cell_namespace cannot be blank")
    if max_attack_regenerations < 0:
        raise ValueError("max_attack_regenerations cannot be negative")
    if attack_method not in ATTACK_METHODS:
        raise ValueError(f"unknown attack method: {attack_method}")
    if max_attack_regenerations and attack_method != "pair":
        raise ValueError("attack regeneration is available only for pair")

    require_verified(items, allow_assumed=allow_assumed)

    ordered = sorted(items, key=lambda item: (item.category, item.language, item.stable_id))
    cells = tuple(
        StudyCell(
            cell_id=_cell_id(
                item,
                max_rounds,
                namespace=cell_namespace,
                max_attack_regenerations=max_attack_regenerations,
                attack_method=attack_method,
            ),
            pair_id=item.pair_id,
            language=item.language,
            stratum=item.stratum,
            category=item.category,
            item_stable_id=item.stable_id,
            item_content_sha256=item.content_sha256,
            expected_behavior=item.expected_behavior,
            reviewer_status=item.reviewer_status,
            max_rounds=max_rounds,
        )
        for item in ordered
    )
    if len({cell.cell_id for cell in cells}) != len(cells):
        raise ValueError("cell identities must be unique")

    calls = len(cells) * max_rounds * calls_per_round(max_attack_regenerations, attack_method)
    envelope = CostEnvelope(
        cells=len(cells),
        max_rounds=max_rounds,
        worst_case_calls=calls,
        worst_case_usd_low=calls * COST_PER_CALL_LOW,
        worst_case_usd_high=calls * COST_PER_CALL_HIGH,
    )
    return BilingualStudyManifest(
        manifest_version=MANIFEST_VERSION,
        dataset_sha256=dataset_sha256,
        subset_plan_sha256=subset_plan_sha256,
        assumed_validation=any(cell.reviewer_status == ASSUMED_VALID for cell in cells),
        attacker=attacker,
        target=target,
        judge=judge,
        cells=cells,
        envelope=envelope,
        enforce_judge_language=enforce_judge_language,
        cell_namespace=cell_namespace,
        max_attack_regenerations=max_attack_regenerations,
        attack_method=attack_method,
    )


def check_affordable(manifest: BilingualStudyManifest, *, remaining_usd: float) -> None:

    if manifest.envelope.worst_case_usd_high > remaining_usd:
        raise ValueError(
            f"worst case USD {manifest.envelope.worst_case_usd_high:.2f} exceeds the "
            f"remaining USD {remaining_usd:.2f}; reduce the sample or refresh the window"
        )
