from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from harness.legacy_dedup import DeduplicationManifest


@dataclass(frozen=True)
class ReplicationObservation:
    pair_id: str
    language: str
    model: str
    verdict: str
    response_sha256: str


@dataclass(frozen=True)
class ReplicationComparison:
    canonical_pair_id: str
    language: str
    model: str
    observations: tuple[ReplicationObservation, ...]

    @property
    def verdicts(self) -> tuple[str, ...]:
        return tuple(observation.verdict for observation in self.observations)

    @property
    def agrees(self) -> bool:

        return len(set(self.verdicts)) <= 1

    @property
    def identical_response(self) -> bool:

        return len({observation.response_sha256 for observation in self.observations}) == 1

    def as_json(self) -> dict[str, Any]:
        return {
            "agrees": self.agrees,
            "canonical_pair_id": self.canonical_pair_id,
            "identical_response": self.identical_response,
            "language": self.language,
            "model": self.model,
            "observations": [
                {
                    "pair_id": observation.pair_id,
                    "response_sha256": observation.response_sha256,
                    "verdict": observation.verdict,
                }
                for observation in self.observations
            ],
            "verdicts": list(self.verdicts),
        }


def response_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def compare_groups(
    manifest: DeduplicationManifest,
    observations: Sequence[ReplicationObservation],
) -> tuple[ReplicationComparison, ...]:

    cells: dict[tuple[str, str, str], list[ReplicationObservation]] = {}
    members = {
        member: group.canonical_pair_id for group in manifest.groups for member in group.members
    }
    for observation in observations:
        canonical = members.get(observation.pair_id)
        if canonical is None:
            continue
        key = (canonical, observation.language, observation.model)
        cells.setdefault(key, []).append(observation)

    comparisons = [
        ReplicationComparison(
            canonical_pair_id=canonical,
            language=language,
            model=model,
            observations=tuple(sorted(found, key=lambda item: item.pair_id)),
        )
        for (canonical, language, model), found in sorted(cells.items())
        if len(found) > 1
    ]
    return tuple(comparisons)


def summarize(comparisons: Sequence[ReplicationComparison]) -> dict[str, Any]:

    total = len(comparisons)
    if not total:
        return {
            "comparable_cells": 0,
            "note": "no duplicate group had two or more judged responses",
        }
    agreeing = sum(comparison.agrees for comparison in comparisons)
    identical = sum(comparison.identical_response for comparison in comparisons)
    return {
        "comparable_cells": total,
        "verdict_agreement_count": agreeing,
        "verdict_agreement_rate": round(agreeing / total, 4),
        "identical_response_count": identical,
        "identical_response_rate": round(identical / total, 4),
        "disagreeing_cells": [
            comparison.as_json() for comparison in comparisons if not comparison.agrees
        ],
    }
