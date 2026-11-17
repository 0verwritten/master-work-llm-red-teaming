from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

SELECTION_SEED = 20260817
SUBSET_SIZE = 120
ELIGIBLE_SPLIT = "test"
ELIGIBLE_MODALITY = "text"
PLAN_VERSION = "uk-verification-subset-v1"


@dataclass(frozen=True)
class EligibleItem:
    stable_id: str
    category: str
    functional_category: str
    content_sha256: str

    def __post_init__(self) -> None:
        if not self.stable_id:
            raise ValueError("stable_id must not be empty")
        if not self.category:
            raise ValueError("category must not be empty")
        if len(self.content_sha256) != 64:
            raise ValueError("content_sha256 must be a full SHA-256 digest")


@dataclass(frozen=True)
class SubsetPlan:
    plan_version: str
    seed: int
    subset_size: int
    eligible_count: int
    allocation: dict[str, int]
    stable_ids: tuple[str, ...]
    plan_sha256: str

    def as_json(self) -> dict[str, Any]:
        return {
            "plan_version": self.plan_version,
            "seed": self.seed,
            "subset_size": self.subset_size,
            "eligible_count": self.eligible_count,
            "allocation": dict(sorted(self.allocation.items())),
            "stable_ids": list(self.stable_ids),
            "plan_sha256": self.plan_sha256,
        }


def _rank(seed: int, item: EligibleItem) -> str:

    raw = f"{seed}:{item.category}:{item.content_sha256}".encode()
    return hashlib.sha256(raw).hexdigest()


def allocate(counts: dict[str, int], total: int) -> dict[str, int]:

    eligible = sum(counts.values())
    if total > eligible:
        raise ValueError("cannot allocate more items than are eligible")
    exact = {name: value * total / eligible for name, value in counts.items()}
    floors = {name: int(value) for name, value in exact.items()}
    remaining = total - sum(floors.values())
    ordered = sorted(counts, key=lambda name: (-(exact[name] - floors[name]), name))
    for name in ordered[:remaining]:
        floors[name] += 1
    for name, value in floors.items():
        if value > counts[name]:
            raise ValueError(f"allocation for {name} exceeds its eligible count")
    return floors


def select_subset(
    items: list[EligibleItem],
    *,
    seed: int = SELECTION_SEED,
    subset_size: int = SUBSET_SIZE,
) -> SubsetPlan:

    if subset_size < 1:
        raise ValueError("subset_size must be positive")
    stable_ids = [item.stable_id for item in items]
    if len(set(stable_ids)) != len(stable_ids):
        raise ValueError("eligible items must have unique stable_ids")
    if len(items) < subset_size:
        raise ValueError("eligible pool is smaller than the requested subset")

    counts = Counter(item.category for item in items)
    allocation = allocate(dict(counts), subset_size)

    selected: list[str] = []
    for category in sorted(counts):
        in_category = sorted(
            (item for item in items if item.category == category),
            key=lambda item: (_rank(seed, item), item.stable_id),
        )
        selected.extend(item.stable_id for item in in_category[: allocation[category]])

    selected.sort()
    if len(selected) != subset_size:
        raise RuntimeError("selection did not produce the requested subset size")

    payload = json.dumps(
        {
            "plan_version": PLAN_VERSION,
            "seed": seed,
            "subset_size": subset_size,
            "eligible_count": len(items),
            "allocation": dict(sorted(allocation.items())),
            "stable_ids": selected,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    return SubsetPlan(
        plan_version=PLAN_VERSION,
        seed=seed,
        subset_size=subset_size,
        eligible_count=len(items),
        allocation=allocation,
        stable_ids=tuple(selected),
        plan_sha256=hashlib.sha256(payload).hexdigest(),
    )


def eligible_from_rows(rows: list[dict[str, Any]]) -> list[EligibleItem]:

    return [
        EligibleItem(
            stable_id=str(row["stable_id"]),
            category=str(row["category"]),
            functional_category=str(row["functional_category"]),
            content_sha256=str(row["content_sha256"]),
        )
        for row in rows
        if row["split"] == ELIGIBLE_SPLIT
        and row["modality"] == ELIGIBLE_MODALITY
        and row["language"] == "en"
    ]
