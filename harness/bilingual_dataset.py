from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

HUMAN_VERIFIED = "human_verified"
AI_REVIEWED = "ai_reviewed_exploratory"
ASSUMED_VALID = "machine_translated_assumed_valid"
UNVERIFIED = "machine_translated_unverified"

ACCEPTABLE_WITH_OPT_IN = frozenset({HUMAN_VERIFIED, ASSUMED_VALID, AI_REVIEWED})
DATASET_VERSION = "thesis-bilingual-v1"

HARMFUL_BEHAVIOR = "refuse_or_safe_redirect"
BENIGN_BEHAVIOR = "comply_safely"

Language = Literal["en", "uk"]


@dataclass(frozen=True)
class DatasetItem:
    stable_id: str
    pair_id: str
    language: Language
    stratum: str
    category: str
    prompt_text: str
    context_text: str | None
    rubric_text: str
    expected_behavior: str
    reviewer_status: str
    translation_provenance: str
    license_id: str

    @property
    def content_sha256(self) -> str:
        payload = json.dumps(
            {
                "prompt": self.prompt_text,
                "context": self.context_text,
                "rubric": self.rubric_text,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def as_json(self) -> dict[str, Any]:
        return {
            "stable_id": self.stable_id,
            "pair_id": self.pair_id,
            "language": self.language,
            "stratum": self.stratum,
            "category": self.category,
            "prompt_text": self.prompt_text,
            "context_text": self.context_text,
            "rubric_text": self.rubric_text,
            "expected_behavior": self.expected_behavior,
            "reviewer_status": self.reviewer_status,
            "translation_provenance": self.translation_provenance,
            "license_id": self.license_id,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True)
class BilingualDataset:
    dataset_version: str
    items: tuple[DatasetItem, ...]
    subset_plan_sha256: str
    assumed_validation: bool

    @property
    def pair_ids(self) -> set[str]:
        return {item.pair_id for item in self.items}

    def by_language(self, language: Language) -> list[DatasetItem]:
        return [item for item in self.items if item.language == language]

    def manifest(self) -> dict[str, Any]:
        entries = sorted(
            (
                {"stable_id": item.stable_id, "content_sha256": item.content_sha256}
                for item in self.items
            ),
            key=lambda entry: entry["stable_id"],
        )
        payload = json.dumps(
            {
                "dataset_version": self.dataset_version,
                "subset_plan_sha256": self.subset_plan_sha256,
                "assumed_validation": self.assumed_validation,
                "entries": entries,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        statuses: dict[str, int] = {}
        for item in self.items:
            statuses[item.reviewer_status] = statuses.get(item.reviewer_status, 0) + 1
        return {
            "dataset_version": self.dataset_version,
            "subset_plan_sha256": self.subset_plan_sha256,
            "assumed_validation": self.assumed_validation,
            "item_count": len(self.items),
            "pair_count": len(self.pair_ids),
            "by_language": {
                "en": len(self.by_language("en")),
                "uk": len(self.by_language("uk")),
            },
            "by_stratum": {
                stratum: sum(1 for item in self.items if item.stratum == stratum)
                for stratum in sorted({item.stratum for item in self.items})
            },
            "reviewer_status": dict(sorted(statuses.items())),
            "ukrainian_human_verified_count": sum(
                1
                for item in self.items
                if item.language == "uk" and item.reviewer_status == HUMAN_VERIFIED
            ),
            "ukrainian_item_count": len(self.by_language("uk")),
            "dataset_sha256": hashlib.sha256(payload).hexdigest(),
        }


def build(
    *,
    english_rows: list[dict[str, Any]],
    english_rubric: str,
    ukrainian_records: list[dict[str, Any]],
    subset_stable_ids: list[str],
    subset_plan_sha256: str,
    benign_english_rows: list[dict[str, Any]],
    benign_english_rubric: str,
    benign_ukrainian_records: list[dict[str, Any]],
    ukrainian_status: str = UNVERIFIED,
) -> BilingualDataset:

    if ukrainian_status not in (HUMAN_VERIFIED, ASSUMED_VALID, UNVERIFIED):
        raise ValueError(f"unknown reviewer status {ukrainian_status!r}")

    english_by_id = {str(row["stable_id"]): row for row in english_rows}
    ukrainian_by_id = {str(record["source_stable_id"]): record for record in ukrainian_records}
    selected = list(dict.fromkeys(subset_stable_ids))
    if len(selected) != len(subset_stable_ids):
        raise ValueError("subset contains duplicate stable_ids")

    items: list[DatasetItem] = []
    for stable_id in selected:
        english = english_by_id.get(stable_id)
        ukrainian = ukrainian_by_id.get(stable_id)
        if english is None:
            raise ValueError(f"English source missing for {stable_id}")
        if ukrainian is None:
            raise ValueError(f"Ukrainian translation missing for {stable_id}")
        items.append(
            DatasetItem(
                stable_id=stable_id,
                pair_id=stable_id,
                language="en",
                stratum="harmbench_subset",
                category=str(english["category"]),
                prompt_text=str(english["prompt_text"]),
                context_text=english.get("context_text") or None,
                rubric_text=english_rubric,
                expected_behavior=HARMFUL_BEHAVIOR,
                reviewer_status=HUMAN_VERIFIED,
                translation_provenance="source_language",
                license_id="MIT",
            )
        )
        items.append(
            DatasetItem(
                stable_id=f"{stable_id}:uk",
                pair_id=stable_id,
                language="uk",
                stratum="harmbench_subset",
                category=str(english["category"]),
                prompt_text=str(ukrainian["prompt_text"]),
                context_text=ukrainian.get("context_text") or None,
                rubric_text=str(ukrainian["rubric_text"]),
                expected_behavior=HARMFUL_BEHAVIOR,
                reviewer_status=ukrainian_status,
                translation_provenance="machine_deepl",
                license_id="MIT",
            )
        )

    benign_uk_by_id = {
        str(record["source_stable_id"]): record for record in benign_ukrainian_records
    }
    for row in benign_english_rows:
        stable_id = str(row["stable_id"])
        ukrainian = benign_uk_by_id.get(stable_id)
        if ukrainian is None:
            raise ValueError(f"Ukrainian benign translation missing for {stable_id}")
        items.append(
            DatasetItem(
                stable_id=stable_id,
                pair_id=stable_id,
                language="en",
                stratum="benign_control",
                category=str(row["category"]),
                prompt_text=str(row["prompt_text"]),
                context_text=None,
                rubric_text=benign_english_rubric,
                expected_behavior=BENIGN_BEHAVIOR,
                reviewer_status=HUMAN_VERIFIED,
                translation_provenance="source_language",
                license_id="NOASSERTION",
            )
        )
        items.append(
            DatasetItem(
                stable_id=f"{stable_id}:uk",
                pair_id=stable_id,
                language="uk",
                stratum="benign_control",
                category=str(row["category"]),
                prompt_text=str(ukrainian["prompt_text"]),
                context_text=None,
                rubric_text=str(ukrainian["rubric_text"]),
                expected_behavior=BENIGN_BEHAVIOR,
                reviewer_status=ukrainian_status,
                translation_provenance="machine_deepl",
                license_id="NOASSERTION",
            )
        )

    dataset = BilingualDataset(
        dataset_version=DATASET_VERSION,
        items=tuple(items),
        subset_plan_sha256=subset_plan_sha256,
        assumed_validation=ukrainian_status == ASSUMED_VALID,
    )
    _validate_pairing(dataset)
    return dataset


def _validate_pairing(dataset: BilingualDataset) -> None:

    if len({item.stable_id for item in dataset.items}) != len(dataset.items):
        raise ValueError("dataset contains duplicate stable_ids")
    for pair_id in dataset.pair_ids:
        sides = [item for item in dataset.items if item.pair_id == pair_id]
        languages = sorted(item.language for item in sides)
        if languages != ["en", "uk"]:
            raise ValueError(f"pair {pair_id} is not exactly one en and one uk item")
        if len({item.expected_behavior for item in sides}) != 1:
            raise ValueError(f"pair {pair_id} has mismatched expected behaviour")
        if len({item.category for item in sides}) != 1:
            raise ValueError(f"pair {pair_id} has mismatched category")


def require_verified(items: list[DatasetItem], *, allow_assumed: bool = False) -> None:

    permitted = ACCEPTABLE_WITH_OPT_IN if allow_assumed else frozenset({HUMAN_VERIFIED})
    offenders = sorted(
        item.stable_id
        for item in items
        if item.language == "uk" and item.reviewer_status not in permitted
    )
    if offenders:
        raise ValueError(
            f"{len(offenders)} Ukrainian item(s) are not verified "
            f"(allow_assumed={allow_assumed}): {offenders[:3]}"
        )
