from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

FINGERPRINT_VERSION = "legacy-pair-content-v1"

FINGERPRINT_FIELDS = (
    "prompt_text",
    "context_text",
    "category",
    "expected_behavior",
    "rubric_text",
    "uk_prompt",
    "uk_image_text",
    "uk_rubric",
)


def _normalize(value: str | None) -> str | None:

    if value is None:
        return None
    return unicodedata.normalize("NFC", value)


def content_fingerprint(fields: Mapping[str, str | None]) -> str:

    payload = {
        "fields": {name: _normalize(fields[name]) for name in FINGERPRINT_FIELDS if name in fields},
        "version": FINGERPRINT_VERSION,
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _source_item_id(pair_id: str) -> str:
    return pair_id.rsplit(":", 1)[-1]


def elect_canonical(pair_ids: Iterable[str]) -> str:

    ordered = sorted(pair_ids)
    if not ordered:
        raise ValueError("cannot elect a canonical pair from an empty group")
    return min(ordered, key=lambda pair_id: (len(_source_item_id(pair_id)), pair_id))


@dataclass(frozen=True)
class DuplicateGroup:
    fingerprint: str
    canonical_pair_id: str
    redundant_pair_ids: tuple[str, ...]

    @property
    def members(self) -> tuple[str, ...]:
        return (self.canonical_pair_id, *self.redundant_pair_ids)

    def as_json(self) -> dict[str, Any]:
        return {
            "canonical_pair_id": self.canonical_pair_id,
            "fingerprint": self.fingerprint,
            "redundant_pair_ids": list(self.redundant_pair_ids),
        }


@dataclass(frozen=True)
class DeduplicationManifest:
    fingerprint_version: str
    groups: tuple[DuplicateGroup, ...]
    canonical_pair_ids: tuple[str, ...]
    redundant_pair_ids: tuple[str, ...]

    @property
    def input_count(self) -> int:
        return len(self.canonical_pair_ids) + len(self.redundant_pair_ids)

    @property
    def canonical_count(self) -> int:
        return len(self.canonical_pair_ids)

    def canonical_for(self, pair_id: str) -> str:

        for group in self.groups:
            if pair_id in group.redundant_pair_ids:
                return group.canonical_pair_id
        return pair_id

    def is_redundant(self, pair_id: str) -> bool:
        return pair_id in self.redundant_pair_ids

    def as_json(self) -> dict[str, Any]:
        return {
            "canonical_count": self.canonical_count,
            "canonical_pair_ids": list(self.canonical_pair_ids),
            "fingerprint_version": self.fingerprint_version,
            "groups": [group.as_json() for group in self.groups],
            "input_count": self.input_count,
            "redundant_pair_ids": list(self.redundant_pair_ids),
        }


def build_manifest(items: Mapping[str, Mapping[str, str | None]]) -> DeduplicationManifest:

    by_fingerprint: dict[str, list[str]] = {}
    for pair_id, fields in items.items():
        by_fingerprint.setdefault(content_fingerprint(fields), []).append(pair_id)

    groups: list[DuplicateGroup] = []
    canonical: list[str] = []
    redundant: list[str] = []
    for fingerprint, pair_ids in sorted(by_fingerprint.items()):
        winner = elect_canonical(pair_ids)
        canonical.append(winner)
        if len(pair_ids) > 1:
            losers = tuple(sorted(set(pair_ids) - {winner}))
            redundant.extend(losers)
            groups.append(
                DuplicateGroup(
                    fingerprint=fingerprint,
                    canonical_pair_id=winner,
                    redundant_pair_ids=losers,
                )
            )

    return DeduplicationManifest(
        fingerprint_version=FINGERPRINT_VERSION,
        groups=tuple(sorted(groups, key=lambda group: group.canonical_pair_id)),
        canonical_pair_ids=tuple(sorted(canonical)),
        redundant_pair_ids=tuple(sorted(redundant)),
    )
