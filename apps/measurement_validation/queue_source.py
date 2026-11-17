from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from harness.artificial_reviewer import ArtificialReview, get_reviewer
from harness.legacy_dedup import DeduplicationManifest, build_manifest

QUEUE_SCHEMA_VERSION = 2
QUEUE_FILENAME = "native-review-queue.json"
REVIEWED_TRANSLATIONS_FILENAME = "ai-reviewed-translations.json"
TRANSLATION_PROGRESS_FILENAME = "translation-progress.json"
ENGLISH_SOURCES_FILENAME = "english-source-items.json"

SUBJECT_SEPARATOR = "␟"

ELIGIBLE_PAIR_COUNT = 34

CANONICAL_PAIR_COUNT = 29

SOURCE_DEFECT_EXCLUSIONS = frozenset(
    {
        "legacy-pair:guardrail_seed:8",
        "legacy-pair:guardrail_seed:14",
        "legacy-pair:guardrail_seed:18",
    }
)


def subject_sha256(uk_prompt: str, uk_image_text: str | None, uk_rubric_text: str) -> str:

    joined = SUBJECT_SEPARATOR.join([uk_prompt or "", uk_image_text or "", uk_rubric_text or ""])
    return hashlib.sha256(joined.encode()).hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class AdjudicationSubject:
    pair_id: str
    source_content_sha256: str
    subject_sha256: str
    category: str
    expected_behavior: str
    artificial_review: ArtificialReview

    en_prompt: str = ""
    en_context: str | None = None
    en_rubric: str = ""
    uk_prompt: str = ""
    uk_image_text: str | None = None
    uk_rubric: str = ""

    @property
    def rubric_variant(self) -> str:

        return self.expected_behavior

    def identity(self) -> dict[str, str]:
        return {
            "pair_id": self.pair_id,
            "source_content_sha256": self.source_content_sha256,
            "subject_sha256": self.subject_sha256,
        }

    def content_fields(self) -> dict[str, str | None]:

        return {
            "prompt_text": self.en_prompt,
            "context_text": self.en_context,
            "category": self.category,
            "expected_behavior": self.expected_behavior,
            "rubric_text": self.en_rubric,
            "uk_prompt": self.uk_prompt,
            "uk_image_text": self.uk_image_text,
            "uk_rubric": self.uk_rubric,
        }


@dataclass(frozen=True)
class AdjudicationQueue:
    reviewer_id: str
    queue_sha256: str

    subjects: tuple[AdjudicationSubject, ...]

    translator_ids: frozenset[str]

    duplicates: DeduplicationManifest

    all_subjects: tuple[AdjudicationSubject, ...] = ()

    def by_pair_id(self) -> dict[str, AdjudicationSubject]:
        return {subject.pair_id: subject for subject in self.subjects}

    def find(self, pair_id: str) -> AdjudicationSubject:
        try:
            return self.by_pair_id()[pair_id]
        except KeyError:
            raise KeyError(f"unknown pair ID: {pair_id}") from None


def _parse_review(value: Any) -> ArtificialReview:

    return ArtificialReview.model_validate_json(json.dumps(value, ensure_ascii=False))


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise ValueError(f"missing private workspace file: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _reviewed_translations(root: Path) -> dict[str, dict[str, Any]]:
    records = _read_json(root / REVIEWED_TRANSLATIONS_FILENAME)
    if not isinstance(records, list):
        raise ValueError("reviewed translations must be a JSON array")
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("reviewed translation records must be objects")
        pair_id = str(record.get("pair_id", ""))
        if not pair_id:
            raise ValueError("reviewed translation record is missing its pair ID")
        if pair_id in indexed:
            raise ValueError(f"duplicate reviewed translation: {pair_id}")
        indexed[pair_id] = record
    return indexed


def _english_sources(root: Path) -> dict[str, dict[str, Any]]:

    path = root / ENGLISH_SOURCES_FILENAME
    if not path.exists():
        return {}
    records = _read_json(path)
    rows = records if isinstance(records, list) else records.get("items", [])
    if not isinstance(rows, list):
        raise ValueError("english sources must be a JSON array")
    return {
        str(row["pair_id"]): row for row in rows if isinstance(row, dict) and row.get("pair_id")
    }


def _translator_ids(root: Path) -> frozenset[str]:
    path = root / TRANSLATION_PROGRESS_FILENAME
    if not path.exists():
        return frozenset()
    state = _read_json(path)
    if not isinstance(state, dict):
        raise ValueError("translation progress must be a JSON object")
    identifiers = {
        str(row["translation"]["translator_id"])
        for row in state.get("items", [])
        if isinstance(row, dict) and isinstance(row.get("translation"), dict)
    }
    return frozenset(identifier for identifier in identifiers if identifier)


def load_queue(
    root: Path,
    *,
    expected_count: int = ELIGIBLE_PAIR_COUNT,
    canonical_count: int | None = None,
) -> AdjudicationQueue:

    payload = _read_json(root / QUEUE_FILENAME)
    if not isinstance(payload, dict):
        raise ValueError("adjudication queue must be a JSON object")
    if payload.get("schema_version") != QUEUE_SCHEMA_VERSION:
        raise ValueError("unsupported adjudication queue schema")
    if payload.get("human_signoff_complete") is not False:
        raise ValueError("adjudication queue must be unadjudicated at load time")

    declared_reviewer = payload.get("reviewer")
    if not isinstance(declared_reviewer, dict):
        raise ValueError("adjudication queue is missing its reviewer identity")
    reviewer = get_reviewer(str(declared_reviewer.get("reviewer_id", "")))
    if declared_reviewer != reviewer.model_dump(mode="json"):
        raise ValueError("adjudication queue reviewer does not match the declared identity")

    rows = payload.get("items")
    if not isinstance(rows, list):
        raise ValueError("adjudication queue is missing its items")
    if len(rows) != expected_count:
        raise ValueError(f"adjudication queue must contain exactly {expected_count} records")

    translations = _reviewed_translations(root)
    english = _english_sources(root)
    subjects: list[AdjudicationSubject] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("adjudication queue records must be objects")
        pair_id = str(row.get("pair_id", ""))
        if not pair_id:
            raise ValueError("adjudication queue record is missing its pair ID")
        if pair_id in seen:
            raise ValueError(f"duplicate adjudication queue record: {pair_id}")
        seen.add(pair_id)
        if pair_id in SOURCE_DEFECT_EXCLUSIONS:
            raise ValueError(f"source-defect exclusion is not eligible for adjudication: {pair_id}")
        translation = translations.get(pair_id)
        if translation is None:
            raise ValueError(f"queued pair has no reviewed translation: {pair_id}")
        if row.get("source_content_sha256") != translation.get("source_content_sha256"):
            raise ValueError(f"queued pair source hash mismatch: {pair_id}")

        review = _parse_review(row.get("artificial_review"))
        if review.reviewer.reviewer_id != reviewer.reviewer_id:
            raise ValueError(f"queued judgement carries a foreign reviewer: {pair_id}")
        if review.human_signoff.resolved:
            raise ValueError(f"queued judgement is already adjudicated: {pair_id}")
        expected_subject = subject_sha256(
            str(translation.get("uk_prompt", "")),
            cast(str | None, translation.get("uk_image_text")),
            str(translation.get("uk_rubric_text", "")),
        )
        if review.subject_sha256 != expected_subject:
            raise ValueError(f"queued judgement is stale for its translation: {pair_id}")

        subjects.append(
            AdjudicationSubject(
                pair_id=pair_id,
                source_content_sha256=str(row["source_content_sha256"]),
                subject_sha256=expected_subject,
                category=str(row.get("category", "")),
                expected_behavior=str(row.get("expected_behavior", "")),
                artificial_review=review,
                en_prompt=str(english.get(pair_id, {}).get("prompt_text", "")),
                en_context=cast(str | None, english.get(pair_id, {}).get("context_text")),
                en_rubric=str(english.get(pair_id, {}).get("rubric_text", "")),
                uk_prompt=str(translation.get("uk_prompt", "")),
                uk_image_text=cast(str | None, translation.get("uk_image_text")),
                uk_rubric=str(translation.get("uk_rubric_text", "")),
            )
        )

    unqueued = set(translations) - seen
    if unqueued:
        raise ValueError(f"{len(unqueued)} reviewed translations are missing from the queue")

    ordered = tuple(sorted(subjects, key=lambda subject: subject.pair_id))

    duplicates = build_manifest({subject.pair_id: subject.content_fields() for subject in ordered})
    canonical = tuple(
        subject for subject in ordered if not duplicates.is_redundant(subject.pair_id)
    )
    if canonical_count is not None and len(canonical) != canonical_count:
        raise ValueError(
            f"expected {canonical_count} distinct pairs after deduplication, found {len(canonical)}"
        )

    return AdjudicationQueue(
        reviewer_id=reviewer.reviewer_id,
        queue_sha256=canonical_sha256([subject.identity() for subject in canonical]),
        subjects=canonical,
        translator_ids=_translator_ids(root),
        duplicates=duplicates,
        all_subjects=ordered,
    )
