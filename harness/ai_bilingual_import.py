from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from harness.benchmark_import import BENCHMARK_NAME, BENCHMARK_VERSION
from harness.bilingual_import import content_hash
from harness.persistence.database import create_engine, session_factory
from harness.persistence.models import BenchmarkItem, BenchmarkVersion
from harness.settings import Settings

EXCLUDED_SOURCE_DEFECT_PAIR_IDS = frozenset(
    {
        "legacy-pair:guardrail_seed:8",
        "legacy-pair:guardrail_seed:14",
        "legacy-pair:guardrail_seed:18",
    }
)


class AIReviewRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    reviewer_kind: str
    reviewer_profile: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    reviewed_at: datetime
    translation_revision: int = Field(ge=1)
    translation_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    dimensions: dict[str, str]
    confidence: float = Field(ge=0, le=1)
    notes: str
    disposition: str

    @field_validator("reviewed_at", mode="before")
    @classmethod
    def parse_json_datetime(cls, value: object) -> object:
        return datetime.fromisoformat(value) if isinstance(value, str) else value

    @field_validator("reviewer_kind")
    @classmethod
    def reviewer_must_be_ai(cls, value: str) -> str:
        if value not in {"ai_agent", "ai_subagent"}:
            raise ValueError("exploratory review must be labelled as an AI agent")
        return value

    @field_validator("disposition")
    @classmethod
    def disposition_must_accept(cls, value: str) -> str:
        if value != "accept":
            raise ValueError("imported exploratory records must have accept disposition")
        return value

    @field_validator("dimensions")
    @classmethod
    def dimensions_must_pass(cls, value: dict[str, str]) -> dict[str, str]:
        expected = {
            "semantic_equivalence",
            "naturalness",
            "category_match",
            "rubric_match",
            "expected_behavior_match",
        }
        if set(value) != expected or any(result != "pass" for result in value.values()):
            raise ValueError("all five AI review dimensions must pass")
        return value


class AIReviewedTranslationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    pair_id: str = Field(min_length=1)
    source_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    uk_prompt: str = Field(min_length=1)
    uk_image_text: str | None
    uk_rubric_text: str = Field(min_length=1)
    translation_method: str
    translated_at: datetime
    ai_review: AIReviewRecord
    reviewer_status: str

    @field_validator("translated_at", mode="before")
    @classmethod
    def parse_json_datetime(cls, value: object) -> object:
        return datetime.fromisoformat(value) if isinstance(value, str) else value

    @field_validator("translation_method")
    @classmethod
    def method_must_disclose_ai_revision(cls, value: str) -> str:
        if value not in {"machine_deepl", "machine_deepl_ai_revised"}:
            raise ValueError("exploratory records must disclose machine_deepl provenance")
        return value

    @field_validator("reviewer_status")
    @classmethod
    def status_must_be_exploratory(cls, value: str) -> str:
        if value != "ai_reviewed_exploratory":
            raise ValueError("AI review cannot be labelled approved")
        return value

    @field_validator("uk_prompt", "uk_rubric_text")
    @classmethod
    def require_cyrillic(cls, value: str) -> str:
        if not any("\u0400" <= character <= "\u04ff" for character in value):
            raise ValueError("Ukrainian fields must contain Cyrillic text")
        return value

    @model_validator(mode="after")
    def review_must_match_translation(self) -> AIReviewedTranslationRecord:
        expected = content_hash(self.uk_prompt, self.uk_image_text, self.uk_rubric_text)
        if self.ai_review.translation_hash != expected:
            raise ValueError("AI review hash does not match translated content")
        return self


def load_ai_reviewed_records(path: Path) -> tuple[list[AIReviewedTranslationRecord], str]:
    raw = path.read_bytes()
    decoded: Any = json.loads(raw)
    if not isinstance(decoded, list):
        raise ValueError("AI-reviewed translation file must contain a JSON array")
    records = [AIReviewedTranslationRecord.model_validate(row) for row in decoded]
    if len(records) != 34 or len({record.pair_id for record in records}) != 34:
        raise ValueError("exploratory translation file must contain 34 unique records")
    if {record.pair_id for record in records} & EXCLUDED_SOURCE_DEFECT_PAIR_IDS:
        raise ValueError("source-defect exclusions cannot enter the exploratory set")
    return records, hashlib.sha256(raw).hexdigest()


async def import_ai_reviewed_translations(
    session: AsyncSession,
    records: list[AIReviewedTranslationRecord],
    source_file_sha256: str,
) -> bool:
    async with session.begin():
        version = await session.scalar(
            select(BenchmarkVersion).where(
                BenchmarkVersion.name == BENCHMARK_NAME,
                BenchmarkVersion.version == BENCHMARK_VERSION,
            )
        )
        if version is None or version.status != "draft":
            raise RuntimeError("draft benchmark version is unavailable")
        english = tuple(
            (
                await session.scalars(
                    select(BenchmarkItem).where(
                        BenchmarkItem.benchmark_version_id == version.id,
                        BenchmarkItem.provenance == "historical_seed",
                        BenchmarkItem.language == "en",
                    )
                )
            ).all()
        )
        english_by_pair = {item.pair_id: item for item in english}
        expected_pairs = set(english_by_pair) - EXCLUDED_SOURCE_DEFECT_PAIR_IDS
        if None in english_by_pair or len(english) != 37 or len(english_by_pair) != 37:
            raise RuntimeError("expected 37 unique English source pairs")
        if {record.pair_id for record in records} != expected_pairs:
            raise ValueError("AI-reviewed pair IDs do not match the eligible English set")
        existing = tuple(
            (
                await session.scalars(
                    select(BenchmarkItem).where(
                        BenchmarkItem.benchmark_version_id == version.id,
                        BenchmarkItem.provenance == "ai_reviewed_translation",
                        BenchmarkItem.language == "uk",
                    )
                )
            ).all()
        )
        if existing:
            if len(existing) != 34 or {item.pair_id for item in existing} != expected_pairs:
                raise RuntimeError("incomplete existing AI-reviewed translation import")
            if any(item.source_version != source_file_sha256 for item in existing):
                raise RuntimeError("AI review provenance changed; create a new benchmark version")
            return False
        for record in records:
            source = english_by_pair[record.pair_id]
            if source.content_sha256 != record.source_content_sha256:
                raise ValueError(f"English source hash mismatch: {record.pair_id}")
            if (source.context_text is None) != (record.uk_image_text is None):
                raise ValueError(f"image-text presence mismatch: {record.pair_id}")
            session.add(
                BenchmarkItem(
                    benchmark_version_id=version.id,
                    stable_id=source.stable_id.replace("legacy-en:", "legacy-uk-ai:", 1),
                    source_name="ai_reviewed_translation",
                    source_version=source_file_sha256,
                    source_item_id=source.source_item_id,
                    source_url="unpublished:private-ai-reviewed-bilingual-record",
                    license_id="NOASSERTION",
                    license_url="unpublished:first-party-license-not-found",
                    provenance="ai_reviewed_translation",
                    split=source.split,
                    language="uk",
                    modality=source.modality,
                    category=source.category,
                    functional_category=source.functional_category,
                    prompt_text=record.uk_prompt,
                    context_text=record.uk_image_text,
                    rubric_text=record.uk_rubric_text,
                    expected_behavior=source.expected_behavior,
                    pair_id=record.pair_id,
                    translation_provenance_json=record.model_dump(
                        mode="json", exclude={"uk_prompt", "uk_image_text", "uk_rubric_text"}
                    ),
                    reviewer_status="ai_reviewed_exploratory",
                    release_status="restricted",
                    content_sha256=content_hash(
                        record.uk_prompt,
                        record.uk_image_text,
                        record.uk_rubric_text,
                    ),
                    source_metadata_json={
                        "english_stable_id": source.stable_id,
                        "english_content_sha256": source.content_sha256,
                        "translation_file_sha256": source_file_sha256,
                    },
                )
            )
        await session.flush()
        count = await session.scalar(
            select(func.count())
            .select_from(BenchmarkItem)
            .where(
                BenchmarkItem.benchmark_version_id == version.id,
                BenchmarkItem.provenance == "ai_reviewed_translation",
            )
        )
        if int(count or 0) != 34:
            raise RuntimeError("AI-reviewed translation import count mismatch")
        return True


async def _persist(records: list[AIReviewedTranslationRecord], source_sha256: str) -> bool:
    engine = create_engine(Settings().database_url)
    try:
        async with session_factory(engine)() as session:
            return await import_ai_reviewed_translations(session, records, source_sha256)
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Import exploratory AI-reviewed translations.")
    parser.add_argument("--translations", required=True, type=Path)
    args = parser.parse_args()
    records, source_sha256 = load_ai_reviewed_records(args.translations)
    inserted = asyncio.run(_persist(records, source_sha256))
    print(
        json.dumps(
            {
                "pair_count": len(records),
                "reviewer_status": "ai_reviewed_exploratory",
                "source_sha256": source_sha256,
                "status": "inserted" if inserted else "already_present",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
