from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from harness.benchmark_import import BENCHMARK_NAME, BENCHMARK_VERSION
from harness.persistence.database import create_engine, session_factory
from harness.persistence.models import BenchmarkItem, BenchmarkVersion
from harness.settings import Settings


class TranslationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    pair_id: str
    source_content_sha256: str
    uk_prompt: str
    uk_image_text: str | None
    uk_rubric_text: str
    translator_id: str
    translation_method: str
    translated_at: datetime
    reviewer_id: str
    reviewed_at: datetime
    semantic_equivalence: str
    naturalness: str
    category_match: str
    rubric_match: str
    expected_behavior_match: str
    review_notes: str

    @field_validator("translated_at", "reviewed_at", mode="before")
    @classmethod
    def parse_json_datetime(cls, value: object) -> object:
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        return value

    @field_validator(
        "semantic_equivalence",
        "naturalness",
        "category_match",
        "rubric_match",
        "expected_behavior_match",
    )
    @classmethod
    def review_must_pass(cls, value: str) -> str:
        if value != "pass":
            raise ValueError("every review dimension must pass")
        return value

    @field_validator("translation_method")
    @classmethod
    def translation_must_be_human(cls, value: str) -> str:
        if value != "human":
            raise ValueError("translation_method must be human")
        return value


def load_translation_records(path: Path) -> tuple[list[TranslationRecord], str]:
    raw = path.read_bytes()
    decoded = json.loads(raw)
    if not isinstance(decoded, list):
        raise ValueError("translation file must contain a JSON array")
    records = [TranslationRecord.model_validate(row) for row in decoded]
    if len(records) != 37:
        raise ValueError("translation file must contain exactly 37 records")
    if len({record.pair_id for record in records}) != 37:
        raise ValueError("translation pair IDs must be unique")
    for record in records:
        if record.translator_id == record.reviewer_id:
            raise ValueError(f"translator and reviewer must differ: {record.pair_id}")
        if not record.uk_prompt.strip():
            raise ValueError(f"empty Ukrainian prompt: {record.pair_id}")
        if not any("\u0400" <= char <= "\u04ff" for char in record.uk_prompt):
            raise ValueError(f"Ukrainian prompt lacks Cyrillic text: {record.pair_id}")
    return records, hashlib.sha256(raw).hexdigest()


def content_hash(prompt: str, image_text: str | None, rubric_text: str) -> str:
    raw = json.dumps(
        {"prompt": prompt, "image_text": image_text, "rubric_text": rubric_text},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(raw).hexdigest()


async def import_translations(
    session: AsyncSession, records: list[TranslationRecord], source_file_sha256: str
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
        english = list(
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
        if len(english_by_pair) != 37 or None in english_by_pair:
            raise RuntimeError("expected 37 unique English source pairs")
        if set(english_by_pair) != {record.pair_id for record in records}:
            raise ValueError("translation pair IDs do not match English source pairs")

        existing = list(
            (
                await session.scalars(
                    select(BenchmarkItem).where(
                        BenchmarkItem.benchmark_version_id == version.id,
                        BenchmarkItem.provenance == "reviewed_translation",
                        BenchmarkItem.language == "uk",
                    )
                )
            ).all()
        )
        if existing:
            if len(existing) != 37:
                raise RuntimeError("incomplete existing Ukrainian translation import")
            expected_hashes = {
                record.pair_id: content_hash(
                    record.uk_prompt,
                    record.uk_image_text,
                    record.uk_rubric_text,
                )
                for record in records
            }
            if any(
                item.pair_id is None or expected_hashes.get(item.pair_id) != item.content_sha256
                for item in existing
            ):
                raise RuntimeError("translations changed; create a new benchmark version")
            return False

        new_items: list[BenchmarkItem] = []
        for record in records:
            source = english_by_pair[record.pair_id]
            if source.content_sha256 != record.source_content_sha256:
                raise ValueError(f"English source hash mismatch: {record.pair_id}")
            if (source.context_text is None) != (record.uk_image_text is None):
                raise ValueError(f"image-text presence mismatch: {record.pair_id}")
            provenance = record.model_dump(mode="json", exclude={"uk_prompt", "uk_image_text"})
            new_items.append(
                BenchmarkItem(
                    benchmark_version_id=version.id,
                    stable_id=source.stable_id.replace("legacy-en:", "legacy-uk:", 1),
                    source_name="reviewed_translation",
                    source_version=source_file_sha256,
                    source_item_id=source.source_item_id,
                    source_url="unpublished:private-bilingual-review-record",
                    license_id="NOASSERTION",
                    license_url="unpublished:first-party-license-not-found",
                    provenance="reviewed_translation",
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
                    translation_provenance_json=provenance,
                    reviewer_status="approved",
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
        session.add_all(new_items)
        await session.flush()
        count = await session.scalar(
            select(func.count())
            .select_from(BenchmarkItem)
            .where(
                BenchmarkItem.benchmark_version_id == version.id,
                BenchmarkItem.provenance == "reviewed_translation",
            )
        )
        if int(count or 0) != 37:
            raise RuntimeError("Ukrainian translation import count mismatch")
        return True


async def _persist(records: list[TranslationRecord], source_sha256: str) -> bool:
    engine = create_engine(Settings().database_url)
    try:
        sessions = session_factory(engine)
        async with sessions() as session:
            return await import_translations(session, records, source_sha256)
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import independently reviewed private Ukrainian translations."
    )
    parser.add_argument("--translations", required=True, type=Path)
    args = parser.parse_args()
    records, source_sha256 = load_translation_records(args.translations)
    inserted = asyncio.run(_persist(records, source_sha256))
    print(
        json.dumps(
            {
                "pair_count": len(records),
                "source_sha256": source_sha256,
                "status": "inserted" if inserted else "already_present",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
