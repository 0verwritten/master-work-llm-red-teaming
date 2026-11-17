from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import func, select

from harness.benchmark_import import BENCHMARK_NAME, BENCHMARK_VERSION
from harness.bilingual_import import (
    TranslationRecord,
    import_translations,
    load_translation_records,
)
from harness.persistence.database import create_engine, create_schema, session_factory
from harness.persistence.models import BenchmarkItem, BenchmarkVersion


def _source_hash(index: int) -> str:
    return hashlib.sha256(f"source-{index}".encode()).hexdigest()


def _translation(index: int) -> TranslationRecord:
    return TranslationRecord(
        pair_id=f"pair-{index}",
        source_content_sha256=_source_hash(index),
        uk_prompt=f"Безпечний тестовий запит {index}",
        uk_image_text=None,
        uk_rubric_text="Українська тестова рубрика",
        translator_id="translator",
        translation_method="human",
        translated_at=datetime(2026, 1, 1, tzinfo=UTC),
        reviewer_id="reviewer",
        reviewed_at=datetime(2026, 1, 2, tzinfo=UTC),
        semantic_equivalence="pass",
        naturalness="pass",
        category_match="pass",
        rubric_match="pass",
        expected_behavior_match="pass",
        review_notes="",
    )


def test_translation_json_accepts_iso_timestamps(tmp_path: Path) -> None:
    path = tmp_path / "translations.json"
    records = [_translation(index).model_dump(mode="json") for index in range(37)]
    path.write_text(json.dumps(records, ensure_ascii=False))

    loaded, source_sha256 = load_translation_records(path)

    assert len(loaded) == 37
    assert loaded[0].translated_at == datetime(2026, 1, 1, tzinfo=UTC)
    assert len(source_sha256) == 64


async def test_reviewed_translation_import_is_idempotent() -> None:
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    await create_schema(engine)
    sessions = session_factory(engine)
    async with sessions() as session:
        async with session.begin():
            version = BenchmarkVersion(
                name=BENCHMARK_NAME,
                version=BENCHMARK_VERSION,
                schema_version=1,
                status="draft",
                source_manifest_json={},
            )
            session.add(version)
            await session.flush()
            session.add_all(
                [
                    BenchmarkItem(
                        benchmark_version_id=version.id,
                        stable_id=f"legacy-en:fixture:{index}",
                        source_name="fixture",
                        source_version="fixture",
                        source_item_id=str(index),
                        source_url="fixture",
                        license_id="NOASSERTION",
                        license_url="fixture",
                        provenance="historical_seed",
                        split="paired_held_out",
                        language="en",
                        modality="text",
                        category="safe_fixture",
                        functional_category="legacy_safety",
                        prompt_text=f"sanitized source {index}",
                        context_text=None,
                        rubric_text="English sanitized rubric",
                        expected_behavior="refuse_or_safe_redirect",
                        pair_id=f"pair-{index}",
                        translation_provenance_json={},
                        reviewer_status="hash_verified",
                        release_status="restricted",
                        content_sha256=_source_hash(index),
                        source_metadata_json={},
                    )
                    for index in range(37)
                ]
            )
        records = [_translation(index) for index in range(37)]
        first = await import_translations(session, records, "a" * 64)
        second = await import_translations(session, records, "a" * 64)
        count = await session.scalar(
            select(func.count()).select_from(BenchmarkItem).where(BenchmarkItem.language == "uk")
        )
        imported_rubric = await session.scalar(
            select(BenchmarkItem.rubric_text).where(BenchmarkItem.language == "uk").limit(1)
        )

    assert first is True
    assert second is False
    assert count == 37
    assert imported_rubric == "Українська тестова рубрика"
    await engine.dispose()
