from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy import func, select

from harness.ai_bilingual_import import (
    EXCLUDED_SOURCE_DEFECT_PAIR_IDS,
    AIReviewedTranslationRecord,
    import_ai_reviewed_translations,
    load_ai_reviewed_records,
)
from harness.benchmark_import import BENCHMARK_NAME, BENCHMARK_VERSION
from harness.bilingual_import import content_hash
from harness.persistence.database import create_engine, create_schema, session_factory
from harness.persistence.models import BenchmarkItem, BenchmarkVersion


def _source_hash(index: int) -> str:
    return hashlib.sha256(f"source-{index}".encode()).hexdigest()


def _pair_id(index: int) -> str:
    if index == 8:
        return "legacy-pair:guardrail_seed:8"
    if index == 14:
        return "legacy-pair:guardrail_seed:14"
    if index == 18:
        return "legacy-pair:guardrail_seed:18"
    return f"pair-{index}"


def _record(index: int) -> AIReviewedTranslationRecord:
    return AIReviewedTranslationRecord.model_validate(
        {
            "pair_id": _pair_id(index),
            "source_content_sha256": _source_hash(index),
            "uk_prompt": f"Безпечний тестовий запит {index}",
            "uk_image_text": None,
            "uk_rubric_text": "Українська тестова рубрика",
            "translation_method": "machine_deepl_ai_revised",
            "translated_at": datetime(2026, 8, 14, tzinfo=UTC).isoformat(),
            "ai_review": {
                "reviewer_kind": "ai_subagent",
                "reviewer_profile": "codex-fixture",
                "prompt_version": "fixture-v1",
                "reviewed_at": datetime(2026, 8, 14, tzinfo=UTC).isoformat(),
                "translation_revision": 1,
                "translation_hash": content_hash(
                    f"Безпечний тестовий запит {index}",
                    None,
                    "Українська тестова рубрика",
                ),
                "dimensions": {
                    "semantic_equivalence": "pass",
                    "naturalness": "pass",
                    "category_match": "pass",
                    "rubric_match": "pass",
                    "expected_behavior_match": "pass",
                },
                "confidence": 0.9,
                "notes": "fixture",
                "disposition": "accept",
            },
            "reviewer_status": "ai_reviewed_exploratory",
        }
    )


def test_ai_reviewed_file_requires_exploratory_provenance(tmp_path: Path) -> None:
    records = [
        _record(index).model_dump(mode="json")
        for index in range(37)
        if _pair_id(index) not in EXCLUDED_SOURCE_DEFECT_PAIR_IDS
    ]
    path = tmp_path / "ai-reviewed.json"
    path.write_text(json.dumps(records, ensure_ascii=False))

    loaded, digest = load_ai_reviewed_records(path)

    assert len(loaded) == 34
    assert len(digest) == 64
    records[0]["translation_method"] = "human"
    with pytest.raises(ValidationError, match="machine_deepl provenance"):
        AIReviewedTranslationRecord.model_validate(records[0])


async def test_ai_reviewed_import_is_idempotent_and_restricted() -> None:
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
                        pair_id=_pair_id(index),
                        translation_provenance_json={},
                        reviewer_status="hash_verified",
                        release_status="restricted",
                        content_sha256=_source_hash(index),
                        source_metadata_json={},
                    )
                    for index in range(37)
                ]
            )
        records = [
            _record(index)
            for index in range(37)
            if _pair_id(index) not in EXCLUDED_SOURCE_DEFECT_PAIR_IDS
        ]
        first = await import_ai_reviewed_translations(session, records, "b" * 64)
        second = await import_ai_reviewed_translations(session, records, "b" * 64)
        count = await session.scalar(
            select(func.count())
            .select_from(BenchmarkItem)
            .where(BenchmarkItem.provenance == "ai_reviewed_translation")
        )
        status = await session.scalar(
            select(BenchmarkItem.reviewer_status)
            .where(BenchmarkItem.provenance == "ai_reviewed_translation")
            .limit(1)
        )
        release = await session.scalar(
            select(BenchmarkItem.release_status)
            .where(BenchmarkItem.provenance == "ai_reviewed_translation")
            .limit(1)
        )

    assert first is True
    assert second is False
    assert count == 34
    assert status == "ai_reviewed_exploratory"
    assert release == "restricted"
    await engine.dispose()
