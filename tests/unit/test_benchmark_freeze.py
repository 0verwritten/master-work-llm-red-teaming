from __future__ import annotations

import hashlib
import uuid

from harness.ai_bilingual_import import EXCLUDED_SOURCE_DEFECT_PAIR_IDS
from harness.benchmark_freeze import freeze_benchmark
from harness.benchmark_import import BENCHMARK_NAME, BENCHMARK_VERSION
from harness.persistence.database import create_engine, create_schema, session_factory
from harness.persistence.models import BenchmarkItem, BenchmarkVersion


def _item(
    *,
    version_id: uuid.UUID,
    stable_id: str,
    source_name: str,
    provenance: str,
    language: str,
    pair_id: str | None,
    reviewer_status: str,
) -> BenchmarkItem:
    digest = hashlib.sha256(stable_id.encode()).hexdigest()
    return BenchmarkItem(
        benchmark_version_id=version_id,
        stable_id=stable_id,
        source_name=source_name,
        source_version="c" * 64 if language == "uk" else "fixture",
        source_item_id=stable_id,
        source_url="fixture",
        license_id="NOASSERTION",
        license_url="fixture",
        provenance=provenance,
        split="fixture",
        language=language,
        modality="text",
        category="fixture",
        functional_category="fixture",
        prompt_text="fixture",
        context_text=None,
        rubric_text="fixture",
        expected_behavior="refuse_or_safe_redirect",
        pair_id=pair_id,
        translation_provenance_json={},
        reviewer_status=reviewer_status,
        release_status="restricted",
        content_sha256=digest,
        source_metadata_json={},
    )


async def test_freeze_benchmark_is_content_addressed_and_idempotent() -> None:
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
                source_manifest_json={"fixture": True},
            )
            session.add(version)
            await session.flush()
            session.add_all(
                [
                    _item(
                        version_id=version.id,
                        stable_id=f"harmbench:{index}",
                        source_name="harmbench",
                        provenance="public_anchor",
                        language="en",
                        pair_id=None,
                        reviewer_status="upstream",
                    )
                    for index in range(510)
                ]
            )
            for index in range(37):
                pair_id = (
                    sorted(EXCLUDED_SOURCE_DEFECT_PAIR_IDS)[index] if index < 3 else f"pair-{index}"
                )
                session.add(
                    _item(
                        version_id=version.id,
                        stable_id=f"legacy-en:{index}",
                        source_name="fixture",
                        provenance="historical_seed",
                        language="en",
                        pair_id=pair_id,
                        reviewer_status="hash_verified",
                    )
                )
                if pair_id not in EXCLUDED_SOURCE_DEFECT_PAIR_IDS:
                    session.add(
                        _item(
                            version_id=version.id,
                            stable_id=f"legacy-uk-ai:{index}",
                            source_name="ai_reviewed_translation",
                            provenance="ai_reviewed_translation",
                            language="uk",
                            pair_id=pair_id,
                            reviewer_status="ai_reviewed_exploratory",
                        )
                    )

        first = await freeze_benchmark(session)
        second = await freeze_benchmark(session)

    assert first.changed is True
    assert second.changed is False
    assert first.benchmark_sha256 == second.benchmark_sha256
    assert first.total_count == 581
    assert first.harmbench_count == 510
    assert first.english_pair_count == 37
    assert first.ukrainian_pair_count == 34
    await engine.dispose()
