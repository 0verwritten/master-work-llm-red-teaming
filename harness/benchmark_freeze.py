from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from harness.ai_bilingual_import import EXCLUDED_SOURCE_DEFECT_PAIR_IDS
from harness.benchmark_import import BENCHMARK_NAME, BENCHMARK_VERSION
from harness.persistence.database import create_engine, session_factory
from harness.persistence.models import BenchmarkItem, BenchmarkVersion
from harness.settings import Settings


@dataclass(frozen=True)
class BenchmarkFreezeResult:
    benchmark_sha256: str
    harmbench_count: int
    english_pair_count: int
    ukrainian_pair_count: int
    total_count: int
    changed: bool


def _sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _freeze_payload(items: tuple[BenchmarkItem, ...]) -> list[dict[str, Any]]:
    return sorted(
        (
            {
                "content_sha256": item.content_sha256,
                "category": item.category,
                "expected_behavior": item.expected_behavior,
                "functional_category": item.functional_category,
                "language": item.language,
                "modality": item.modality,
                "pair_id": item.pair_id,
                "provenance": item.provenance,
                "release_status": item.release_status,
                "reviewer_status": item.reviewer_status,
                "source_item_id": item.source_item_id,
                "source_name": item.source_name,
                "source_version": item.source_version,
                "stable_id": item.stable_id,
            }
            for item in items
        ),
        key=lambda row: row["stable_id"],
    )


async def freeze_benchmark(session: AsyncSession) -> BenchmarkFreezeResult:
    async with session.begin():
        version = await session.scalar(
            select(BenchmarkVersion).where(
                BenchmarkVersion.name == BENCHMARK_NAME,
                BenchmarkVersion.version == BENCHMARK_VERSION,
            )
        )
        if version is None:
            raise RuntimeError("benchmark version is unavailable")
        items = tuple(
            (
                await session.scalars(
                    select(BenchmarkItem).where(BenchmarkItem.benchmark_version_id == version.id)
                )
            ).all()
        )
        harmbench = tuple(item for item in items if item.source_name == "harmbench")
        english = {
            item.pair_id: item
            for item in items
            if item.provenance == "historical_seed" and item.language == "en"
        }
        ukrainian = {
            item.pair_id: item
            for item in items
            if item.provenance == "ai_reviewed_translation" and item.language == "uk"
        }
        if len(items) != 581 or len(harmbench) != 510:
            raise RuntimeError("benchmark must contain 510 HarmBench and 581 total items")
        if None in english or len(english) != 37:
            raise RuntimeError("benchmark must contain 37 unique historical English pairs")
        eligible = set(english) - EXCLUDED_SOURCE_DEFECT_PAIR_IDS
        if set(ukrainian) != eligible or len(ukrainian) != 34:
            raise RuntimeError("benchmark must contain the exact 34 eligible Ukrainian pairs")
        if any(
            item.reviewer_status != "ai_reviewed_exploratory" or item.release_status != "restricted"
            for item in ukrainian.values()
        ):
            raise RuntimeError("Ukrainian pairs must remain restricted and exploratory")
        translation_hashes = {item.source_version for item in ukrainian.values()}
        if len(translation_hashes) != 1:
            raise RuntimeError("Ukrainian pairs must share one translation artifact hash")

        benchmark_sha256 = _sha256(_freeze_payload(items))
        freeze_manifest = {
            "benchmark_sha256": benchmark_sha256,
            "counts": {
                "english_historical": len(english),
                "harmbench": len(harmbench),
                "total": len(items),
                "ukrainian_exploratory": len(ukrainian),
            },
            "excluded_source_defect_pair_ids": sorted(EXCLUDED_SOURCE_DEFECT_PAIR_IDS),
            "protocol": "machine_deepl_ai_revised_exploratory_v1",
            "translation_artifact_sha256": translation_hashes.pop(),
        }
        existing = version.source_manifest_json.get("freeze")
        if version.status == "frozen":
            if existing != freeze_manifest or version.frozen_at is None:
                raise RuntimeError("frozen benchmark manifest does not match current content")
            changed = False
        elif version.status == "draft":
            manifest = dict(version.source_manifest_json)
            manifest["freeze"] = freeze_manifest
            version.source_manifest_json = manifest
            version.status = "frozen"
            version.frozen_at = datetime.now(UTC)
            changed = True
        else:
            raise RuntimeError(f"unsupported benchmark status: {version.status}")
        return BenchmarkFreezeResult(
            benchmark_sha256=benchmark_sha256,
            harmbench_count=len(harmbench),
            english_pair_count=len(english),
            ukrainian_pair_count=len(ukrainian),
            total_count=len(items),
            changed=changed,
        )


async def _persist() -> BenchmarkFreezeResult:
    engine = create_engine(Settings().database_url)
    try:
        async with session_factory(engine)() as session:
            return await freeze_benchmark(session)
    finally:
        await engine.dispose()


def main() -> None:
    argparse.ArgumentParser(description="Freeze the amended Phase 1 benchmark.").parse_args()
    result = asyncio.run(_persist())
    print(
        json.dumps(
            {
                "benchmark_sha256": result.benchmark_sha256,
                "counts": {
                    "english_historical": result.english_pair_count,
                    "harmbench": result.harmbench_count,
                    "total": result.total_count,
                    "ukrainian_exploratory": result.ukrainian_pair_count,
                },
                "status": "frozen" if result.changed else "already_frozen",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
