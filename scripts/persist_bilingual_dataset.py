"""Persist the ADR-0025 bilingual dataset into loopback PostgreSQL.

Creates a NEW benchmark version. `thesis-core/phase1-v1-draft` is frozen and
carries the Phase 3-5 results; it is never modified here.

The insert is transactional and idempotent: re-running with an unchanged dataset
reports `already_present` and writes nothing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from harness.bilingual_dataset import ASSUMED_VALID, DATASET_VERSION
from harness.persistence.database import create_engine, session_factory
from harness.persistence.models import BenchmarkItem, BenchmarkVersion
from harness.settings import Settings

BENCHMARK_NAME = "thesis-core"
DEFAULT_DATASET = Path(".translation-workspace") / "bilingual-dataset-v1.json"


async def _persist(session: AsyncSession, payload: dict[str, Any]) -> dict[str, Any]:
    manifest = payload["manifest"]
    items = payload["items"]

    existing = (
        await session.execute(
            select(BenchmarkVersion).where(
                BenchmarkVersion.name == BENCHMARK_NAME,
                BenchmarkVersion.version == DATASET_VERSION,
            )
        )
    ).scalar_one_or_none()

    if existing is not None:
        stored = existing.source_manifest_json.get("dataset_sha256")
        if stored == manifest["dataset_sha256"]:
            return {"status": "already_present", "dataset_sha256": stored}
        raise RuntimeError(
            "a different bilingual dataset is already stored under this version; "
            "bump DATASET_VERSION rather than mutating a frozen version"
        )

    version = BenchmarkVersion(
        name=BENCHMARK_NAME,
        version=DATASET_VERSION,
        schema_version=1,
        status="draft",
        source_manifest_json=manifest,
    )
    session.add(version)
    await session.flush()

    for item in items:
        session.add(
            BenchmarkItem(
                benchmark_version_id=version.id,
                stable_id=item["stable_id"],
                source_name="harmbench" if item["stratum"] == "harmbench_subset" else "legacy",
                source_version=manifest["subset_plan_sha256"],
                source_item_id=item["pair_id"],
                source_url="",
                license_id=item["license_id"],
                license_url="",
                provenance=item["translation_provenance"],
                split=item["stratum"],
                language=item["language"],
                modality="text",
                category=item["category"],
                functional_category=item["stratum"],
                prompt_text=item["prompt_text"],
                context_text=item["context_text"],
                rubric_text=item["rubric_text"],
                expected_behavior=item["expected_behavior"],
                pair_id=item["pair_id"],
                translation_provenance_json={
                    "provenance": item["translation_provenance"],
                    "reviewer_status": item["reviewer_status"],
                    "assumed_validation": manifest["assumed_validation"],
                },
                reviewer_status=item["reviewer_status"],
                release_status="restricted",
                content_sha256=item["content_sha256"],
                source_metadata_json={"stratum": item["stratum"]},
            )
        )
    await session.commit()
    return {"status": "inserted", "items": len(items)}


async def run(payload: dict[str, Any]) -> dict[str, Any]:
    engine = create_engine(Settings().database_url)
    try:
        factory = session_factory(engine)
        async with factory() as session:
            result = await _persist(session, payload)
    finally:
        await engine.dispose()
    manifest = payload["manifest"]
    return {
        **result,
        "benchmark": f"{BENCHMARK_NAME}/{DATASET_VERSION}",
        "item_count": manifest["item_count"],
        "pair_count": manifest["pair_count"],
        "assumed_validation": manifest["assumed_validation"],
        "ukrainian_human_verified_count": manifest["ukrainian_human_verified_count"],
        "dataset_sha256": manifest["dataset_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    payload = json.loads(args.dataset.read_text(encoding="utf-8"))
    if payload["manifest"]["assumed_validation"]:
        print(
            "NOTE: this dataset carries reviewer_status "
            f"{ASSUMED_VALID!r} on every Ukrainian item. "
            "It is provisional and is not independent human review.\n"
        )
    print(json.dumps(asyncio.run(run(payload)), indent=2))


if __name__ == "__main__":
    main()
