"""Export the English HarmBench rows for translation and subset selection.

Writes a private JSON file into the Git-ignored translation workspace. Only the
frozen `thesis-core/phase1-v1-draft` English HarmBench rows are exported; the
historical legacy strata are excluded, because ADR-0025 builds the Ukrainian arm
on the MIT-licensed anchor only.

This is the one step in the ADR-0025 pipeline that needs loopback PostgreSQL.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from sqlalchemy import select

from harness.benchmark_import import (
    BENCHMARK_NAME,
    BENCHMARK_VERSION,
    RUBRIC_TEXT,
    fetch_pinned_sources,
    parse_harmbench_sources,
)
from harness.experiments.verification_subset import eligible_from_rows, select_subset
from harness.persistence.database import create_engine, session_factory
from harness.persistence.models import BenchmarkItem, BenchmarkVersion
from harness.settings import Settings

DEFAULT_OUTPUT = Path(".translation-workspace") / "harmbench-english-source.json"
HARMBENCH_SOURCE_NAME = "harmbench"


def _write_private(output: Path, rows: list[dict[str, Any]]) -> None:
    payload = json.dumps(
        {
            "benchmark_name": BENCHMARK_NAME,
            "benchmark_version": BENCHMARK_VERSION,
            "rubric_text": RUBRIC_TEXT,
            "row_count": len(rows),
            "rows": rows,
        },
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, output)
    output.chmod(0o600)


def export_from_upstream(output: Path) -> dict[str, Any]:

    items = parse_harmbench_sources(fetch_pinned_sources())
    rows = [
        {
            "stable_id": item.stable_id,
            "source_item_id": item.source_item_id,
            "prompt_text": item.prompt_text,
            "context_text": item.context_text,
            "content_sha256": item.content_sha256,
            "category": item.category,
            "functional_category": item.functional_category,
            "split": item.split,
            "modality": item.modality,
            "language": "en",
            "expected_behavior": "refuse_or_safe_redirect",
            "license_id": "MIT",
            "license_url": item.source_url,
        }
        for item in items
    ]
    _write_private(output, rows)
    plan = select_subset(eligible_from_rows(rows))
    return {
        "source": "upstream_pinned",
        "exported_rows": len(rows),
        "eligible_pool": plan.eligible_count,
        "subset_size": plan.subset_size,
        "allocation": dict(sorted(plan.allocation.items())),
        "plan_sha256": plan.plan_sha256,
    }


async def export(output: Path) -> dict[str, Any]:
    settings = Settings()
    engine = create_engine(settings.database_url)
    factory = session_factory(engine)
    try:
        async with factory() as session:
            version = (
                await session.execute(
                    select(BenchmarkVersion).where(
                        BenchmarkVersion.name == BENCHMARK_NAME,
                        BenchmarkVersion.version == BENCHMARK_VERSION,
                    )
                )
            ).scalar_one()
            items = (
                (
                    await session.execute(
                        select(BenchmarkItem)
                        .where(
                            BenchmarkItem.benchmark_version_id == version.id,
                            BenchmarkItem.language == "en",
                            BenchmarkItem.source_name == HARMBENCH_SOURCE_NAME,
                        )
                        .order_by(BenchmarkItem.stable_id)
                    )
                )
                .scalars()
                .all()
            )
    finally:
        await engine.dispose()

    rows = [
        {
            "stable_id": item.stable_id,
            "source_item_id": item.source_item_id,
            "prompt_text": item.prompt_text,
            "context_text": item.context_text,
            "content_sha256": item.content_sha256,
            "category": item.category,
            "functional_category": item.functional_category,
            "split": item.split,
            "modality": item.modality,
            "language": item.language,
            "expected_behavior": item.expected_behavior,
            "license_id": item.license_id,
            "license_url": item.license_url,
        }
        for item in items
    ]

    _write_private(output, rows)
    plan = select_subset(eligible_from_rows(rows))
    return {
        "exported_rows": len(rows),
        "eligible_pool": plan.eligible_count,
        "subset_size": plan.subset_size,
        "allocation": dict(sorted(plan.allocation.items())),
        "plan_sha256": plan.plan_sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--from-upstream",
        action="store_true",
        help="Fetch the commit-pinned CSVs instead of reading PostgreSQL.",
    )
    args = parser.parse_args()
    if args.from_upstream:
        print(json.dumps(export_from_upstream(args.output), indent=2))
        return
    print(json.dumps(asyncio.run(export(args.output)), indent=2))


if __name__ == "__main__":
    main()
