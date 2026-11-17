from __future__ import annotations

import argparse
import asyncio
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from sqlalchemy import select

from harness.agents.native.adversarial import ExperimentLanguage
from harness.benchmark_import import BENCHMARK_NAME, BENCHMARK_VERSION
from harness.experiments.adversarial_matrix import MatrixItem
from harness.experiments.bilingual_pilot import (
    ReviewedPair,
    build_bilingual_pilot_plan,
    estimate_pilot_cost,
)
from harness.experiments.model_roster import PHASE2_MODEL_ROSTER_V1
from harness.persistence.database import create_engine, session_factory
from harness.persistence.models import BenchmarkItem, BenchmarkVersion
from harness.settings import Settings

BENCHMARK_SHA256 = "c895708fef6fc918b927350998655defd4e78031c925c320611da31a1cfd0c4c"
SELECTION_SEED = 20260728
RUN_SEED = 17
RUBRIC_VERSION = "pilot-v1"
TARGET_KEY = "qwen36-27b-fp8"
CROSS_ATTACKER_KEY = "gemma4-31b-it"
STAGE_PAIR_ID = "legacy-pair:adversarial_pairs:h1"


async def build_frozen_manifest() -> dict[str, Any]:
    engine = create_engine(Settings().database_url)
    try:
        async with session_factory(engine)() as session:
            version = await session.scalar(
                select(BenchmarkVersion).where(
                    BenchmarkVersion.name == BENCHMARK_NAME,
                    BenchmarkVersion.version == BENCHMARK_VERSION,
                )
            )
            if version is None or version.status != "frozen":
                raise RuntimeError("frozen benchmark is unavailable")
            freeze = version.source_manifest_json.get("freeze", {})
            if freeze.get("benchmark_sha256") != BENCHMARK_SHA256:
                raise RuntimeError("benchmark hash does not match the pilot contract")
            items = tuple(
                (
                    await session.scalars(
                        select(BenchmarkItem).where(
                            BenchmarkItem.benchmark_version_id == version.id,
                            BenchmarkItem.pair_id.is_not(None),
                        )
                    )
                ).all()
            )
    finally:
        await engine.dispose()

    english = {
        item.pair_id: item
        for item in items
        if item.language == "en" and item.provenance == "historical_seed"
    }
    ukrainian = {
        item.pair_id: item
        for item in items
        if item.language == "uk" and item.provenance == "ai_reviewed_translation"
    }
    if set(ukrainian) - set(english) or len(ukrainian) != 34:
        raise RuntimeError("expected exactly 34 matched exploratory pairs")
    pairs = tuple(
        ReviewedPair(
            pair_id=pair_id or "",
            english=_matrix_item(english[pair_id], ExperimentLanguage.ENGLISH),
            ukrainian=_matrix_item(item, ExperimentLanguage.UKRAINIAN),
            expected_behavior=item.expected_behavior,
        )
        for pair_id, item in sorted(ukrainian.items())
    )
    roster = PHASE2_MODEL_ROSTER_V1
    profiles = {profile.key: profile for profile in roster.matrix_profiles}
    plan = build_bilingual_pilot_plan(
        pairs=pairs,
        target=profiles[TARGET_KEY],
        cross_attacker=profiles[CROSS_ATTACKER_KEY],
        judge=roster.judge_profile,
        selection_seed=SELECTION_SEED,
        run_seed=RUN_SEED,
        rubric_version=RUBRIC_VERSION,
        review_basis="ai_reviewed_exploratory",
    )
    cost = estimate_pilot_cost(
        plan,
        startup_seconds_per_profile=300,
        request_seconds_per_call=30,
        active_cost_usd_per_second=Decimal("0.00150"),
        noncompute_allowance_usd=Decimal("0.75"),
        observed_window_use_usd=Decimal("5.05167558"),
        provider_cap_usd=Decimal("30.00"),
        required_reserve_usd=Decimal("6.00"),
    )
    stage_cells = sorted(
        cell.cell_id for cell in plan.matrix.cells if cell.item.pair_id == STAGE_PAIR_ID
    )
    if len(stage_cells) != 4:
        raise RuntimeError("staging pair must expand to exactly four run cells")
    return {
        "manifest_schema_version": 2,
        "frozen_on": "2026-08-14",
        "benchmark": {
            "name": BENCHMARK_NAME,
            "sha256": BENCHMARK_SHA256,
            "version": BENCHMARK_VERSION,
        },
        "cost_envelope": cost.manifest_json(),
        "cross_attacker_key": CROSS_ATTACKER_KEY,
        "execution_protocol": {
            "code_revision_policy": (
                "record git HEAD and require execution-critical paths to match it"
            ),
            "decoding": {
                "attacker_temperature": 0.7,
                "judge_temperature": 0.0,
                "target_temperature": 0.0,
                "top_p": None,
            },
            "per_run_budgets": {
                "max_estimated_cost_usd": 1.5,
                "max_input_tokens": 200000,
                "max_output_tokens": 20000,
                "max_total_llm_calls": 9,
                "max_wall_time_seconds": 3600,
            },
            "stage": {
                "cell_ids": stage_cells,
                "max_llm_calls": 36,
                "max_target_queries": 12,
                "name": "paired_four_run_preflight",
                "pair_id": STAGE_PAIR_ID,
            },
            "ukrainian_incremental_budget": {
                "core_english_item_count": 357,
                "incremental_ukrainian_item_count": 34,
                "ratio": "0.09523809523809524",
                "shared_profile_startup_increment": "0",
            },
        },
        "pilot": plan.manifest_json(),
        "protocol": "adr-0018-ai-reviewed-exploratory",
        "roster_sha256": roster.roster_sha256,
        "target_key": TARGET_KEY,
    }


def _matrix_item(item: BenchmarkItem, language: ExperimentLanguage) -> MatrixItem:
    return MatrixItem(
        stable_id=item.stable_id,
        content_sha256=item.content_sha256,
        language=language,
        category=item.category,
        pair_id=item.pair_id,
        reviewer_status=item.reviewer_status,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the exploratory bilingual pilot.")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    repository_root = Path(__file__).resolve().parents[2]
    if not output.is_relative_to(repository_root):
        raise ValueError("pilot manifest must remain inside the repository")
    manifest = asyncio.run(build_frozen_manifest())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "cell_count": len(manifest["pilot"]["matrix"]["cells"]),
                "output": str(output),
                "plan_sha256": manifest["pilot"]["plan_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
