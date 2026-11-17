"""Run a deterministic fraction of the ADR-0025 bilingual dataset.

This is a measurement run, not the smoke. Two settings differ:

- `enforce_judge_language` is ON. The smoke deliberately disabled it to measure
  the adherence rate; now that the contract is validated at 30/30, a rationale in
  the wrong language is a defect and must stop its session (ADR-0026).
- `max_rounds` defaults to 3, the frozen PAIR design, rather than the smoke's 1.

The fraction is drawn deterministically from English item identity, stratified by
category, and covers both strata: harmful HarmBench items and the benign controls
that make over-refusal measurable. Cells execute in category order and every
completed cell is checkpointed, so an interrupted run resumes without re-paying.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from harness.agents.native.modal_curl_provider import ModalCurlClient
from harness.agents.native.modal_provider import (
    GPT_OSS_20B_JUDGE_CANDIDATE,
    QWEN36_27B_FP8_CANDIDATE,
)
from harness.agents.native.types import ModelClient
from harness.bilingual_dataset import DatasetItem
from harness.experiments.bilingual_execute import execute_smoke
from harness.experiments.bilingual_manifest import (
    BilingualStudyManifest,
    StudyCell,
    build_manifest,
)
from harness.experiments.bilingual_smoke import select_smoke_pairs, smoke_items
from harness.experiments.checkpoint import CheckpointLog
from harness.experiments.modal_cli import ModalCli, stop_all
from harness.experiments.model_roster import PHASE2_MODEL_ROSTER_V1
from harness.experiments.warmup import wait_until_ready
from harness.persistence.database import create_engine, create_schema, session_factory
from harness.persistence.repository import TraceRepository

WORKSPACE = Path(".translation-workspace")
DEFAULT_DATASET = WORKSPACE / "bilingual-dataset-v1.json"
APP_PATHS = {
    "target": "modal_apps/vllm_qwen36/app.py",
    "judge": "modal_apps/vllm_gpt_oss_judge/app.py",
}


def load_items(path: Path) -> tuple[list[DatasetItem], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = [
        DatasetItem(
            stable_id=row["stable_id"],
            pair_id=row["pair_id"],
            language=row["language"],
            stratum=row["stratum"],
            category=row["category"],
            prompt_text=row["prompt_text"],
            context_text=row["context_text"],
            rubric_text=row["rubric_text"],
            expected_behavior=row["expected_behavior"],
            reviewer_status=row["reviewer_status"],
            translation_provenance=row["translation_provenance"],
            license_id=row["license_id"],
        )
        for row in payload["items"]
    ]
    return items, payload["manifest"]


def select_fraction(items: list[DatasetItem], *, seed: int, fraction: float) -> list[DatasetItem]:

    chosen: list[DatasetItem] = []
    for stratum in sorted({item.stratum for item in items}):
        available = {item.pair_id for item in items if item.stratum == stratum}
        wanted = max(1, round(len(available) * fraction))
        pair_ids = select_smoke_pairs(items, seed=seed, pairs=wanted, stratum=stratum)
        chosen.extend(smoke_items([i for i in items if i.stratum == stratum], pair_ids))
    return sorted(chosen, key=lambda item: (item.category, item.language, item.stable_id))


def build(args: argparse.Namespace) -> tuple[BilingualStudyManifest, list[DatasetItem]]:
    items, dataset_manifest = load_items(args.dataset)
    selected = select_fraction(items, seed=args.seed, fraction=args.fraction)
    roster = PHASE2_MODEL_ROSTER_V1
    target_profile = {p.key: p for p in roster.matrix_profiles}["qwen36-27b-fp8"]
    manifest = build_manifest(
        selected,
        dataset_sha256=dataset_manifest["dataset_sha256"],
        subset_plan_sha256=dataset_manifest["subset_plan_sha256"],
        attacker=target_profile,
        target=target_profile,
        judge=roster.judge_profile,
        max_rounds=args.rounds,
        allow_assumed=dataset_manifest["assumed_validation"],
        enforce_judge_language=not args.no_enforce_language,
    )
    return manifest, selected


async def _execute(
    args: argparse.Namespace, manifest: BilingualStudyManifest, selected: list[DatasetItem]
) -> None:
    cli = ModalCli()
    endpoints: dict[str, str] = {}
    for role, app_path in sorted(APP_PATHS.items()):
        print(f"deploying {role}: {app_path}", flush=True)
        endpoints[role] = cli.deploy(app_path)
        print(f"  endpoint: {endpoints[role]}", flush=True)

    serving = {"target": QWEN36_27B_FP8_CANDIDATE, "judge": GPT_OSS_20B_JUDGE_CANDIDATE}

    def client_factory(*, role: str, cell: StudyCell) -> ModelClient:
        del cell
        key = "judge" if role == "judge" else "target"
        return ModalCurlClient(profile=serving[key], endpoint_url=endpoints[key])

    for role, url in sorted(endpoints.items()):
        print(f"warming {role} ...", flush=True)
        waited = await wait_until_ready(url, app_name=serving[role].app_name)
        print(f"  {role} ready after {waited:.0f}s", flush=True)

    engine = create_engine(args.database_url)
    await create_schema(engine)
    checkpoint = CheckpointLog(args.checkpoint)
    print(f"checkpoint: {checkpoint.summary()}", flush=True)
    make_session = session_factory(engine)

    @asynccontextmanager
    async def repository_factory() -> AsyncIterator[TraceRepository]:
        async with make_session() as session:
            yield TraceRepository(session)

    try:
        async with make_session() as session:
            result = await execute_smoke(
                manifest,
                items=selected,
                repository=TraceRepository(session),
                client_factory=client_factory,
                checkpoint=checkpoint,
                experiment_name="bilingual half-dataset run",
                max_concurrency=args.cell_concurrency,
                repository_factory=repository_factory,
            )
        print(json.dumps(result, indent=2))
    finally:
        await engine.dispose()
        names = [QWEN36_27B_FP8_CANDIDATE.app_name, GPT_OSS_20B_JUDGE_CANDIDATE.app_name]
        print("stopping apps:", names, flush=True)
        stop_all(cli, names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument(
        "--cell-concurrency",
        type=int,
        default=8,
        help=(
            "independent PAIR cells in flight; each cell remains serial internally "
            "and uses its own trace transaction (default: 8)"
        ),
    )
    parser.add_argument("--no-enforce-language", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--checkpoint", type=Path, default=WORKSPACE / "half-study-checkpoint.jsonl"
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default="sqlite+aiosqlite:///.translation-workspace/half-study.sqlite3",
    )
    args = parser.parse_args()
    if args.cell_concurrency < 1:
        parser.error("--cell-concurrency must be positive")

    manifest, selected = build(args)
    summary = {
        "mode": "plan" if not args.execute else "execute",
        "cells": manifest.envelope.cells,
        "pairs": len({c.pair_id for c in manifest.cells}),
        "by_stratum": {
            s: sum(1 for c in manifest.cells if c.stratum == s)
            for s in sorted({c.stratum for c in manifest.cells})
        },
        "by_language": {
            lang: sum(1 for c in manifest.cells if c.language == lang) for lang in ("en", "uk")
        },
        "cells_by_category": manifest.cells_by_category(),
        "max_rounds": manifest.envelope.max_rounds,
        "cell_concurrency": args.cell_concurrency,
        "enforce_judge_language": manifest.enforce_judge_language,
        "assumed_validation": manifest.assumed_validation,
        "envelope": manifest.envelope.as_json(),
        "manifest_sha256": manifest.manifest_sha256,
    }
    print(json.dumps(summary, indent=2))
    if not args.execute:
        print("\nPlan only; nothing spent. Re-run with --execute.")
        return
    asyncio.run(_execute(args, manifest, selected))


if __name__ == "__main__":
    main()
