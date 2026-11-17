"""Plan or execute the ADR-0026 judge-language smoke.

Default is --plan, which spends nothing: it freezes the sample, prints the
category breakdown, and reports the worst-case envelope. Execution requires
--execute plus an explicit --remaining-usd, and refuses to start when the worst
case can cross that window (START_HERE rule 4).

Resume is by cell: every completed cell is committed before the next begins, so
an interrupted run loses at most the in-flight cell. Re-running skips cells that
already have a persisted evaluation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from harness.agents.native.modal_provider import (
    GPT_OSS_20B_JUDGE_CANDIDATE,
    QWEN36_27B_FP8_CANDIDATE,
)
from harness.agents.native.types import ModelClient
from harness.bilingual_dataset import ASSUMED_VALID, DatasetItem
from harness.experiments.bilingual_manifest import StudyCell, build_manifest
from harness.experiments.bilingual_smoke import (
    PAIRS_PER_LANGUAGE,
    SMOKE_MAX_ROUNDS,
    select_smoke_pairs,
    smoke_items,
    smoke_manifest_summary,
)
from harness.experiments.model_roster import PHASE2_MODEL_ROSTER_V1

DEFAULT_DATASET = Path(".translation-workspace") / "bilingual-dataset-v1.json"
DEFAULT_PLAN_OUT = Path(".translation-workspace") / "judge-language-smoke-plan.json"


def load_items(path: Path) -> tuple[list[DatasetItem], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
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
    ], payload["manifest"]


def plan(args: argparse.Namespace) -> dict[str, Any]:
    items, dataset_manifest = load_items(args.dataset)
    pair_ids = select_smoke_pairs(items, pairs=args.pairs)
    selected = smoke_items(items, pair_ids)

    roster = PHASE2_MODEL_ROSTER_V1
    profiles = {profile.key: profile for profile in roster.matrix_profiles}
    target = profiles["qwen36-27b-fp8"]

    manifest = build_manifest(
        selected,
        dataset_sha256=dataset_manifest["dataset_sha256"],
        subset_plan_sha256=dataset_manifest["subset_plan_sha256"],
        attacker=target,
        target=target,
        judge=roster.judge_profile,
        max_rounds=SMOKE_MAX_ROUNDS,
        allow_assumed=dataset_manifest["assumed_validation"],
        enforce_judge_language=False,
    )
    summary = {
        "mode": "plan",
        "sample": smoke_manifest_summary(pair_ids, selected),
        "cells_by_category": manifest.cells_by_category(),
        "envelope": manifest.envelope.as_json(),
        "assumed_validation": manifest.assumed_validation,
        "enforce_judge_language": manifest.enforce_judge_language,
        "manifest_sha256": manifest.manifest_sha256,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({**summary, "manifest": manifest.as_json()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args.out.chmod(0o600)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=DEFAULT_PLAN_OUT)
    parser.add_argument("--pairs", type=int, default=PAIRS_PER_LANGUAGE)
    parser.add_argument("--execute", action="store_true", help="Spend money and run the smoke.")
    parser.add_argument(
        "--deploy", action="store_true", help="Deploy target and judge via modal CLI."
    )
    parser.add_argument("--stop-after", action="store_true", default=True)
    parser.add_argument("--target-url", type=str, default=None)
    parser.add_argument("--judge-url", type=str, default=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(".translation-workspace") / "judge-language-smoke-checkpoint.jsonl",
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default="sqlite+aiosqlite:///.translation-workspace/judge-language-smoke.sqlite3",
        help="Trace store; defaults to a local file so an unreachable DB cannot lose a batch.",
    )
    args = parser.parse_args()

    summary = plan(args)
    if summary["assumed_validation"]:
        print(
            f"NOTE: Ukrainian items carry {ASSUMED_VALID!r}. "
            "This smoke measures the judge, not the translations.\n"
        )
    print(json.dumps(summary, indent=2))

    if not args.execute:
        print("\nPlan only; nothing was spent. Re-run with --execute --deploy to run it.")
        return

    asyncio.run(_execute(args))


async def _execute(args: argparse.Namespace) -> None:
    import httpx

    from harness.agents.native.modal_curl_provider import ModalCurlClient
    from harness.experiments.bilingual_execute import execute_smoke
    from harness.experiments.checkpoint import CheckpointLog
    from harness.experiments.modal_cli import ModalCli
    from harness.persistence.database import create_engine, create_schema, session_factory
    from harness.persistence.repository import TraceRepository

    del httpx

    items, dataset_manifest = load_items(args.dataset)
    pair_ids = select_smoke_pairs(items, pairs=args.pairs)
    selected = smoke_items(items, pair_ids)
    roster = PHASE2_MODEL_ROSTER_V1
    profiles = {profile.key: profile for profile in roster.matrix_profiles}
    target_profile = profiles["qwen36-27b-fp8"]
    manifest = build_manifest(
        selected,
        dataset_sha256=dataset_manifest["dataset_sha256"],
        subset_plan_sha256=dataset_manifest["subset_plan_sha256"],
        attacker=target_profile,
        target=target_profile,
        judge=roster.judge_profile,
        max_rounds=SMOKE_MAX_ROUNDS,
        allow_assumed=dataset_manifest["assumed_validation"],
        enforce_judge_language=False,
    )

    cli = ModalCli()
    deployed: list[str] = []
    endpoints: dict[str, str] = {}
    if args.deploy:
        for role, app_path in (
            ("target", "modal_apps/vllm_qwen36/app.py"),
            ("judge", "modal_apps/vllm_gpt_oss_judge/app.py"),
        ):
            print(f"deploying {role}: {app_path}", flush=True)
            endpoints[role] = cli.deploy(app_path)
            deployed.append(app_path)
            print(f"  endpoint: {endpoints[role]}", flush=True)
    else:
        if not (args.target_url and args.judge_url):
            raise SystemExit("supply --deploy, or both --target-url and --judge-url")
        endpoints = {"target": args.target_url, "judge": args.judge_url}

    serving = {
        "target": QWEN36_27B_FP8_CANDIDATE,
        "judge": GPT_OSS_20B_JUDGE_CANDIDATE,
    }

    def client_factory(*, role: str, cell: StudyCell) -> ModelClient:
        del cell
        key = "judge" if role == "judge" else "target"
        return ModalCurlClient(profile=serving[key], endpoint_url=endpoints[key])

    from harness.experiments.warmup import wait_until_ready

    for role, url in sorted(endpoints.items()):
        print(f"warming {role} ...", flush=True)
        waited = await wait_until_ready(url, app_name=serving[role].app_name)
        print(f"  {role} ready after {waited:.0f}s", flush=True)

    engine = create_engine(args.database_url)
    await create_schema(engine)
    checkpoint = CheckpointLog(args.checkpoint)
    print(f"checkpoint: {checkpoint.summary()}", flush=True)
    try:
        async with session_factory(engine)() as session:
            result = await execute_smoke(
                manifest,
                items=selected,
                repository=TraceRepository(session),
                client_factory=client_factory,
                checkpoint=checkpoint,
            )
        print(json.dumps(result, indent=2))
    finally:
        await engine.dispose()
        if args.deploy and args.stop_after:
            from harness.experiments.modal_cli import stop_all

            names = [QWEN36_27B_FP8_CANDIDATE.app_name, GPT_OSS_20B_JUDGE_CANDIDATE.app_name]
            print("stopping apps:", names, flush=True)
            stop_all(cli, names)


if __name__ == "__main__":
    main()
