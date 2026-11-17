"""ADR-0039: run the ADR-0037 repeat-rejection test on Modal with a concurrent control.

Two Ukrainian arms over the 120 human-verified HarmBench items share one Qwen
attacker/target deployment and one GPT-OSS judge deployment:

- control: the original PAIR protocol;
- intervention: repeat rejection with two regenerations.

Endpoints are reached with ``modal curl`` (local Modal credentials only), and the
trace is persisted to a local SQLite database created with the harness schema, so
every provider request still has one ``llm_calls`` row. Cells run in small
batches; no new batch starts after the wall-clock deadline, and both apps are
stopped on every exit path.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
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
from harness.experiments.checkpoint import CheckpointLog
from harness.experiments.modal_cli import ModalCli, stop_all
from harness.experiments.model_roster import PHASE2_MODEL_ROSTER_V1
from harness.experiments.warmup import wait_until_ready
from harness.persistence.database import create_engine, create_schema, session_factory
from harness.persistence.repository import TraceRepository

WORKSPACE = Path(".translation-workspace")
DATASET = WORKSPACE / "bilingual-dataset-v1-harmbench-reviewed.json"
DATABASE_URL = "sqlite+aiosqlite:///.translation-workspace/adr0039-modal.sqlite3"
NAMESPACE = "modal-harmbench-uk-human-verified-adr0039"
APP_PATHS = {
    "target": "modal_apps/vllm_qwen36/app.py",
    "judge": "modal_apps/vllm_gpt_oss_judge/app.py",
}
SERVING = {"target": QWEN36_27B_FP8_CANDIDATE, "judge": GPT_OSS_20B_JUDGE_CANDIDATE}
ARMS = {"control": 0, "intervention": 2}


def load_items(path: Path) -> tuple[list[DatasetItem], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = [
        DatasetItem(**{key: row[key] for key in DatasetItem.__dataclass_fields__})
        for row in payload["items"]
    ]
    return items, payload["manifest"]


def build(
    dataset: Path, *, rounds: int = 3
) -> tuple[dict[str, BilingualStudyManifest], list[DatasetItem]]:
    items, dataset_manifest = load_items(dataset)
    selected = [
        item
        for item in items
        if item.language == "uk"
        and item.stratum == "harmbench_subset"
        and item.reviewer_status == "human_verified"
    ]
    if len(selected) != 120:
        raise RuntimeError(f"expected 120 human-verified Ukrainian items, found {len(selected)}")
    qwen = {p.key: p for p in PHASE2_MODEL_ROSTER_V1.matrix_profiles}["qwen36-27b-fp8"]
    manifests = {
        arm: build_manifest(
            selected,
            dataset_sha256=dataset_manifest["dataset_sha256"],
            subset_plan_sha256=dataset_manifest["subset_plan_sha256"],
            attacker=qwen,
            target=qwen,
            judge=PHASE2_MODEL_ROSTER_V1.judge_profile,
            max_rounds=rounds,
            enforce_judge_language=True,
            cell_namespace=NAMESPACE,
            max_attack_regenerations=regenerations,
        )
        for arm, regenerations in ARMS.items()
    }
    return manifests, selected


def checkpoint_path(arm: str) -> Path:
    return WORKSPACE / f"adr0039-modal-uk-{arm}-checkpoint.jsonl"


def pending_cells(manifest: BilingualStudyManifest, checkpoint: CheckpointLog) -> int:
    done = checkpoint.completed_cell_ids()
    return sum(cell.cell_id not in done for cell in manifest.cells)


async def execute(args: argparse.Namespace) -> None:
    manifests, selected = build(args.dataset)
    cli = ModalCli()
    print(json.dumps({"modal_profile": cli.profile()}), flush=True)
    deadline = time.monotonic() + args.max_hours * 3600
    try:
        endpoints = {}
        for role, app_path in sorted(APP_PATHS.items()):
            print(f"deploying {role}: {app_path}", flush=True)
            endpoints[role] = cli.deploy(app_path)
        for role, url in sorted(endpoints.items()):
            waited = await wait_until_ready(url, app_name=SERVING[role].app_name)
            print(json.dumps({"ready": role, "seconds": round(waited)}), flush=True)

        def client_factory(*, role: str, cell: StudyCell) -> ModelClient:
            del cell
            key = "judge" if role == "judge" else "target"
            return ModalCurlClient(profile=SERVING[key], endpoint_url=endpoints[key])

        engine = create_engine(args.database_url)
        await create_schema(engine)
        make_session = session_factory(engine)

        @asynccontextmanager
        async def repository_factory() -> AsyncIterator[TraceRepository]:
            async with make_session() as session:
                yield TraceRepository(session)

        async def run_arm(arm: str, max_cells: int) -> dict[str, Any]:
            async with make_session() as session:
                return await execute_smoke(
                    manifests[arm],
                    items=selected,
                    repository=TraceRepository(session),
                    client_factory=client_factory,
                    checkpoint=CheckpointLog(checkpoint_path(arm)),
                    experiment_name=f"ADR-0039 Modal Ukrainian {arm}",
                    max_concurrency=min(args.cell_concurrency, max_cells),
                    repository_factory=repository_factory,
                    max_cells=max_cells,
                )

        try:
            await asyncio.gather(*(run_arm(arm, 1) for arm in ARMS))
            for arm in ARMS:
                records = CheckpointLog(checkpoint_path(arm)).records()
                last = records[-1] if records else None
                print(
                    json.dumps(
                        {
                            "smoke": arm,
                            "status": last.status if last else None,
                            "stop_reason": last.stop_reason if last else None,
                            "judge_language": last.judge_language_adherence if last else None,
                        }
                    ),
                    flush=True,
                )
                if (
                    last is None
                    or last.status not in {"completed", "stopped"}
                    or (last.stop_reason == "role_request_error")
                ):
                    raise RuntimeError(f"smoke failed for {arm}; no batch released")
            if args.smoke:
                return
            while time.monotonic() < deadline:
                remaining = {
                    arm: pending_cells(manifests[arm], CheckpointLog(checkpoint_path(arm)))
                    for arm in ARMS
                }
                print(json.dumps({"pending": remaining}), flush=True)
                active = [arm for arm, count in remaining.items() if count]
                if not active:
                    break
                results = await asyncio.gather(*(run_arm(arm, args.batch_cells) for arm in active))
                if all(result["executed_cells"] == 0 for result in results):
                    print("no cell completed in this batch; stopping", flush=True)
                    break
            else:
                print("wall-clock deadline reached; no new batch started", flush=True)
        finally:
            await engine.dispose()
    finally:
        names = [profile.app_name for profile in SERVING.values()]
        print(json.dumps({"stopping_apps": names}), flush=True)
        stop_all(cli, names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--database-url", default=DATABASE_URL)
    parser.add_argument("--cell-concurrency", type=int, default=8)
    parser.add_argument("--batch-cells", type=int, default=16)
    parser.add_argument("--max-hours", type=float, default=3.0)
    parser.add_argument("--smoke", action="store_true", help="one cell per arm, then stop")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    manifests, selected = build(args.dataset)
    print(
        json.dumps(
            {
                "items": len(selected),
                "manifests": {arm: value.manifest_sha256 for arm, value in manifests.items()},
                "worst_case_calls": {
                    arm: value.envelope.worst_case_calls for arm, value in manifests.items()
                },
                "pending": {
                    arm: pending_cells(value, CheckpointLog(checkpoint_path(arm)))
                    for arm, value in manifests.items()
                },
            },
            indent=2,
        )
    )
    if not args.execute:
        print("\nPlan only; nothing deployed.")
        return
    asyncio.run(execute(args))


if __name__ == "__main__":
    main()
