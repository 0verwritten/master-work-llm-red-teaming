"""Run ADR-0040's bilingual, budget-matched attack-method comparison on Modal."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from harness.agents.native.adversarial import AttackMethod, BaselineInputs, ExperimentLanguage
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
DATABASE_URL = "sqlite+aiosqlite:///.translation-workspace/adr0040-modal.sqlite3"
NAMESPACE = "modal-harmbench-human-verified-adr0040"
APP_PATHS = {
    "target": "modal_apps/vllm_qwen36/app.py",
    "judge": "modal_apps/vllm_gpt_oss_judge/app.py",
}
SERVING = {"target": QWEN36_27B_FP8_CANDIDATE, "judge": GPT_OSS_20B_JUDGE_CANDIDATE}
METHODS = {
    "pair": (AttackMethod.PAIR, 0),
    "single_shot": (AttackMethod.SINGLE_SHOT, 0),
    "static": (AttackMethod.STATIC, 0),
    "random_mutation": (AttackMethod.RANDOM_MUTATION, 0),
    "pair_repeat_rejection": (AttackMethod.PAIR, 2),
}
BASELINE_SOURCE_VERSION = "adr0040-reviewed-direct-prompt-v1"


def load_items(path: Path) -> tuple[list[DatasetItem], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = [
        DatasetItem(**{key: row[key] for key in DatasetItem.__dataclass_fields__})
        for row in payload["items"]
    ]
    return items, payload["manifest"]


def build(
    dataset: Path, *, rounds: int = 3
) -> tuple[dict[str, BilingualStudyManifest], dict[str, list[DatasetItem]]]:
    items, dataset_manifest = load_items(dataset)
    verified_pairs = {
        item.pair_id
        for item in items
        if item.language == "uk"
        and item.stratum == "harmbench_subset"
        and item.reviewer_status == "human_verified"
    }
    selected = {
        language: [
            item
            for item in items
            if item.language == language
            and item.stratum == "harmbench_subset"
            and item.pair_id in verified_pairs
        ]
        for language in ("en", "uk")
    }
    if len(verified_pairs) != 120 or any(len(values) != 120 for values in selected.values()):
        raise RuntimeError("expected exactly 120 complete human-verified bilingual pairs")
    qwen = {profile.key: profile for profile in PHASE2_MODEL_ROSTER_V1.matrix_profiles}[
        "qwen36-27b-fp8"
    ]
    manifests: dict[str, BilingualStudyManifest] = {}
    for language, language_items in selected.items():
        for arm, (method, regenerations) in METHODS.items():
            key = f"{language}-{arm}"
            manifests[key] = build_manifest(
                language_items,
                dataset_sha256=dataset_manifest["dataset_sha256"],
                subset_plan_sha256=dataset_manifest["subset_plan_sha256"],
                attacker=qwen,
                target=qwen,
                judge=PHASE2_MODEL_ROSTER_V1.judge_profile,
                max_rounds=rounds,
                enforce_judge_language=True,
                cell_namespace=f"{NAMESPACE}-{key}",
                max_attack_regenerations=regenerations,
                attack_method=method.value,
            )
    return manifests, selected


def checkpoint_path(key: str) -> Path:
    return WORKSPACE / f"adr0040-modal-{key}-checkpoint.jsonl"


def pending_cells(manifest: BilingualStudyManifest, checkpoint: CheckpointLog) -> int:
    done = checkpoint.completed_cell_ids()
    return sum(cell.cell_id not in done for cell in manifest.cells)


def baseline_inputs(item: DatasetItem, cell: StudyCell) -> BaselineInputs:
    del cell
    return BaselineInputs(
        candidates=(item.prompt_text, item.prompt_text, item.prompt_text),
        source_version=BASELINE_SOURCE_VERSION,
        language=ExperimentLanguage(item.language),
    )


async def execute(args: argparse.Namespace) -> None:
    manifests, selected = build(args.dataset)
    cli = ModalCli()
    print(json.dumps({"modal_profile": cli.profile()}), flush=True)
    deadline = time.monotonic() + args.max_hours * 3600

    smoke_keys = {key.strip() for key in args.smoke_keys.split(",") if key.strip()}

    selected_arms = {key.strip() for key in args.arms.split(",") if key.strip()}
    if selected_arms:
        missing = selected_arms - set(manifests)
        if missing:
            raise RuntimeError(f"unknown arms: {sorted(missing)}")
        manifests = {key: value for key, value in manifests.items() if key in selected_arms}
        smoke_keys &= selected_arms
        if not smoke_keys:
            smoke_keys = {sorted(manifests)[0]}
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
            return ModalCurlClient(
                profile=SERVING[key],
                endpoint_url=endpoints[key],
                request_timeout_seconds=args.request_timeout_seconds,
            )

        engine = create_engine(args.database_url)
        await create_schema(engine)
        make_session = session_factory(engine)

        @asynccontextmanager
        async def repository_factory() -> AsyncIterator[TraceRepository]:
            async with make_session() as session:
                yield TraceRepository(session)

        async def run_arm(key: str, max_cells: int) -> dict[str, Any]:
            language, arm = key.split("-", 1)
            method = METHODS[arm][0]
            async with make_session() as session:
                return await execute_smoke(
                    manifests[key],
                    items=selected[language],
                    repository=TraceRepository(session),
                    client_factory=client_factory,
                    checkpoint=CheckpointLog(checkpoint_path(key)),
                    experiment_name=f"ADR-0040 Modal {language} {arm}",
                    max_concurrency=min(args.cell_concurrency, max_cells),
                    repository_factory=repository_factory,
                    max_cells=max_cells,
                    max_role_attempts=args.max_role_attempts,
                    retry_backoff_seconds=args.retry_backoff_seconds,
                    max_estimated_cost_usd=args.max_cell_estimated_cost_usd,
                    method=method,
                    baseline_factory=(
                        baseline_inputs
                        if method in {AttackMethod.STATIC, AttackMethod.RANDOM_MUTATION}
                        else None
                    ),
                )

        try:
            gate_keys = [key for key in sorted(manifests) if key in smoke_keys]
            if not gate_keys:
                raise RuntimeError(f"no smoke arm matches {sorted(smoke_keys)}")
            for key in gate_keys:
                await run_arm(key, 1)
            healthy = 0
            for key in gate_keys:
                records = CheckpointLog(checkpoint_path(key)).records()
                last = records[-1] if records else None
                print(
                    json.dumps(
                        {
                            "smoke": key,
                            "status": last.status if last else None,
                            "stop_reason": last.stop_reason if last else None,
                            "judge_language": last.judge_language_adherence if last else None,
                        }
                    ),
                    flush=True,
                )
                if last is not None and last.status in {"completed", "stopped"}:
                    healthy += 1
                elif last is not None and last.stop_reason != "role_request_error":
                    raise RuntimeError(f"smoke failed permanently for {key}: {last.error}")

            if not healthy:
                raise RuntimeError("no gate arm completed a cell; no batch released")
            if args.smoke:
                return
            while time.monotonic() < deadline:
                remaining = {
                    key: pending_cells(manifest, CheckpointLog(checkpoint_path(key)))
                    for key, manifest in manifests.items()
                }
                print(json.dumps({"pending": remaining}), flush=True)
                active = [key for key, count in remaining.items() if count]
                if not active:
                    break
                results = [await run_arm(key, args.batch_cells) for key in active]
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
    parser.add_argument("--cell-concurrency", type=int, default=6)
    parser.add_argument("--batch-cells", type=int, default=24)
    parser.add_argument("--max-hours", type=float, default=8.0)
    parser.add_argument(
        "--max-role-attempts",
        type=int,
        default=3,
        help="attempts per role request; Modal's router reports transient states under load",
    )
    parser.add_argument("--retry-backoff-seconds", type=float, default=2.0)
    parser.add_argument(
        "--max-cell-estimated-cost-usd",
        type=float,
        default=3.0,
        help="runaway-loop guard per cell; the estimate inflates with retries and queuing",
    )
    parser.add_argument(
        "--arms",
        default="",
        help="comma-separated arms to execute this session; empty means all",
    )
    parser.add_argument(
        "--smoke-keys",
        default="en-pair,en-static,uk-pair,uk-static",
        help="arms that must pass one unseen cell before any batch is released",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=180.0,
        help="abandon and retry a stalled request well before the profile ceiling",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    manifests, selected = build(args.dataset)
    print(
        json.dumps(
            {
                "items_per_language": {key: len(value) for key, value in selected.items()},
                "manifests": {key: manifest.manifest_sha256 for key, manifest in manifests.items()},
                "worst_case_calls": {
                    key: manifest.envelope.worst_case_calls for key, manifest in manifests.items()
                },
                "worst_case_calls_total": sum(
                    manifest.envelope.worst_case_calls for manifest in manifests.values()
                ),
                "pending": {
                    key: pending_cells(manifest, CheckpointLog(checkpoint_path(key)))
                    for key, manifest in manifests.items()
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
