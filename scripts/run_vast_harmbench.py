"""Plan or execute a frozen HarmBench Vast run.

Rental creation, SSH tunnels, destruction, and spend recording deliberately stay
outside this process. The runner accepts only loopback endpoints and exactly
labelled running instances through ``VastOpenAIClient``.
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

import httpx
from sqlalchemy import text

from harness.agents.native.types import ModelClient
from harness.agents.native.vast_provider import VastOpenAIClient, VastServingProfile
from harness.bilingual_dataset import DatasetItem
from harness.experiments.bilingual_execute import execute_smoke
from harness.experiments.bilingual_manifest import BilingualStudyManifest, StudyCell, build_manifest
from harness.experiments.bilingual_smoke import select_smoke_pairs
from harness.experiments.checkpoint import CheckpointLog
from harness.experiments.vast_harmbench_profiles import (
    VAST_GPT_OSS_L40S,
    VAST_GPT_OSS_L40S_FROZEN,
    VAST_GPT_OSS_RTX6000_ADA,
    VAST_GPT_OSS_RTX6000_ADA_FROZEN,
    VAST_MISTRAL_SMALL_H100,
    VAST_MISTRAL_SMALL_H100_FROZEN,
    VAST_QWEN_H100,
    VAST_QWEN_H100_FROZEN,
)
from harness.persistence.database import create_engine, session_factory
from harness.persistence.repository import TraceRepository
from harness.settings import Settings
from harness.spend.vast import VastInstance, fetch_instances

WORKSPACE = Path(".translation-workspace")
DEFAULT_DATASET = WORKSPACE / "bilingual-dataset-v1.json"
REVIEWED_DATASET = WORKSPACE / "bilingual-dataset-v1-harmbench-reviewed.json"
FULL_DATASET = WORKSPACE / "bilingual-dataset-v1-harmbench-full-510.json"


def _judge_profile(args: argparse.Namespace) -> VastServingProfile:
    if args.judge_gpu == "rtx6000ada":
        return VAST_GPT_OSS_RTX6000_ADA
    return VAST_GPT_OSS_RTX6000_ADA if args.language == "uk" else VAST_GPT_OSS_L40S


ATTACKERS = {
    "qwen": (VAST_QWEN_H100, VAST_QWEN_H100_FROZEN),
    "mistral": (VAST_MISTRAL_SMALL_H100, VAST_MISTRAL_SMALL_H100_FROZEN),
}


def _separate_attacker(args: argparse.Namespace) -> bool:

    return bool(args.attacker != "qwen")


def _judge_frozen(args: argparse.Namespace):  # type: ignore[no-untyped-def]
    return (
        VAST_GPT_OSS_RTX6000_ADA_FROZEN
        if _judge_profile(args) == VAST_GPT_OSS_RTX6000_ADA
        else VAST_GPT_OSS_L40S_FROZEN
    )


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


def build(args: argparse.Namespace) -> tuple[BilingualStudyManifest, list[DatasetItem]]:
    items, dataset_manifest = load_items(args.dataset)
    pair_ids = set(
        select_smoke_pairs(
            items,
            seed=args.seed,
            pairs=args.records,
            stratum="harmbench_subset",
        )
    )
    selected = sorted(
        (
            item
            for item in items
            if item.language == args.language
            and item.stratum == "harmbench_subset"
            and item.pair_id in pair_ids
        ),
        key=lambda item: (item.category, item.stable_id),
    )
    if len(selected) != args.records:
        raise RuntimeError(
            f"{args.language} HarmBench selection did not produce the requested records"
        )
    default_namespace = (
        "vast-harmbench-uk-qwen-h100-gptoss-rtx6000ada-v1"
        if args.language == "uk"
        else "vast-harmbench-en-qwen-h100-gptoss-l40s-v1"
    )
    if args.retry_latest_errors_from is not None:
        reference = build_manifest(
            selected,
            dataset_sha256=dataset_manifest["dataset_sha256"],
            subset_plan_sha256=dataset_manifest["subset_plan_sha256"],
            attacker=VAST_QWEN_H100_FROZEN,
            target=VAST_QWEN_H100_FROZEN,
            judge=(
                VAST_GPT_OSS_RTX6000_ADA_FROZEN
                if args.language == "uk"
                else VAST_GPT_OSS_L40S_FROZEN
            ),
            max_rounds=args.rounds,
            allow_assumed=args.allow_ai_reviewed,
            enforce_judge_language=True,
            cell_namespace=default_namespace,
        )
        latest = {
            record.cell_id: record
            for record in CheckpointLog(args.retry_latest_errors_from).records()
        }
        retry_ids = {
            cell_id
            for cell_id, record in latest.items()
            if record.status == "error"
            or (record.status == "stopped" and record.stop_reason == "role_request_error")
        }
        retry_stable_ids = {
            cell.item_stable_id for cell in reference.cells if cell.cell_id in retry_ids
        }
        selected = [item for item in selected if item.stable_id in retry_stable_ids]
        if not selected:
            raise RuntimeError("reference checkpoint has no retryable latest cells")
    if args.cell_namespace:
        namespace = args.cell_namespace
    else:
        namespace = default_namespace
        if _separate_attacker(args):
            namespace += f"-attacker-{args.attacker}"
        if args.max_attack_regenerations:
            namespace += f"-reject-repeats-r{args.max_attack_regenerations}"
    manifest = build_manifest(
        selected,
        dataset_sha256=dataset_manifest["dataset_sha256"],
        subset_plan_sha256=dataset_manifest["subset_plan_sha256"],
        attacker=ATTACKERS[args.attacker][1],
        target=VAST_QWEN_H100_FROZEN,
        judge=_judge_frozen(args),
        max_rounds=args.rounds,
        allow_assumed=args.allow_ai_reviewed,
        enforce_judge_language=True,
        cell_namespace=namespace,
        max_attack_regenerations=args.max_attack_regenerations,
    )
    return manifest, selected


def _instance(instance_id: int, profile: VastServingProfile) -> VastInstance:
    matches = [instance for instance in fetch_instances() if instance.id == instance_id]
    if len(matches) != 1:
        raise RuntimeError(f"Vast instance {instance_id} is not present")
    instance = matches[0]
    if instance.gpu_name != profile.gpu_name:
        raise RuntimeError(
            f"Vast instance {instance_id} is {instance.gpu_name}, expected {profile.gpu_name}"
        )
    return instance


async def _wait_ready(client: httpx.AsyncClient, url: str, *, timeout_seconds: float) -> float:
    started = time.monotonic()
    while time.monotonic() - started < timeout_seconds:
        try:
            response = await client.get(f"{url.rstrip('/')}/health", timeout=10)
            if response.status_code == 200:
                return time.monotonic() - started
        except httpx.TransportError:
            pass
        await asyncio.sleep(5)
    raise TimeoutError(f"Vast endpoint did not become ready within {timeout_seconds:.0f}s")


async def _execute(
    args: argparse.Namespace,
    manifest: BilingualStudyManifest,
    selected: list[DatasetItem],
) -> None:
    assert args.target_instance_id is not None
    assert args.judge_instance_id is not None
    target_instance = _instance(args.target_instance_id, VAST_QWEN_H100)
    attacker_profile = ATTACKERS[args.attacker][0]
    attacker_instance = (
        _instance(args.attacker_instance_id, attacker_profile)
        if _separate_attacker(args)
        else target_instance
    )
    attacker_url = args.attacker_url if _separate_attacker(args) else args.target_url
    judge_profile = _judge_profile(args)
    judge_instance = _instance(args.judge_instance_id, judge_profile)
    settings = Settings()
    engine = create_engine(settings.database_url)
    make_session = session_factory(engine)
    checkpoint = CheckpointLog(args.checkpoint)

    @asynccontextmanager
    async def repository_factory() -> AsyncIterator[TraceRepository]:
        async with make_session() as session:
            yield TraceRepository(session)

    limits = httpx.Limits(
        max_connections=max(16, args.cell_concurrency * 2),
        max_keepalive_connections=0,
    )
    try:
        async with make_session() as session:
            await session.execute(text("SELECT 1"))
        async with httpx.AsyncClient(limits=limits) as http_client:
            target_wait = await _wait_ready(
                http_client, args.target_url, timeout_seconds=args.readiness_timeout_seconds
            )
            judge_wait = await _wait_ready(
                http_client, args.judge_url, timeout_seconds=args.readiness_timeout_seconds
            )
            attacker_wait = (
                await _wait_ready(
                    http_client, attacker_url, timeout_seconds=args.readiness_timeout_seconds
                )
                if _separate_attacker(args)
                else target_wait
            )
            print(
                json.dumps(
                    {
                        "target_ready_seconds": round(target_wait, 1),
                        "judge_ready_seconds": round(judge_wait, 1),
                        "attacker_ready_seconds": round(attacker_wait, 1),
                    }
                ),
                flush=True,
            )

            def client_factory(*, role: str, cell: StudyCell) -> ModelClient:
                del cell
                if role == "judge":
                    return VastOpenAIClient(
                        profile=judge_profile,
                        endpoint_url=args.judge_url,
                        instance=judge_instance,
                        http_client=http_client,
                        vllm_api_key=settings.vast_vllm_api_key_value(),
                    )
                if role == "attacker":
                    return VastOpenAIClient(
                        profile=attacker_profile,
                        endpoint_url=attacker_url,
                        instance=attacker_instance,
                        http_client=http_client,
                        vllm_api_key=settings.vast_vllm_api_key_value(),
                    )
                return VastOpenAIClient(
                    profile=VAST_QWEN_H100,
                    endpoint_url=args.target_url,
                    instance=target_instance,
                    http_client=http_client,
                    vllm_api_key=settings.vast_vllm_api_key_value(),
                )

            async with make_session() as session:
                result = await execute_smoke(
                    manifest,
                    items=selected,
                    repository=TraceRepository(session),
                    client_factory=client_factory,
                    checkpoint=checkpoint,
                    experiment_name=f"Vast {args.language} HarmBench 100 v1",
                    max_concurrency=args.cell_concurrency,
                    repository_factory=repository_factory,
                    max_cells=args.max_cells,
                )
        print(json.dumps(result, indent=2))
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--language", choices=("en", "uk"), default="en")
    parser.add_argument("--records", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--cell-concurrency", type=int, default=8)
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--retry-latest-errors-from", type=Path)
    parser.add_argument("--cell-namespace")
    parser.add_argument(
        "--max-attack-regenerations",
        type=int,
        default=0,
        help="ADR-0037: reject repeated PAIR attacks and regenerate up to this many times",
    )
    parser.add_argument("--judge-gpu", choices=("default", "rtx6000ada"), default="default")
    parser.add_argument("--target-url", default="http://127.0.0.1:18000")
    parser.add_argument("--judge-url", default="http://127.0.0.1:18001")
    parser.add_argument("--target-instance-id", type=int)
    parser.add_argument("--judge-instance-id", type=int)
    parser.add_argument(
        "--attacker",
        choices=tuple(ATTACKERS),
        default="qwen",
        help="ADR-0038: qwen reuses the target endpoint; others need their own rental",
    )
    parser.add_argument("--attacker-url", default="http://127.0.0.1:18002")
    parser.add_argument("--attacker-instance-id", type=int)
    parser.add_argument("--readiness-timeout-seconds", type=float, default=1200)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--allow-ai-reviewed",
        action="store_true",
        help="explicitly permit ai_reviewed_exploratory Ukrainian rows",
    )
    args = parser.parse_args()
    if args.dataset is None:
        args.dataset = REVIEWED_DATASET if args.language == "uk" else DEFAULT_DATASET
    if args.checkpoint is None:
        args.checkpoint = WORKSPACE / f"vast-harmbench-{args.language}-100-checkpoint.jsonl"
    if not 1 <= args.records <= 510:
        parser.error("--records must be between 1 and 510")
    if args.rounds < 1 or args.cell_concurrency < 1:
        parser.error("--rounds and --cell-concurrency must be positive")
    if args.max_attack_regenerations < 0:
        parser.error("--max-attack-regenerations cannot be negative")
    if (
        args.max_attack_regenerations or args.attacker != "qwen"
    ) and args.retry_latest_errors_from is not None:
        parser.error("--retry-latest-errors-from applies only to the original protocol")
    if args.execute and args.attacker != "qwen" and args.attacker_instance_id is None:
        parser.error("--attacker other than qwen requires --attacker-instance-id")
    if args.max_cells is not None and args.max_cells < 1:
        parser.error("--max-cells must be positive")
    if args.execute and (args.target_instance_id is None or args.judge_instance_id is None):
        parser.error("--execute requires --target-instance-id and --judge-instance-id")

    manifest, selected = build(args)
    summary = {
        "mode": "execute" if args.execute else "plan",
        "records": len(selected),
        "language": args.language,
        "stratum": "harmbench_subset",
        "max_rounds": args.rounds,
        "cell_concurrency": args.cell_concurrency,
        "max_cells": args.max_cells,
        "max_attack_regenerations": args.max_attack_regenerations,
        "cell_namespace": manifest.cell_namespace,
        "manifest_sha256": manifest.manifest_sha256,
        "attacker_profile_id": ATTACKERS[args.attacker][0].profile_id,
        "target_profile_id": VAST_QWEN_H100.profile_id,
        "judge_profile_id": (_judge_profile(args).profile_id),
        "worst_case_calls": manifest.envelope.worst_case_calls,
        "human_verified_items": all(item.reviewer_status == "human_verified" for item in selected),
        "reviewer_status_counts": {
            status: sum(item.reviewer_status == status for item in selected)
            for status in sorted({item.reviewer_status for item in selected})
        },
    }
    print(json.dumps(summary, indent=2))
    if not args.execute:
        print("\nPlan only; no provider request was made.")
        return
    asyncio.run(_execute(args, manifest, selected))


if __name__ == "__main__":
    main()
