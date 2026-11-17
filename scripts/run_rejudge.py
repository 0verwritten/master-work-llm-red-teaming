"""Re-score the frozen bilingual responses with a second judge.

No regeneration: attacker and target are never deployed. Only the second judge
runs, against `target_response` values already stored by the bilingual run. One
GPU app instead of two, one call per stored round.

The question is not which judge is correct — there are no reference labels. It is
whether the `partial` asymmetry (12/60 English, 0/60 Ukrainian) survives a change
of judge, which separates a target property from a judge threshold.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

sys.path.insert(0, "scripts")

from harness.agents.native.gateway import TracedModelGateway
from harness.agents.native.modal_curl_provider import ModalCurlClient
from harness.agents.native.modal_provider import (
    GEMMA4_31B_JUDGE_V2,
    GPT_OSS_20B_JUDGE_CANDIDATE,
)
from harness.experiments.batch_guard import ConsecutiveFailureBreaker, with_retries
from harness.experiments.bilingual_config import build_cell_config
from harness.experiments.checkpoint import CheckpointLog, CheckpointRecord
from harness.experiments.diagnose import diagnose, render
from harness.experiments.graded_rubric import GRADED_RUBRIC_VERSION, rubric_for, rubric_sha256
from harness.experiments.modal_cli import ModalCli, stop_all
from harness.experiments.rejudge import load_frozen_responses, rescore, summarise
from harness.experiments.warmup import wait_until_ready
from harness.persistence.database import create_engine, session_factory
from harness.persistence.repository import TraceRepository

WORKSPACE = Path(".translation-workspace")
JUDGES = {
    "gemma4": (GEMMA4_31B_JUDGE_V2, "modal_apps/vllm_gemma4_judge/app.py"),
    "gpt-oss": (GPT_OSS_20B_JUDGE_CANDIDATE, "modal_apps/vllm_gpt_oss_judge/app.py"),
}
DATASET_ARGS = Namespace(
    dataset=WORKSPACE / "bilingual-dataset-v1.json",
    fraction=0.5,
    rounds=3,
    seed=20260819,
    no_enforce_language=False,
)


def manifest_and_prompts() -> tuple[Any, dict[str, str]]:
    from run_bilingual_study import build

    manifest, selected = build(DATASET_ARGS)
    by_stable = {item.stable_id: item.prompt_text for item in selected}
    return manifest, {c.cell_id: by_stable[c.item_stable_id] for c in manifest.cells}


def key(cell_id: str, response_sha256: str) -> str:
    return f"{cell_id}:{response_sha256[:8]}"


async def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest, prompts = manifest_and_prompts()
    engine = create_engine(args.database_url)
    checkpoint = CheckpointLog(args.checkpoint)
    done = {r.cell_id for r in checkpoint.records() if r.status == "completed"}

    async with session_factory(engine)() as session:
        frozen = await load_frozen_responses(TraceRepository(session), prompts)
    pending = [f for f in frozen if key(f.cell_id, f.response_sha256) not in done]
    print(f"stored rounds {len(frozen)} | already re-judged {len(done)} | pending {len(pending)}")

    if args.plan:
        await engine.dispose()
        return {
            "mode": "plan",
            "pending": len(pending),
            "judge": JUDGES[args.judge][0].name,
            "rubric": GRADED_RUBRIC_VERSION if args.graded else "original-binary",
            "rubric_sha256": rubric_sha256() if args.graded else None,
        }

    profile, judge_app = JUDGES[args.judge]
    cli = ModalCli()
    print(f"deploying judge {args.judge}: {judge_app}", flush=True)
    endpoint = cli.deploy(judge_app)
    print(f"  endpoint: {endpoint}", flush=True)
    waited = await wait_until_ready(endpoint, app_name=profile.app_name)
    print(f"  ready after {waited:.0f}s", flush=True)

    records: list[dict[str, Any]] = []
    breaker = ConsecutiveFailureBreaker()
    stopped_early = False
    config = build_cell_config(manifest, manifest.cells[0])
    try:
        async with session_factory(engine)() as session:
            repository = TraceRepository(session)
            experiment = await repository.create_experiment(
                name="second-judge rescoring",
                research_question="Does the partial asymmetry survive a change of judge?",
                hypothesis="If partial is a target property it reappears under a second judge.",
                created_by="rejudge",
            )
            snapshot = await repository.get_or_create_config(name="rejudge-v1", config=config)
            for index, item in enumerate(pending):
                run_row = await repository.create_run(
                    experiment_id=experiment.id,
                    config_id=snapshot.id,
                    benchmark_item_id=item.cell_id,
                    random_seed=17,
                    code_version="rejudge-v1",
                )
                gateway = TracedModelGateway(
                    ModalCurlClient(profile=profile, endpoint_url=endpoint),
                    repository,
                )
                try:

                    async def attempt(
                        it: Any = item,
                        rr: Any = run_row,
                        gw: Any = gateway,
                        ix: int = index,
                    ) -> dict[str, Any]:
                        return await rescore(
                            it,
                            gateway=gw,
                            run=rr,
                            repository=repository,
                            step_index=ix,
                            rubric_override=rubric_for(it.language) if args.graded else None,
                        )

                    record = await with_retries(attempt)
                except Exception as exc:
                    breaker.record_failure(f"{type(exc).__name__}: {exc}")
                    checkpoint.append(
                        CheckpointRecord(
                            cell_id=key(item.cell_id, item.response_sha256),
                            language=item.language,
                            category="rejudge",
                            status="error",
                            error=f"{type(exc).__name__}: {exc}"[:300],
                        )
                    )

                    if breaker.tripped:
                        print(
                            f"\nstopping after {breaker.consecutive} consecutive failures",
                            flush=True,
                        )
                        stopped_early = True
                        break
                    continue
                breaker.record_success()
                records.append(record)
                checkpoint.append(
                    CheckpointRecord(
                        cell_id=key(item.cell_id, item.response_sha256),
                        language=item.language,
                        category="rejudge",
                        status="completed",
                        judge_language_adherence=record["judge_language_adherence"],
                        label=record["rejudge_label"],
                        estimated_cost_usd=record["estimated_cost_usd"],
                        stop_reason=record["original_label"],
                    )
                )
                if index % 25 == 0:
                    print(f"  {index}/{len(pending)}", flush=True)
    finally:
        await engine.dispose()
        print("stopping judge app", flush=True)
        stop_all(cli, [profile.app_name])

    result: dict[str, Any] = {
        "mode": "execute",
        "judge": profile.profile_id,
        "rubric": GRADED_RUBRIC_VERSION if args.graded else "original-binary",
        "rescored": len(records),
        "stopped_early": stopped_early,
        "failures": breaker.as_json(),
        "checkpoint": checkpoint.summary(),
        "summary": summarise(records) if records else None,
    }
    if stopped_early:
        print("\n=== why the batch stopped ===", flush=True)
        report = diagnose(profile.app_name)
        print(render(report), flush=True)
        result["diagnosis"] = report
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--judge", choices=sorted(JUDGES), default="gemma4")
    parser.add_argument(
        "--graded",
        action="store_true",
        help="Use the rubric that defines all three levels (graded-three-level-v1).",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--database-url",
        type=str,
        default="sqlite+aiosqlite:///.translation-workspace/half-study.sqlite3",
    )
    args = parser.parse_args()
    if args.checkpoint is None:
        rubric = "graded" if args.graded else "original"
        args.checkpoint = WORKSPACE / f"rejudge-{args.judge}-{rubric}-checkpoint.jsonl"
    print(json.dumps(asyncio.run(run(args)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
