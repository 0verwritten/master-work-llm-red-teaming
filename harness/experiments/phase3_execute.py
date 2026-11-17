from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Literal, cast

import httpx
from sqlalchemy import func, select

from harness.agents.native.adversarial import ExperimentLanguage
from harness.agents.native.modal_provider import (
    GEMMA4_31B_CANDIDATE,
    GPT_OSS_20B_JUDGE_CANDIDATE,
    QWEN36_27B_FP8_CANDIDATE,
    ModalOpenAIClient,
    ModalServingProfile,
)
from harness.experiments.adversarial_matrix import FrozenModelProfile
from harness.experiments.phase3_pilot import (
    STAGE_NAME,
    CellRecord,
    HydratedPilotCell,
    Phase3PilotManifest,
    PilotClientFactory,
    build_cell_config,
    execute_cell,
    hydrate_pilot,
    hydrate_stage,
    load_phase3_manifest,
)
from harness.language import detect_language_adherence
from harness.persistence.database import create_engine, session_factory
from harness.persistence.models import (
    AdversarialRound,
    AdversarialSession,
    CallStatus,
    Evaluation,
    Experiment,
    LLMCall,
    Phase3ExecutionBatch,
    Phase3ExecutionCell,
    Phase3ExecutionCellStatus,
    Run,
    RunStatus,
)
from harness.persistence.repository import Phase3ExecutionCellSpec, TraceRepository
from harness.settings import Settings

PROFILE_BY_ID: dict[str, ModalServingProfile] = {
    profile.profile_id: profile
    for profile in (
        QWEN36_27B_FP8_CANDIDATE,
        GEMMA4_31B_CANDIDATE,
        GPT_OSS_20B_JUDGE_CANDIDATE,
    )
}
EXECUTION_CRITICAL_PATHS = (
    "config/phase3-bilingual-pilot.json",
    "harness/agents/native/adversarial.py",
    "harness/agents/native/gateway.py",
    "harness/agents/native/modal_provider.py",
    "harness/config/schema.py",
    "harness/experiments",
    "harness/persistence",
    "harness/settings.py",
    "modal_apps/vllm_gemma4/app.py",
    "modal_apps/vllm_gpt_oss_judge/app.py",
    "modal_apps/vllm_qwen36/app.py",
)


def committed_clean_revision() -> str:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *EXECUTION_CRITICAL_PATHS],
        check=True,
    )
    if len(revision) != 40:
        raise RuntimeError("Git did not return a full commit revision")
    return revision


def _cell_specs(manifest: Phase3PilotManifest) -> tuple[Phase3ExecutionCellSpec, ...]:
    stage_ids = set(manifest.execution_protocol.stage.cell_ids)
    stage_index = {
        cell_id: index for index, cell_id in enumerate(manifest.execution_protocol.stage.cell_ids)
    }
    pilot_index = 0
    specs: list[Phase3ExecutionCellSpec] = []
    for cell in manifest.pilot.matrix.cells:
        stage = STAGE_NAME if cell.cell_id in stage_ids else "pilot"
        ordinal = stage_index[cell.cell_id] if stage == STAGE_NAME else pilot_index
        pilot_index += int(stage == "pilot")
        payload = cell.model_dump(mode="json")
        specs.append(
            Phase3ExecutionCellSpec(
                cell_id=cell.cell_id,
                stage=stage,
                ordinal=ordinal,
                cell_sha256=cell.cell_id,
                cell_spec_json=payload,
            )
        )
    return tuple(specs)


async def materialize_stage(
    *,
    manifest: Phase3PilotManifest,
    repository: TraceRepository,
) -> tuple[Phase3ExecutionBatch, bool]:
    return await repository.materialize_phase3_execution(
        manifest_sha256=manifest.manifest_sha256,
        plan_sha256=manifest.pilot.plan_sha256,
        benchmark_sha256=manifest.benchmark.sha256,
        roster_sha256=manifest.roster_sha256,
        execution_envelope_json=manifest.execution_protocol.model_dump(mode="json"),
        cells=_cell_specs(manifest),
    )


def _endpoint_by_profile(settings: Settings) -> dict[str, str]:
    bindings = {
        QWEN36_27B_FP8_CANDIDATE.profile_id: settings.modal_qwen36_endpoint_url,
        GEMMA4_31B_CANDIDATE.profile_id: settings.modal_gemma4_endpoint_url,
        GPT_OSS_20B_JUDGE_CANDIDATE.profile_id: settings.modal_gpt_oss_judge_endpoint_url,
    }
    missing = [profile_id for profile_id, value in bindings.items() if value is None]
    if missing:
        raise RuntimeError("all three frozen Modal endpoint URLs are required")
    return {key: value for key, value in bindings.items() if value is not None}


async def execute_real_stage(
    manifest_path: Path,
    *,
    stage: Literal["paired_four_run_preflight", "pilot"] = "paired_four_run_preflight",
) -> dict[str, Any]:
    revision = committed_clean_revision()
    manifest = load_phase3_manifest(manifest_path)
    settings = Settings()
    token_id, token_secret = settings.require_modal_proxy_credentials()
    endpoints = _endpoint_by_profile(settings)
    engine = create_engine(settings.database_url)
    stage_results: list[dict[str, Any]] = []
    try:
        async with session_factory(engine)() as session, httpx.AsyncClient() as http:
            repository = TraceRepository(session)
            hydrate = hydrate_stage if stage == STAGE_NAME else hydrate_pilot
            hydrated = {
                cell.manifest_cell.cell_id: cell for cell in await hydrate(manifest, session)
            }
            batch, _ = await materialize_stage(manifest=manifest, repository=repository)
            if stage == "pilot":
                await _require_completed_stage(repository=repository, batch=batch)
            experiment = await _get_or_create_experiment(repository, manifest)
            await session.commit()

            def factory(
                *,
                profile: FrozenModelProfile,
                role: Literal["attacker", "target", "judge"],
                cell: CellRecord,
            ) -> ModalOpenAIClient:
                del role, cell
                if profile.endpoint_profile is None:
                    raise RuntimeError("frozen Modal profile has no endpoint identity")
                serving = PROFILE_BY_ID.get(profile.endpoint_profile)
                if serving is None or serving.checkpoint_revision != profile.revision:
                    raise RuntimeError("serving profile does not match the frozen roster")
                return ModalOpenAIClient(
                    profile=serving,
                    endpoint_url=endpoints[profile.endpoint_profile],
                    proxy_token_id=token_id,
                    proxy_token_secret=token_secret,
                    http_client=http,
                )

            while True:
                cell_state = await repository.claim_next_phase3_execution_cell(
                    batch_id=batch.id,
                    stage=stage,
                )
                if cell_state is None:
                    break
                await session.commit()
                hydrated_cell = hydrated[cell_state.cell_id]
                try:
                    run = await _prepare_run(
                        repository=repository,
                        experiment=experiment,
                        cell=hydrated_cell,
                        revision=revision,
                    )
                    await repository.link_phase3_execution_run(cell=cell_state, run=run)
                    await session.commit()
                    result = await execute_cell(
                        manifest=manifest,
                        cell=hydrated_cell,
                        repository=repository,
                        experiment=experiment,
                        client_factory=cast(PilotClientFactory, factory),
                        code_revision=revision,
                        run=run,
                    )
                    await repository.mark_phase3_execution_cell_terminal(
                        cell=cell_state,
                        run=run,
                        stop_reason=result.result.stop_reason,
                    )
                    await session.commit()
                    if result.result.status != RunStatus.COMPLETED:
                        raise RuntimeError("staged run did not complete")
                    stage_results.append(
                        {
                            "attacker_language_adherence": list(result.attacker_language_adherence),
                            "cell_id": result.cell_id,
                            "estimated_cost_usd": result.result.estimated_cost_usd,
                            "llm_calls": result.result.llm_calls,
                            "rounds": result.result.rounds,
                            "status": result.result.status,
                            "target_language_adherence": list(result.target_language_adherence),
                        }
                    )
                except Exception:
                    if cell_state.status != Phase3ExecutionCellStatus.TERMINAL:
                        await repository.mark_phase3_execution_cell_indeterminate(
                            cell=cell_state,
                            stop_reason="execution_requires_manual_reconciliation",
                        )
                        await session.commit()
                    raise
            batch = await repository.reconcile_phase3_execution_batch(batch_id=batch.id)
            await session.commit()
            audit = await _stage_audit(repository=repository, batch=batch, stage=stage)
            return {
                "audit": audit,
                "batch_status": str(batch.status),
                "manifest_sha256": manifest.manifest_sha256,
                "results": stage_results,
            }
    finally:
        await engine.dispose()


async def _get_or_create_experiment(
    repository: TraceRepository,
    manifest: Phase3PilotManifest,
) -> Experiment:
    name = f"Phase 3 staged pilot {manifest.manifest_sha256[:12]}"
    existing = await repository.session.scalar(select(Experiment).where(Experiment.name == name))
    if existing is not None:
        return existing
    return await repository.create_experiment(
        name=name,
        research_question="RQ-A staged exploratory bilingual calibration",
        hypothesis="The native staged block is operationally trace-complete and judgeable.",
        created_by="phase3-executor",
    )


async def _prepare_run(
    *,
    repository: TraceRepository,
    experiment: Experiment,
    cell: HydratedPilotCell,
    revision: str,
) -> Run:
    config = build_cell_config(cell)
    snapshot = await repository.get_or_create_config(
        name=f"phase3-{cell.manifest_cell.cell_id}",
        config=config,
    )
    return await repository.create_run(
        experiment_id=experiment.id,
        config_id=snapshot.id,
        benchmark_item_id=cell.manifest_cell.item.stable_id,
        random_seed=cell.manifest_cell.seed,
        code_version=revision,
    )


async def _stage_audit(
    *,
    repository: TraceRepository,
    batch: Phase3ExecutionBatch,
    stage: str,
) -> dict[str, Any]:
    cells = tuple(
        (
            await repository.session.scalars(
                select(Phase3ExecutionCell).where(
                    Phase3ExecutionCell.batch_id == batch.id,
                    Phase3ExecutionCell.stage == stage,
                )
            )
        ).all()
    )
    run_ids = [cell.run_id for cell in cells if cell.run_id is not None]
    call_count = await repository.session.scalar(
        select(func.count()).select_from(LLMCall).where(LLMCall.run_id.in_(run_ids))
    )
    unresolved_calls = await repository.session.scalar(
        select(func.count())
        .select_from(LLMCall)
        .where(LLMCall.run_id.in_(run_ids), LLMCall.status == "started")
    )
    rationale_rows = tuple(
        (
            await repository.session.execute(
                select(AdversarialSession.language, Evaluation.rationale)
                .join(AdversarialRound, AdversarialRound.session_id == AdversarialSession.id)
                .join(Evaluation, Evaluation.round_id == AdversarialRound.id)
                .where(AdversarialSession.run_id.in_(run_ids))
            )
        ).all()
    )
    return {
        "batch_status": str(batch.status),
        "cell_count": len(cells),
        "cell_statuses": sorted({str(cell.status) for cell in cells}),
        "judge_language_adherence": _judge_language_counts(
            [(ExperimentLanguage(language), rationale) for language, rationale in rationale_rows]
        ),
        "persisted_llm_calls": int(call_count or 0),
        "raw_content_exported": False,
        "unresolved_started_calls": int(unresolved_calls or 0),
    }


async def _require_completed_stage(
    *,
    repository: TraceRepository,
    batch: Phase3ExecutionBatch,
) -> None:
    cells = tuple(
        (
            await repository.session.scalars(
                select(Phase3ExecutionCell).where(
                    Phase3ExecutionCell.batch_id == batch.id,
                    Phase3ExecutionCell.stage == STAGE_NAME,
                )
            )
        ).all()
    )
    if len(cells) != 4 or any(cell.status != Phase3ExecutionCellStatus.TERMINAL for cell in cells):
        raise RuntimeError("paired four-run preflight is not complete")
    judge_rationales: list[tuple[ExperimentLanguage, str]] = []
    for cell in cells:
        if cell.run_id is None:
            raise RuntimeError("paired four-run preflight lacks run identity")
        run = await repository.session.get(Run, cell.run_id)
        if run is None or run.status != RunStatus.COMPLETED or run.finished_at is None:
            raise RuntimeError("paired four-run preflight run is not completed")
        language = ExperimentLanguage(str(cell.cell_spec_json["item"]["language"]))
        adversarial_session = await repository.session.scalar(
            select(AdversarialSession).where(AdversarialSession.run_id == run.id)
        )
        if (
            adversarial_session is None
            or adversarial_session.status != RunStatus.COMPLETED
            or adversarial_session.language != language.value
            or adversarial_session.method != "pair"
        ):
            raise RuntimeError("paired four-run preflight session identity is incomplete")
        rounds = tuple(
            (
                await repository.session.scalars(
                    select(AdversarialRound)
                    .where(AdversarialRound.session_id == adversarial_session.id)
                    .order_by(AdversarialRound.round_number)
                )
            ).all()
        )
        if not rounds or [round_.round_number for round_ in rounds] != list(
            range(1, len(rounds) + 1)
        ):
            raise RuntimeError("paired four-run preflight rounds are incomplete")
        if len(rounds) > adversarial_session.max_rounds:
            raise RuntimeError("paired four-run preflight exceeded its round budget")
        evaluations = tuple(
            (
                await repository.session.scalars(
                    select(Evaluation)
                    .join(AdversarialRound)
                    .where(AdversarialRound.session_id == adversarial_session.id)
                )
            ).all()
        )
        if len(evaluations) != len(rounds):
            raise RuntimeError("paired four-run preflight evaluation trace is incomplete")
        evaluation_by_round = {evaluation.round_id: evaluation for evaluation in evaluations}
        if set(evaluation_by_round) != {round_.id for round_ in rounds}:
            raise RuntimeError("paired four-run preflight evaluation binding is incomplete")

        calls = tuple(
            (
                await repository.session.scalars(select(LLMCall).where(LLMCall.run_id == run.id))
            ).all()
        )
        expected_steps = {
            step_id: role
            for round_ in rounds
            for step_id, role in (
                (round_.attacker_step_id, "attacker"),
                (round_.target_step_id, "target"),
                (round_.judge_step_id, "judge"),
            )
        }
        if (
            len(calls) != 3 * len(rounds)
            or any(call.status != CallStatus.SUCCEEDED for call in calls)
            or {call.step_id: call.purpose for call in calls} != expected_steps
        ):
            raise RuntimeError("paired four-run preflight request trace is incomplete")
        judge_rationales.extend(
            (language, evaluation_by_round[round_.id].rationale) for round_ in rounds
        )
    _judge_language_diagnostic(judge_rationales)


def _judge_language_diagnostic(
    rationales: list[tuple[ExperimentLanguage, str]],
) -> dict[str, dict[str, int]]:

    result = _judge_language_counts(rationales)
    for language in ExperimentLanguage:
        counts = result[language.value]
        total = counts["total"]
        if total < 2:
            raise RuntimeError(
                f"judge language diagnostic is insufficient for language={language.value}"
            )
        if counts["fail"] >= 2 and counts["fail"] > counts["pass"]:
            raise RuntimeError(
                f"systematic judge language switch detected for language={language.value}"
            )
        if counts["pass"] == 0:
            raise RuntimeError(
                f"judge language diagnostic is insufficient for language={language.value}"
            )
    return result


def _judge_language_counts(
    rationales: list[tuple[ExperimentLanguage, str]],
) -> dict[str, dict[str, int]]:
    aggregate: dict[str, Counter[str]] = {
        language.value: Counter() for language in ExperimentLanguage
    }
    for language, rationale in rationales:
        aggregate[language.value][detect_language_adherence(rationale, language)] += 1
    result: dict[str, dict[str, int]] = {}
    for language in ExperimentLanguage:
        counts = aggregate[language.value]
        total = sum(counts.values())
        result[language.value] = {
            "fail": counts["fail"],
            "indeterminate": counts["indeterminate"],
            "pass": counts["pass"],
            "total": total,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute the frozen Phase 3 stage.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--stage", choices=(STAGE_NAME, "pilot"), default=STAGE_NAME)
    args = parser.parse_args()
    print(
        json.dumps(
            asyncio.run(execute_real_stage(args.manifest, stage=args.stage)),
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
