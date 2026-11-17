from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from typing import Any, cast

import httpx
from sqlalchemy import func, select

from harness.agents.native.adversarial import ExperimentLanguage
from harness.agents.native.gateway import TracedModelGateway
from harness.agents.native.modal_provider import QWEN36_27B_FP8_CANDIDATE, ModalOpenAIClient
from harness.agents.native.types import ModelClient
from harness.config.schema import (
    AgentConfig,
    BudgetsConfig,
    ExperimentConfig,
    ModelConfig,
    ModelsConfig,
    Provider,
    ReportingConfig,
    TaskConfig,
    ToolsConfig,
)
from harness.experiments.phase3_pilot import Phase3PilotManifest
from harness.experiments.phase4_self_judge import (
    TARGET_PROFILE,
    Phase4SelfJudgeProtocol,
    benchmark_goal,
    execute_target_self_judge,
    input_bundle_sha256,
    reconcile_succeeded_target_self_judge,
    source_evaluation_sha256,
    target_self_judge_messages,
)
from harness.persistence.models import (
    AdversarialRound,
    AdversarialSession,
    BenchmarkItem,
    BenchmarkVersion,
    CallStatus,
    ConfigSnapshot,
    Evaluation,
    Experiment,
    LLMCall,
    Phase3ExecutionBatch,
    Phase3ExecutionCell,
    Phase3ExecutionCellStatus,
    Phase4InstrumentBatch,
    Phase4InstrumentItem,
    Phase4InstrumentItemStatus,
    Run,
    RunStatus,
)
from harness.persistence.repository import Phase4InstrumentItemSpec, TraceRepository
from harness.settings import Settings

EXECUTION_CRITICAL_PATHS = (
    "config/phase3-bilingual-pilot.json",
    "harness/agents/native/adversarial.py",
    "harness/agents/native/gateway.py",
    "harness/agents/native/modal_provider.py",
    "harness/experiments/phase3_pilot.py",
    "harness/experiments/phase4_execute.py",
    "harness/experiments/phase4_self_judge.py",
    "harness/persistence",
    "harness/settings.py",
    "modal_apps/vllm_qwen36/app.py",
)


@dataclass(frozen=True)
class FrozenPhase4Inputs:
    batch: Phase4InstrumentBatch
    changed: bool
    manifest_sha256: str
    response_set_sha256: str
    item_count: int


def committed_phase4_revision() -> str:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *EXECUTION_CRITICAL_PATHS],
        check=True,
    )
    if len(revision) != 40:
        raise RuntimeError("Git did not return a full committed revision")
    return revision


async def materialize_phase4_inputs(
    *,
    phase3_manifest: Phase3PilotManifest,
    repository: TraceRepository,
    protocol: Phase4SelfJudgeProtocol | None = None,
) -> FrozenPhase4Inputs:

    protocol = protocol or Phase4SelfJudgeProtocol()
    phase3_batch = await repository.session.scalar(
        select(Phase3ExecutionBatch).where(
            Phase3ExecutionBatch.manifest_sha256 == phase3_manifest.manifest_sha256
        )
    )
    if phase3_batch is None:
        raise RuntimeError("frozen Phase 3 execution batch is unavailable")
    if (
        str(phase3_batch.status) != "indeterminate"
        or phase3_batch.stop_reason != "pilot_abort_systematic_ukrainian_judge_language_switch"
    ):
        raise RuntimeError("Phase 3 stopped-batch identity differs from the Phase 4 contract")
    all_states = tuple(
        (
            await repository.session.scalars(
                select(Phase3ExecutionCell).where(Phase3ExecutionCell.batch_id == phase3_batch.id)
            )
        ).all()
    )
    state_counts = {
        status: sum(str(state.status) == status for state in all_states)
        for status in ("terminal", "indeterminate")
    }
    if len(all_states) != 40 or state_counts != {"terminal": 7, "indeterminate": 33}:
        raise RuntimeError("Phase 3 stopped cell counts differ from the frozen contract")
    states = tuple(
        (
            await repository.session.scalars(
                select(Phase3ExecutionCell).where(
                    Phase3ExecutionCell.batch_id == phase3_batch.id,
                    Phase3ExecutionCell.status == Phase3ExecutionCellStatus.TERMINAL,
                    Phase3ExecutionCell.run_id.is_not(None),
                )
            )
        ).all()
    )
    if len(states) != 7:
        raise RuntimeError("Phase 4 requires exactly seven terminal Phase 3 runs")
    manifest_cells = {cell.cell_id: cell for cell in phase3_manifest.pilot.matrix.cells}
    if any(
        state.cell_id not in manifest_cells
        or state.cell_spec_json != manifest_cells[state.cell_id].model_dump(mode="json")
        for state in states
    ):
        raise RuntimeError("Phase 3 cell identity changed before Phase 4 freeze")
    run_ids = [state.run_id for state in states if state.run_id is not None]
    calls = tuple(
        (await repository.session.scalars(select(LLMCall).where(LLMCall.run_id.in_(run_ids)))).all()
    )
    if len(calls) != 63 or any(call.status != CallStatus.SUCCEEDED for call in calls):
        raise RuntimeError("Phase 3 frozen source trace must contain 63 succeeded calls")
    calls_by_step = {call.step_id: call for call in calls}
    if len(calls_by_step) != 63:
        raise RuntimeError("Phase 3 source calls do not bind uniquely to role steps")

    version = await repository.session.scalar(
        select(BenchmarkVersion).where(
            BenchmarkVersion.name == phase3_manifest.benchmark.name,
            BenchmarkVersion.version == phase3_manifest.benchmark.version,
        )
    )
    if (
        version is None
        or version.status != "frozen"
        or version.source_manifest_json.get("freeze", {}).get("benchmark_sha256")
        != phase3_manifest.benchmark.sha256
    ):
        raise RuntimeError("Phase 3 benchmark binding changed before Phase 4 freeze")
    specs: list[Phase4InstrumentItemSpec] = []
    public_rows: list[dict[str, Any]] = []
    ordinal = 0
    source_rows: list[tuple[str, Phase3ExecutionCell, Run]] = []
    for state in states:
        assert state.run_id is not None
        run = await repository.session.get(Run, state.run_id)
        if run is None:
            raise RuntimeError("Phase 4 source run is unavailable")
        source_rows.append((str(run.id), state, run))
    for _, state, run in sorted(source_rows, key=lambda value: value[0]):
        cell = manifest_cells[state.cell_id]
        if run.status != RunStatus.COMPLETED:
            raise RuntimeError("Phase 4 source run must be completed")
        config = await repository.session.get(ConfigSnapshot, run.config_id)
        if (
            config is None
            or config.config_json.get("task", {}).get("benchmark_item_id") != cell.item.stable_id
        ):
            raise RuntimeError("Phase 4 source run config identity changed")
        benchmark = await repository.session.scalar(
            select(BenchmarkItem).where(
                BenchmarkItem.benchmark_version_id == version.id,
                BenchmarkItem.stable_id == cell.item.stable_id,
            )
        )
        if benchmark is None or benchmark.content_sha256 != cell.item.content_sha256:
            raise RuntimeError("Phase 4 source benchmark item changed")
        source_session = await repository.session.scalar(
            select(AdversarialSession).where(AdversarialSession.run_id == run.id)
        )
        if source_session is None or source_session.language != cell.item.language:
            raise RuntimeError("Phase 4 source adversarial session changed")
        rounds = tuple(
            (
                await repository.session.scalars(
                    select(AdversarialRound)
                    .where(AdversarialRound.session_id == source_session.id)
                    .order_by(AdversarialRound.round_number)
                )
            ).all()
        )
        if len(rounds) != 3:
            raise RuntimeError("each frozen Phase 4 source run must contain three rounds")
        for round_ in rounds:
            evaluation = await repository.session.scalar(
                select(Evaluation).where(Evaluation.round_id == round_.id)
            )
            if evaluation is None:
                raise RuntimeError("Phase 4 source cross-evaluation is unavailable")
            role_steps = {
                "attacker": round_.attacker_step_id,
                "target": round_.target_step_id,
                "judge": round_.judge_step_id,
            }
            if any(
                step_id is None
                or step_id not in calls_by_step
                or calls_by_step[step_id].purpose != role
                for role, step_id in role_steps.items()
            ):
                raise RuntimeError("Phase 3 source role/round trace binding changed")
            source_judge_call = calls_by_step[round_.judge_step_id]
            source_target_call = calls_by_step[round_.target_step_id]
            if (
                source_target_call.response_json is None
                or source_target_call.response_json.get("content") != round_.target_response
                or source_judge_call.response_json is None
                or source_judge_call.response_json.get("content")
                != evaluation.raw_json.get("content")
                or evaluation.judge_provider != cell.judge.provider
                or evaluation.judge_endpoint_profile != cell.judge.endpoint_profile
                or evaluation.rubric_version != cell.rubric_version
                or source_judge_call.provider != cell.judge.provider
                or source_judge_call.endpoint_profile != cell.judge.endpoint_profile
            ):
                raise RuntimeError("Phase 3 source response/evaluation identity changed")
            expected_messages = target_self_judge_messages(
                language=ExperimentLanguage(source_session.language),
                benchmark_prompt=benchmark_goal(benchmark),
                rubric_text=benchmark.rubric_text,
                target_response=round_.target_response,
            )
            if source_judge_call.request_messages_json != expected_messages:
                raise RuntimeError("Phase 4 prompt reconstruction mismatch")
            response_hash = hashlib.sha256(round_.target_response.encode()).hexdigest()
            evaluation_hash = source_evaluation_sha256(evaluation)
            bundle_hash = input_bundle_sha256(
                source_round_id=str(round_.id),
                source_evaluation_id=str(evaluation.id),
                source_evaluation_sha256=evaluation_hash,
                language=source_session.language,
                benchmark_prompt=benchmark_goal(benchmark),
                context_text=benchmark.context_text,
                rubric_text=benchmark.rubric_text,
                target_response=round_.target_response,
            )
            specs.append(
                Phase4InstrumentItemSpec(
                    source_round_id=round_.id,
                    source_evaluation_id=evaluation.id,
                    ordinal=ordinal,
                    source_language=source_session.language,
                    source_response_sha256=response_hash,
                    source_evaluation_sha256=evaluation_hash,
                    input_bundle_sha256=bundle_hash,
                )
            )
            public_rows.append(
                {
                    "input_bundle_sha256": bundle_hash,
                    "ordinal": ordinal,
                    "source_evaluation_id": str(evaluation.id),
                    "source_evaluation_sha256": evaluation_hash,
                    "source_language": source_session.language,
                    "source_response_sha256": response_hash,
                    "source_round_id": str(round_.id),
                }
            )
            ordinal += 1
    if len(specs) != 21:
        raise RuntimeError("Phase 4 freeze did not resolve exactly 21 source rounds")
    response_set_sha256 = _json_sha256(public_rows)
    manifest_sha256 = _json_sha256(
        {
            "protocol": protocol.model_dump(mode="json"),
            "response_set_sha256": response_set_sha256,
            "source_phase3_manifest_sha256": phase3_manifest.manifest_sha256,
        }
    )
    batch, changed = await repository.materialize_phase4_instrument_batch(
        manifest_sha256=manifest_sha256,
        source_phase3_manifest_sha256=phase3_manifest.manifest_sha256,
        response_set_sha256=response_set_sha256,
        target_endpoint_profile=protocol.target_endpoint_profile,
        protocol_json=protocol.model_dump(mode="json"),
        items=tuple(specs),
    )
    return FrozenPhase4Inputs(
        batch=batch,
        changed=changed,
        manifest_sha256=manifest_sha256,
        response_set_sha256=response_set_sha256,
        item_count=len(specs),
    )


def phase4_run_config(item: Phase4InstrumentItem) -> ExperimentConfig:
    qwen = ModelConfig(
        provider=Provider.MODAL,
        endpoint_profile=TARGET_PROFILE.endpoint_profile,
        temperature=0,
    )
    return ExperimentConfig(
        task=TaskConfig(benchmark_item_id=str(item.source_round_id)),
        agent=AgentConfig(max_wall_time_seconds=1200),
        models=ModelsConfig(
            researcher=qwen,
            attacker=qwen,
            target=qwen,
            judge=qwen,
        ),
        tools=ToolsConfig(enabled=("finish_experiment",)),
        budgets=BudgetsConfig(
            max_total_llm_calls=1,
            max_input_tokens=200000,
            max_output_tokens=20000,
            max_estimated_cost_usd=1.5,
        ),
        reporting=ReportingConfig(telegram=False, include_private_trace_link=False),
    )


async def execute_phase4_batch(
    *,
    repository: TraceRepository,
    batch: Phase4InstrumentBatch,
    phase3_manifest: Phase3PilotManifest,
    settings: Settings,
    http_client: httpx.AsyncClient,
    code_revision: str,
) -> dict[str, Any]:

    token_id, token_secret = settings.require_modal_proxy_credentials()
    if code_revision != committed_phase4_revision():
        raise RuntimeError("Phase 4 execution revision is not the clean committed HEAD")
    if batch.source_phase3_manifest_sha256 != phase3_manifest.manifest_sha256:
        raise RuntimeError("Phase 4 batch does not match the frozen Phase 3 manifest")
    if settings.modal_qwen36_endpoint_url is None:
        raise RuntimeError("frozen Qwen endpoint URL is required")
    protocol = Phase4SelfJudgeProtocol.model_validate(batch.protocol_json, strict=True)
    experiment = await _get_or_create_phase4_experiment(repository, batch)
    while True:
        claimed = await repository.session.scalar(
            select(Phase4InstrumentItem).where(
                Phase4InstrumentItem.batch_id == batch.id,
                Phase4InstrumentItem.status == Phase4InstrumentItemStatus.CLAIMED,
            )
        )
        if claimed is not None:
            if claimed.run_id is None:
                raise RuntimeError("claimed Phase 4 item lacks a run; reconcile manually")
            existing_run = await repository.session.get(Run, claimed.run_id)
            if existing_run is None:
                raise RuntimeError("claimed Phase 4 run is unavailable")
            _, _, _, verified_messages = await _load_verified_source(
                repository, claimed, phase3_manifest
            )
            await reconcile_succeeded_target_self_judge(
                repository=repository,
                item=claimed,
                run=existing_run,
                protocol=protocol,
                verified_messages=verified_messages,
            )
            continue
        item = await repository.claim_next_phase4_instrument_item(batch_id=batch.id)
        if item is None:
            break
        config = phase4_run_config(item)
        snapshot = await repository.get_or_create_config(
            name=f"phase4-self-judge-{item.ordinal}", config=config
        )
        run = await repository.create_run(
            experiment_id=experiment.id,
            config_id=snapshot.id,
            benchmark_item_id=str(item.source_round_id),
            random_seed=17,
            code_version=code_revision,
        )
        await repository.link_phase4_instrument_run(item=item, run=run)
        await repository.session.commit()
        try:
            (
                source_round,
                source_evaluation,
                benchmark,
                verified_messages,
            ) = await _load_verified_source(repository, item, phase3_manifest)
        except Exception:
            await repository.mark_phase4_instrument_item_indeterminate(
                item=item, stop_reason="phase4_source_identity_requires_reconciliation"
            )
            await repository.session.commit()
            raise
        client = ModalOpenAIClient(
            profile=QWEN36_27B_FP8_CANDIDATE,
            endpoint_url=settings.modal_qwen36_endpoint_url,
            proxy_token_id=token_id,
            proxy_token_secret=token_secret,
            http_client=http_client,
        )
        await execute_target_self_judge(
            repository=repository,
            item=item,
            run=run,
            source_round=source_round,
            source_evaluation=source_evaluation,
            benchmark_item=benchmark,
            gateway=TracedModelGateway(cast(ModelClient, client), repository),
            protocol=protocol,
            verified_messages=verified_messages,
        )
    completed = await repository.session.scalar(
        select(func.count())
        .select_from(Phase4InstrumentItem)
        .where(
            Phase4InstrumentItem.batch_id == batch.id,
            Phase4InstrumentItem.status == Phase4InstrumentItemStatus.TERMINAL,
        )
    )
    return {
        "aggregate_only": True,
        "completed_items": int(completed or 0),
        "expected_items": 21,
        "manifest_sha256": batch.manifest_sha256,
        "raw_content_exported": False,
        "status": str(batch.status),
    }


async def _get_or_create_phase4_experiment(
    repository: TraceRepository, batch: Phase4InstrumentBatch
) -> Experiment:
    name = f"Phase 4 target self-judge {batch.manifest_sha256[:12]}"
    existing = await repository.session.scalar(select(Experiment).where(Experiment.name == name))
    if existing is not None:
        return existing
    return await repository.create_experiment(
        name=name,
        research_question="How do automated instruments agree on frozen Phase 3 responses?",
        hypothesis="Instrument agreement is reported without gold-label claims.",
        created_by="phase4-self-judge-executor",
    )


def _json_sha256(value: Any) -> str:
    canonical = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


async def _load_verified_source(
    repository: TraceRepository,
    item: Phase4InstrumentItem,
    manifest: Phase3PilotManifest,
) -> tuple[AdversarialRound, Evaluation, BenchmarkItem, list[dict[str, Any]]]:
    source_round = await repository.session.get(AdversarialRound, item.source_round_id)
    source_evaluation = await repository.session.get(Evaluation, item.source_evaluation_id)
    if (
        source_round is None
        or source_evaluation is None
        or source_evaluation.round_id != source_round.id
    ):
        raise RuntimeError("Phase 4 frozen source rows are unavailable")
    source_session = await repository.session.get(AdversarialSession, source_round.session_id)
    source_run = (
        await repository.session.get(Run, source_session.run_id)
        if source_session is not None
        else None
    )
    source_version = await repository.session.scalar(
        select(BenchmarkVersion).where(
            BenchmarkVersion.name == manifest.benchmark.name,
            BenchmarkVersion.version == manifest.benchmark.version,
        )
    )
    benchmark = (
        await repository.session.scalar(
            select(BenchmarkItem).where(
                BenchmarkItem.benchmark_version_id == source_version.id,
                BenchmarkItem.stable_id == source_run.benchmark_item_id,
            )
        )
        if source_run is not None and source_version is not None
        else None
    )
    source_judge_call = await repository.session.scalar(
        select(LLMCall).where(LLMCall.step_id == source_round.judge_step_id)
    )
    source_target_call = await repository.session.scalar(
        select(LLMCall).where(LLMCall.step_id == source_round.target_step_id)
    )
    source_cell = (
        await repository.session.scalar(
            select(Phase3ExecutionCell).where(Phase3ExecutionCell.run_id == source_run.id)
        )
        if source_run is not None
        else None
    )
    manifest_cell = (
        next(
            (cell for cell in manifest.pilot.matrix.cells if cell.cell_id == source_cell.cell_id),
            None,
        )
        if source_cell is not None
        else None
    )
    if (
        benchmark is None
        or source_session is None
        or source_judge_call is None
        or source_target_call is None
        or source_cell is None
        or manifest_cell is None
        or benchmark.content_sha256 != manifest_cell.item.content_sha256
        or source_cell.cell_spec_json != manifest_cell.model_dump(mode="json")
    ):
        raise RuntimeError("Phase 4 frozen source identity is unavailable")
    messages = target_self_judge_messages(
        language=ExperimentLanguage(source_session.language),
        benchmark_prompt=benchmark_goal(benchmark),
        rubric_text=benchmark.rubric_text,
        target_response=source_round.target_response,
    )
    bundle_hash = input_bundle_sha256(
        source_round_id=str(source_round.id),
        source_evaluation_id=str(source_evaluation.id),
        source_evaluation_sha256=source_evaluation_sha256(source_evaluation),
        language=source_session.language,
        benchmark_prompt=benchmark_goal(benchmark),
        context_text=benchmark.context_text,
        rubric_text=benchmark.rubric_text,
        target_response=source_round.target_response,
    )
    if (
        hashlib.sha256(source_round.target_response.encode()).hexdigest()
        != item.source_response_sha256
        or source_evaluation_sha256(source_evaluation) != item.source_evaluation_sha256
        or bundle_hash != item.input_bundle_sha256
        or source_judge_call.request_messages_json != messages
        or source_target_call.response_json is None
        or source_target_call.response_json.get("content") != source_round.target_response
        or source_target_call.provider != manifest_cell.target.provider
        or source_target_call.endpoint_profile != manifest_cell.target.endpoint_profile
        or source_judge_call.response_json is None
        or source_judge_call.response_json.get("content")
        != source_evaluation.raw_json.get("content")
        or source_judge_call.provider != manifest_cell.judge.provider
        or source_judge_call.endpoint_profile != manifest_cell.judge.endpoint_profile
        or source_evaluation.judge_provider != manifest_cell.judge.provider
        or source_evaluation.judge_endpoint_profile != manifest_cell.judge.endpoint_profile
        or source_evaluation.rubric_version != manifest_cell.rubric_version
    ):
        raise RuntimeError("Phase 4 frozen source identity changed after materialization")
    return source_round, source_evaluation, benchmark, messages
