from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import select

from harness.agents.native.adversarial import ExperimentLanguage
from harness.config.schema import ExperimentConfig
from harness.experiments.phase3_execute import (
    _judge_language_diagnostic,
    _require_completed_stage,
)
from harness.experiments.phase3_pilot import STAGE_NAME
from harness.persistence.database import create_engine, create_schema, session_factory
from harness.persistence.models import (
    AdversarialRound,
    AdversarialSession,
    CallStatus,
    Evaluation,
    LLMCall,
    Phase3ExecutionBatch,
    RunStatus,
)
from harness.persistence.repository import Phase3ExecutionCellSpec, TraceRepository


def test_judge_language_diagnostic_rejects_systematic_ukrainian_switch() -> None:
    rationales = [
        (ExperimentLanguage.ENGLISH, "This is an English safety rationale.") for _ in range(6)
    ] + [(ExperimentLanguage.UKRAINIAN, "This rationale switched to English.") for _ in range(6)]

    with pytest.raises(RuntimeError, match=r"systematic judge language switch.*uk"):
        _judge_language_diagnostic(rationales)


async def _persist_early_success_preflight(
    repository: TraceRepository,
    config: ExperimentConfig,
) -> Phase3ExecutionBatch:
    cells = tuple(
        Phase3ExecutionCellSpec(
            cell_id=f"{index + 1:064x}",
            stage=STAGE_NAME,
            ordinal=index,
            cell_sha256=f"{index + 101:064x}",
            cell_spec_json={"item": {"language": "en" if index % 2 == 0 else "uk"}},
        )
        for index in range(4)
    )
    batch, _ = await repository.materialize_phase3_execution(
        manifest_sha256="a" * 64,
        plan_sha256="b" * 64,
        benchmark_sha256="c" * 64,
        roster_sha256="d" * 64,
        execution_envelope_json={"protocol": "phase3-fixture-v1"},
        cells=cells,
    )
    experiment = await repository.create_experiment(
        name="Phase 3 early-success fixture",
        research_question="Can early success retain complete traces?",
        hypothesis="Yes.",
        created_by="pytest",
    )
    snapshot = await repository.get_or_create_config(name="phase3-early-success", config=config)
    for index in range(4):
        language = "en" if index % 2 == 0 else "uk"
        cell = await repository.claim_next_phase3_execution_cell(
            batch_id=batch.id,
            stage=STAGE_NAME,
        )
        assert cell is not None
        run = await repository.create_run(
            experiment_id=experiment.id,
            config_id=snapshot.id,
            benchmark_item_id=f"fixture-{index}",
            random_seed=17,
            code_version="a" * 40,
        )
        await repository.link_phase3_execution_run(cell=cell, run=run)
        await repository.start_run(run)
        session = AdversarialSession(
            run_id=run.id,
            method="pair",
            language=language,
            max_rounds=3,
            rubric_version="pilot-v1",
            rubric_text="fixture rubric",
            spec_json={},
            status=RunStatus.COMPLETED,
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
        )
        repository.session.add(session)
        await repository.session.flush()
        steps = []
        for sequence, role in enumerate(("attacker", "target", "judge"), start=1):
            step = await repository.create_step(
                run_id=run.id,
                sequence_number=sequence,
                step_type="adversarial_role",
                purpose=role,
            )
            call = await repository.begin_llm_call(
                run_id=run.id,
                step_id=step.id,
                purpose=role,
                provider="mock",
                model="fixture",
                endpoint_profile=None,
                messages=[],
                tools=[],
                parameters={},
            )
            await repository.succeed_llm_call(
                call,
                response={},
                input_tokens=1,
                output_tokens=1,
                cached_input_tokens=0,
                reasoning_tokens=0,
                latency_ms=1,
                estimated_cost_usd=0,
            )
            await repository.finish_step(step)
            steps.append(step)
        round_ = AdversarialRound(
            session_id=session.id,
            round_number=1,
            attacker_step_id=steps[0].id,
            target_step_id=steps[1].id,
            judge_step_id=steps[2].id,
            attack_text="private fixture",
            target_response="private fixture",
            candidate_metadata_json={},
        )
        repository.session.add(round_)
        await repository.session.flush()
        repository.session.add(
            Evaluation(
                round_id=round_.id,
                rubric_version="pilot-v1",
                judge_provider="mock",
                judge_model="fixture",
                judge_endpoint_profile=None,
                label="complied",
                rationale=(
                    "This response complies with the rubric."
                    if language == "en"
                    else "Ця відповідь відповідає визначеній рубриці."
                ),
                confidence=1,
                raw_json={},
            )
        )
        await repository.finish_run(
            run,
            status=RunStatus.COMPLETED,
            stop_reason="validated_success",
            estimated_cost_usd=0,
        )
        await repository.mark_phase3_execution_cell_terminal(
            cell=cell,
            run=run,
            stop_reason="validated_success",
        )
    await repository.session.commit()
    return batch


async def test_preflight_gate_accepts_complete_early_success_and_rejects_trace_loss(
    experiment_config: ExperimentConfig,
) -> None:
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    await create_schema(engine)
    async with session_factory(engine)() as session:
        repository = TraceRepository(session)
        batch = await _persist_early_success_preflight(repository, experiment_config)

        await _require_completed_stage(repository=repository, batch=batch)

        call = await session.scalar(select(LLMCall).limit(1))
        assert call is not None
        call.status = CallStatus.STARTED
        await session.flush()
        with pytest.raises(RuntimeError, match="request trace is incomplete"):
            await _require_completed_stage(repository=repository, batch=batch)

    await engine.dispose()
