from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import select

from harness.experiments.phase3_audit import (
    SYSTEMATIC_UKRAINIAN_SWITCH,
    audit_stopped_phase3_pilot,
    clopper_pearson_95,
)
from harness.experiments.phase3_pilot import (
    STAGE_NAME,
    HydratedPilotCell,
    build_cell_config,
    load_phase3_manifest,
)
from harness.persistence.database import create_engine, create_schema, session_factory
from harness.persistence.models import (
    AdversarialRound,
    AdversarialSession,
    CallStatus,
    Evaluation,
    LLMCall,
    Phase3ExecutionBatchStatus,
    Phase3ExecutionCell,
    Phase3ExecutionCellStatus,
    RunStatus,
    Step,
    utc_now,
)
from harness.persistence.repository import Phase3ExecutionCellSpec, TraceRepository

MANIFEST_PATH = Path(__file__).resolve().parents[2] / "config" / "phase3-bilingual-pilot.json"


def test_clopper_pearson_interval_matches_standard_exact_values() -> None:
    assert clopper_pearson_95(0, 0) is None
    assert clopper_pearson_95(0, 10) == pytest.approx((0.0, 0.3084971078))
    assert clopper_pearson_95(5, 10) == pytest.approx((0.1870860284, 0.8129139716))
    assert clopper_pearson_95(10, 10) == pytest.approx((0.6915028922, 1.0))
    with pytest.raises(ValueError, match="binomial counts"):
        clopper_pearson_95(2, 1)


async def test_stopped_pilot_audit_is_aggregate_only_and_detects_ukrainian_switch() -> None:
    manifest = load_phase3_manifest(MANIFEST_PATH)
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    await create_schema(engine)
    async with session_factory(engine)() as session:
        repository = TraceRepository(session)
        stage_ids = set(manifest.execution_protocol.stage.cell_ids)
        pilot_ordinal = 0
        specs: list[Phase3ExecutionCellSpec] = []
        for cell in manifest.pilot.matrix.cells:
            is_stage = cell.cell_id in stage_ids
            stage = STAGE_NAME if is_stage else "pilot"
            ordinal = (
                manifest.execution_protocol.stage.cell_ids.index(cell.cell_id)
                if is_stage
                else pilot_ordinal
            )
            pilot_ordinal += int(not is_stage)
            specs.append(
                Phase3ExecutionCellSpec(
                    cell_id=cell.cell_id,
                    stage=stage,
                    ordinal=ordinal,
                    cell_sha256=cell.cell_id,
                    cell_spec_json=cell.model_dump(mode="json"),
                )
            )
        batch, _ = await repository.materialize_phase3_execution(
            manifest_sha256=manifest.manifest_sha256,
            plan_sha256=manifest.pilot.plan_sha256,
            benchmark_sha256=manifest.benchmark.sha256,
            roster_sha256=manifest.roster_sha256,
            execution_envelope_json=manifest.execution_protocol.model_dump(mode="json"),
            cells=tuple(specs),
        )
        batch.status = Phase3ExecutionBatchStatus.INDETERMINATE
        batch.stop_reason = SYSTEMATIC_UKRAINIAN_SWITCH
        batch.finished_at = utc_now()
        experiment = await repository.create_experiment(
            name="aggregate pilot audit fixture",
            research_question="Can the stopped pilot be audited without exporting text?",
            hypothesis="Only aggregates leave the audit boundary.",
            created_by="pytest",
        )
        await session.flush()
        states = {
            state.cell_id: state
            for state in (
                await session.scalars(
                    select(Phase3ExecutionCell).where(Phase3ExecutionCell.batch_id == batch.id)
                )
            ).all()
        }

        for cell in manifest.stage_cells:
            hydrated = HydratedPilotCell(
                manifest_cell=cell,
                execution_protocol=manifest.execution_protocol,
                prompt_text="PRIVATE PROMPT MUST NEVER BE EXPORTED",
                context_text=None,
                rubric_text="PRIVATE RUBRIC MUST NEVER BE EXPORTED",
                expected_behavior="refuse_or_safe_redirect",
            )
            config = await repository.get_or_create_config(
                name=f"audit-{cell.cell_id}", config=build_cell_config(hydrated)
            )
            run = await repository.create_run(
                experiment_id=experiment.id,
                config_id=config.id,
                benchmark_item_id=cell.item.stable_id,
                random_seed=cell.seed,
                code_version="a" * 40,
            )
            now = utc_now()
            run.status = RunStatus.COMPLETED
            run.started_at = now
            run.finished_at = now
            run.stop_reason = "validated_success"
            state = states[cell.cell_id]
            state.run_id = run.id
            state.status = Phase3ExecutionCellStatus.TERMINAL
            state.terminal_at = now
            state.stop_reason = "validated_success"
            protocol_session = AdversarialSession(
                run_id=run.id,
                method=cell.method,
                language=cell.item.language,
                max_rounds=cell.max_rounds,
                rubric_version=cell.rubric_version,
                rubric_text="PRIVATE RUBRIC MUST NEVER BE EXPORTED",
                spec_json={},
                status="completed",
                started_at=now,
                finished_at=now,
            )
            session.add(protocol_session)
            await session.flush()

            steps: dict[str, Step] = {}
            for sequence, role in enumerate(("attacker", "target", "judge"), start=1):
                step = Step(
                    run_id=run.id,
                    sequence_number=sequence,
                    step_type="llm_request",
                    purpose=role,
                    started_at=now,
                    finished_at=now,
                    status=CallStatus.SUCCEEDED,
                )
                session.add(step)
                await session.flush()
                profile = getattr(cell, role)
                session.add(
                    LLMCall(
                        run_id=run.id,
                        step_id=step.id,
                        purpose=role,
                        provider=profile.provider,
                        model=f"fixture:{profile.key}",
                        endpoint_profile=profile.endpoint_profile,
                        request_messages_json=[{"content": "PRIVATE REQUEST"}],
                        request_tools_json=[],
                        request_parameters_json={},
                        response_json={"content": "PRIVATE RESPONSE"},
                        input_tokens=1,
                        output_tokens=1,
                        estimated_cost_usd=0.01,
                        started_at=now,
                        finished_at=now,
                        status=CallStatus.SUCCEEDED,
                    )
                )
                steps[role] = step

            is_ukrainian = cell.item.language == "uk"
            round_record = AdversarialRound(
                session_id=protocol_session.id,
                round_number=1,
                attacker_step_id=steps["attacker"].id,
                target_step_id=steps["target"].id,
                judge_step_id=steps["judge"].id,
                attack_text=(
                    "Приватний український текст атаки"
                    if is_ukrainian
                    else "Private English attack text"
                ),
                target_response=(
                    "Приватна українська відповідь"
                    if is_ukrainian
                    else "Private English target response"
                ),
                candidate_metadata_json={},
            )
            session.add(round_record)
            await session.flush()
            session.add(
                Evaluation(
                    round_id=round_record.id,
                    rubric_version=cell.rubric_version,
                    judge_provider=cell.judge.provider,
                    judge_model=f"fixture:{cell.judge.key}",
                    judge_endpoint_profile=cell.judge.endpoint_profile,
                    label="complied",
                    rationale="Private English judge rationale",
                    confidence=0.9,
                    raw_json={"content": "PRIVATE RAW JUDGE RESPONSE"},
                )
            )
            if cell.cell_id == manifest.stage_cells[0].cell_id:
                second_steps: dict[str, Step] = {}
                for sequence, role in enumerate(("attacker", "target", "judge"), start=4):
                    step = Step(
                        run_id=run.id,
                        sequence_number=sequence,
                        step_type="llm_request",
                        purpose=role,
                        started_at=now,
                        finished_at=now,
                        status=CallStatus.SUCCEEDED,
                    )
                    session.add(step)
                    await session.flush()
                    profile = getattr(cell, role)
                    session.add(
                        LLMCall(
                            run_id=run.id,
                            step_id=step.id,
                            purpose=role,
                            provider=profile.provider,
                            model=f"fixture:{profile.key}",
                            endpoint_profile=profile.endpoint_profile,
                            request_messages_json=[{"content": "PRIVATE REQUEST"}],
                            request_tools_json=[],
                            request_parameters_json={},
                            response_json={"content": "PRIVATE RESPONSE"},
                            input_tokens=1,
                            output_tokens=1,
                            estimated_cost_usd=0.01,
                            started_at=now,
                            finished_at=now,
                            status=CallStatus.SUCCEEDED,
                        )
                    )
                    second_steps[role] = step
                second_round = AdversarialRound(
                    session_id=protocol_session.id,
                    round_number=2,
                    attacker_step_id=second_steps["attacker"].id,
                    target_step_id=second_steps["target"].id,
                    judge_step_id=second_steps["judge"].id,
                    attack_text="Private English second attack text",
                    target_response="Private English second target response",
                    candidate_metadata_json={},
                )
                session.add(second_round)
                await session.flush()
                session.add(
                    Evaluation(
                        round_id=second_round.id,
                        rubric_version=cell.rubric_version,
                        judge_provider=cell.judge.provider,
                        judge_model=f"fixture:{cell.judge.key}",
                        judge_endpoint_profile=cell.judge.endpoint_profile,
                        label="complied",
                        rationale="Private English second judge rationale",
                        confidence=0.9,
                        raw_json={"content": "PRIVATE RAW JUDGE RESPONSE"},
                    )
                )
        await session.commit()

        report = await audit_stopped_phase3_pilot(manifest, repository)
        serialized = json.dumps(report.model_dump(mode="json"), sort_keys=True)

        assert report.expected_cells == 40
        assert report.materialized_cells == 40
        assert report.batch_status == "indeterminate"
        assert report.batch_stop_reason == SYSTEMATIC_UKRAINIAN_SWITCH
        assert report.runs == 4
        assert report.calls == 15
        assert report.rounds == 5
        assert report.evaluations == 5
        assert report.consumed_target_queries == 5
        assert report.parse_rate.numerator == 5
        assert report.fixed_judge_labels == {"complied": 5}
        assert report.fixed_judge_asr_at_k.numerator == 4
        assert report.fixed_judge_asr_at_k.denominator == 4
        assert report.language_adherence["judge_rationale"] == {"fail": 2, "pass": 3}
        assert report.gate.decision == "stop"
        assert report.gate.reasons == (
            SYSTEMATIC_UKRAINIAN_SWITCH,
            "batch_explicitly_stopped",
            "pilot_incomplete",
        )
        assert report.gate.validated_asr_claimed is False
        assert report.gate.inferential_claims_authorized is False
        assert report.identity_violation_counts == {}
        assert "PRIVATE" not in serialized
        assert "prompt" not in serialized.lower()
        assert "response" not in serialized.lower()

    await engine.dispose()
