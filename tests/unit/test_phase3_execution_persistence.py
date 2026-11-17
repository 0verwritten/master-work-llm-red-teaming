from __future__ import annotations

from sqlalchemy import func, select

from harness.config.schema import ExperimentConfig
from harness.persistence.database import create_engine, create_schema, session_factory
from harness.persistence.models import (
    Phase3ExecutionBatch,
    Phase3ExecutionBatchStatus,
    Phase3ExecutionCell,
    Phase3ExecutionCellStatus,
    Run,
    RunStatus,
)
from harness.persistence.repository import Phase3ExecutionCellSpec, TraceRepository


def _cell(
    index: int,
    *,
    stage: str = "four_run_preflight",
    ordinal: int | None = None,
) -> Phase3ExecutionCellSpec:
    return Phase3ExecutionCellSpec(
        cell_id=f"{index + 1:064x}",
        stage=stage,
        ordinal=index if ordinal is None else ordinal,
        cell_sha256=f"{index + 101:064x}",
        cell_spec_json={"cell_id": f"{index + 1:064x}", "stage": stage},
    )


async def _new_run(
    repository: TraceRepository,
    config: ExperimentConfig,
    *,
    suffix: str,
) -> Run:
    experiment = await repository.create_experiment(
        name=f"Phase 3 fixture {suffix}",
        research_question="Does persistence prevent duplicate paid cells?",
        hypothesis="It does.",
        created_by="pytest",
    )
    snapshot = await repository.get_or_create_config(name=f"phase3-{suffix}", config=config)
    return await repository.create_run(
        experiment_id=experiment.id,
        config_id=snapshot.id,
        benchmark_item_id=f"fixture-{suffix}",
        random_seed=17,
        code_version="test-sha",
    )


async def test_phase3_manifest_materialization_is_idempotent_and_rejects_drift() -> None:
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    await create_schema(engine)
    async with session_factory(engine)() as session:
        repository = TraceRepository(session)
        cells = (_cell(0), _cell(1), _cell(2, stage="pilot", ordinal=0))
        first, first_changed = await repository.materialize_phase3_execution(
            manifest_sha256="a" * 64,
            plan_sha256="b" * 64,
            benchmark_sha256="c" * 64,
            roster_sha256="d" * 64,
            execution_envelope_json={"protocol": "phase3-fixture-v1"},
            cells=cells,
        )
        await session.commit()
        second, second_changed = await repository.materialize_phase3_execution(
            manifest_sha256="a" * 64,
            plan_sha256="b" * 64,
            benchmark_sha256="c" * 64,
            roster_sha256="d" * 64,
            execution_envelope_json={"protocol": "phase3-fixture-v1"},
            cells=cells,
        )

        assert first.id == second.id
        assert first_changed is True
        assert second_changed is False
        assert await session.scalar(select(func.count()).select_from(Phase3ExecutionBatch)) == 1
        assert await session.scalar(select(func.count()).select_from(Phase3ExecutionCell)) == 3

        try:
            await repository.materialize_phase3_execution(
                manifest_sha256="a" * 64,
                plan_sha256="b" * 64,
                benchmark_sha256="c" * 64,
                roster_sha256="d" * 64,
                execution_envelope_json={"protocol": "changed"},
                cells=cells,
            )
        except RuntimeError as exc:
            assert "identity changed" in str(exc)
        else:
            raise AssertionError("changed execution envelope was accepted")

        try:
            await repository.materialize_phase3_execution(
                manifest_sha256="a" * 64,
                plan_sha256="b" * 64,
                benchmark_sha256="c" * 64,
                roster_sha256="d" * 64,
                execution_envelope_json={"protocol": "phase3-fixture-v1"},
                cells=(*cells[:-1], _cell(3, stage="pilot", ordinal=0)),
            )
        except RuntimeError as exc:
            assert "cell set changed" in str(exc)
        else:
            raise AssertionError("changed execution cell set was accepted")

        try:
            await repository.materialize_phase3_execution(
                manifest_sha256="e" * 64,
                plan_sha256="b" * 64,
                benchmark_sha256="c" * 64,
                roster_sha256="d" * 64,
                execution_envelope_json={"protocol": "phase3-fixture-v1"},
                cells=(_cell(0), _cell(1, ordinal=0)),
            )
        except ValueError as exc:
            assert "stage ordinals" in str(exc)
        else:
            raise AssertionError("duplicate stage position was accepted")

    await engine.dispose()


async def test_phase3_cells_claim_link_finish_and_stop_on_indeterminate(
    experiment_config: ExperimentConfig,
) -> None:
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    await create_schema(engine)
    async with session_factory(engine)() as session:
        repository = TraceRepository(session)
        batch, _ = await repository.materialize_phase3_execution(
            manifest_sha256="a" * 64,
            plan_sha256="b" * 64,
            benchmark_sha256="c" * 64,
            roster_sha256="d" * 64,
            execution_envelope_json={"protocol": "phase3-fixture-v1"},
            cells=(_cell(0), _cell(1), _cell(2, stage="pilot", ordinal=0)),
        )
        await session.commit()

        first = await repository.claim_next_phase3_execution_cell(
            batch_id=batch.id,
            stage="four_run_preflight",
        )
        assert first is not None
        assert first.ordinal == 0
        assert first.status == Phase3ExecutionCellStatus.CLAIMED
        assert batch.status == Phase3ExecutionBatchStatus.RUNNING

        run = await _new_run(repository, experiment_config, suffix="first")
        await repository.link_phase3_execution_run(cell=first, run=run)
        await repository.start_run(run)
        await repository.finish_run(
            run,
            status=RunStatus.COMPLETED,
            stop_reason="max_rounds",
            estimated_cost_usd=0.25,
        )
        await repository.mark_phase3_execution_cell_terminal(
            cell=first,
            run=run,
            stop_reason="max_rounds",
        )
        await repository.mark_phase3_execution_cell_terminal(
            cell=first,
            run=run,
            stop_reason="max_rounds",
        )
        await session.commit()
        assert first.status == Phase3ExecutionCellStatus.TERMINAL
        assert first.run_id == run.id

        second = await repository.claim_next_phase3_execution_cell(
            batch_id=batch.id,
            stage="four_run_preflight",
        )
        assert second is not None
        assert second.ordinal == 1
        try:
            await repository.link_phase3_execution_run(cell=second, run=run)
        except RuntimeError as exc:
            assert "another Phase 3 execution cell" in str(exc)
        else:
            raise AssertionError("one run was linked to two execution cells")

        await repository.mark_phase3_execution_cell_indeterminate(
            cell=second,
            stop_reason="started_provider_call_requires_reconciliation",
        )
        await repository.mark_phase3_execution_cell_indeterminate(
            cell=second,
            stop_reason="started_provider_call_requires_reconciliation",
        )
        await session.commit()
        assert second.status == Phase3ExecutionCellStatus.INDETERMINATE
        assert batch.status == Phase3ExecutionBatchStatus.INDETERMINATE
        assert (
            await repository.claim_next_phase3_execution_cell(
                batch_id=batch.id,
                stage="pilot",
            )
            is None
        )

    await engine.dispose()


async def test_phase3_batch_settles_only_after_every_cell_completes(
    experiment_config: ExperimentConfig,
) -> None:
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    await create_schema(engine)
    async with session_factory(engine)() as session:
        repository = TraceRepository(session)
        batch, _ = await repository.materialize_phase3_execution(
            manifest_sha256="a" * 64,
            plan_sha256="b" * 64,
            benchmark_sha256="c" * 64,
            roster_sha256="d" * 64,
            execution_envelope_json={"protocol": "phase3-fixture-v1"},
            cells=(_cell(0), _cell(1)),
        )

        for index in range(2):
            cell = await repository.claim_next_phase3_execution_cell(
                batch_id=batch.id,
                stage="four_run_preflight",
            )
            assert cell is not None
            run = await _new_run(repository, experiment_config, suffix=f"complete-{index}")
            await repository.link_phase3_execution_run(cell=cell, run=run)
            await repository.start_run(run)
            await repository.finish_run(
                run,
                status=RunStatus.COMPLETED,
                stop_reason="max_rounds",
                estimated_cost_usd=0.25,
            )
            await repository.mark_phase3_execution_cell_terminal(
                cell=cell,
                run=run,
                stop_reason="max_rounds",
            )
            assert batch.status == (
                Phase3ExecutionBatchStatus.RUNNING
                if index == 0
                else Phase3ExecutionBatchStatus.COMPLETED
            )

        assert batch.finished_at is not None
        assert batch.stop_reason == "all_cells_completed"
        settled = await repository.reconcile_phase3_execution_batch(batch_id=batch.id)
        assert settled.status == Phase3ExecutionBatchStatus.COMPLETED
        assert settled.stop_reason == "all_cells_completed"

    await engine.dispose()


async def test_phase3_failed_terminal_run_aborts_batch_without_claiming_more(
    experiment_config: ExperimentConfig,
) -> None:
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    await create_schema(engine)
    async with session_factory(engine)() as session:
        repository = TraceRepository(session)
        batch, _ = await repository.materialize_phase3_execution(
            manifest_sha256="a" * 64,
            plan_sha256="b" * 64,
            benchmark_sha256="c" * 64,
            roster_sha256="d" * 64,
            execution_envelope_json={"protocol": "phase3-fixture-v1"},
            cells=(_cell(0), _cell(1)),
        )
        cell = await repository.claim_next_phase3_execution_cell(
            batch_id=batch.id,
            stage="four_run_preflight",
        )
        assert cell is not None
        run = await _new_run(repository, experiment_config, suffix="failed")
        await repository.link_phase3_execution_run(cell=cell, run=run)
        await repository.start_run(run)
        await repository.finish_run(
            run,
            status=RunStatus.FAILED,
            stop_reason="judge_parse_error",
            estimated_cost_usd=0.25,
        )
        await repository.mark_phase3_execution_cell_terminal(
            cell=cell,
            run=run,
            stop_reason="judge_parse_error",
        )

        assert batch.status == Phase3ExecutionBatchStatus.ABORTED
        assert batch.finished_at is not None
        assert batch.stop_reason == "terminal_cell_not_completed"
        assert (
            await repository.claim_next_phase3_execution_cell(
                batch_id=batch.id,
                stage="four_run_preflight",
            )
            is None
        )

    await engine.dispose()


async def test_phase3_resume_refuses_to_skip_an_existing_claim() -> None:
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    await create_schema(engine)
    async with session_factory(engine)() as session:
        repository = TraceRepository(session)
        batch, _ = await repository.materialize_phase3_execution(
            manifest_sha256="a" * 64,
            plan_sha256="b" * 64,
            benchmark_sha256="c" * 64,
            roster_sha256="d" * 64,
            execution_envelope_json={"protocol": "phase3-fixture-v1"},
            cells=(_cell(0), _cell(1)),
        )
        first = await repository.claim_next_phase3_execution_cell(
            batch_id=batch.id,
            stage="four_run_preflight",
        )
        assert first is not None

        try:
            await repository.claim_next_phase3_execution_cell(
                batch_id=batch.id,
                stage="four_run_preflight",
            )
        except RuntimeError as exc:
            assert "manual reconciliation" in str(exc)
        else:
            raise AssertionError("a queued cell was claimed past an unresolved claim")

        queued = await session.scalar(
            select(func.count())
            .select_from(Phase3ExecutionCell)
            .where(
                Phase3ExecutionCell.batch_id == batch.id,
                Phase3ExecutionCell.status == Phase3ExecutionCellStatus.QUEUED,
            )
        )
        assert queued == 1

    await engine.dispose()


def test_phase3_cell_specs_reject_invalid_hashes() -> None:
    try:
        Phase3ExecutionCellSpec(
            cell_id="not-a-hash",
            stage="four_run_preflight",
            ordinal=0,
            cell_sha256="a" * 64,
            cell_spec_json={},
        )
    except ValueError as exc:
        assert "cell_id" in str(exc)
    else:
        raise AssertionError("invalid cell ID was accepted")
