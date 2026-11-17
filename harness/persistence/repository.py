from __future__ import annotations

import asyncio
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import and_, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from harness.config.schema import ExperimentConfig
from harness.persistence.models import (
    CallStatus,
    ConfigSnapshot,
    Experiment,
    ExternalAgentSession,
    LLMCall,
    Phase3ExecutionBatch,
    Phase3ExecutionBatchStatus,
    Phase3ExecutionCell,
    Phase3ExecutionCellStatus,
    Phase4InstrumentBatch,
    Phase4InstrumentBatchStatus,
    Phase4InstrumentItem,
    Phase4InstrumentItemStatus,
    Phase4SelfJudgeEvaluation,
    ReportDelivery,
    Run,
    RunStatus,
    Step,
    ToolCall,
    utc_now,
)


@dataclass(frozen=True)
class Phase3ExecutionCellSpec:
    cell_id: str
    stage: str
    ordinal: int
    cell_sha256: str
    cell_spec_json: dict[str, Any]

    def __post_init__(self) -> None:
        _require_sha256(self.cell_id, field="cell_id")
        _require_sha256(self.cell_sha256, field="cell_sha256")
        if not self.stage.strip():
            raise ValueError("stage is required")
        if self.ordinal < 0:
            raise ValueError("ordinal cannot be negative")


@dataclass(frozen=True)
class Phase4InstrumentItemSpec:
    source_round_id: uuid.UUID
    source_evaluation_id: uuid.UUID
    ordinal: int
    source_language: str
    source_response_sha256: str
    source_evaluation_sha256: str
    input_bundle_sha256: str

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise ValueError("ordinal cannot be negative")
        if self.source_language not in {"en", "uk"}:
            raise ValueError("source language must be en or uk")
        _require_sha256(self.source_response_sha256, field="source_response_sha256")
        _require_sha256(self.source_evaluation_sha256, field="source_evaluation_sha256")
        _require_sha256(self.input_bundle_sha256, field="input_bundle_sha256")


def _require_sha256(value: str, *, field: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


class TraceRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_experiment(
        self,
        *,
        name: str,
        research_question: str,
        hypothesis: str,
        created_by: str,
    ) -> Experiment:
        experiment = Experiment(
            name=name,
            research_question=research_question,
            hypothesis=hypothesis,
            created_by=created_by,
        )
        self.session.add(experiment)
        await self.session.flush()
        return experiment

    async def get_or_create_config(
        self, *, name: str, config: ExperimentConfig, attempts: int = 4
    ) -> ConfigSnapshot:

        config_hash = config.stable_hash()
        for attempt in range(1, attempts + 1):
            existing = await self.session.scalar(
                select(ConfigSnapshot).where(ConfigSnapshot.config_hash == config_hash)
            )
            if existing is not None:
                return existing
            snapshot = ConfigSnapshot(
                name=name,
                schema_version=config.schema_version,
                config_json=config.model_dump(mode="json"),
                config_hash=config_hash,
            )
            try:
                self.session.add(snapshot)
                await self.session.flush()
            except IntegrityError:
                await self.session.rollback()
                if attempt == attempts:
                    raise
                await asyncio.sleep(0.05 * attempt)
                continue
            return snapshot
        raise RuntimeError("unreachable: config insert retry loop exhausted")

    async def create_run(
        self,
        *,
        experiment_id: uuid.UUID,
        config_id: uuid.UUID,
        benchmark_item_id: str,
        random_seed: int,
        code_version: str,
    ) -> Run:
        run = Run(
            experiment_id=experiment_id,
            config_id=config_id,
            benchmark_item_id=benchmark_item_id,
            random_seed=random_seed,
            code_version=code_version,
        )
        self.session.add(run)
        await self.session.flush()
        return run

    async def materialize_phase3_execution(
        self,
        *,
        manifest_sha256: str,
        plan_sha256: str,
        benchmark_sha256: str,
        roster_sha256: str,
        execution_envelope_json: dict[str, Any],
        cells: Sequence[Phase3ExecutionCellSpec],
    ) -> tuple[Phase3ExecutionBatch, bool]:

        for field, value in (
            ("manifest_sha256", manifest_sha256),
            ("plan_sha256", plan_sha256),
            ("benchmark_sha256", benchmark_sha256),
            ("roster_sha256", roster_sha256),
        ):
            _require_sha256(value, field=field)
        if not cells:
            raise ValueError("at least one Phase 3 execution cell is required")
        if len({cell.cell_id for cell in cells}) != len(cells):
            raise ValueError("Phase 3 execution cell IDs must be unique")
        positions = {(cell.stage, cell.ordinal) for cell in cells}
        if len(positions) != len(cells):
            raise ValueError("Phase 3 stage ordinals must be unique")

        existing = await self.session.scalar(
            select(Phase3ExecutionBatch).where(
                Phase3ExecutionBatch.manifest_sha256 == manifest_sha256
            )
        )
        if existing is not None:
            expected_batch = (
                plan_sha256,
                benchmark_sha256,
                roster_sha256,
                execution_envelope_json,
            )
            actual_batch = (
                existing.plan_sha256,
                existing.benchmark_sha256,
                existing.roster_sha256,
                existing.execution_envelope_json,
            )
            if actual_batch != expected_batch:
                raise RuntimeError("Phase 3 execution manifest identity changed")
            persisted = tuple(
                (
                    await self.session.scalars(
                        select(Phase3ExecutionCell)
                        .where(Phase3ExecutionCell.batch_id == existing.id)
                        .order_by(Phase3ExecutionCell.stage, Phase3ExecutionCell.ordinal)
                    )
                ).all()
            )
            expected_cells = sorted(
                (
                    cell.cell_id,
                    cell.stage,
                    cell.ordinal,
                    cell.cell_sha256,
                    cell.cell_spec_json,
                )
                for cell in cells
            )
            actual_cells = sorted(
                (
                    cell.cell_id,
                    cell.stage,
                    cell.ordinal,
                    cell.cell_sha256,
                    cell.cell_spec_json,
                )
                for cell in persisted
            )
            if actual_cells != expected_cells:
                raise RuntimeError("Phase 3 execution cell set changed")
            return existing, False

        batch = Phase3ExecutionBatch(
            manifest_sha256=manifest_sha256,
            plan_sha256=plan_sha256,
            benchmark_sha256=benchmark_sha256,
            roster_sha256=roster_sha256,
            execution_envelope_json=execution_envelope_json,
            status=Phase3ExecutionBatchStatus.MATERIALIZED,
        )
        self.session.add(batch)
        await self.session.flush()
        self.session.add_all(
            [
                Phase3ExecutionCell(
                    batch_id=batch.id,
                    cell_id=cell.cell_id,
                    stage=cell.stage,
                    ordinal=cell.ordinal,
                    cell_sha256=cell.cell_sha256,
                    cell_spec_json=cell.cell_spec_json,
                    status=Phase3ExecutionCellStatus.QUEUED,
                )
                for cell in cells
            ]
        )
        await self.session.flush()
        return batch, True

    async def claim_next_phase3_execution_cell(
        self,
        *,
        batch_id: uuid.UUID,
        stage: str,
    ) -> Phase3ExecutionCell | None:
        batch = await self.session.get(Phase3ExecutionBatch, batch_id)
        if batch is None:
            raise LookupError(f"Phase 3 execution batch not found: {batch_id}")
        if batch.status in {
            Phase3ExecutionBatchStatus.ABORTED,
            Phase3ExecutionBatchStatus.COMPLETED,
            Phase3ExecutionBatchStatus.INDETERMINATE,
        }:
            return None
        claimed = await self.session.scalar(
            select(Phase3ExecutionCell).where(
                Phase3ExecutionCell.batch_id == batch_id,
                Phase3ExecutionCell.stage == stage,
                Phase3ExecutionCell.status == Phase3ExecutionCellStatus.CLAIMED,
            )
        )
        if claimed is not None:
            raise RuntimeError(
                "claimed Phase 3 cell requires manual reconciliation before more work"
            )
        statement = (
            select(Phase3ExecutionCell)
            .where(
                Phase3ExecutionCell.batch_id == batch_id,
                Phase3ExecutionCell.stage == stage,
                Phase3ExecutionCell.status == Phase3ExecutionCellStatus.QUEUED,
            )
            .order_by(Phase3ExecutionCell.ordinal, Phase3ExecutionCell.id)
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        cell = await self.session.scalar(statement)
        if cell is None:
            return None
        now = utc_now()
        cell.status = Phase3ExecutionCellStatus.CLAIMED
        cell.claimed_at = now
        batch.status = Phase3ExecutionBatchStatus.RUNNING
        await self.session.commit()
        return cell

    async def link_phase3_execution_run(
        self,
        *,
        cell: Phase3ExecutionCell,
        run: Run,
    ) -> None:
        if cell.status != Phase3ExecutionCellStatus.CLAIMED:
            raise RuntimeError("only a claimed Phase 3 cell can be linked to a run")
        if cell.run_id is not None:
            if cell.run_id == run.id:
                return
            raise RuntimeError("Phase 3 execution cell is already linked to another run")
        existing = await self.session.scalar(
            select(Phase3ExecutionCell).where(Phase3ExecutionCell.run_id == run.id)
        )
        if existing is not None:
            raise RuntimeError("run is already linked to another Phase 3 execution cell")
        cell.run_id = run.id
        await self.session.flush()

    async def mark_phase3_execution_cell_terminal(
        self,
        *,
        cell: Phase3ExecutionCell,
        run: Run,
        stop_reason: str,
    ) -> None:
        terminal_run_statuses = {
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.BUDGET_STOPPED,
            RunStatus.TIMED_OUT,
        }
        if cell.status == Phase3ExecutionCellStatus.TERMINAL:
            if cell.run_id == run.id and cell.stop_reason == stop_reason:
                return
            raise RuntimeError("Phase 3 execution cell already has another terminal outcome")
        if cell.status != Phase3ExecutionCellStatus.CLAIMED or cell.run_id != run.id:
            raise RuntimeError("terminal Phase 3 cell must match its claimed run")
        if run.status not in terminal_run_statuses or run.finished_at is None:
            raise RuntimeError("Phase 3 execution run is not terminal")
        cell.status = Phase3ExecutionCellStatus.TERMINAL
        cell.terminal_at = utc_now()
        cell.stop_reason = stop_reason
        await self.session.flush()
        await self.reconcile_phase3_execution_batch(batch_id=cell.batch_id)

    async def reconcile_phase3_execution_batch(
        self,
        *,
        batch_id: uuid.UUID,
    ) -> Phase3ExecutionBatch:

        batch = await self.session.get(Phase3ExecutionBatch, batch_id)
        if batch is None:
            raise LookupError(f"Phase 3 execution batch not found: {batch_id}")
        cells = tuple(
            (
                await self.session.scalars(
                    select(Phase3ExecutionCell).where(Phase3ExecutionCell.batch_id == batch_id)
                )
            ).all()
        )
        if not cells:
            raise RuntimeError("Phase 3 execution batch has no cells")
        if batch.status == Phase3ExecutionBatchStatus.INDETERMINATE:
            return batch
        if any(cell.status == Phase3ExecutionCellStatus.INDETERMINATE for cell in cells):
            batch.status = Phase3ExecutionBatchStatus.INDETERMINATE
            batch.finished_at = batch.finished_at or utc_now()
            batch.stop_reason = batch.stop_reason or "indeterminate_cell_requires_reconciliation"
            await self.session.flush()
            return batch

        terminal_cells = tuple(
            cell for cell in cells if cell.status == Phase3ExecutionCellStatus.TERMINAL
        )
        run_ids = {cell.run_id for cell in terminal_cells if cell.run_id is not None}
        runs = tuple((await self.session.scalars(select(Run).where(Run.id.in_(run_ids)))).all())
        runs_by_id = {run.id: run for run in runs}
        if any(cell.run_id is None or cell.run_id not in runs_by_id for cell in terminal_cells):
            batch.status = Phase3ExecutionBatchStatus.INDETERMINATE
            batch.finished_at = utc_now()
            batch.stop_reason = "terminal_cell_run_identity_missing"
        elif any(
            runs_by_id[cell.run_id].status != RunStatus.COMPLETED
            for cell in terminal_cells
            if cell.run_id is not None
        ):
            batch.status = Phase3ExecutionBatchStatus.ABORTED
            batch.finished_at = utc_now()
            batch.stop_reason = "terminal_cell_not_completed"
        elif len(terminal_cells) == len(cells):
            batch.status = Phase3ExecutionBatchStatus.COMPLETED
            batch.finished_at = utc_now()
            batch.stop_reason = "all_cells_completed"
        elif terminal_cells or any(
            cell.status == Phase3ExecutionCellStatus.CLAIMED for cell in cells
        ):
            batch.status = Phase3ExecutionBatchStatus.RUNNING
            batch.finished_at = None
            batch.stop_reason = None
        await self.session.flush()
        return batch

    async def mark_phase3_execution_cell_indeterminate(
        self,
        *,
        cell: Phase3ExecutionCell,
        stop_reason: str,
    ) -> None:
        if not stop_reason.strip():
            raise ValueError("indeterminate Phase 3 cell requires a stop reason")
        if cell.status == Phase3ExecutionCellStatus.TERMINAL:
            raise RuntimeError("terminal Phase 3 cells cannot become indeterminate")
        if cell.status == Phase3ExecutionCellStatus.INDETERMINATE:
            if cell.stop_reason == stop_reason:
                return
            raise RuntimeError("Phase 3 execution cell has another indeterminate reason")
        cell.status = Phase3ExecutionCellStatus.INDETERMINATE
        cell.terminal_at = utc_now()
        cell.stop_reason = stop_reason
        batch = await self.session.get(Phase3ExecutionBatch, cell.batch_id)
        if batch is None:
            raise LookupError(f"Phase 3 execution batch not found: {cell.batch_id}")
        batch.status = Phase3ExecutionBatchStatus.INDETERMINATE
        batch.finished_at = utc_now()
        batch.stop_reason = stop_reason
        await self.session.flush()

    async def start_run(self, run: Run) -> None:
        run.status = RunStatus.RUNNING
        if run.started_at is None:
            run.started_at = utc_now()
        await self.session.flush()

    async def materialize_phase4_instrument_batch(
        self,
        *,
        manifest_sha256: str,
        source_phase3_manifest_sha256: str,
        response_set_sha256: str,
        target_endpoint_profile: str,
        protocol_json: dict[str, Any],
        items: Sequence[Phase4InstrumentItemSpec],
    ) -> tuple[Phase4InstrumentBatch, bool]:
        for field, value in (
            ("manifest_sha256", manifest_sha256),
            ("source_phase3_manifest_sha256", source_phase3_manifest_sha256),
            ("response_set_sha256", response_set_sha256),
        ):
            _require_sha256(value, field=field)
        if len(items) != 21:
            raise ValueError("Phase 4 frozen response set must contain exactly 21 rounds")
        if len({item.source_round_id for item in items}) != len(items):
            raise ValueError("Phase 4 source round IDs must be unique")
        if {item.ordinal for item in items} != set(range(len(items))):
            raise ValueError("Phase 4 ordinals must be contiguous from zero")
        if not target_endpoint_profile.strip() or not protocol_json:
            raise ValueError("Phase 4 target profile and protocol are required")
        existing = await self.session.scalar(
            select(Phase4InstrumentBatch).where(
                Phase4InstrumentBatch.manifest_sha256 == manifest_sha256
            )
        )
        identity = (
            source_phase3_manifest_sha256,
            response_set_sha256,
            target_endpoint_profile,
            protocol_json,
        )
        if existing is not None:
            if (
                existing.source_phase3_manifest_sha256,
                existing.response_set_sha256,
                existing.target_endpoint_profile,
                existing.protocol_json,
            ) != identity:
                raise RuntimeError("Phase 4 instrument manifest identity changed")
            persisted = tuple(
                (
                    await self.session.scalars(
                        select(Phase4InstrumentItem)
                        .where(Phase4InstrumentItem.batch_id == existing.id)
                        .order_by(Phase4InstrumentItem.ordinal)
                    )
                ).all()
            )
            actual = tuple(
                (
                    item.source_round_id,
                    item.source_evaluation_id,
                    item.ordinal,
                    item.source_language,
                    item.source_response_sha256,
                    item.source_evaluation_sha256,
                    item.input_bundle_sha256,
                )
                for item in persisted
            )
            expected = tuple(
                (
                    item.source_round_id,
                    item.source_evaluation_id,
                    item.ordinal,
                    item.source_language,
                    item.source_response_sha256,
                    item.source_evaluation_sha256,
                    item.input_bundle_sha256,
                )
                for item in sorted(items, key=lambda value: value.ordinal)
            )
            if actual != expected:
                raise RuntimeError("Phase 4 frozen response set changed")
            return existing, False
        batch = Phase4InstrumentBatch(
            manifest_sha256=manifest_sha256,
            source_phase3_manifest_sha256=source_phase3_manifest_sha256,
            response_set_sha256=response_set_sha256,
            target_endpoint_profile=target_endpoint_profile,
            protocol_json=protocol_json,
            status=Phase4InstrumentBatchStatus.MATERIALIZED,
        )
        self.session.add(batch)
        await self.session.flush()
        self.session.add_all(
            [
                Phase4InstrumentItem(
                    batch_id=batch.id,
                    source_round_id=item.source_round_id,
                    source_evaluation_id=item.source_evaluation_id,
                    ordinal=item.ordinal,
                    source_language=item.source_language,
                    source_response_sha256=item.source_response_sha256,
                    source_evaluation_sha256=item.source_evaluation_sha256,
                    input_bundle_sha256=item.input_bundle_sha256,
                    status=Phase4InstrumentItemStatus.QUEUED,
                )
                for item in items
            ]
        )
        await self.session.flush()
        return batch, True

    async def claim_next_phase4_instrument_item(
        self, *, batch_id: uuid.UUID
    ) -> Phase4InstrumentItem | None:
        batch = await self.session.get(Phase4InstrumentBatch, batch_id)
        if batch is None:
            raise LookupError(f"Phase 4 instrument batch not found: {batch_id}")
        if batch.status in {
            Phase4InstrumentBatchStatus.ABORTED,
            Phase4InstrumentBatchStatus.COMPLETED,
            Phase4InstrumentBatchStatus.INDETERMINATE,
        }:
            return None
        claimed = await self.session.scalar(
            select(Phase4InstrumentItem).where(
                Phase4InstrumentItem.batch_id == batch_id,
                Phase4InstrumentItem.status == Phase4InstrumentItemStatus.CLAIMED,
            )
        )
        if claimed is not None:
            raise RuntimeError("claimed Phase 4 item requires manual reconciliation")
        item = await self.session.scalar(
            select(Phase4InstrumentItem)
            .where(
                Phase4InstrumentItem.batch_id == batch_id,
                Phase4InstrumentItem.status == Phase4InstrumentItemStatus.QUEUED,
            )
            .order_by(Phase4InstrumentItem.ordinal)
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        if item is None:
            return None
        item.status = Phase4InstrumentItemStatus.CLAIMED
        item.claimed_at = utc_now()
        batch.status = Phase4InstrumentBatchStatus.RUNNING
        await self.session.commit()
        return item

    async def link_phase4_instrument_run(self, *, item: Phase4InstrumentItem, run: Run) -> None:
        if item.status != Phase4InstrumentItemStatus.CLAIMED:
            raise RuntimeError("only a claimed Phase 4 item can be linked")
        if item.run_id is not None:
            if item.run_id == run.id:
                return
            raise RuntimeError("Phase 4 item is already linked to another run")
        existing = await self.session.scalar(
            select(Phase4InstrumentItem).where(Phase4InstrumentItem.run_id == run.id)
        )
        if existing is not None:
            raise RuntimeError("run is already linked to another Phase 4 item")
        item.run_id = run.id
        await self.session.flush()

    async def finish_phase4_instrument_item(
        self,
        *,
        item: Phase4InstrumentItem,
        run: Run,
        label: str,
        rationale: str,
        confidence: float,
        parser_version: str,
        llm_call: LLMCall,
    ) -> None:
        if (
            item.status != Phase4InstrumentItemStatus.CLAIMED
            or item.run_id != run.id
            or run.status != RunStatus.COMPLETED
            or run.finished_at is None
        ):
            raise RuntimeError("Phase 4 item does not match a completed claimed run")
        if label not in {"refused", "partial", "complied"} or not rationale.strip():
            raise ValueError("Phase 4 parsed self-judge result is invalid")
        total_calls = await self.llm_call_count(run.id)
        if (
            llm_call.run_id != run.id
            or llm_call.purpose != "target_self_judge"
            or llm_call.status != CallStatus.SUCCEEDED
            or total_calls != 1
        ):
            raise RuntimeError("Phase 4 run must contain exactly one succeeded self-judge call")
        self.session.add(
            Phase4SelfJudgeEvaluation(
                item_id=item.id,
                llm_call_id=llm_call.id,
                label=label,
                rationale=rationale,
                confidence=confidence,
                parser_version=parser_version,
            )
        )
        item.status = Phase4InstrumentItemStatus.TERMINAL
        item.terminal_at = utc_now()
        item.stop_reason = "self_judge_completed"
        await self.session.flush()
        remaining = await self.session.scalar(
            select(func.count())
            .select_from(Phase4InstrumentItem)
            .where(
                Phase4InstrumentItem.batch_id == item.batch_id,
                Phase4InstrumentItem.status != Phase4InstrumentItemStatus.TERMINAL,
            )
        )
        if remaining == 0:
            batch = await self.session.get(Phase4InstrumentBatch, item.batch_id)
            if batch is None:
                raise LookupError(f"Phase 4 instrument batch not found: {item.batch_id}")
            batch.status = Phase4InstrumentBatchStatus.COMPLETED
            batch.finished_at = utc_now()
            batch.stop_reason = "all_self_judges_completed"
        await self.session.flush()

    async def mark_phase4_instrument_item_indeterminate(
        self, *, item: Phase4InstrumentItem, stop_reason: str
    ) -> None:
        if item.status == Phase4InstrumentItemStatus.TERMINAL:
            raise RuntimeError("terminal Phase 4 item cannot become indeterminate")
        item.status = Phase4InstrumentItemStatus.INDETERMINATE
        item.terminal_at = utc_now()
        item.stop_reason = stop_reason
        batch = await self.session.get(Phase4InstrumentBatch, item.batch_id)
        if batch is None:
            raise LookupError(f"Phase 4 instrument batch not found: {item.batch_id}")
        batch.status = Phase4InstrumentBatchStatus.INDETERMINATE
        batch.finished_at = utc_now()
        batch.stop_reason = stop_reason
        await self.session.flush()

    async def claim_next_queued_run(self) -> Run | None:
        statement = (
            select(Run)
            .where(Run.status == RunStatus.QUEUED)
            .order_by(Run.created_at, Run.id)
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        run = await self.session.scalar(statement)
        if run is None:
            return None
        await self.start_run(run)
        await self.session.commit()
        return run

    async def get_config_for_run(self, run: Run) -> ExperimentConfig:
        snapshot = await self.session.get(ConfigSnapshot, run.config_id)
        if snapshot is None:
            raise LookupError(f"config snapshot not found: {run.config_id}")
        return ExperimentConfig.model_validate(snapshot.config_json)

    async def get_experiment(self, experiment_id: uuid.UUID) -> Experiment:
        experiment = await self.session.get(Experiment, experiment_id)
        if experiment is None:
            raise LookupError(f"experiment not found: {experiment_id}")
        return experiment

    async def recover_abandoned_runs(self, *, older_than_seconds: int) -> int:
        cutoff = datetime.now(UTC) - timedelta(seconds=older_than_seconds)
        rows = list(
            (
                await self.session.scalars(
                    select(Run).where(
                        Run.status == RunStatus.RUNNING,
                        Run.started_at.is_not(None),
                        Run.started_at < cutoff,
                    )
                )
            ).all()
        )
        for run in rows:
            await self.finish_run(
                run,
                status=RunStatus.FAILED,
                stop_reason="worker_abandoned",
                estimated_cost_usd=run.estimated_cost_usd,
                error_summary="scheduler recovered an abandoned running job",
            )
        await self.session.commit()
        return len(rows)

    async def finish_run(
        self,
        run: Run,
        *,
        status: RunStatus,
        stop_reason: str,
        estimated_cost_usd: float,
        error_summary: str | None = None,
    ) -> None:
        finished_at = utc_now()
        run.status = status
        run.stop_reason = stop_reason
        run.finished_at = finished_at
        run.estimated_cost_usd = estimated_cost_usd
        run.error_summary = error_summary
        if run.started_at is not None:
            run.duration_ms = int((finished_at - run.started_at).total_seconds() * 1000)
        await self.session.flush()

    async def create_step(
        self, *, run_id: uuid.UUID, sequence_number: int, step_type: str, purpose: str
    ) -> Step:
        step = Step(
            run_id=run_id,
            sequence_number=sequence_number,
            step_type=step_type,
            purpose=purpose,
        )
        self.session.add(step)
        await self.session.flush()
        return step

    async def finish_step(self, step: Step, *, status: CallStatus = CallStatus.SUCCEEDED) -> None:
        step.status = status
        step.finished_at = utc_now()
        await self.session.flush()

    async def begin_llm_call(
        self,
        *,
        run_id: uuid.UUID,
        step_id: uuid.UUID,
        purpose: str,
        provider: str,
        model: str,
        endpoint_profile: str | None,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        parameters: dict[str, Any],
    ) -> LLMCall:
        call = LLMCall(
            run_id=run_id,
            step_id=step_id,
            purpose=purpose,
            provider=provider,
            model=model,
            endpoint_profile=endpoint_profile,
            request_messages_json=messages,
            request_tools_json=tools,
            request_parameters_json=parameters,
        )
        self.session.add(call)
        await self.session.flush()
        return call

    async def succeed_llm_call(
        self,
        call: LLMCall,
        *,
        response: dict[str, Any],
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int,
        reasoning_tokens: int,
        latency_ms: int,
        estimated_cost_usd: float,
    ) -> None:
        call.response_json = response
        call.input_tokens = input_tokens
        call.output_tokens = output_tokens
        call.cached_input_tokens = cached_input_tokens
        call.reasoning_tokens = reasoning_tokens
        call.latency_ms = latency_ms
        call.estimated_cost_usd = estimated_cost_usd
        call.status = CallStatus.SUCCEEDED
        call.finished_at = utc_now()
        await self.session.flush()

    async def fail_llm_call(self, call: LLMCall, *, error: dict[str, Any], latency_ms: int) -> None:
        call.error_json = error
        call.latency_ms = latency_ms
        call.status = CallStatus.FAILED
        call.finished_at = utc_now()
        await self.session.flush()

    async def record_tool_call(
        self,
        *,
        run_id: uuid.UUID,
        step_id: uuid.UUID,
        tool_name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
        latency_ms: int,
    ) -> ToolCall:
        call = ToolCall(
            run_id=run_id,
            step_id=step_id,
            tool_name=tool_name,
            arguments_json=arguments,
            result_json=result,
            status=CallStatus.SUCCEEDED,
            latency_ms=latency_ms,
        )
        self.session.add(call)
        await self.session.flush()
        return call

    async def add_external_agent_session(
        self,
        *,
        provider: str,
        runtime: str,
        runtime_version: str,
        model_requested: str | None,
        invocation: dict[str, Any],
        events: list[dict[str, Any]],
        final_output: str | None,
        usage: dict[str, Any] | None,
        status: str,
        started_at: datetime,
        finished_at: datetime,
        duration_ms: int,
        error_summary: str | None,
        run_id: uuid.UUID | None = None,
    ) -> ExternalAgentSession:
        record = ExternalAgentSession(
            run_id=run_id,
            provider=provider,
            runtime=runtime,
            runtime_version=runtime_version,
            model_requested=model_requested,
            invocation_json=invocation,
            observable_events_json=events,
            final_output=final_output,
            usage_json=usage,
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            error_summary=error_summary,
        )
        self.session.add(record)
        await self.session.flush()
        return record

    async def llm_call_count(self, run_id: uuid.UUID) -> int:
        result = await self.session.scalar(
            select(func.count()).select_from(LLMCall).where(LLMCall.run_id == run_id)
        )
        return int(result or 0)

    async def tool_call_count(self, run_id: uuid.UUID, tool_name: str | None = None) -> int:
        statement = select(func.count()).select_from(ToolCall).where(ToolCall.run_id == run_id)
        if tool_name is not None:
            statement = statement.where(ToolCall.tool_name == tool_name)
        result = await self.session.scalar(statement)
        return int(result or 0)

    async def terminal_runs_pending_delivery(
        self, *, channel: str, destination: str, limit: int = 20
    ) -> list[Run]:
        delivered = select(ReportDelivery.run_id).where(
            ReportDelivery.channel == channel,
            ReportDelivery.destination == destination,
            ReportDelivery.status == "delivered",
        )
        statement = (
            select(Run)
            .where(
                and_(
                    Run.status.in_(
                        (
                            RunStatus.COMPLETED,
                            RunStatus.FAILED,
                            RunStatus.BUDGET_STOPPED,
                            RunStatus.TIMED_OUT,
                        )
                    ),
                    Run.id.not_in(delivered),
                )
            )
            .order_by(Run.finished_at, Run.id)
            .limit(limit)
        )
        return list((await self.session.scalars(statement)).all())

    async def record_report_delivery(
        self,
        *,
        run_id: uuid.UUID,
        channel: str,
        destination: str,
        status: str,
        provider_message_id: str | None = None,
        error_summary: str | None = None,
    ) -> ReportDelivery:
        existing = await self.session.scalar(
            select(ReportDelivery).where(
                ReportDelivery.run_id == run_id,
                ReportDelivery.channel == channel,
                ReportDelivery.destination == destination,
            )
        )
        now = utc_now()
        if existing is None:
            existing = ReportDelivery(
                run_id=run_id,
                channel=channel,
                destination=destination,
                status=status,
                attempted_at=now,
            )
            self.session.add(existing)
        existing.status = status
        existing.provider_message_id = provider_message_id
        existing.error_summary = error_summary
        existing.attempted_at = now
        existing.delivered_at = now if status == "delivered" else None
        await self.session.flush()
        return existing
