from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Any, Protocol

from sqlalchemy import or_, select

from harness.agents.native.adversarial import (
    AdversarialRunner,
    AdversarialSpec,
    AttackMethod,
    BaselineInputs,
    ExperimentLanguage,
    RoleGateways,
)
from harness.agents.native.gateway import TracedModelGateway
from harness.agents.native.types import ModelClient
from harness.bilingual_dataset import DatasetItem
from harness.experiments.bilingual_manifest import BilingualStudyManifest, StudyCell
from harness.experiments.bilingual_smoke import JudgeLanguageObservation, summarise
from harness.experiments.checkpoint import (
    RETRYABLE_STOP_REASONS,
    CheckpointLog,
    CheckpointRecord,
)
from harness.persistence.models import (
    AdversarialRound,
    AdversarialSession,
    Evaluation,
    Run,
    RunStatus,
)
from harness.persistence.repository import TraceRepository


class ClientFactory(Protocol):
    def __call__(self, *, role: str, cell: StudyCell) -> ModelClient: ...


RepositoryFactory = Callable[[], AbstractAsyncContextManager[TraceRepository]]
BaselineFactory = Callable[[DatasetItem, StudyCell], BaselineInputs | None]

TERMINAL_STATUSES = frozenset({RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.BUDGET_STOPPED})


async def _completed_cell_ids(repository: TraceRepository) -> set[str]:

    rows = (
        await repository.session.execute(
            select(AdversarialSession.spec_json)
            .join(Run, Run.id == AdversarialSession.run_id)
            .join(AdversarialRound, AdversarialRound.session_id == AdversarialSession.id)
            .join(Evaluation, Evaluation.round_id == AdversarialRound.id)
            .where(Run.status.in_(tuple(TERMINAL_STATUSES)))
            .where(
                or_(
                    Run.stop_reason.is_(None),
                    Run.stop_reason.not_in(tuple(RETRYABLE_STOP_REASONS)),
                )
            )
        )
    ).all()
    return {
        str(spec.get("cell_id"))
        for (spec,) in rows
        if isinstance(spec, dict) and spec.get("cell_id")
    }


async def execute_smoke(
    manifest: BilingualStudyManifest,
    *,
    items: list[DatasetItem],
    repository: TraceRepository,
    client_factory: ClientFactory,
    checkpoint: CheckpointLog,
    experiment_name: str = "judge language smoke",
    max_concurrency: int = 1,
    repository_factory: RepositoryFactory | None = None,
    max_cells: int | None = None,
    method: AttackMethod = AttackMethod.PAIR,
    baseline_factory: BaselineFactory | None = None,
    max_role_attempts: int = 1,
    retry_backoff_seconds: float = 0.0,
    max_estimated_cost_usd: float = 1.0,
) -> dict[str, Any]:

    if max_concurrency < 1:
        raise ValueError("max_concurrency must be positive")
    if method.value != manifest.attack_method:
        raise ValueError("execution method does not match frozen manifest")
    if max_concurrency > 1 and repository_factory is None:
        raise ValueError("repository_factory is required when max_concurrency > 1")
    if max_cells is not None and max_cells < 1:
        raise ValueError("max_cells must be positive")

    by_stable_id = {item.stable_id: item for item in items}
    done = checkpoint.completed_cell_ids() | await _completed_cell_ids(repository)

    experiment = await repository.create_experiment(
        name=experiment_name,
        research_question="Can the pinned judge sustain rationales in the requested language?",
        hypothesis="Under an explicit contract the judge answers in the requested language.",
        created_by="bilingual_execute",
    )

    await repository.session.commit()

    pending = [cell for cell in manifest.cells if cell.cell_id not in done]
    skipped = len(manifest.cells) - len(pending)
    if max_cells is not None:
        pending = pending[:max_cells]

    async def execute_one(cell: StudyCell, cell_repository: TraceRepository) -> bool:
        try:
            record = await _execute_cell(
                manifest=manifest,
                cell=cell,
                item=by_stable_id[cell.item_stable_id],
                repository=cell_repository,
                client_factory=client_factory,
                experiment_id=experiment.id,
                method=method,
                baseline_factory=baseline_factory,
                max_role_attempts=max_role_attempts,
                retry_backoff_seconds=retry_backoff_seconds,
                max_estimated_cost_usd=max_estimated_cost_usd,
            )
        except Exception as exc:
            await cell_repository.session.rollback()
            checkpoint.append(
                CheckpointRecord(
                    cell_id=cell.cell_id,
                    language=cell.language,
                    category=cell.category,
                    status="error",
                    error=f"{type(exc).__name__}: {exc}"[:400],
                )
            )
            return False
        checkpoint.append(record)
        return True

    if max_concurrency == 1:
        outcomes = [await execute_one(cell, repository) for cell in pending]
    else:
        assert repository_factory is not None
        semaphore = asyncio.Semaphore(max_concurrency)

        async def execute_isolated(cell: StudyCell) -> bool:
            async with semaphore:
                async with repository_factory() as cell_repository:
                    return await execute_one(cell, cell_repository)

        outcomes = await asyncio.gather(*(execute_isolated(cell) for cell in pending))
    executed = sum(outcomes)

    latest_records = {record.cell_id: record for record in checkpoint.records()}
    observations = [
        JudgeLanguageObservation(
            cell_id=record.cell_id,
            language=record.language,
            category=record.category,
            adherence=record.judge_language_adherence or "indeterminate",
        )
        for record in latest_records.values()
        if record.status in {"completed", "stopped"} and record.judge_language_adherence
    ]
    report = summarise(observations) if observations else None
    return {
        "executed_cells": executed,
        "skipped_cells": skipped,
        "max_concurrency": max_concurrency,
        "max_cells": max_cells,
        "manifest_sha256": manifest.manifest_sha256,
        "checkpoint": checkpoint.summary(),
        "report": report.as_json() if report else None,
    }


async def _execute_cell(
    *,
    manifest: BilingualStudyManifest,
    cell: StudyCell,
    item: DatasetItem,
    repository: TraceRepository,
    client_factory: ClientFactory,
    experiment_id: Any,
    method: AttackMethod = AttackMethod.PAIR,
    baseline_factory: BaselineFactory | None = None,
    max_role_attempts: int = 1,
    retry_backoff_seconds: float = 0.0,
    max_estimated_cost_usd: float = 1.0,
) -> CheckpointRecord:
    from harness.experiments.bilingual_config import build_cell_config

    config = build_cell_config(
        manifest,
        cell,
        max_role_attempts=max_role_attempts,
        max_estimated_cost_usd=max_estimated_cost_usd,
    )
    snapshot = await repository.get_or_create_config(name=cell.cell_id[:32], config=config)
    run = await repository.create_run(
        experiment_id=experiment_id,
        config_id=snapshot.id,
        benchmark_item_id=cell.item_stable_id,
        random_seed=17,
        code_version=manifest.manifest_sha256[:12],
    )
    attacker = (
        None
        if method in {AttackMethod.STATIC, AttackMethod.RANDOM_MUTATION}
        else client_factory(role="attacker", cell=cell)
    )
    target = client_factory(role="target", cell=cell)
    judge = client_factory(role="judge", cell=cell)
    result = await AdversarialRunner(
        repository=repository,
        gateways=RoleGateways(
            attacker=(TracedModelGateway(attacker, repository) if attacker is not None else None),
            target=TracedModelGateway(target, repository),
            judge=TracedModelGateway(judge, repository),
        ),
    ).run(
        run=run,
        config=config,
        benchmark_prompt=item.prompt_text,
        spec=AdversarialSpec(
            method=method,
            language=ExperimentLanguage(cell.language),
            max_rounds=cell.max_rounds,
            rubric_version=manifest.manifest_version,
            rubric_text=item.rubric_text,
            rubric_language=ExperimentLanguage(cell.language),
            benchmark_content_sha256=cell.item_content_sha256,
            enforce_judge_language=manifest.enforce_judge_language,
            reject_repeated_attacks=(
                method == AttackMethod.PAIR and manifest.max_attack_regenerations > 0
            ),
            max_attack_regenerations=manifest.max_attack_regenerations,
            max_role_attempts=max_role_attempts,
            retry_backoff_seconds=retry_backoff_seconds,
        ),
        baseline_inputs=(baseline_factory(item, cell) if baseline_factory else None),
    )
    session = (
        await repository.session.execute(
            select(AdversarialSession).where(AdversarialSession.run_id == run.id)
        )
    ).scalar_one()
    session.spec_json = {**session.spec_json, "cell_id": cell.cell_id}
    await repository.session.commit()

    evaluation = (
        (
            await repository.session.execute(
                select(Evaluation)
                .join(AdversarialRound, AdversarialRound.id == Evaluation.round_id)
                .where(AdversarialRound.session_id == session.id)
                .order_by(Evaluation.created_at.desc())
            )
        )
        .scalars()
        .first()
    )
    adherence = None
    label = None
    rationale_sha = None
    if evaluation is not None:
        label = evaluation.label
        adherence = (evaluation.raw_json or {}).get("judge_language_adherence")
        rationale_sha = hashlib.sha256((evaluation.rationale or "").encode()).hexdigest()

    if evaluation is None:
        status = "error"
    elif result.status == RunStatus.COMPLETED:
        status = "completed"
    else:
        status = "stopped"
    return CheckpointRecord(
        cell_id=cell.cell_id,
        language=cell.language,
        category=cell.category,
        status=status,
        judge_language_adherence=adherence,
        label=label,
        rationale_sha256=rationale_sha,
        estimated_cost_usd=result.estimated_cost_usd,
        stop_reason=result.stop_reason,
        error=None if evaluation is not None else f"no evaluation: {result.stop_reason}",
    )
