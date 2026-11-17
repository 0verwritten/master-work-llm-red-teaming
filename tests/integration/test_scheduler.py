from __future__ import annotations

from harness.config.schema import ExperimentConfig
from harness.persistence.database import create_engine, create_schema, session_factory
from harness.persistence.models import RunStatus
from harness.persistence.repository import TraceRepository
from harness.scheduler import Scheduler


async def test_scheduler_claims_and_completes_queued_run(
    experiment_config: ExperimentConfig,
) -> None:
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    await create_schema(engine)
    sessions = session_factory(engine)
    async with sessions() as session:
        repository = TraceRepository(session)
        experiment = await repository.create_experiment(
            name="Scheduled demo",
            research_question="Does the queue drive the native loop?",
            hypothesis="The queued run completes.",
            created_by="pytest",
        )
        snapshot = await repository.get_or_create_config(name="scheduled", config=experiment_config)
        run = await repository.create_run(
            experiment_id=experiment.id,
            config_id=snapshot.id,
            benchmark_item_id=experiment_config.task.benchmark_item_id,
            random_seed=7,
            code_version="test",
        )
        run_id = run.id
        await session.commit()

    scheduler = Scheduler(sessions=sessions)
    result = await scheduler.run_once()

    assert result is not None
    assert result.run_id == run_id
    assert result.status == RunStatus.COMPLETED
    assert result.llm_calls == 2
    assert await scheduler.run_once() is None
    await engine.dispose()
