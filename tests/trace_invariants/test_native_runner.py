from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine

from harness.agents.native.gateway import TracedModelGateway
from harness.agents.native.mock_provider import DeterministicMockClient
from harness.agents.native.runner import NativeRunner
from harness.agents.native.tools import ToolRegistry
from harness.config.schema import ExperimentConfig
from harness.persistence.database import create_engine, create_schema, session_factory
from harness.persistence.models import RunStatus
from harness.persistence.repository import TraceRepository


async def _make_engine() -> AsyncEngine:
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    await create_schema(engine)
    return engine


async def test_every_provider_request_has_exactly_one_trace(
    experiment_config: ExperimentConfig,
) -> None:
    engine = await _make_engine()
    factory = session_factory(engine)
    async with factory() as session:
        repository = TraceRepository(session)
        experiment = await repository.create_experiment(
            name="Invariant",
            research_question="Are requests traced?",
            hypothesis="Yes",
            created_by="pytest",
        )
        snapshot = await repository.get_or_create_config(name="test", config=experiment_config)
        run = await repository.create_run(
            experiment_id=experiment.id,
            config_id=snapshot.id,
            benchmark_item_id="benchmark-001",
            random_seed=1,
            code_version="test",
        )
        client = DeterministicMockClient()
        runner = NativeRunner(
            repository=repository,
            gateway=TracedModelGateway(client, repository),
            tools=ToolRegistry(),
        )

        result = await runner.run(
            run=run,
            config=experiment_config,
            initial_prompt="Run the invariant test.",
        )

        assert result.status == RunStatus.COMPLETED
        assert result.stop_reason == "finish_experiment"
        assert client.actual_request_count == 2
        assert result.llm_calls == client.actual_request_count
        assert result.tool_calls == 2
        assert await repository.tool_call_count(run.id, "finish_experiment") == 1
    await engine.dispose()


async def test_no_natural_language_completion_counts_as_finished(
    experiment_config: ExperimentConfig,
) -> None:
    constrained = experiment_config.model_copy(
        update={"agent": experiment_config.agent.model_copy(update={"max_steps": 1})}
    )
    engine = await _make_engine()
    factory = session_factory(engine)
    async with factory() as session:
        repository = TraceRepository(session)
        experiment = await repository.create_experiment(
            name="Explicit finish",
            research_question="Is finish explicit?",
            hypothesis="Yes",
            created_by="pytest",
        )
        snapshot = await repository.get_or_create_config(name="one-step", config=constrained)
        run = await repository.create_run(
            experiment_id=experiment.id,
            config_id=snapshot.id,
            benchmark_item_id="benchmark-001",
            random_seed=1,
            code_version="test",
        )
        client = DeterministicMockClient()
        result = await NativeRunner(
            repository=repository,
            gateway=TracedModelGateway(client, repository),
            tools=ToolRegistry(),
        ).run(run=run, config=constrained, initial_prompt="Try once.")

        assert result.status == RunStatus.BUDGET_STOPPED
        assert result.stop_reason == "max_steps"
        assert await repository.tool_call_count(run.id, "finish_experiment") == 0
    await engine.dispose()
