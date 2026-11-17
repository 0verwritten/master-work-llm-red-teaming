from __future__ import annotations

import asyncio
from collections.abc import Callable

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from harness.agents.native.gateway import TracedModelGateway
from harness.agents.native.mock_provider import DeterministicMockClient
from harness.agents.native.runner import NativeRunner, NativeRunResult
from harness.agents.native.tools import ToolRegistry
from harness.agents.native.types import ModelClient
from harness.config.schema import ExperimentConfig, Provider
from harness.persistence.models import RunStatus
from harness.persistence.repository import TraceRepository

ModelClientFactory = Callable[[ExperimentConfig], ModelClient]


def default_model_client_factory(config: ExperimentConfig) -> ModelClient:
    researcher = config.models.researcher
    if researcher.provider == Provider.MOCK:
        return DeterministicMockClient()
    raise RuntimeError(
        f"provider adapter is not configured: {researcher.provider}/{researcher.model}"
    )


class Scheduler:
    def __init__(
        self,
        *,
        sessions: async_sessionmaker[AsyncSession],
        model_client_factory: ModelClientFactory = default_model_client_factory,
        abandoned_after_seconds: int = 3_600,
    ) -> None:
        self.sessions = sessions
        self.model_client_factory = model_client_factory
        self.abandoned_after_seconds = abandoned_after_seconds

    async def recover_abandoned(self) -> int:
        async with self.sessions() as session:
            return await TraceRepository(session).recover_abandoned_runs(
                older_than_seconds=self.abandoned_after_seconds
            )

    async def run_once(self) -> NativeRunResult | None:
        async with self.sessions() as session:
            repository = TraceRepository(session)
            run = await repository.claim_next_queued_run()
            if run is None:
                return None
            try:
                config = await repository.get_config_for_run(run)
                experiment = await repository.get_experiment(run.experiment_id)
                client = self.model_client_factory(config)
                runner = NativeRunner(
                    repository=repository,
                    gateway=TracedModelGateway(client, repository),
                    tools=ToolRegistry(),
                )
                prompt = (
                    f"Research question: {experiment.research_question}\n"
                    f"Hypothesis: {experiment.hypothesis}\n"
                    "Complete the experiment using only registered tools."
                )
                return await runner.run(run=run, config=config, initial_prompt=prompt)
            except Exception as exc:
                await repository.finish_run(
                    run,
                    status=RunStatus.FAILED,
                    stop_reason="scheduler_error",
                    estimated_cost_usd=run.estimated_cost_usd,
                    error_summary=str(exc),
                )
                await session.commit()
                return NativeRunResult(
                    run_id=run.id,
                    status=RunStatus.FAILED,
                    stop_reason="scheduler_error",
                    llm_calls=await repository.llm_call_count(run.id),
                    tool_calls=await repository.tool_call_count(run.id),
                    input_tokens=0,
                    output_tokens=0,
                    estimated_cost_usd=run.estimated_cost_usd,
                )

    async def run_forever(self, *, poll_seconds: float, stop: asyncio.Event) -> None:
        await self.recover_abandoned()
        while not stop.is_set():
            result = await self.run_once()
            if result is None:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=poll_seconds)
                except TimeoutError:
                    pass
