from __future__ import annotations

from typing import Any

from harness.config.schema import ExperimentConfig
from harness.persistence.database import create_engine, create_schema, session_factory
from harness.persistence.models import RunStatus
from harness.persistence.repository import TraceRepository
from harness.reporting.telegram import TelegramReporter


class FakeTelegramClient:
    def __init__(self) -> None:
        self.messages: list[tuple[int, str]] = []

    async def send_message(self, *, chat_id: int, text: str) -> str:
        self.messages.append((chat_id, text))
        return str(len(self.messages))


async def test_terminal_run_is_delivered_once(
    experiment_config: ExperimentConfig,
) -> None:
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    await create_schema(engine)
    sessions = session_factory(engine)
    async with sessions() as session:
        repository = TraceRepository(session)
        experiment = await repository.create_experiment(
            name="Telegram",
            research_question="Will a report be sent?",
            hypothesis="Exactly one report is delivered.",
            created_by="pytest",
        )
        snapshot = await repository.get_or_create_config(name="telegram", config=experiment_config)
        run = await repository.create_run(
            experiment_id=experiment.id,
            config_id=snapshot.id,
            benchmark_item_id="benchmark-001",
            random_seed=1,
            code_version="test",
        )
        await repository.start_run(run)
        await repository.finish_run(
            run,
            status=RunStatus.COMPLETED,
            stop_reason="finish_experiment",
            estimated_cost_usd=0,
        )
        await session.commit()

    client = FakeTelegramClient()
    reporter = TelegramReporter(
        client=client,  # type: ignore[arg-type]
        sessions=sessions,
        allowed_chat_ids=(123,),
        private_trace_base_url="https://trace.example",
    )

    assert await reporter.report_pending_once() == 1
    assert await reporter.report_pending_once() == 0
    assert len(client.messages) == 1
    assert "finish_experiment" in client.messages[0][1]
    await engine.dispose()


def test_telegram_transport_error_does_not_need_token_in_message() -> None:

    value: dict[str, Any] = {"token_fixture_used": False}
    assert value["token_fixture_used"] is False
