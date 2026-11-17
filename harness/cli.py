from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from alembic import command
from alembic.config import Config
from sqlalchemy import text

from harness.agents.native.gateway import TracedModelGateway
from harness.agents.native.mock_provider import DeterministicMockClient
from harness.agents.native.runner import NativeRunner
from harness.agents.native.tools import ToolRegistry
from harness.agents.subscription.runner import SubscriptionBaselineRunner, SubscriptionRuntime
from harness.code_execution import CodeExecutionGateway, CodeExecutionRequest
from harness.code_execution.modal_backend import ModalSandboxBackend
from harness.config.schema import ExperimentConfig
from harness.persistence.database import create_engine, create_schema, session_factory
from harness.persistence.repository import TraceRepository
from harness.reporting import TelegramClient, TelegramReporter
from harness.scheduler import Scheduler
from harness.settings import Settings
from harness.spend import AttemptLog, SpendEntry, SpendLedger
from harness.spend.backfill import backfill
from harness.spend.modal import ModalCliUnavailableError
from harness.spend.modal import status_report as modal_status_report
from harness.spend.vast import VastCliUnavailableError
from harness.spend.vast import status_report as vast_status_report

app = typer.Typer(help="Self-adversarial LLM research harness.")
baseline_app = typer.Typer(help="Opaque Claude Code and Codex subscription baselines.")
db_app = typer.Typer(help="Database migration and connectivity commands.")
experiment_app = typer.Typer(help="Create and queue experiments.")
scheduler_app = typer.Typer(help="PostgreSQL queue worker commands.")
modal_app = typer.Typer(help="Modal connectivity and sandbox smoke tests.")
telegram_app = typer.Typer(help="Telegram Bot API connectivity and reporting.")
spend_app = typer.Typer(help="Cross-provider spend ledger and per-attempt cost logs.")
app.add_typer(baseline_app, name="baseline")
app.add_typer(db_app, name="db")
app.add_typer(experiment_app, name="experiment")
app.add_typer(scheduler_app, name="scheduler")
app.add_typer(modal_app, name="modal")
app.add_typer(telegram_app, name="telegram")
app.add_typer(spend_app, name="spend")


def _load_config(path: Path) -> ExperimentConfig:
    raw: Any = yaml.safe_load(path.read_text())
    return ExperimentConfig.model_validate(raw)


@app.command()
def validate(config: Annotated[Path, typer.Argument(exists=True, readable=True)]) -> None:

    parsed = _load_config(config)
    typer.echo(json.dumps({"valid": True, "config_hash": parsed.stable_hash()}, indent=2))


@app.command()
def demo() -> None:

    typer.echo(json.dumps(asyncio.run(_demo()), indent=2))


async def _demo() -> dict[str, Any]:
    config = _load_config(Path(__file__).parent / "config" / "example.yaml")
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    await create_schema(engine)
    factory = session_factory(engine)
    async with factory() as session:
        repository = TraceRepository(session)
        experiment = await repository.create_experiment(
            name="Trace invariant demo",
            research_question="Can the native boundary persist every provider request?",
            hypothesis="Every provider request has exactly one llm_calls row.",
            created_by="cli",
        )
        snapshot = await repository.get_or_create_config(name="demo-v1", config=config)
        run = await repository.create_run(
            experiment_id=experiment.id,
            config_id=snapshot.id,
            benchmark_item_id=config.task.benchmark_item_id,
            random_seed=42,
            code_version="working-tree",
        )
        client = DeterministicMockClient()
        runner = NativeRunner(
            repository=repository,
            gateway=TracedModelGateway(client, repository),
            tools=ToolRegistry(),
        )
        result = await runner.run(
            run=run,
            config=config,
            initial_prompt="Demonstrate trace completeness and finish explicitly.",
        )
        output = {
            "run_id": str(result.run_id),
            "status": result.status,
            "stop_reason": result.stop_reason,
            "actual_provider_requests": client.actual_request_count,
            "persisted_llm_calls": result.llm_calls,
            "persisted_tool_calls": result.tool_calls,
            "trace_invariant_holds": client.actual_request_count == result.llm_calls,
        }
    await engine.dispose()
    return output


@baseline_app.command("status")
def baseline_status() -> None:

    typer.echo(json.dumps(asyncio.run(SubscriptionBaselineRunner.status()), indent=2))


@baseline_app.command("run")
def baseline_run(
    runtime: Annotated[SubscriptionRuntime, typer.Argument()],
    model: Annotated[str | None, typer.Option(help="Optional CLI model alias or ID.")] = None,
    timeout_seconds: Annotated[int, typer.Option(min=1, max=3_600)] = 600,
    database_url: Annotated[
        str, typer.Option(envvar="DATABASE_URL")
    ] = "sqlite+aiosqlite:///./research_harness.sqlite3",
) -> None:

    prompt = sys.stdin.read()
    if not prompt.strip():
        raise typer.BadParameter("a non-empty prompt must be piped through stdin")
    result = asyncio.run(
        _baseline_run(
            runtime=runtime,
            prompt=prompt,
            model=model,
            timeout_seconds=timeout_seconds,
            database_url=database_url,
        )
    )
    typer.echo(
        json.dumps(
            {
                "session_id": result["session_id"],
                "status": result["status"],
                "runtime": runtime,
                "duration_ms": result["duration_ms"],
                "final_output": result["final_output"],
                "trace_completeness": "opaque_runtime",
            },
            indent=2,
        )
    )


async def _baseline_run(
    *,
    runtime: SubscriptionRuntime,
    prompt: str,
    model: str | None,
    timeout_seconds: int,
    database_url: str,
) -> dict[str, Any]:
    baseline = SubscriptionBaselineRunner(timeout_seconds=timeout_seconds)
    result = await baseline.run(runtime, prompt=prompt, model=model)
    engine = create_engine(database_url)
    await create_schema(engine)
    factory = session_factory(engine)
    async with factory() as session:
        repository = TraceRepository(session)
        record = await repository.add_external_agent_session(
            provider="openai" if runtime == SubscriptionRuntime.CODEX else "anthropic",
            runtime=f"{runtime.value}_cli",
            runtime_version=result.runtime_version,
            model_requested=model,
            invocation=result.invocation,
            events=result.events,
            final_output=result.final_output,
            usage=result.usage,
            status=result.status,
            started_at=result.started_at,
            finished_at=result.finished_at,
            duration_ms=result.duration_ms,
            error_summary=result.error_summary,
        )
        await session.commit()
        output = {
            "session_id": str(record.id),
            "status": result.status,
            "duration_ms": result.duration_ms,
            "final_output": result.final_output,
        }
    await engine.dispose()
    return output


@db_app.command("upgrade")
def db_upgrade() -> None:

    command.upgrade(Config("alembic.ini"), "head")
    typer.echo(json.dumps({"status": "upgraded", "revision": "head"}, indent=2))


@db_app.command("status")
def db_status() -> None:

    typer.echo(json.dumps(asyncio.run(_db_status()), indent=2))


async def _db_status() -> dict[str, Any]:
    settings = Settings()
    engine = create_engine(settings.database_url)
    try:
        async with engine.connect() as connection:
            value = await connection.scalar(text("SELECT 1"))
        return {"connected": value == 1, "driver": engine.url.drivername}
    finally:
        await engine.dispose()


@experiment_app.command("enqueue")
def experiment_enqueue(
    config: Annotated[Path, typer.Option(exists=True, readable=True)],
    name: Annotated[str, typer.Option()],
    research_question: Annotated[str, typer.Option()],
    hypothesis: Annotated[str, typer.Option()],
    seed: Annotated[int, typer.Option()] = 42,
) -> None:

    output = asyncio.run(
        _experiment_enqueue(
            config=_load_config(config),
            name=name,
            research_question=research_question,
            hypothesis=hypothesis,
            seed=seed,
        )
    )
    typer.echo(json.dumps(output, indent=2))


async def _experiment_enqueue(
    *,
    config: ExperimentConfig,
    name: str,
    research_question: str,
    hypothesis: str,
    seed: int,
) -> dict[str, str]:
    settings = Settings()
    engine = create_engine(settings.database_url)
    try:
        async with session_factory(engine)() as session:
            repository = TraceRepository(session)
            experiment = await repository.create_experiment(
                name=name,
                research_question=research_question,
                hypothesis=hypothesis,
                created_by="cli",
            )
            snapshot = await repository.get_or_create_config(name=f"{name}-config", config=config)
            run = await repository.create_run(
                experiment_id=experiment.id,
                config_id=snapshot.id,
                benchmark_item_id=config.task.benchmark_item_id,
                random_seed=seed,
                code_version="working-tree",
            )
            await session.commit()
            return {
                "experiment_id": str(experiment.id),
                "run_id": str(run.id),
                "status": run.status,
                "config_hash": snapshot.config_hash,
            }
    finally:
        await engine.dispose()


@scheduler_app.command("once")
def scheduler_once() -> None:

    typer.echo(json.dumps(asyncio.run(_scheduler_once()), indent=2, default=str))


async def _scheduler_once() -> dict[str, Any]:
    settings = Settings()
    engine = create_engine(settings.database_url)
    try:
        scheduler = Scheduler(
            sessions=session_factory(engine),
            abandoned_after_seconds=settings.scheduler_abandoned_after_seconds,
        )
        result = await scheduler.run_once()
        if result is None:
            return {"claimed": False}
        return {
            "claimed": True,
            "run_id": str(result.run_id),
            "status": result.status,
            "stop_reason": result.stop_reason,
            "llm_calls": result.llm_calls,
            "tool_calls": result.tool_calls,
        }
    finally:
        await engine.dispose()


@modal_app.command("status")
def modal_status() -> None:

    settings = Settings()
    backend = ModalSandboxBackend(app_name=settings.modal_app_name)
    typer.echo(json.dumps(asyncio.run(backend.connectivity()), indent=2))


@modal_app.command("smoke")
def modal_smoke() -> None:

    typer.echo(json.dumps(asyncio.run(_modal_smoke()), indent=2, default=str))


async def _modal_smoke() -> dict[str, Any]:
    settings = Settings()
    gateway = CodeExecutionGateway(ModalSandboxBackend(app_name=settings.modal_app_name))
    source = """\
import os
import urllib.request

sensitive = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET",
    "TELEGRAM_BOT_TOKEN",
    "DATABASE_URL",
)
assert not any(os.getenv(name) for name in sensitive), "credential leaked into sandbox"
try:
    urllib.request.urlopen("https://example.com", timeout=3)
except Exception:
    print("modal connectivity ok; network blocked; credentials absent")
else:
    raise RuntimeError("sandbox unexpectedly reached the public network")
"""
    result = await gateway.execute(
        CodeExecutionRequest(
            run_id=uuid.uuid4(),
            step_id=uuid.uuid4(),
            files={"main.py": source},
            entrypoint="main.py",
            timeout_seconds=60,
            cpu=0.25,
            memory_mb=512,
            max_output_bytes=10_000,
        )
    )
    return result.model_dump(mode="json")


@telegram_app.command("status")
def telegram_status() -> None:

    typer.echo(json.dumps(asyncio.run(_telegram_status()), indent=2))


async def _telegram_status() -> dict[str, str | int | bool]:
    settings = Settings()
    if settings.telegram_bot_token is None:
        return {
            "configured": False,
            "connected": False,
            "allowed_chat_count": len(settings.allowed_chat_ids),
        }
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    client = TelegramClient(settings.require_telegram_token())
    try:
        reporter = TelegramReporter(
            client=client,
            sessions=session_factory(engine),
            allowed_chat_ids=settings.allowed_chat_ids,
            private_trace_base_url=settings.private_trace_base_url,
            poll_timeout_seconds=settings.telegram_poll_timeout_seconds,
        )
        return await reporter.connectivity()
    finally:
        await client.close()
        await engine.dispose()


@telegram_app.command("report-once")
def telegram_report_once() -> None:

    typer.echo(json.dumps(asyncio.run(_telegram_report_once()), indent=2))


async def _telegram_report_once() -> dict[str, int]:
    settings = Settings()
    engine = create_engine(settings.database_url)
    client = TelegramClient(settings.require_telegram_token())
    try:
        reporter = TelegramReporter(
            client=client,
            sessions=session_factory(engine),
            allowed_chat_ids=settings.allowed_chat_ids,
            private_trace_base_url=settings.private_trace_base_url,
            poll_timeout_seconds=settings.telegram_poll_timeout_seconds,
        )
        return {"delivered": await reporter.report_pending_once()}
    finally:
        await client.close()
        await engine.dispose()


@spend_app.command("status")
def spend_status() -> None:

    ledger = SpendLedger.at_repository_root(Path.cwd())
    typer.echo(
        json.dumps(
            {
                "ledger": str(ledger.path),
                "totals": ledger.totals().model_dump(mode="json"),
                "by_account": {
                    account: totals.model_dump(mode="json")
                    for account, totals in ledger.totals_by_account().items()
                },
            },
            indent=2,
        )
    )


@spend_app.command("vast")
def spend_vast(
    record: Annotated[bool, typer.Option(help="Append the running rentals to the ledger.")] = False,
    purpose: Annotated[str, typer.Option(help="Why the instance is rented.")] = "unrecorded rental",
    evidence: Annotated[
        str, typer.Option(help="Command or document substantiating the amount.")
    ] = "vastai show instances --raw",
) -> None:

    try:
        report = vast_status_report()
    except VastCliUnavailableError as error:
        typer.echo(json.dumps({"error": str(error)}, indent=2))
        raise typer.Exit(code=1) from error
    if record:
        from harness.spend.vast import fetch_account, fetch_instances

        account = fetch_account()
        ledger = SpendLedger.at_repository_root(Path.cwd())
        recorded = [
            ledger.append(
                instance.to_ledger_entry(
                    account=account.account_label, purpose=purpose, evidence=evidence
                )
            ).reference
            for instance in fetch_instances()
            if instance.is_running and instance.is_ours
        ]
        report["recorded"] = recorded
    typer.echo(json.dumps(report, indent=2))


@spend_app.command("attempts")
def spend_attempts(run_id: Annotated[str, typer.Argument()]) -> None:

    typer.echo(json.dumps(AttemptLog.at_repository_root(Path.cwd(), run_id).summary(), indent=2))


@spend_app.command("record")
def spend_record(
    provider: Annotated[str, typer.Option()],
    account: Annotated[str, typer.Option()],
    amount_usd: Annotated[float, typer.Option()],
    reference: Annotated[str, typer.Option()],
    purpose: Annotated[str, typer.Option()],
    evidence: Annotated[str, typer.Option()],
    kind: Annotated[
        str, typer.Option(help="rental, call_batch, storage, or adjustment.")
    ] = "call_batch",
    basis: Annotated[str, typer.Option(help="estimated, observed, or reconciled.")] = "observed",
) -> None:

    entry = SpendEntry.model_validate(
        {
            "provider": provider,
            "account": account,
            "kind": kind,
            "basis": basis,
            "amount_usd": amount_usd,
            "reference": reference,
            "purpose": purpose,
            "evidence": evidence,
        }
    )
    SpendLedger.at_repository_root(Path.cwd()).append(entry)
    typer.echo(
        json.dumps({"appended": entry.entry_sha256, "amount_usd": entry.amount_usd}, indent=2)
    )


@spend_app.command("modal")
def spend_modal() -> None:

    try:
        typer.echo(json.dumps(modal_status_report(), indent=2))
    except ModalCliUnavailableError as error:
        typer.echo(
            json.dumps(
                {
                    "error": str(error),
                    "note": (
                        "Modal's control plane is unreachable. Ledger totals below are "
                        "unaffected; only the live app listing needs it."
                    ),
                },
                indent=2,
            )
        )
        raise typer.Exit(code=1) from error


@spend_app.command("backfill")
def spend_backfill(
    dry_run: Annotated[bool, typer.Option(help="Show what would be appended.")] = False,
) -> None:

    typer.echo(
        json.dumps(backfill(SpendLedger.at_repository_root(Path.cwd()), dry_run=dry_run), indent=2)
    )


if __name__ == "__main__":
    app()
