from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from harness.config.schema import ExperimentConfig


def test_config_hash_is_stable(config_data: dict[str, Any]) -> None:
    first = ExperimentConfig.model_validate(config_data)
    reordered = {
        "tools": config_data["tools"],
        "models": config_data["models"],
        "task": config_data["task"],
        "schema_version": 1,
        "reporting": config_data["reporting"],
    }
    second = ExperimentConfig.model_validate(reordered)
    assert first.stable_hash() == second.stable_hash()


def test_config_is_immutable(experiment_config: ExperimentConfig) -> None:
    with pytest.raises(ValidationError):
        experiment_config.agent.max_steps = 50  # type: ignore[misc]


def test_finish_tool_is_required(config_data: dict[str, Any]) -> None:
    config_data["tools"]["enabled"] = ["note_finding"]
    with pytest.raises(ValidationError, match="finish_experiment is required"):
        ExperimentConfig.model_validate(config_data)


def test_modal_requires_endpoint_profile(config_data: dict[str, Any]) -> None:
    config_data["models"]["target"] = {"provider": "modal", "model": "not-valid"}
    with pytest.raises(ValidationError, match="endpoint_profile"):
        ExperimentConfig.model_validate(config_data)


def test_vast_requires_content_addressed_endpoint_profile(config_data: dict[str, Any]) -> None:
    config_data["models"]["target"] = {
        "provider": "vast.ai",
        "model": None,
        "endpoint_profile": f"vast-fixture@{'a' * 64}",
    }
    config = ExperimentConfig.model_validate(config_data)
    assert config.models.target.provider.value == "vast.ai"

    config_data["models"]["target"] = {"provider": "vast.ai", "model": "not-valid"}
    with pytest.raises(ValidationError, match="endpoint_profile"):
        ExperimentConfig.model_validate(config_data)


def test_summary_interval_matches_context_mode(config_data: dict[str, Any]) -> None:
    config_data["agent"] = {"context_mode": "summarize"}
    with pytest.raises(ValidationError, match="summarize_every_n_steps"):
        ExperimentConfig.model_validate(config_data)


async def test_concurrent_config_inserts_resolve_to_one_snapshot(
    experiment_config: ExperimentConfig,
) -> None:

    import asyncio

    from harness.persistence.database import create_engine, create_schema, session_factory
    from harness.persistence.repository import TraceRepository

    engine = create_engine("sqlite+aiosqlite:///:memory:?cache=shared&uri=true")
    await create_schema(engine)
    sessions = session_factory(engine)

    async def insert() -> str:
        async with sessions() as session:
            repository = TraceRepository(session)
            snapshot = await repository.get_or_create_config(name="race", config=experiment_config)
            await session.commit()
            return str(snapshot.config_hash)

    hashes = await asyncio.gather(*(insert() for _ in range(4)))
    assert len(set(hashes)) == 1
    await engine.dispose()
