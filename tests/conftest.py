from __future__ import annotations

from typing import Any

import pytest

from harness.config.schema import ExperimentConfig


@pytest.fixture
def config_data() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": {"benchmark_item_id": "benchmark-001"},
        "models": {
            "researcher": {"provider": "mock", "model": "deterministic-v1"},
            "attacker": {"provider": "mock", "model": "deterministic-v1"},
            "target": {"provider": "mock", "model": "deterministic-v1"},
            "judge": {"provider": "mock", "model": "deterministic-v1"},
        },
        "tools": {"enabled": ["note_finding", "finish_experiment"]},
        "reporting": {"telegram": False, "include_private_trace_link": False},
    }


@pytest.fixture
def experiment_config(config_data: dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig.model_validate(config_data)
