from __future__ import annotations

from harness.config.schema import (
    AgentConfig,
    BudgetsConfig,
    ExperimentConfig,
    ModelConfig,
    ModelsConfig,
    Provider,
    ReportingConfig,
    TaskConfig,
    ToolsConfig,
)
from harness.experiments.adversarial_matrix import FrozenModelProfile
from harness.experiments.bilingual_manifest import BilingualStudyManifest, StudyCell


def _model(profile: FrozenModelProfile, *, temperature: float) -> ModelConfig:
    return ModelConfig(
        provider=Provider(profile.provider),
        model=profile.model,
        endpoint_profile=profile.endpoint_profile,
        temperature=temperature,
    )


def build_cell_config(
    manifest: BilingualStudyManifest,
    cell: StudyCell,
    *,
    max_role_attempts: int = 1,
    max_estimated_cost_usd: float = 1.0,
) -> ExperimentConfig:
    attacker = _model(manifest.attacker, temperature=0.7)
    return ExperimentConfig(
        task=TaskConfig(benchmark_item_id=cell.item_stable_id),
        agent=AgentConfig(max_wall_time_seconds=900),
        models=ModelsConfig(
            researcher=attacker,
            attacker=attacker,
            target=_model(manifest.target, temperature=0.0),
            judge=_model(manifest.judge, temperature=0.0),
        ),
        tools=ToolsConfig(enabled=("finish_experiment",)),
        budgets=BudgetsConfig(
            max_total_llm_calls=cell.max_rounds * manifest.calls_per_round * max_role_attempts,
            max_input_tokens=200_000,
            max_output_tokens=40_000,
            max_estimated_cost_usd=max_estimated_cost_usd,
        ),
        reporting=ReportingConfig(telegram=False, include_private_trace_link=False),
    )
