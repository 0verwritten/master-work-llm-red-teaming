from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Provider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MODAL = "modal"
    VAST = "vast.ai"
    MOCK = "mock"


class Strategy(StrEnum):
    REACTIVE = "reactive"
    PLANNER_FIRST = "planner_first"


class ContextMode(StrEnum):
    FULL_HISTORY = "full_history"
    SUMMARIZE = "summarize"
    SCRATCHPAD = "scratchpad"


class RunnerConfig(FrozenModel):
    type: Literal["native"] = "native"
    experiment_class: Literal["controlled"] = "controlled"


class TaskConfig(FrozenModel):
    benchmark_item_id: str = Field(min_length=1, max_length=200)


class AgentConfig(FrozenModel):
    strategy: Strategy = Strategy.REACTIVE
    system_prompt_variant: str = Field(default="baseline-v1", min_length=1)
    context_mode: ContextMode = ContextMode.FULL_HISTORY
    summarize_every_n_steps: int | None = Field(default=None, ge=1)
    max_steps: int = Field(default=20, ge=1, le=1_000)
    max_wall_time_seconds: int = Field(default=1_800, ge=1, le=86_400)
    max_code_executions: int = Field(default=20, ge=0, le=1_000)

    @model_validator(mode="after")
    def validate_summary_interval(self) -> AgentConfig:
        if self.context_mode == ContextMode.SUMMARIZE and self.summarize_every_n_steps is None:
            raise ValueError("summarize_every_n_steps is required for summarize context mode")
        if self.context_mode != ContextMode.SUMMARIZE and self.summarize_every_n_steps is not None:
            raise ValueError("summarize_every_n_steps is only valid for summarize context mode")
        return self


class ModelConfig(FrozenModel):
    provider: Provider
    model: str | None = None
    endpoint_profile: str | None = None
    temperature: float = Field(default=0, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)

    @model_validator(mode="after")
    def validate_locator(self) -> ModelConfig:
        if self.provider in {Provider.MODAL, Provider.VAST}:
            if not self.endpoint_profile or self.model:
                raise ValueError("profiled serving providers require endpoint_profile and no model")
        elif not self.model or self.endpoint_profile:
            raise ValueError("non-profiled providers require model and no endpoint_profile")
        return self


class ModelsConfig(FrozenModel):
    researcher: ModelConfig
    attacker: ModelConfig
    target: ModelConfig
    judge: ModelConfig
    verifier: ModelConfig | None = None


ALLOWED_TOOLS = frozenset(
    {
        "search_literature",
        "fetch_document",
        "note_finding",
        "query_model",
        "run_code",
        "get_run_artifact",
        "finish_experiment",
    }
)


class ToolsConfig(FrozenModel):
    enabled: tuple[str, ...]

    @model_validator(mode="after")
    def validate_tools(self) -> ToolsConfig:
        unknown = set(self.enabled) - ALLOWED_TOOLS
        if unknown:
            raise ValueError(f"unknown tools: {', '.join(sorted(unknown))}")
        if len(set(self.enabled)) != len(self.enabled):
            raise ValueError("enabled tools must be unique")
        if "finish_experiment" not in self.enabled:
            raise ValueError("finish_experiment is required")
        return self


class BudgetsConfig(FrozenModel):
    max_total_llm_calls: int = Field(default=300, ge=1)
    max_input_tokens: int = Field(default=1_000_000, ge=1)
    max_output_tokens: int = Field(default=200_000, ge=1)
    max_estimated_cost_usd: float = Field(default=10.0, ge=0)


class CodeExecutionConfig(FrozenModel):
    provider: Literal["modal"] = "modal"
    language: Literal["python"] = "python"
    network: Literal["disabled"] = "disabled"
    cpu: float = Field(default=2, gt=0, le=16)
    memory_mb: int = Field(default=4_096, ge=128, le=131_072)
    timeout_seconds: int = Field(default=600, ge=1, le=3_600)
    max_files: int = Field(default=20, ge=1, le=100)
    max_input_bytes: int = Field(default=1_000_000, ge=1)
    max_output_bytes: int = Field(default=10_000_000, ge=1)


class ReportingConfig(FrozenModel):
    telegram: bool = True
    include_private_trace_link: bool = True


class ExperimentConfig(FrozenModel):
    schema_version: Literal[1] = 1
    runner: RunnerConfig = RunnerConfig()
    task: TaskConfig
    agent: AgentConfig = AgentConfig()
    models: ModelsConfig
    tools: ToolsConfig
    budgets: BudgetsConfig = BudgetsConfig()
    code_execution: CodeExecutionConfig = CodeExecutionConfig()
    reporting: ReportingConfig = ReportingConfig()

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def stable_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()
