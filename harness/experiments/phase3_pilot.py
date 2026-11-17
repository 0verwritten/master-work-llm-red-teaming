from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from harness.agents.native.adversarial import (
    AdversarialResult,
    AdversarialRunner,
    AdversarialSpec,
    AttackMethod,
    ExperimentLanguage,
    RoleGateways,
)
from harness.agents.native.gateway import TracedModelGateway
from harness.agents.native.types import ModelClient
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
from harness.experiments.adversarial_matrix import FrozenModelProfile, MatrixItem
from harness.experiments.bilingual_pilot import ReviewedPair, build_bilingual_pilot_plan
from harness.experiments.freeze_bilingual_pilot import (
    BENCHMARK_SHA256,
    CROSS_ATTACKER_KEY,
    RUBRIC_VERSION,
    RUN_SEED,
    SELECTION_SEED,
    STAGE_PAIR_ID,
    TARGET_KEY,
)
from harness.experiments.model_roster import PHASE2_MODEL_ROSTER_V1
from harness.language import LanguageAdherence, detect_language_adherence
from harness.persistence.models import (
    AdversarialRound,
    AdversarialSession,
    BenchmarkItem,
    BenchmarkVersion,
    Experiment,
    Run,
)
from harness.persistence.repository import TraceRepository

SHA256_PATTERN = r"^[0-9a-f]{64}$"
MANIFEST_SCHEMA_VERSION = 2
STAGE_NAME = "paired_four_run_preflight"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ProfileRecord(StrictModel):
    endpoint_profile: str = Field(min_length=1)
    key: str = Field(min_length=1)
    model: None
    provider: Literal["modal"]
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")

    def frozen_profile(self) -> FrozenModelProfile:
        return FrozenModelProfile(**self.model_dump())


class ItemRecord(StrictModel):
    category: str = Field(min_length=1)
    content_sha256: str = Field(pattern=SHA256_PATTERN)
    language: Literal["en", "uk"]
    pair_id: str = Field(min_length=1)
    reviewer_status: Literal["hash_verified", "ai_reviewed_exploratory"]
    stable_id: str = Field(min_length=1)

    def matrix_item(self) -> MatrixItem:
        return MatrixItem(**self.model_dump())


class CellRecord(StrictModel):
    attacker: ProfileRecord
    cell_id: str = Field(pattern=SHA256_PATTERN)
    item: ItemRecord
    judge: ProfileRecord
    max_rounds: Literal[3]
    max_role_attempts: Literal[1]
    method: Literal["pair"]
    rubric_version: Literal["pilot-v1"]
    review_basis: Literal["ai_reviewed_exploratory"]
    seed: Literal[17]
    target: ProfileRecord


class MatrixRecord(StrictModel):
    cells: tuple[CellRecord, ...] = Field(min_length=40, max_length=40)
    max_llm_calls: Literal[360]
    max_target_queries: Literal[120]
    plan_sha256: str = Field(pattern=SHA256_PATTERN)


class PairRecord(StrictModel):
    category: str = Field(min_length=1)
    english: ItemRecord
    expected_behavior: Literal["refuse_or_safe_redirect", "comply_safely"]
    pair_id: str = Field(min_length=1)
    ukrainian: ItemRecord

    def reviewed_pair(self) -> ReviewedPair:
        return ReviewedPair(
            pair_id=self.pair_id,
            english=self.english.matrix_item(),
            ukrainian=self.ukrainian.matrix_item(),
            expected_behavior=self.expected_behavior,
        )


class PilotRecord(StrictModel):
    matrix: MatrixRecord
    plan_sha256: str = Field(pattern=SHA256_PATTERN)
    selected_pairs: tuple[PairRecord, ...] = Field(min_length=10, max_length=10)
    selection_seed: Literal[20260728]


class BenchmarkBinding(StrictModel):
    name: Literal["thesis-core"]
    sha256: str = Field(pattern=SHA256_PATTERN)
    version: Literal["phase1-v1-draft"]


class CostEnvelope(StrictModel):
    active_cost_usd_per_second: Literal["0.00150"]
    estimated_incremental_usd: Literal["18.30000"]
    max_llm_calls: Literal[360]
    noncompute_allowance_usd: Literal["0.75"]
    observed_window_use_usd: Literal["5.05167558"]
    projected_reserve_usd: Literal["6.64832442"]
    projected_window_use_usd: Literal["23.35167558"]
    provider_cap_usd: Literal["30.00"]
    request_seconds_per_call: Literal[30]
    required_reserve_usd: Literal["6.00"]
    startup_seconds_per_profile: Literal[300]
    unique_profile_count: Literal[3]


class DecodingProtocol(StrictModel):
    attacker_temperature: float
    judge_temperature: float
    target_temperature: float
    top_p: None

    @model_validator(mode="after")
    def require_frozen_decoding(self) -> DecodingProtocol:
        if (
            self.attacker_temperature,
            self.judge_temperature,
            self.target_temperature,
        ) != (0.7, 0.0, 0.0):
            raise ValueError("decoding parameters differ from the frozen protocol")
        return self


class PerRunBudgets(StrictModel):
    max_estimated_cost_usd: float
    max_input_tokens: Literal[200000]
    max_output_tokens: Literal[20000]
    max_total_llm_calls: Literal[9]
    max_wall_time_seconds: Literal[3600]

    @model_validator(mode="after")
    def require_frozen_cost_limit(self) -> PerRunBudgets:
        if self.max_estimated_cost_usd != 1.5:
            raise ValueError("cost limit differs from the frozen protocol")
        return self


class StageProtocol(StrictModel):
    cell_ids: tuple[str, ...] = Field(min_length=4, max_length=4)
    max_llm_calls: Literal[36]
    max_target_queries: Literal[12]
    name: Literal["paired_four_run_preflight"]
    pair_id: Literal["legacy-pair:adversarial_pairs:h1"]


class UkrainianBudget(StrictModel):
    core_english_item_count: Literal[357]
    incremental_ukrainian_item_count: Literal[34]
    ratio: Literal["0.09523809523809524"]
    shared_profile_startup_increment: Literal["0"]


class ExecutionProtocol(StrictModel):
    code_revision_policy: Literal[
        "record git HEAD and require execution-critical paths to match it"
    ]
    decoding: DecodingProtocol
    per_run_budgets: PerRunBudgets
    stage: StageProtocol
    ukrainian_incremental_budget: UkrainianBudget


class Phase3PilotManifest(StrictModel):
    manifest_schema_version: Literal[2]
    frozen_on: Literal["2026-08-14"]
    benchmark: BenchmarkBinding
    cost_envelope: CostEnvelope
    cross_attacker_key: Literal["gemma4-31b-it"]
    execution_protocol: ExecutionProtocol
    pilot: PilotRecord
    protocol: Literal["adr-0018-ai-reviewed-exploratory"]
    roster_sha256: str = Field(pattern=SHA256_PATTERN)
    target_key: Literal["qwen36-27b-fp8"]

    @model_validator(mode="after")
    def verify_derived_bindings(self) -> Phase3PilotManifest:
        if self.benchmark.sha256 != BENCHMARK_SHA256:
            raise ValueError("manifest benchmark hash does not match the frozen benchmark")
        if self.roster_sha256 != PHASE2_MODEL_ROSTER_V1.roster_sha256:
            raise ValueError("manifest roster hash does not match the frozen roster")
        profiles = {profile.key: profile for profile in PHASE2_MODEL_ROSTER_V1.matrix_profiles}
        plan = build_bilingual_pilot_plan(
            pairs=tuple(pair.reviewed_pair() for pair in self.pilot.selected_pairs),
            target=profiles[TARGET_KEY],
            cross_attacker=profiles[CROSS_ATTACKER_KEY],
            judge=PHASE2_MODEL_ROSTER_V1.judge_profile,
            selection_seed=SELECTION_SEED,
            run_seed=RUN_SEED,
            sample_size=10,
            max_rounds=3,
            max_role_attempts=1,
            rubric_version=RUBRIC_VERSION,
            review_basis="ai_reviewed_exploratory",
        )
        if plan.manifest_json() != self.pilot.model_dump(mode="json"):
            raise ValueError("pilot plan does not recompute from its frozen inputs")
        stage_cells = tuple(
            sorted(cell.cell_id for cell in plan.matrix.cells if cell.item.pair_id == STAGE_PAIR_ID)
        )
        if stage_cells != tuple(self.execution_protocol.stage.cell_ids):
            raise ValueError("stage cell IDs do not match the frozen staging pair")
        selected_stage = [
            cell for cell in self.pilot.matrix.cells if cell.cell_id in set(stage_cells)
        ]
        conditions = {
            (cell.item.language, cell.attacker.key, cell.target.key) for cell in selected_stage
        }
        expected = {
            (language, attacker, TARGET_KEY)
            for language in ("en", "uk")
            for attacker in (TARGET_KEY, CROSS_ATTACKER_KEY)
        }
        if conditions != expected:
            raise ValueError("stage must contain both languages and both attacker conditions")
        return self

    @property
    def manifest_sha256(self) -> str:
        return _json_sha256(self.model_dump(mode="json"))

    @property
    def stage_cells(self) -> tuple[CellRecord, ...]:
        by_id = {cell.cell_id: cell for cell in self.pilot.matrix.cells}
        return tuple(by_id[cell_id] for cell_id in self.execution_protocol.stage.cell_ids)


@dataclass(frozen=True)
class HydratedPilotCell:
    manifest_cell: CellRecord
    execution_protocol: ExecutionProtocol
    prompt_text: str
    context_text: str | None
    rubric_text: str
    expected_behavior: str

    @property
    def benchmark_prompt(self) -> str:
        if self.context_text is None:
            return self.prompt_text
        return json.dumps(
            {"context": self.context_text, "prompt": self.prompt_text},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


@dataclass(frozen=True)
class CellExecutionResult:
    cell_id: str
    result: AdversarialResult
    attacker_language_adherence: tuple[LanguageAdherence, ...]
    target_language_adherence: tuple[LanguageAdherence, ...]


class PilotClientFactory(Protocol):
    def __call__(
        self,
        *,
        profile: FrozenModelProfile,
        role: Literal["attacker", "target", "judge"],
        cell: CellRecord,
    ) -> ModelClient: ...


def load_phase3_manifest(path: Path) -> Phase3PilotManifest:

    resolved = path.resolve()
    repository_root = Path(__file__).resolve().parents[2]
    if not resolved.is_relative_to(repository_root):
        raise ValueError("Phase 3 manifest must be inside the repository")
    return Phase3PilotManifest.model_validate_json(resolved.read_bytes(), strict=True)


async def hydrate_stage(
    manifest: Phase3PilotManifest,
    session: AsyncSession,
) -> tuple[HydratedPilotCell, ...]:

    return await _hydrate_cells(manifest, session, manifest.stage_cells)


async def hydrate_pilot(
    manifest: Phase3PilotManifest,
    session: AsyncSession,
) -> tuple[HydratedPilotCell, ...]:

    return await _hydrate_cells(manifest, session, manifest.pilot.matrix.cells)


async def _hydrate_cells(
    manifest: Phase3PilotManifest,
    session: AsyncSession,
    cells: tuple[CellRecord, ...],
) -> tuple[HydratedPilotCell, ...]:
    version = await session.scalar(
        select(BenchmarkVersion).where(
            BenchmarkVersion.name == manifest.benchmark.name,
            BenchmarkVersion.version == manifest.benchmark.version,
        )
    )
    if version is None or version.status != "frozen":
        raise RuntimeError("frozen pilot benchmark is unavailable")
    freeze = version.source_manifest_json.get("freeze", {})
    if freeze.get("benchmark_sha256") != manifest.benchmark.sha256:
        raise RuntimeError("database benchmark does not match the pilot manifest")
    stable_ids = {cell.item.stable_id for cell in cells}
    items = tuple(
        (
            await session.scalars(
                select(BenchmarkItem).where(
                    BenchmarkItem.benchmark_version_id == version.id,
                    BenchmarkItem.stable_id.in_(stable_ids),
                )
            )
        ).all()
    )
    by_id = {item.stable_id: item for item in items}
    if set(by_id) != stable_ids:
        raise RuntimeError("one or more staged benchmark items are unavailable")

    hydrated: list[HydratedPilotCell] = []
    for cell in cells:
        item = by_id[cell.item.stable_id]
        expected_metadata = cell.item.model_dump(mode="json")
        actual_metadata = {
            "category": item.category,
            "content_sha256": item.content_sha256,
            "language": item.language,
            "pair_id": item.pair_id,
            "reviewer_status": item.reviewer_status,
            "stable_id": item.stable_id,
        }
        if actual_metadata != expected_metadata:
            raise RuntimeError(f"benchmark item binding mismatch: {item.stable_id}")
        if item.release_status != "restricted":
            raise RuntimeError("staged benchmark material must remain restricted")
        hydrated.append(
            HydratedPilotCell(
                manifest_cell=cell,
                execution_protocol=manifest.execution_protocol,
                prompt_text=item.prompt_text,
                context_text=item.context_text,
                rubric_text=item.rubric_text,
                expected_behavior=item.expected_behavior,
            )
        )
    return tuple(hydrated)


def build_cell_config(cell: HydratedPilotCell) -> ExperimentConfig:
    protocol = cell.manifest_cell
    decoding = cell.execution_protocol.decoding
    budgets = cell.execution_protocol.per_run_budgets
    attacker = _model_config(protocol.attacker, temperature=decoding.attacker_temperature)
    return ExperimentConfig(
        task=TaskConfig(benchmark_item_id=protocol.item.stable_id),
        agent=AgentConfig(max_wall_time_seconds=budgets.max_wall_time_seconds),
        models=ModelsConfig(
            researcher=attacker,
            attacker=attacker,
            target=_model_config(protocol.target, temperature=decoding.target_temperature),
            judge=_model_config(protocol.judge, temperature=decoding.judge_temperature),
        ),
        tools=ToolsConfig(enabled=("finish_experiment",)),
        budgets=BudgetsConfig(
            max_total_llm_calls=budgets.max_total_llm_calls,
            max_input_tokens=budgets.max_input_tokens,
            max_output_tokens=budgets.max_output_tokens,
            max_estimated_cost_usd=budgets.max_estimated_cost_usd,
        ),
        reporting=ReportingConfig(telegram=False, include_private_trace_link=False),
    )


async def execute_cell(
    *,
    manifest: Phase3PilotManifest,
    cell: HydratedPilotCell,
    repository: TraceRepository,
    experiment: Experiment,
    client_factory: PilotClientFactory,
    code_revision: str,
    run: Run | None = None,
) -> CellExecutionResult:

    manifest_cell_ids = {record.cell_id for record in manifest.pilot.matrix.cells}
    if cell.manifest_cell.cell_id not in manifest_cell_ids:
        raise ValueError("cell is outside the frozen pilot")
    if not re.fullmatch(r"[0-9a-f]{40}", code_revision):
        raise ValueError("code_revision must be a committed Git SHA-1")
    config = build_cell_config(cell)
    snapshot = await repository.get_or_create_config(
        name=f"phase3-{cell.manifest_cell.cell_id}",
        config=config,
    )
    if run is None:
        run = await repository.create_run(
            experiment_id=experiment.id,
            config_id=snapshot.id,
            benchmark_item_id=cell.manifest_cell.item.stable_id,
            random_seed=cell.manifest_cell.seed,
            code_version=code_revision,
        )
    elif any(
        (
            run.experiment_id != experiment.id,
            run.config_id != snapshot.id,
            run.benchmark_item_id != cell.manifest_cell.item.stable_id,
            run.random_seed != cell.manifest_cell.seed,
            run.code_version != code_revision,
        )
    ):
        raise ValueError("prepared run does not match the frozen cell")
    gateways = RoleGateways(
        attacker=TracedModelGateway(
            _make_client(client_factory, cell, "attacker", cell.manifest_cell.attacker),
            repository,
        ),
        target=TracedModelGateway(
            _make_client(client_factory, cell, "target", cell.manifest_cell.target),
            repository,
        ),
        judge=TracedModelGateway(
            _make_client(client_factory, cell, "judge", cell.manifest_cell.judge),
            repository,
        ),
    )
    language = ExperimentLanguage(cell.manifest_cell.item.language)
    result = await AdversarialRunner(repository=repository, gateways=gateways).run(
        run=run,
        config=config,
        benchmark_prompt=cell.benchmark_prompt,
        spec=AdversarialSpec(
            method=AttackMethod.PAIR,
            language=language,
            max_rounds=cell.manifest_cell.max_rounds,
            rubric_version=cell.manifest_cell.rubric_version,
            rubric_text=cell.rubric_text,
            rubric_language=language,
            benchmark_content_sha256=cell.manifest_cell.item.content_sha256,
            max_role_attempts=cell.manifest_cell.max_role_attempts,
        ),
    )
    rounds = tuple(
        (
            await repository.session.scalars(
                select(AdversarialRound)
                .join(AdversarialSession)
                .where(AdversarialSession.run_id == run.id)
                .order_by(AdversarialRound.round_number)
            )
        ).all()
    )
    return CellExecutionResult(
        cell_id=cell.manifest_cell.cell_id,
        result=result,
        attacker_language_adherence=tuple(
            detect_language_adherence(round_.attack_text, language) for round_ in rounds
        ),
        target_language_adherence=tuple(
            detect_language_adherence(round_.target_response, language) for round_ in rounds
        ),
    )


def _make_client(
    factory: PilotClientFactory,
    cell: HydratedPilotCell,
    role: Literal["attacker", "target", "judge"],
    record: ProfileRecord,
) -> ModelClient:
    profile = record.frozen_profile()
    client = factory(profile=profile, role=role, cell=cell.manifest_cell)
    if client.provider != profile.provider or client.endpoint_profile != profile.endpoint_profile:
        raise RuntimeError(f"{role} client identity does not match the frozen profile")
    return client


def _model_config(profile: ProfileRecord, *, temperature: float) -> ModelConfig:
    return ModelConfig(
        provider=Provider.MODAL,
        endpoint_profile=profile.endpoint_profile,
        temperature=temperature,
        top_p=None,
    )


def _json_sha256(value: Any) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()
