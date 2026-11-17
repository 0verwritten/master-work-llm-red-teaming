from __future__ import annotations

import hashlib
import json
import uuid
from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select

from harness.experiments.phase3_pilot import Phase3PilotManifest
from harness.persistence.models import (
    AdversarialRound,
    AdversarialSession,
    CallStatus,
    Evaluation,
    LLMCall,
    Phase3ExecutionBatch,
    Phase3ExecutionCell,
    Phase3ExecutionCellStatus,
    Run,
    RunStatus,
    VulnerabilityAssignment,
    VulnerabilityClusteringRun,
)
from harness.persistence.repository import TraceRepository


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class Phase5PrivateRoundIdentity(FrozenModel):
    source_run_id: uuid.UUID
    source_round_id: uuid.UUID
    phase3_cell_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    round_number: int = Field(ge=1, le=3)
    language: Literal["en", "uk"]
    method: Literal["pair"]
    category: str = Field(min_length=1)
    attacker_key: str = Field(min_length=1)
    target_key: str = Field(min_length=1)
    attack_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_metadata_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Phase5InputSet(FrozenModel):
    schema_version: Literal[1] = 1
    source_phase3_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rounds: tuple[Phase5PrivateRoundIdentity, ...] = Field(min_length=21, max_length=21)


class Phase5AggregateManifest(FrozenModel):
    schema_version: Literal[1] = 1
    aggregate_only: Literal[True] = True
    raw_content_exported: Literal[False] = False
    source_phase3_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_count: Literal[7]
    round_count: Literal[21]
    round_numbers: dict[int, int]
    languages: dict[str, int]
    methods: dict[str, int]
    categories: dict[str, int]
    attackers: dict[str, int]
    language_attacker_cells: dict[str, int]
    unique_attack_hashes: int
    unique_response_hashes: int
    attack_duplicate_group_sizes: tuple[int, ...]
    response_duplicate_group_sizes: tuple[int, ...]
    persisted_clustering_runs: int
    persisted_clustering_assignments_for_input: int
    input_fully_clustered: bool
    rq_c_estimability: Literal["descriptive_trace_only"] = "descriptive_trace_only"


async def materialize_phase5_inputs(
    *,
    manifest: Phase3PilotManifest,
    repository: TraceRepository,
) -> tuple[Phase5InputSet, Phase5AggregateManifest]:

    batch = await repository.session.scalar(
        select(Phase3ExecutionBatch).where(
            Phase3ExecutionBatch.manifest_sha256 == manifest.manifest_sha256
        )
    )
    if (
        batch is None
        or str(batch.status) != "indeterminate"
        or batch.stop_reason != "pilot_abort_systematic_ukrainian_judge_language_switch"
    ):
        raise RuntimeError("Phase 5 source batch is not the frozen stopped Phase 3 batch")
    states = tuple(
        (
            await repository.session.scalars(
                select(Phase3ExecutionCell).where(Phase3ExecutionCell.batch_id == batch.id)
            )
        ).all()
    )
    if Counter(str(state.status) for state in states) != Counter(
        {"terminal": 7, "indeterminate": 33}
    ):
        raise RuntimeError("Phase 5 source cell disposition changed")
    terminal = [state for state in states if state.status == Phase3ExecutionCellStatus.TERMINAL]
    manifest_cells = {cell.cell_id: cell for cell in manifest.pilot.matrix.cells}
    entries: list[Phase5PrivateRoundIdentity] = []
    for state in terminal:
        if state.run_id is None or state.cell_id not in manifest_cells:
            raise RuntimeError("Phase 5 terminal source lacks frozen identity")
        cell = manifest_cells[state.cell_id]
        if state.cell_spec_json != cell.model_dump(mode="json"):
            raise RuntimeError("Phase 5 source cell specification changed")
        run = await repository.session.get(Run, state.run_id)
        source_session = await repository.session.scalar(
            select(AdversarialSession).where(AdversarialSession.run_id == state.run_id)
        )
        if (
            run is None
            or run.status != RunStatus.COMPLETED
            or source_session is None
            or source_session.method != "pair"
            or source_session.language != cell.item.language
        ):
            raise RuntimeError("Phase 5 source run/session identity changed")
        rounds = tuple(
            (
                await repository.session.scalars(
                    select(AdversarialRound)
                    .where(AdversarialRound.session_id == source_session.id)
                    .order_by(AdversarialRound.round_number)
                )
            ).all()
        )
        if [round_.round_number for round_ in rounds] != [1, 2, 3]:
            raise RuntimeError("Phase 5 requires three contiguous rounds per run")
        calls = tuple(
            (
                await repository.session.scalars(select(LLMCall).where(LLMCall.run_id == run.id))
            ).all()
        )
        calls_by_step = {call.step_id: call for call in calls}
        if len(calls) != 9 or any(call.status != CallStatus.SUCCEEDED for call in calls):
            raise RuntimeError("Phase 5 source run trace is incomplete")
        for round_ in rounds:
            role_steps = {
                "attacker": round_.attacker_step_id,
                "target": round_.target_step_id,
                "judge": round_.judge_step_id,
            }
            if any(
                step_id is None
                or step_id not in calls_by_step
                or calls_by_step[step_id].purpose != role
                for role, step_id in role_steps.items()
            ):
                raise RuntimeError("Phase 5 round/role trace binding changed")
            target_call = calls_by_step[round_.target_step_id]
            evaluation = await repository.session.scalar(
                select(Evaluation).where(Evaluation.round_id == round_.id)
            )
            if (
                evaluation is None
                or target_call.response_json is None
                or target_call.response_json.get("content") != round_.target_response
            ):
                raise RuntimeError("Phase 5 target response/evaluation trace changed")
            entries.append(
                Phase5PrivateRoundIdentity(
                    source_run_id=run.id,
                    source_round_id=round_.id,
                    phase3_cell_id=state.cell_id,
                    round_number=round_.round_number,
                    language=source_session.language,
                    method="pair",
                    category=cell.item.category,
                    attacker_key=cell.attacker.key,
                    target_key=cell.target.key,
                    attack_sha256=hashlib.sha256(round_.attack_text.encode()).hexdigest(),
                    response_sha256=hashlib.sha256(round_.target_response.encode()).hexdigest(),
                    candidate_metadata_sha256=_json_sha256(round_.candidate_metadata_json),
                    evaluation_sha256=_json_sha256(
                        {
                            "confidence": evaluation.confidence,
                            "judge_endpoint_profile": evaluation.judge_endpoint_profile,
                            "label": evaluation.label,
                            "rationale": evaluation.rationale,
                            "rubric_version": evaluation.rubric_version,
                        }
                    ),
                )
            )
    ordered = tuple(sorted(entries, key=lambda item: (str(item.source_run_id), item.round_number)))
    if len(ordered) != 21 or len({item.source_round_id for item in ordered}) != 21:
        raise RuntimeError("Phase 5 input set must contain 21 unique round identities")
    input_set_sha256 = _json_sha256([item.model_dump(mode="json") for item in ordered])
    private_set = Phase5InputSet(
        source_phase3_manifest_sha256=manifest.manifest_sha256,
        input_set_sha256=input_set_sha256,
        rounds=ordered,
    )
    cluster_runs = int(
        await repository.session.scalar(
            select(func.count()).select_from(VulnerabilityClusteringRun)
        )
        or 0
    )
    assigned = int(
        await repository.session.scalar(
            select(func.count())
            .select_from(VulnerabilityAssignment)
            .where(VulnerabilityAssignment.round_id.in_([item.source_round_id for item in ordered]))
        )
        or 0
    )
    aggregate = build_phase5_aggregate(
        private_set,
        persisted_clustering_runs=cluster_runs,
        persisted_clustering_assignments=assigned,
    )
    return private_set, aggregate


def build_phase5_aggregate(
    input_set: Phase5InputSet,
    *,
    persisted_clustering_runs: int,
    persisted_clustering_assignments: int,
) -> Phase5AggregateManifest:
    rounds = input_set.rounds
    attack_counts = Counter(item.attack_sha256 for item in rounds)
    response_counts = Counter(item.response_sha256 for item in rounds)
    return Phase5AggregateManifest(
        source_phase3_manifest_sha256=input_set.source_phase3_manifest_sha256,
        input_set_sha256=input_set.input_set_sha256,
        run_count=7,
        round_count=21,
        round_numbers=dict(sorted(Counter(item.round_number for item in rounds).items())),
        languages=dict(sorted(Counter(item.language for item in rounds).items())),
        methods=dict(sorted(Counter(item.method for item in rounds).items())),
        categories=dict(sorted(Counter(item.category for item in rounds).items())),
        attackers=dict(sorted(Counter(item.attacker_key for item in rounds).items())),
        language_attacker_cells=dict(
            sorted(Counter(f"{item.language}:{item.attacker_key}" for item in rounds).items())
        ),
        unique_attack_hashes=len(attack_counts),
        unique_response_hashes=len(response_counts),
        attack_duplicate_group_sizes=tuple(
            sorted(count for count in attack_counts.values() if count > 1)
        ),
        response_duplicate_group_sizes=tuple(
            sorted(count for count in response_counts.values() if count > 1)
        ),
        persisted_clustering_runs=persisted_clustering_runs,
        persisted_clustering_assignments_for_input=persisted_clustering_assignments,
        input_fully_clustered=persisted_clustering_assignments == len(rounds),
    )


def _json_sha256(value: Any) -> str:
    canonical = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()
