from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class ExperimentStatus(StrEnum):
    DRAFT = "draft"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BUDGET_STOPPED = "budget_stopped"
    TIMED_OUT = "timed_out"


class CallStatus(StrEnum):
    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Phase3ExecutionBatchStatus(StrEnum):
    MATERIALIZED = "materialized"
    RUNNING = "running"
    COMPLETED = "completed"
    ABORTED = "aborted"
    INDETERMINATE = "indeterminate"


class Phase3ExecutionCellStatus(StrEnum):
    QUEUED = "queued"
    CLAIMED = "claimed"
    TERMINAL = "terminal"
    INDETERMINATE = "indeterminate"


class Phase4InstrumentBatchStatus(StrEnum):
    MATERIALIZED = "materialized"
    RUNNING = "running"
    COMPLETED = "completed"
    ABORTED = "aborted"
    INDETERMINATE = "indeterminate"


class Phase4InstrumentItemStatus(StrEnum):
    QUEUED = "queued"
    CLAIMED = "claimed"
    TERMINAL = "terminal"
    INDETERMINATE = "indeterminate"


class UUIDPrimaryKey:
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)


class CreatedAt:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


class Experiment(UUIDPrimaryKey, CreatedAt, Base):
    __tablename__ = "experiments"

    name: Mapped[str] = mapped_column(String(300), nullable=False)
    research_question: Mapped[str] = mapped_column(Text, nullable=False)
    hypothesis: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default=ExperimentStatus.DRAFT, nullable=False)
    created_by: Mapped[str] = mapped_column(String(200), nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    runs: Mapped[list[Run]] = relationship(back_populates="experiment")


class ConfigSnapshot(UUIDPrimaryKey, CreatedAt, Base):
    __tablename__ = "configs"
    __table_args__ = (UniqueConstraint("config_hash", name="uq_configs_hash"),)

    name: Mapped[str] = mapped_column(String(300), nullable=False)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False)
    config_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)


class Run(UUIDPrimaryKey, CreatedAt, Base):
    __tablename__ = "runs"

    experiment_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False
    )
    benchmark_item_id: Mapped[str] = mapped_column(String(200), nullable=False)
    config_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("configs.id"), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default=RunStatus.QUEUED, nullable=False)
    stop_reason: Mapped[str | None] = mapped_column(String(100))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    random_seed: Mapped[int] = mapped_column(Integer, nullable=False)
    code_version: Mapped[str] = mapped_column(String(100), nullable=False)
    container_image_digest: Mapped[str | None] = mapped_column(String(200))
    error_summary: Mapped[str | None] = mapped_column(Text)

    experiment: Mapped[Experiment] = relationship(back_populates="runs")
    steps: Mapped[list[Step]] = relationship(back_populates="run")
    phase3_execution_cell: Mapped[Phase3ExecutionCell | None] = relationship(
        back_populates="run",
        uselist=False,
    )
    phase4_instrument_item: Mapped[Phase4InstrumentItem | None] = relationship(
        back_populates="run",
        uselist=False,
    )


class Phase3ExecutionBatch(UUIDPrimaryKey, CreatedAt, Base):
    __tablename__ = "phase3_execution_batches"
    __table_args__ = (
        UniqueConstraint("manifest_sha256", name="uq_phase3_execution_manifest"),
        CheckConstraint(
            "status IN ('materialized', 'running', 'completed', 'aborted', 'indeterminate')",
            name="ck_phase3_execution_batch_status",
        ),
    )

    manifest_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    plan_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    benchmark_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    roster_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    execution_envelope_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(
        String(32),
        default=Phase3ExecutionBatchStatus.MATERIALIZED,
        nullable=False,
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    stop_reason: Mapped[str | None] = mapped_column(String(100))

    cells: Mapped[list[Phase3ExecutionCell]] = relationship(
        back_populates="batch",
        cascade="all, delete-orphan",
    )


class Phase3ExecutionCell(UUIDPrimaryKey, CreatedAt, Base):
    __tablename__ = "phase3_execution_cells"
    __table_args__ = (
        UniqueConstraint("batch_id", "cell_id", name="uq_phase3_execution_cell"),
        UniqueConstraint("batch_id", "stage", "ordinal", name="uq_phase3_execution_stage_order"),
        UniqueConstraint("run_id", name="uq_phase3_execution_cell_run"),
        CheckConstraint("ordinal >= 0", name="ck_phase3_execution_cell_ordinal"),
        CheckConstraint(
            "status IN ('queued', 'claimed', 'terminal', 'indeterminate')",
            name="ck_phase3_execution_cell_status",
        ),
        Index("ix_phase3_execution_claim", "batch_id", "stage", "status", "ordinal"),
    )

    batch_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("phase3_execution_batches.id", ondelete="CASCADE"),
        nullable=False,
    )
    cell_id: Mapped[str] = mapped_column(String(64), nullable=False)
    stage: Mapped[str] = mapped_column(String(50), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    cell_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    cell_spec_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    run_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("runs.id"))
    status: Mapped[str] = mapped_column(
        String(32),
        default=Phase3ExecutionCellStatus.QUEUED,
        nullable=False,
    )
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    terminal_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    stop_reason: Mapped[str | None] = mapped_column(String(100))

    batch: Mapped[Phase3ExecutionBatch] = relationship(back_populates="cells")
    run: Mapped[Run | None] = relationship(back_populates="phase3_execution_cell")


class Phase4InstrumentBatch(UUIDPrimaryKey, CreatedAt, Base):
    __tablename__ = "phase4_instrument_batches"
    __table_args__ = (
        UniqueConstraint("manifest_sha256", name="uq_phase4_instrument_manifest"),
        CheckConstraint(
            "status IN ('materialized', 'running', 'completed', 'aborted', 'indeterminate')",
            name="ck_phase4_instrument_batch_status",
        ),
    )

    manifest_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    source_phase3_manifest_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    response_set_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    target_endpoint_profile: Mapped[str] = mapped_column(String(300), nullable=False)
    protocol_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), default=Phase4InstrumentBatchStatus.MATERIALIZED, nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    stop_reason: Mapped[str | None] = mapped_column(String(100))

    items: Mapped[list[Phase4InstrumentItem]] = relationship(
        back_populates="batch", cascade="all, delete-orphan"
    )


class Phase4InstrumentItem(UUIDPrimaryKey, CreatedAt, Base):
    __tablename__ = "phase4_instrument_items"
    __table_args__ = (
        UniqueConstraint("batch_id", "source_round_id", name="uq_phase4_instrument_round"),
        UniqueConstraint("batch_id", "ordinal", name="uq_phase4_instrument_ordinal"),
        UniqueConstraint("run_id", name="uq_phase4_instrument_run"),
        CheckConstraint("ordinal >= 0", name="ck_phase4_instrument_ordinal"),
        CheckConstraint("source_language IN ('en', 'uk')", name="ck_phase4_instrument_language"),
        CheckConstraint(
            "status IN ('queued', 'claimed', 'terminal', 'indeterminate')",
            name="ck_phase4_instrument_item_status",
        ),
        Index("ix_phase4_instrument_claim", "batch_id", "status", "ordinal"),
    )

    batch_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("phase4_instrument_batches.id", ondelete="CASCADE"), nullable=False
    )
    source_round_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("adversarial_rounds.id"), nullable=False
    )
    source_evaluation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("evaluations.id"), nullable=False
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    source_language: Mapped[str] = mapped_column(String(20), nullable=False)
    source_response_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    source_evaluation_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    input_bundle_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    run_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("runs.id"))
    status: Mapped[str] = mapped_column(
        String(32), default=Phase4InstrumentItemStatus.QUEUED, nullable=False
    )
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    terminal_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    stop_reason: Mapped[str | None] = mapped_column(String(100))

    batch: Mapped[Phase4InstrumentBatch] = relationship(back_populates="items")
    run: Mapped[Run | None] = relationship(back_populates="phase4_instrument_item")
    evaluation: Mapped[Phase4SelfJudgeEvaluation | None] = relationship(
        back_populates="item", uselist=False, cascade="all, delete-orphan"
    )


class Phase4SelfJudgeEvaluation(UUIDPrimaryKey, CreatedAt, Base):
    __tablename__ = "phase4_self_judge_evaluations"
    __table_args__ = (
        UniqueConstraint("item_id", name="uq_phase4_self_judge_item"),
        CheckConstraint(
            "label IN ('refused', 'partial', 'complied')",
            name="ck_phase4_self_judge_label",
        ),
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_phase4_self_judge_confidence",
        ),
    )

    item_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("phase4_instrument_items.id", ondelete="CASCADE"), nullable=False
    )
    llm_call_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("llm_calls.id"), unique=True, nullable=False
    )
    label: Mapped[str] = mapped_column(String(32), nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    parser_version: Mapped[str] = mapped_column(String(100), nullable=False)

    item: Mapped[Phase4InstrumentItem] = relationship(back_populates="evaluation")


class Step(UUIDPrimaryKey, Base):
    __tablename__ = "steps"
    __table_args__ = (UniqueConstraint("run_id", "sequence_number", name="uq_step_sequence"),)

    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), nullable=False
    )
    sequence_number: Mapped[int] = mapped_column(Integer, nullable=False)
    step_type: Mapped[str] = mapped_column(String(100), nullable=False)
    purpose: Mapped[str] = mapped_column(String(300), nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(32), default=CallStatus.STARTED, nullable=False)

    run: Mapped[Run] = relationship(back_populates="steps")


class LLMCall(UUIDPrimaryKey, Base):
    __tablename__ = "llm_calls"

    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), nullable=False
    )
    step_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("steps.id", ondelete="CASCADE"), nullable=False
    )
    purpose: Mapped[str] = mapped_column(String(300), nullable=False)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    model: Mapped[str] = mapped_column(String(200), nullable=False)
    endpoint_profile: Mapped[str | None] = mapped_column(String(300))
    request_messages_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    request_tools_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    request_parameters_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    response_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    cached_input_tokens: Mapped[int | None] = mapped_column(Integer)
    reasoning_tokens: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(32), default=CallStatus.STARTED, nullable=False)
    error_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)


class ToolCall(UUIDPrimaryKey, Base):
    __tablename__ = "tool_calls"

    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), nullable=False
    )
    step_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("steps.id", ondelete="CASCADE"), nullable=False
    )
    tool_name: Mapped[str] = mapped_column(String(100), nullable=False)
    arguments_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    result_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(32), default=CallStatus.STARTED, nullable=False)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    error_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)


class Artifact(UUIDPrimaryKey, CreatedAt, Base):
    __tablename__ = "artifacts"

    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[str] = mapped_column(String(100), nullable=False)
    storage_uri: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(200), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)


class CodeExecution(UUIDPrimaryKey, Base):
    __tablename__ = "code_executions"

    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), nullable=False
    )
    step_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("steps.id", ondelete="CASCADE"), nullable=False
    )
    platform: Mapped[str] = mapped_column(String(50), nullable=False)
    image_digest: Mapped[str] = mapped_column(String(200), nullable=False)
    entrypoint: Mapped[str] = mapped_column(String(500), nullable=False)
    source_artifact_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("artifacts.id"))
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    exit_code: Mapped[int | None] = mapped_column(Integer)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    cpu_limit: Mapped[float] = mapped_column(Float, nullable=False)
    memory_limit_mb: Mapped[int] = mapped_column(Integer, nullable=False)
    timeout_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    stdout_artifact_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("artifacts.id"))
    stderr_artifact_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("artifacts.id"))
    policy_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    error_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)


class ExternalAgentSession(UUIDPrimaryKey, Base):
    __tablename__ = "external_agent_sessions"

    run_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("runs.id", ondelete="SET NULL"))
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    runtime: Mapped[str] = mapped_column(String(50), nullable=False)
    runtime_version: Mapped[str] = mapped_column(String(100), nullable=False)
    model_requested: Mapped[str | None] = mapped_column(String(200))
    invocation_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    observable_events_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    final_output: Mapped[str | None] = mapped_column(Text)
    usage_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    error_summary: Mapped[str | None] = mapped_column(Text)
    trace_completeness: Mapped[str] = mapped_column(
        String(50), default="opaque_runtime", nullable=False
    )


class ReportDelivery(UUIDPrimaryKey, CreatedAt, Base):
    __tablename__ = "report_deliveries"
    __table_args__ = (
        UniqueConstraint("run_id", "channel", "destination", name="uq_report_delivery"),
    )

    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), nullable=False
    )
    channel: Mapped[str] = mapped_column(String(50), nullable=False)
    destination: Mapped[str] = mapped_column(String(200), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    provider_message_id: Mapped[str | None] = mapped_column(String(200))
    attempted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_summary: Mapped[str | None] = mapped_column(Text)


class LegacyImportBatch(UUIDPrimaryKey, CreatedAt, Base):
    __tablename__ = "legacy_import_batches"
    __table_args__ = (
        UniqueConstraint("source_sha256", name="uq_legacy_import_source_sha256"),
        CheckConstraint("provenance = 'legacy'", name="ck_legacy_import_provenance"),
    )

    provenance: Mapped[str] = mapped_column(String(32), default="legacy", nullable=False)
    source_filename: Mapped[str] = mapped_column(String(300), nullable=False)
    source_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    source_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    expected_record_count: Mapped[int] = mapped_column(Integer, nullable=False)
    importer_version: Mapped[str] = mapped_column(String(32), nullable=False)

    records: Mapped[list[LegacyRecord]] = relationship(back_populates="batch")


class LegacyRecord(UUIDPrimaryKey, Base):
    __tablename__ = "legacy_records"
    __table_args__ = (
        UniqueConstraint("batch_id", "source_row_number", name="uq_legacy_record_source_row"),
    )

    batch_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("legacy_import_batches.id", ondelete="CASCADE"), nullable=False
    )
    source_row_number: Mapped[int] = mapped_column(Integer, nullable=False)
    record_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    benchmark_item_id: Mapped[str] = mapped_column(String(200), nullable=False)
    source_collection: Mapped[str] = mapped_column(String(100), nullable=False)
    category: Mapped[str] = mapped_column(String(100), nullable=False)
    claimed_model: Mapped[str] = mapped_column(String(200), nullable=False)
    repetition: Mapped[int] = mapped_column(Integer, nullable=False)
    temperature: Mapped[float] = mapped_column(Float, nullable=False)
    recorded_timestamp: Mapped[str] = mapped_column(String(40), nullable=False)
    text_refused: Mapped[bool] = mapped_column(Boolean, nullable=False)
    image_refused: Mapped[bool] = mapped_column(Boolean, nullable=False)
    text_judge_verdict: Mapped[str] = mapped_column(String(32), nullable=False)
    image_judge_verdict: Mapped[str] = mapped_column(String(32), nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

    batch: Mapped[LegacyImportBatch] = relationship(back_populates="records")


class BenchmarkVersion(UUIDPrimaryKey, CreatedAt, Base):
    __tablename__ = "benchmark_versions"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_benchmark_name_version"),)

    name: Mapped[str] = mapped_column(String(200), nullable=False)
    version: Mapped[str] = mapped_column(String(100), nullable=False)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    source_manifest_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    frozen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    items: Mapped[list[BenchmarkItem]] = relationship(back_populates="benchmark_version")


class BenchmarkItem(UUIDPrimaryKey, CreatedAt, Base):
    __tablename__ = "benchmark_items"
    __table_args__ = (
        UniqueConstraint("benchmark_version_id", "stable_id", name="uq_benchmark_item_stable_id"),
        Index("ix_benchmark_item_split", "benchmark_version_id", "split"),
        Index("ix_benchmark_item_pair", "benchmark_version_id", "pair_id"),
    )

    benchmark_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("benchmark_versions.id", ondelete="CASCADE"), nullable=False
    )
    stable_id: Mapped[str] = mapped_column(String(300), nullable=False)
    source_name: Mapped[str] = mapped_column(String(100), nullable=False)
    source_version: Mapped[str] = mapped_column(String(100), nullable=False)
    source_item_id: Mapped[str] = mapped_column(String(300), nullable=False)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    license_id: Mapped[str] = mapped_column(String(100), nullable=False)
    license_url: Mapped[str] = mapped_column(Text, nullable=False)
    provenance: Mapped[str] = mapped_column(String(50), nullable=False)
    split: Mapped[str] = mapped_column(String(50), nullable=False)
    language: Mapped[str] = mapped_column(String(20), nullable=False)
    modality: Mapped[str] = mapped_column(String(50), nullable=False)
    category: Mapped[str] = mapped_column(String(100), nullable=False)
    functional_category: Mapped[str] = mapped_column(String(100), nullable=False)
    prompt_text: Mapped[str] = mapped_column(Text, nullable=False)
    context_text: Mapped[str | None] = mapped_column(Text)
    rubric_text: Mapped[str] = mapped_column(Text, nullable=False)
    expected_behavior: Mapped[str] = mapped_column(String(100), nullable=False)
    pair_id: Mapped[str | None] = mapped_column(String(200))
    translation_provenance_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, default=dict, nullable=False
    )
    reviewer_status: Mapped[str] = mapped_column(String(50), nullable=False)
    release_status: Mapped[str] = mapped_column(String(50), nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    source_metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

    benchmark_version: Mapped[BenchmarkVersion] = relationship(back_populates="items")


class ResearchInput(UUIDPrimaryKey, CreatedAt, Base):
    __tablename__ = "research_inputs"
    __table_args__ = (UniqueConstraint("stable_id", name="uq_research_input_stable_id"),)

    stable_id: Mapped[str] = mapped_column(String(300), nullable=False)
    role: Mapped[str] = mapped_column(String(100), nullable=False)
    source_name: Mapped[str] = mapped_column(String(100), nullable=False)
    source_version: Mapped[str] = mapped_column(String(100), nullable=False)
    source_item_id: Mapped[str] = mapped_column(String(300), nullable=False)
    source_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    language: Mapped[str] = mapped_column(String(20), nullable=False)
    category: Mapped[str] = mapped_column(String(100), nullable=False)
    subcategory: Mapped[str] = mapped_column(String(100), nullable=False)
    content_text: Mapped[str] = mapped_column(Text, nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    source_metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    release_status: Mapped[str] = mapped_column(String(50), nullable=False)


class AdversarialSession(UUIDPrimaryKey, CreatedAt, Base):
    __tablename__ = "adversarial_sessions"
    __table_args__ = (UniqueConstraint("run_id", name="uq_adversarial_session_run"),)

    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), nullable=False
    )
    method: Mapped[str] = mapped_column(String(50), nullable=False)
    language: Mapped[str] = mapped_column(String(20), nullable=False)
    max_rounds: Mapped[int] = mapped_column(Integer, nullable=False)
    rubric_version: Mapped[str] = mapped_column(String(100), nullable=False)
    rubric_text: Mapped[str] = mapped_column(Text, nullable=False)
    spec_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    rounds: Mapped[list[AdversarialRound]] = relationship(back_populates="session")


class AdversarialRound(UUIDPrimaryKey, CreatedAt, Base):
    __tablename__ = "adversarial_rounds"
    __table_args__ = (
        UniqueConstraint("session_id", "round_number", name="uq_adversarial_round_number"),
    )

    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("adversarial_sessions.id", ondelete="CASCADE"), nullable=False
    )
    round_number: Mapped[int] = mapped_column(Integer, nullable=False)
    attacker_step_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("steps.id"))
    target_step_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("steps.id"), nullable=False)
    judge_step_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("steps.id"), nullable=False)
    attack_text: Mapped[str] = mapped_column(Text, nullable=False)
    target_response: Mapped[str] = mapped_column(Text, nullable=False)
    candidate_metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, default=dict, nullable=False
    )

    session: Mapped[AdversarialSession] = relationship(back_populates="rounds")
    evaluation: Mapped[Evaluation | None] = relationship(back_populates="round")


class Evaluation(UUIDPrimaryKey, CreatedAt, Base):
    __tablename__ = "evaluations"
    __table_args__ = (UniqueConstraint("round_id", name="uq_evaluation_round"),)

    round_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("adversarial_rounds.id", ondelete="CASCADE"), nullable=False
    )
    rubric_version: Mapped[str] = mapped_column(String(100), nullable=False)
    judge_provider: Mapped[str] = mapped_column(String(50), nullable=False)
    judge_model: Mapped[str] = mapped_column(String(200), nullable=False)
    judge_endpoint_profile: Mapped[str | None] = mapped_column(String(300))
    label: Mapped[str] = mapped_column(String(32), nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    raw_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

    round: Mapped[AdversarialRound] = relationship(back_populates="evaluation")


class VulnerabilityClusteringRun(UUIDPrimaryKey, CreatedAt, Base):
    __tablename__ = "vulnerability_clustering_runs"
    __table_args__ = (
        UniqueConstraint(
            "experiment_id",
            "input_sha256",
            "spec_sha256",
            name="uq_vulnerability_clustering_input_spec",
        ),
    )

    experiment_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False
    )
    algorithm: Mapped[str] = mapped_column(String(100), nullable=False)
    algorithm_version: Mapped[str] = mapped_column(String(50), nullable=False)
    embedding_provider: Mapped[str] = mapped_column(String(50), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(200), nullable=False)
    embedding_profile: Mapped[str | None] = mapped_column(String(300))
    similarity_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    input_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    spec_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    spec_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    observation_count: Mapped[int] = mapped_column(Integer, nullable=False)
    cluster_count: Mapped[int] = mapped_column(Integer, nullable=False)
    finished_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class VulnerabilityCluster(UUIDPrimaryKey, CreatedAt, Base):
    __tablename__ = "vulnerability_clusters"
    __table_args__ = (
        UniqueConstraint(
            "clustering_run_id",
            "stable_key",
            name="uq_vulnerability_cluster_key",
        ),
    )

    clustering_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("vulnerability_clustering_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    stable_key: Mapped[str] = mapped_column(String(64), nullable=False)
    stratum: Mapped[str] = mapped_column(String(300), nullable=False)
    representative_round_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("adversarial_rounds.id"), nullable=False
    )
    member_count: Mapped[int] = mapped_column(Integer, nullable=False)


class VulnerabilityAssignment(UUIDPrimaryKey, CreatedAt, Base):
    __tablename__ = "vulnerability_assignments"
    __table_args__ = (
        UniqueConstraint(
            "clustering_run_id",
            "round_id",
            name="uq_vulnerability_assignment_round",
        ),
    )

    clustering_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("vulnerability_clustering_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    cluster_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("vulnerability_clusters.id", ondelete="CASCADE"), nullable=False
    )
    round_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("adversarial_rounds.id"), nullable=False)
    text_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    similarity_to_representative: Mapped[float] = mapped_column(Float, nullable=False)
    embedding_json: Mapped[list[float]] = mapped_column(JSON, nullable=False)
