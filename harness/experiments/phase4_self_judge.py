from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select

from harness.agents.native.adversarial import (
    AdversarialRunner,
    ExperimentLanguage,
    JudgeDecision,
)
from harness.agents.native.gateway import TracedModelGateway
from harness.agents.native.modal_provider import QWEN36_27B_FP8_CANDIDATE
from harness.experiments.model_roster import PHASE2_MODEL_ROSTER_V1
from harness.persistence.models import (
    AdversarialRound,
    BenchmarkItem,
    CallStatus,
    Evaluation,
    LLMCall,
    Phase4InstrumentItem,
    Phase4InstrumentItemStatus,
    Run,
    RunStatus,
    Step,
)
from harness.persistence.repository import TraceRepository

TARGET_PROFILE = next(
    profile for profile in PHASE2_MODEL_ROSTER_V1.matrix_profiles if profile.key == "qwen36-27b-fp8"
)
PARSER_VERSION = "phase4-target-self-judge-v1"
PROMPT_BUILDER_VERSION = "native-adversarial-judge-messages-v1"


class Phase4SelfJudgeProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: int = Field(default=1, ge=1, le=1)
    purpose: str = Field(default="target_self_judge", pattern=r"^target_self_judge$")
    parser_version: str = Field(default=PARSER_VERSION, pattern=r"^phase4-target-self-judge-v1$")
    prompt_builder_version: str = Field(
        default=PROMPT_BUILDER_VERSION,
        pattern=r"^native-adversarial-judge-messages-v1$",
    )
    target_endpoint_profile: str = Field(
        default=TARGET_PROFILE.endpoint_profile or "", min_length=1
    )
    target_revision: str = Field(default=TARGET_PROFILE.revision, pattern=r"^[0-9a-f]{40}$")
    target_model: str = Field(default=QWEN36_27B_FP8_CANDIDATE.served_model, min_length=1)
    temperature: float = Field(default=0, ge=0, le=0)
    response_count: int = Field(default=21, ge=21, le=21)

    @model_validator(mode="after")
    def require_frozen_target(self) -> Phase4SelfJudgeProtocol:
        if (
            self.target_endpoint_profile != TARGET_PROFILE.endpoint_profile
            or self.target_revision != TARGET_PROFILE.revision
            or self.target_model != QWEN36_27B_FP8_CANDIDATE.served_model
        ):
            raise ValueError("Phase 4 protocol target identity differs from frozen Qwen")
        return self

    @property
    def identity_sha256(self) -> str:
        return _json_sha256(self.model_dump(mode="json"))


def input_bundle_sha256(
    *,
    source_round_id: str,
    source_evaluation_id: str,
    source_evaluation_sha256: str,
    language: str,
    benchmark_prompt: str,
    context_text: str | None,
    rubric_text: str,
    target_response: str,
) -> str:
    messages_sha256 = _json_sha256(
        target_self_judge_messages(
            language=ExperimentLanguage(language),
            benchmark_prompt=benchmark_prompt,
            rubric_text=rubric_text,
            target_response=target_response,
        )
    )
    return _json_sha256(
        {
            "benchmark_prompt": benchmark_prompt,
            "context_text": context_text,
            "language": language,
            "prompt_builder_version": PROMPT_BUILDER_VERSION,
            "request_messages_sha256": messages_sha256,
            "rubric_text": rubric_text,
            "source_round_id": source_round_id,
            "source_evaluation_id": source_evaluation_id,
            "source_evaluation_sha256": source_evaluation_sha256,
            "target_response": target_response,
        }
    )


def source_evaluation_sha256(evaluation: Evaluation) -> str:
    return _json_sha256(
        {
            "confidence": evaluation.confidence,
            "judge_endpoint_profile": evaluation.judge_endpoint_profile,
            "judge_model": evaluation.judge_model,
            "judge_provider": evaluation.judge_provider,
            "label": evaluation.label,
            "rationale": evaluation.rationale,
            "rubric_version": evaluation.rubric_version,
        }
    )


def target_self_judge_messages(
    *,
    language: ExperimentLanguage,
    benchmark_prompt: str,
    rubric_text: str,
    target_response: str,
) -> list[dict[str, Any]]:
    return AdversarialRunner._judge_messages(
        benchmark_prompt=benchmark_prompt,
        target_response=target_response,
        rubric=rubric_text,
        language=language,
    )


def benchmark_goal(item: BenchmarkItem) -> str:
    if item.context_text is None:
        return item.prompt_text
    return json.dumps(
        {"context": item.context_text, "prompt": item.prompt_text},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


async def execute_target_self_judge(
    *,
    repository: TraceRepository,
    item: Phase4InstrumentItem,
    run: Run,
    source_round: AdversarialRound,
    source_evaluation: Evaluation,
    benchmark_item: BenchmarkItem,
    gateway: TracedModelGateway,
    protocol: Phase4SelfJudgeProtocol,
    verified_messages: list[dict[str, Any]] | None = None,
) -> JudgeDecision:

    if (
        item.run_id != run.id
        or item.source_round_id != source_round.id
        or item.source_evaluation_id != source_evaluation.id
        or source_evaluation.round_id != source_round.id
    ):
        raise ValueError("Phase 4 run/item/source-round identity mismatch")
    if benchmark_item.language != item.source_language:
        raise ValueError("Phase 4 source language mismatch")
    reconstructed_messages = target_self_judge_messages(
        language=ExperimentLanguage(item.source_language),
        benchmark_prompt=benchmark_goal(benchmark_item),
        rubric_text=benchmark_item.rubric_text,
        target_response=source_round.target_response,
    )
    if verified_messages is not None and verified_messages != reconstructed_messages:
        raise ValueError("Phase 4 verified messages differ from the frozen prompt")
    response_sha256 = hashlib.sha256(source_round.target_response.encode()).hexdigest()
    evaluation_sha256 = source_evaluation_sha256(source_evaluation)
    bundle_sha256 = input_bundle_sha256(
        source_round_id=str(source_round.id),
        source_evaluation_id=str(source_evaluation.id),
        source_evaluation_sha256=evaluation_sha256,
        language=item.source_language,
        benchmark_prompt=benchmark_goal(benchmark_item),
        context_text=benchmark_item.context_text,
        rubric_text=benchmark_item.rubric_text,
        target_response=source_round.target_response,
    )
    if (
        response_sha256 != item.source_response_sha256
        or evaluation_sha256 != item.source_evaluation_sha256
        or bundle_sha256 != item.input_bundle_sha256
    ):
        raise ValueError("Phase 4 frozen input hash mismatch")
    if (
        gateway.client.provider != "modal"
        or gateway.client.endpoint_profile != protocol.target_endpoint_profile
        or protocol.target_endpoint_profile != TARGET_PROFILE.endpoint_profile
    ):
        raise ValueError("Phase 4 evaluator must use the frozen Qwen target profile")
    await repository.start_run(run)
    step = await repository.create_step(
        run_id=run.id,
        sequence_number=1,
        step_type="instrument_request",
        purpose="target_self_judge",
    )
    await repository.session.commit()
    request_cost = 0.0
    try:
        response = await gateway.request(
            run_id=run.id,
            step_id=step.id,
            purpose="target_self_judge",
            messages=verified_messages or reconstructed_messages,
            tools=[],
            parameters={
                "experiment_language": item.source_language,
                "seed": 17,
                "temperature": protocol.temperature,
            },
        )
        request_cost = response.estimated_cost_usd
        decision = JudgeDecision.model_validate_json(response.content)
    except Exception as exc:
        await repository.finish_step(step, status=CallStatus.FAILED)
        await repository.finish_run(
            run,
            status=RunStatus.FAILED,
            stop_reason="target_self_judge_failed",
            estimated_cost_usd=request_cost,
            error_summary=type(exc).__name__,
        )
        await repository.mark_phase4_instrument_item_indeterminate(
            item=item,
            stop_reason="target_self_judge_requires_reconciliation",
        )
        await repository.session.commit()
        raise
    await repository.finish_step(step)
    await repository.finish_run(
        run,
        status=RunStatus.COMPLETED,
        stop_reason="target_self_judge_completed",
        estimated_cost_usd=response.estimated_cost_usd,
    )
    llm_call = await repository.session.scalar(select(LLMCall).where(LLMCall.run_id == run.id))
    if llm_call is None:
        raise RuntimeError("persisted Phase 4 self-judge call is unavailable")
    await repository.finish_phase4_instrument_item(
        item=item,
        run=run,
        label=decision.label,
        rationale=decision.rationale,
        confidence=decision.confidence,
        parser_version=protocol.parser_version,
        llm_call=llm_call,
    )
    await repository.session.commit()
    return decision


async def reconcile_succeeded_target_self_judge(
    *,
    repository: TraceRepository,
    item: Phase4InstrumentItem,
    run: Run,
    protocol: Phase4SelfJudgeProtocol,
    verified_messages: list[dict[str, Any]],
) -> JudgeDecision:

    calls = tuple(
        (await repository.session.scalars(select(LLMCall).where(LLMCall.run_id == run.id))).all()
    )
    if len(calls) != 1:
        raise RuntimeError("Phase 4 reconciliation requires exactly one persisted call")
    call = calls[0]
    step = await repository.session.get(Step, call.step_id)
    if (
        item.status != Phase4InstrumentItemStatus.CLAIMED
        or item.run_id != run.id
        or step is None
        or step.run_id != run.id
        or step.purpose != "target_self_judge"
        or call.purpose != "target_self_judge"
        or call.status != CallStatus.SUCCEEDED
        or call.response_json is None
        or call.provider != "modal"
        or call.model != protocol.target_model
        or call.endpoint_profile != protocol.target_endpoint_profile
        or call.request_messages_json != verified_messages
        or call.request_parameters_json
        != {
            "experiment_language": item.source_language,
            "seed": 17,
            "temperature": protocol.temperature,
        }
    ):
        raise RuntimeError("Phase 4 persisted self-judge call is not reconcilable")
    response_content = call.response_json.get("content")
    if not isinstance(response_content, str):
        raise RuntimeError("Phase 4 persisted response content is unavailable")
    decision = JudgeDecision.model_validate_json(response_content)
    await repository.finish_step(step)
    if run.status != RunStatus.COMPLETED:
        await repository.finish_run(
            run,
            status=RunStatus.COMPLETED,
            stop_reason="target_self_judge_reconciled",
            estimated_cost_usd=call.estimated_cost_usd,
        )
    await repository.finish_phase4_instrument_item(
        item=item,
        run=run,
        label=decision.label,
        rationale=decision.rationale,
        confidence=decision.confidence,
        parser_version=protocol.parser_version,
        llm_call=call,
    )
    await repository.session.commit()
    return decision


def _json_sha256(value: Any) -> str:
    canonical = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()
