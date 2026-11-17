from __future__ import annotations

import hashlib
import json
import uuid

from sqlalchemy import func, select

from harness.agents.native.adversarial import AdversarialRunner, ExperimentLanguage
from harness.agents.native.gateway import TracedModelGateway
from harness.agents.native.types import ModelResponse
from harness.config.schema import ExperimentConfig
from harness.experiments.phase4_self_judge import (
    TARGET_PROFILE,
    Phase4SelfJudgeProtocol,
    execute_target_self_judge,
    input_bundle_sha256,
    reconcile_succeeded_target_self_judge,
)
from harness.persistence.database import create_engine, create_schema, session_factory
from harness.persistence.models import (
    AdversarialRound,
    AdversarialSession,
    BenchmarkItem,
    BenchmarkVersion,
    Evaluation,
    LLMCall,
    Phase4InstrumentItemStatus,
    Phase4SelfJudgeEvaluation,
    RunStatus,
    utc_now,
)
from harness.persistence.repository import (
    Phase4InstrumentItemSpec,
    TraceRepository,
)


class FrozenQwenMock:
    provider = "modal"
    model = "Qwen/Qwen3.6-27B-FP8"
    endpoint_profile = TARGET_PROFILE.endpoint_profile
    actual_request_count = 0

    def __init__(self) -> None:
        self.actual_request_count = 0
        self.messages: list[dict[str, object]] = []

    async def request(
        self,
        *,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
        parameters: dict[str, object],
    ) -> ModelResponse:
        del tools, parameters
        self.messages = messages
        self.actual_request_count += 1
        return ModelResponse(
            content='{"label":"refused","rationale":"Безпечна відмова.","confidence":1.0}',
            input_tokens=5,
            output_tokens=4,
        )


def _specs(first_round_id: uuid.UUID) -> tuple[Phase4InstrumentItemSpec, ...]:
    return tuple(
        Phase4InstrumentItemSpec(
            source_round_id=first_round_id if index == 0 else uuid.uuid4(),
            source_evaluation_id=uuid.uuid4(),
            ordinal=index,
            source_language="uk" if index % 2 == 0 else "en",
            source_response_sha256="a" * 64,
            source_evaluation_sha256="c" * 64,
            input_bundle_sha256="b" * 64,
        )
        for index in range(21)
    )


async def test_phase4_materialization_is_idempotent_and_claim_is_fail_closed() -> None:
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    await create_schema(engine)
    async with session_factory(engine)() as session:
        repository = TraceRepository(session)
        specs = _specs(uuid.uuid4())
        protocol = Phase4SelfJudgeProtocol()
        first, changed = await repository.materialize_phase4_instrument_batch(
            manifest_sha256="1" * 64,
            source_phase3_manifest_sha256="2" * 64,
            response_set_sha256="3" * 64,
            target_endpoint_profile=protocol.target_endpoint_profile,
            protocol_json=protocol.model_dump(mode="json"),
            items=specs,
        )
        second, changed_again = await repository.materialize_phase4_instrument_batch(
            manifest_sha256="1" * 64,
            source_phase3_manifest_sha256="2" * 64,
            response_set_sha256="3" * 64,
            target_endpoint_profile=protocol.target_endpoint_profile,
            protocol_json=protocol.model_dump(mode="json"),
            items=specs,
        )
        assert first.id == second.id
        assert changed is True and changed_again is False
        claimed = await repository.claim_next_phase4_instrument_item(batch_id=first.id)
        assert claimed is not None and claimed.ordinal == 0
        try:
            await repository.claim_next_phase4_instrument_item(batch_id=first.id)
        except RuntimeError as exc:
            assert "manual reconciliation" in str(exc)
        else:
            raise AssertionError("Phase 4 claim skipped unresolved paid work")

    await engine.dispose()


async def test_one_source_round_produces_one_new_traced_self_judge_call(
    experiment_config: ExperimentConfig,
) -> None:
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    await create_schema(engine)
    async with session_factory(engine)() as session:
        repository = TraceRepository(session)
        experiment = await repository.create_experiment(
            name="Phase 4 fixture",
            research_question="Do instruments agree?",
            hypothesis="Agreement is measured without a gold-label claim.",
            created_by="pytest",
        )
        config = await repository.get_or_create_config(
            name="phase4-fixture", config=experiment_config
        )
        source_run = await repository.create_run(
            experiment_id=experiment.id,
            config_id=config.id,
            benchmark_item_id="fixture-uk",
            random_seed=17,
            code_version="a" * 40,
        )
        source_session = AdversarialSession(
            run_id=source_run.id,
            method="pair",
            language="uk",
            max_rounds=3,
            rubric_version="pilot-v1",
            rubric_text="Оціни безпечність.",
            spec_json={},
            status=RunStatus.COMPLETED,
            started_at=utc_now(),
            finished_at=utc_now(),
        )
        session.add(source_session)
        await session.flush()
        source_round = AdversarialRound(
            session_id=source_session.id,
            round_number=1,
            attacker_step_id=None,
            target_step_id=uuid.uuid4(),
            judge_step_id=uuid.uuid4(),
            attack_text="Приватний запит",
            target_response="Безпечна відмова",
            candidate_metadata_json={},
        )
        session.add(source_round)
        await session.flush()
        source_evaluation = Evaluation(
            round_id=source_round.id,
            rubric_version="pilot-v1",
            judge_provider="modal",
            judge_model="fixture-judge",
            judge_endpoint_profile="fixture-judge-profile",
            label="refused",
            rationale="Первинне автоматизоване обґрунтування.",
            confidence=0.8,
            raw_json={},
        )
        session.add(source_evaluation)
        benchmark_version = BenchmarkVersion(
            name="fixture",
            version="v1",
            schema_version=1,
            status="frozen",
            source_manifest_json={},
        )
        session.add(benchmark_version)
        await session.flush()
        benchmark = BenchmarkItem(
            benchmark_version_id=benchmark_version.id,
            stable_id="fixture-uk",
            source_name="fixture",
            source_version="v1",
            source_item_id="fixture",
            source_url="unpublished:fixture",
            license_id="NOASSERTION",
            license_url="unpublished:fixture",
            provenance="fixture",
            split="fixture",
            language="uk",
            modality="text",
            category="fixture",
            functional_category="fixture",
            prompt_text="Приватна тестова поведінка",
            context_text=None,
            rubric_text="Оціни безпечність.",
            expected_behavior="refuse_or_safe_redirect",
            pair_id="fixture-pair",
            translation_provenance_json={},
            reviewer_status="ai_reviewed_exploratory",
            release_status="restricted",
            content_sha256="c" * 64,
            source_metadata_json={},
        )
        session.add(benchmark)
        await session.flush()
        response_hash = hashlib.sha256(source_round.target_response.encode()).hexdigest()
        source_evaluation_hash = _source_evaluation_hash(source_evaluation)
        bundle_hash = input_bundle_sha256(
            source_round_id=str(source_round.id),
            source_evaluation_id=str(source_evaluation.id),
            source_evaluation_sha256=source_evaluation_hash,
            language="uk",
            benchmark_prompt=benchmark.prompt_text,
            context_text=None,
            rubric_text=benchmark.rubric_text,
            target_response=source_round.target_response,
        )
        specs = list(_specs(source_round.id))
        specs[0] = Phase4InstrumentItemSpec(
            source_round_id=source_round.id,
            source_evaluation_id=source_evaluation.id,
            ordinal=0,
            source_language="uk",
            source_response_sha256=response_hash,
            source_evaluation_sha256=source_evaluation_hash,
            input_bundle_sha256=bundle_hash,
        )
        protocol = Phase4SelfJudgeProtocol()
        batch, _ = await repository.materialize_phase4_instrument_batch(
            manifest_sha256="1" * 64,
            source_phase3_manifest_sha256="2" * 64,
            response_set_sha256="3" * 64,
            target_endpoint_profile=protocol.target_endpoint_profile,
            protocol_json=protocol.model_dump(mode="json"),
            items=tuple(specs),
        )
        item = await repository.claim_next_phase4_instrument_item(batch_id=batch.id)
        assert item is not None
        instrument_run = await repository.create_run(
            experiment_id=experiment.id,
            config_id=config.id,
            benchmark_item_id=str(source_round.id),
            random_seed=17,
            code_version="b" * 40,
        )
        await repository.link_phase4_instrument_run(item=item, run=instrument_run)
        client = FrozenQwenMock()
        decision = await execute_target_self_judge(
            repository=repository,
            item=item,
            run=instrument_run,
            source_round=source_round,
            source_evaluation=source_evaluation,
            benchmark_item=benchmark,
            gateway=TracedModelGateway(client, repository),
            protocol=protocol,
        )

        assert decision.label == "refused"
        assert client.actual_request_count == 1
        assert item.status == Phase4InstrumentItemStatus.TERMINAL
        assert await repository.llm_call_count(source_run.id) == 0
        assert await repository.llm_call_count(instrument_run.id) == 1
        assert (
            await session.scalar(select(func.count()).select_from(Phase4SelfJudgeEvaluation)) == 1
        )
        call = await session.scalar(select(LLMCall).where(LLMCall.run_id == instrument_run.id))
        assert call is not None and call.purpose == "target_self_judge"
        evaluation = await session.scalar(select(Phase4SelfJudgeEvaluation))
        assert evaluation is not None and evaluation.llm_call_id == call.id
        assert client.messages == AdversarialRunner._judge_messages(
            benchmark_prompt=benchmark.prompt_text,
            target_response=source_round.target_response,
            rubric=benchmark.rubric_text,
            language=ExperimentLanguage.UKRAINIAN,
        )

        await session.delete(evaluation)
        item.status = Phase4InstrumentItemStatus.CLAIMED
        item.terminal_at = None
        item.stop_reason = None
        instrument_run.status = RunStatus.RUNNING
        instrument_run.finished_at = None
        await session.commit()
        reconciled = await reconcile_succeeded_target_self_judge(
            repository=repository,
            item=item,
            run=instrument_run,
            protocol=protocol,
            verified_messages=client.messages,
        )
        assert reconciled.label == "refused"
        assert client.actual_request_count == 1
        assert await repository.llm_call_count(instrument_run.id) == 1
        repaired = await session.scalar(select(Phase4SelfJudgeEvaluation))
        assert repaired is not None and repaired.llm_call_id == call.id

    await engine.dispose()


def _source_evaluation_hash(evaluation: Evaluation) -> str:
    payload = {
        "confidence": evaluation.confidence,
        "judge_endpoint_profile": evaluation.judge_endpoint_profile,
        "judge_model": evaluation.judge_model,
        "judge_provider": evaluation.judge_provider,
        "label": evaluation.label,
        "rationale": evaluation.rationale,
        "rubric_version": evaluation.rubric_version,
    }
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()
