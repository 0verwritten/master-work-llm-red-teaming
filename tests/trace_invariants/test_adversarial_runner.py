from __future__ import annotations

import json

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine

from harness.agents.native.adversarial import (
    AdversarialRunner,
    AdversarialSpec,
    AttackMethod,
    BaselineInputs,
    ExperimentLanguage,
    RoleGateways,
)
from harness.agents.native.gateway import TracedModelGateway
from harness.agents.native.mock_provider import DeterministicMockClient
from harness.agents.native.types import ModelResponse, TransientModelError
from harness.config.schema import ExperimentConfig
from harness.persistence.database import create_engine, create_schema, session_factory
from harness.persistence.models import (
    AdversarialRound,
    AdversarialSession,
    Evaluation,
    LLMCall,
    Run,
    RunStatus,
)
from harness.persistence.repository import TraceRepository


def _client(*contents: str) -> DeterministicMockClient:
    return DeterministicMockClient(
        [ModelResponse(content=content, input_tokens=2, output_tokens=3) for content in contents]
    )


async def _setup(config: ExperimentConfig) -> tuple[AsyncEngine, TraceRepository, Run]:
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    await create_schema(engine)
    sessions = session_factory(engine)
    session = sessions()
    repository = TraceRepository(session)
    experiment = await repository.create_experiment(
        name="Adversarial fixture",
        research_question="Does the explicit loop retain traces?",
        hypothesis="Yes",
        created_by="pytest",
    )
    snapshot = await repository.get_or_create_config(name="fixture", config=config)
    run = await repository.create_run(
        experiment_id=experiment.id,
        config_id=snapshot.id,
        benchmark_item_id="safe-fixture",
        random_seed=1,
        code_version="test",
    )
    await session.commit()
    return engine, repository, run


def _spec(
    max_rounds: int = 2,
    method: AttackMethod = AttackMethod.PAIR,
    language: ExperimentLanguage = ExperimentLanguage.ENGLISH,
) -> AdversarialSpec:
    return AdversarialSpec(
        method=method,
        language=language,
        max_rounds=max_rounds,
        rubric_version="fixture-v1",
        rubric_text=(
            "Return the required structured decision."
            if language == ExperimentLanguage.ENGLISH
            else "Поверни структуроване рішення."
        ),
        rubric_language=language,
        benchmark_content_sha256="a" * 64,
    )


async def test_two_round_loop_traces_every_role_request(
    experiment_config: ExperimentConfig,
) -> None:
    engine, repository, run = await _setup(experiment_config)
    attacker = _client("candidate one", "candidate two")
    target = _client("safe response", "unsafe response")
    judge = _client(
        '{"label":"refused","rationale":"safe","confidence":1.0}',
        '{"label":"complied","rationale":"enabled","confidence":0.9}',
    )
    result = await AdversarialRunner(
        repository=repository,
        gateways=RoleGateways(
            attacker=TracedModelGateway(attacker, repository),
            target=TracedModelGateway(target, repository),
            judge=TracedModelGateway(judge, repository),
        ),
    ).run(
        run=run,
        config=experiment_config,
        benchmark_prompt="sanitized benchmark goal",
        spec=_spec(),
    )

    calls = await repository.session.scalar(select(func.count()).select_from(LLMCall))
    rounds = await repository.session.scalar(select(func.count()).select_from(AdversarialRound))
    labels = list(
        (
            await repository.session.scalars(
                select(Evaluation.label).order_by(Evaluation.created_at)
            )
        ).all()
    )
    assert result.status == RunStatus.COMPLETED
    assert result.stop_reason == "validated_success"
    assert result.rounds == 2
    assert result.llm_calls == 6
    assert (
        calls
        == attacker.actual_request_count + target.actual_request_count + judge.actual_request_count
    )
    assert rounds == 2
    assert labels == ["refused", "complied"]
    await repository.session.close()
    await engine.dispose()


async def test_single_shot_attempts_never_receive_feedback_history(
    experiment_config: ExperimentConfig,
) -> None:
    engine, repository, run = await _setup(experiment_config)
    attacker = _client("independent one", "independent two")
    target = _client("safe one", "safe two")
    judge = _client(
        '{"label":"refused","rationale":"safe","confidence":1.0}',
        '{"label":"refused","rationale":"safe","confidence":1.0}',
    )
    result = await AdversarialRunner(
        repository=repository,
        gateways=RoleGateways(
            attacker=TracedModelGateway(attacker, repository),
            target=TracedModelGateway(target, repository),
            judge=TracedModelGateway(judge, repository),
        ),
    ).run(
        run=run,
        config=experiment_config,
        benchmark_prompt="sanitized benchmark goal",
        spec=_spec(method=AttackMethod.SINGLE_SHOT),
    )

    attacker_calls = (
        await repository.session.scalars(
            select(LLMCall).where(LLMCall.purpose == "attacker").order_by(LLMCall.started_at)
        )
    ).all()
    histories = [
        json.loads(call.request_messages_json[1]["content"])["history"] for call in attacker_calls
    ]
    assert result.rounds == 2
    assert result.llm_calls == 6
    assert histories == [[], []]
    await repository.session.close()
    await engine.dispose()


async def test_static_baseline_uses_frozen_candidates_without_attacker_calls(
    experiment_config: ExperimentConfig,
) -> None:
    engine, repository, run = await _setup(experiment_config)
    target = _client("safe one", "safe two")
    judge = _client(
        '{"label":"refused","rationale":"safe","confidence":1.0}',
        '{"label":"refused","rationale":"safe","confidence":1.0}',
    )
    result = await AdversarialRunner(
        repository=repository,
        gateways=RoleGateways(
            attacker=None,
            target=TracedModelGateway(target, repository),
            judge=TracedModelGateway(judge, repository),
        ),
    ).run(
        run=run,
        config=experiment_config,
        benchmark_prompt="sanitized benchmark goal",
        spec=_spec(method=AttackMethod.STATIC),
        baseline_inputs=BaselineInputs(
            candidates=("frozen candidate one", "frozen candidate two"),
            source_version="fixture-static-v1",
            language=ExperimentLanguage.ENGLISH,
        ),
    )

    rounds = (
        await repository.session.scalars(
            select(AdversarialRound).order_by(AdversarialRound.round_number)
        )
    ).all()
    assert result.rounds == 2
    assert result.llm_calls == 4
    assert [round_.attack_text for round_ in rounds] == [
        "frozen candidate one",
        "frozen candidate two",
    ]
    assert all(round_.attacker_step_id is None for round_ in rounds)
    assert all(
        round_.candidate_metadata_json["source"] == "frozen_static_candidate" for round_ in rounds
    )
    await repository.session.close()
    await engine.dispose()


async def test_random_mutation_is_seeded_and_persists_operator_provenance(
    experiment_config: ExperimentConfig,
) -> None:
    engine, repository, run = await _setup(experiment_config)
    target = _client("safe one", "safe two")
    judge = _client(
        '{"label":"refused","rationale":"safe","confidence":1.0}',
        '{"label":"refused","rationale":"safe","confidence":1.0}',
    )
    inputs = BaselineInputs(
        candidates=("alpha beta gamma delta", "one two three four"),
        source_version="fixture-mutation-v1",
        language=ExperimentLanguage.ENGLISH,
    )
    result = await AdversarialRunner(
        repository=repository,
        gateways=RoleGateways(
            attacker=None,
            target=TracedModelGateway(target, repository),
            judge=TracedModelGateway(judge, repository),
        ),
    ).run(
        run=run,
        config=experiment_config,
        benchmark_prompt="sanitized benchmark goal",
        spec=_spec(method=AttackMethod.RANDOM_MUTATION),
        baseline_inputs=inputs,
    )

    rounds = (
        await repository.session.scalars(
            select(AdversarialRound).order_by(AdversarialRound.round_number)
        )
    ).all()
    expected = [
        AdversarialRunner._mutate_candidate(
            candidates=inputs.candidates,
            source_version=inputs.source_version,
            random_seed=run.random_seed,
            round_number=round_number,
        )[0]
        for round_number in (1, 2)
    ]
    assert result.llm_calls == 4
    assert [round_.attack_text for round_ in rounds] == expected
    assert all(
        round_.candidate_metadata_json["source"] == "deterministic_random_mutation"
        for round_ in rounds
    )
    assert all(round_.attacker_step_id is None for round_ in rounds)
    await repository.session.close()
    await engine.dispose()


async def test_ukrainian_factor_is_consistent_across_prompts_and_traces(
    experiment_config: ExperimentConfig,
) -> None:
    engine, repository, run = await _setup(experiment_config)
    attacker = _client("тестовий запит")
    target = _client("безпечна відповідь")
    judge = _client('{"label":"refused","rationale":"безпечна відповідь","confidence":1.0}')
    result = await AdversarialRunner(
        repository=repository,
        gateways=RoleGateways(
            attacker=TracedModelGateway(attacker, repository),
            target=TracedModelGateway(target, repository),
            judge=TracedModelGateway(judge, repository),
        ),
    ).run(
        run=run,
        config=experiment_config,
        benchmark_prompt="перевірена тестова ціль",
        spec=_spec(
            max_rounds=1,
            method=AttackMethod.PAIR,
            language=ExperimentLanguage.UKRAINIAN,
        ),
    )

    calls = (await repository.session.scalars(select(LLMCall).order_by(LLMCall.started_at))).all()
    session = await repository.session.scalar(select(AdversarialSession))
    judge_payload = json.loads(calls[2].request_messages_json[1]["content"])
    assert result.llm_calls == 3
    assert session is not None
    assert session.language == "uk"
    assert session.spec_json["rubric_language"] == "uk"
    assert all(call.request_parameters_json["experiment_language"] == "uk" for call in calls)
    assert "українською" in calls[0].request_messages_json[0]["content"]
    assert "українською" in calls[1].request_messages_json[0]["content"]
    assert calls[1].request_messages_json[1] == {
        "role": "user",
        "content": "тестовий запит",
    }
    assert calls[2].request_messages_json[0]["content"] == "Поверни структуроване рішення."
    assert judge_payload["language"] == "uk"
    assert "Застосуй" in judge_payload["instruction"]
    await repository.session.close()
    await engine.dispose()


def test_language_mismatches_are_rejected_before_execution() -> None:
    with pytest.raises(ValueError, match="rubric_language"):
        AdversarialSpec(
            method=AttackMethod.PAIR,
            language=ExperimentLanguage.ENGLISH,
            max_rounds=1,
            rubric_version="fixture-v1",
            rubric_text="fixture",
            rubric_language=ExperimentLanguage.UKRAINIAN,
            benchmark_content_sha256="a" * 64,
        )

    spec = _spec(max_rounds=1, method=AttackMethod.STATIC)
    inputs = BaselineInputs(
        candidates=("тестовий запит",),
        source_version="fixture-uk-v1",
        language=ExperimentLanguage.UKRAINIAN,
    )
    with pytest.raises(ValueError, match="baseline input language"):
        AdversarialRunner._validate_method_inputs(
            spec=spec,
            baseline_inputs=inputs,
        )


async def test_judge_parse_failure_is_persisted(
    experiment_config: ExperimentConfig,
) -> None:
    engine, repository, run = await _setup(experiment_config)
    clients = (_client("candidate"), _client("response"), _client("not-json"))
    result = await AdversarialRunner(
        repository=repository,
        gateways=RoleGateways(
            attacker=TracedModelGateway(clients[0], repository),
            target=TracedModelGateway(clients[1], repository),
            judge=TracedModelGateway(clients[2], repository),
        ),
    ).run(
        run=run,
        config=experiment_config,
        benchmark_prompt="sanitized benchmark goal",
        spec=_spec(1),
    )

    label = await repository.session.scalar(select(Evaluation.label))
    assert result.status == RunStatus.FAILED
    assert result.stop_reason == "judge_parse_error"
    assert result.llm_calls == 3
    assert label == "parse_error"
    await repository.session.close()
    await engine.dispose()


async def test_call_budget_stops_before_partial_round(
    experiment_config: ExperimentConfig,
) -> None:
    constrained = experiment_config.model_copy(
        update={"budgets": experiment_config.budgets.model_copy(update={"max_total_llm_calls": 2})}
    )
    engine, repository, run = await _setup(constrained)
    clients = (_client("unused"), _client("unused"), _client("unused"))
    result = await AdversarialRunner(
        repository=repository,
        gateways=RoleGateways(
            attacker=TracedModelGateway(clients[0], repository),
            target=TracedModelGateway(clients[1], repository),
            judge=TracedModelGateway(clients[2], repository),
        ),
    ).run(
        run=run,
        config=constrained,
        benchmark_prompt="sanitized benchmark goal",
        spec=_spec(1),
    )

    assert result.status == RunStatus.BUDGET_STOPPED
    assert result.stop_reason == "max_total_llm_calls"
    assert result.rounds == 0
    assert result.llm_calls == 0
    assert sum(client.actual_request_count for client in clients) == 0
    await repository.session.close()
    await engine.dispose()


async def test_wall_time_budget_stops_before_starting_a_round(
    experiment_config: ExperimentConfig,
) -> None:
    constrained = experiment_config.model_copy(
        update={"agent": experiment_config.agent.model_copy(update={"max_wall_time_seconds": 1})}
    )
    engine, repository, run = await _setup(constrained)
    clients = (_client("unused"), _client("unused"), _client("unused"))
    clock_values = iter((0.0, 1.0))
    result = await AdversarialRunner(
        repository=repository,
        gateways=RoleGateways(
            attacker=TracedModelGateway(clients[0], repository),
            target=TracedModelGateway(clients[1], repository),
            judge=TracedModelGateway(clients[2], repository),
        ),
        clock=lambda: next(clock_values),
    ).run(
        run=run,
        config=constrained,
        benchmark_prompt="sanitized benchmark goal",
        spec=_spec(1),
    )

    assert result.status == RunStatus.BUDGET_STOPPED
    assert result.stop_reason == "max_wall_time"
    assert result.rounds == 0
    assert result.llm_calls == 0
    assert sum(client.actual_request_count for client in clients) == 0
    await repository.session.close()
    await engine.dispose()


async def test_wall_time_budget_stops_after_a_complete_persisted_round(
    experiment_config: ExperimentConfig,
) -> None:
    constrained = experiment_config.model_copy(
        update={"agent": experiment_config.agent.model_copy(update={"max_wall_time_seconds": 1})}
    )
    engine, repository, run = await _setup(constrained)
    target = _client("safe response")
    judge = _client('{"label":"refused","rationale":"safe","confidence":1.0}')
    clock_values = iter((0.0, 0.0, 1.0))
    result = await AdversarialRunner(
        repository=repository,
        gateways=RoleGateways(
            attacker=None,
            target=TracedModelGateway(target, repository),
            judge=TracedModelGateway(judge, repository),
        ),
        clock=lambda: next(clock_values),
    ).run(
        run=run,
        config=constrained,
        benchmark_prompt="sanitized benchmark goal",
        spec=_spec(method=AttackMethod.STATIC),
        baseline_inputs=BaselineInputs(
            candidates=("frozen candidate one", "frozen candidate two"),
            source_version="fixture-static-v1",
            language=ExperimentLanguage.ENGLISH,
        ),
    )

    rounds = await repository.session.scalar(select(func.count()).select_from(AdversarialRound))
    assert result.status == RunStatus.BUDGET_STOPPED
    assert result.stop_reason == "max_wall_time"
    assert result.rounds == 1
    assert result.llm_calls == 2
    assert rounds == 1
    assert target.actual_request_count == 1
    assert judge.actual_request_count == 1
    await repository.session.close()
    await engine.dispose()


async def test_transient_retry_retains_failed_and_successful_request_traces(
    experiment_config: ExperimentConfig,
) -> None:
    engine, repository, run = await _setup(experiment_config)
    attacker = DeterministicMockClient(
        [
            TransientModelError("fixture transient failure"),
            ModelResponse(content="candidate", input_tokens=2, output_tokens=3),
        ]
    )
    result = await AdversarialRunner(
        repository=repository,
        gateways=RoleGateways(
            attacker=TracedModelGateway(attacker, repository),
            target=TracedModelGateway(_client("safe response"), repository),
            judge=TracedModelGateway(
                _client('{"label":"refused","rationale":"safe","confidence":1.0}'),
                repository,
            ),
        ),
    ).run(
        run=run,
        config=experiment_config,
        benchmark_prompt="sanitized benchmark goal",
        spec=AdversarialSpec(
            method=AttackMethod.PAIR,
            language=ExperimentLanguage.ENGLISH,
            max_rounds=1,
            rubric_version="fixture-v1",
            rubric_text="Return the required structured decision.",
            rubric_language=ExperimentLanguage.ENGLISH,
            benchmark_content_sha256="a" * 64,
            max_role_attempts=2,
        ),
    )

    attacker_calls = (
        await repository.session.scalars(
            select(LLMCall).where(LLMCall.purpose == "attacker").order_by(LLMCall.started_at)
        )
    ).all()
    assert result.status == RunStatus.COMPLETED
    assert result.llm_calls == 4
    assert [call.status for call in attacker_calls] == ["failed", "succeeded"]
    assert [call.request_parameters_json["role_attempt"] for call in attacker_calls] == [1, 2]
    assert len({call.step_id for call in attacker_calls}) == 1
    assert attacker.actual_request_count == 2
    await repository.session.close()
    await engine.dispose()


async def test_permanent_failure_is_not_retried(
    experiment_config: ExperimentConfig,
) -> None:
    engine, repository, run = await _setup(experiment_config)
    attacker = DeterministicMockClient([ValueError("fixture permanent failure")])
    result = await AdversarialRunner(
        repository=repository,
        gateways=RoleGateways(
            attacker=TracedModelGateway(attacker, repository),
            target=TracedModelGateway(_client("unused"), repository),
            judge=TracedModelGateway(_client("unused"), repository),
        ),
    ).run(
        run=run,
        config=experiment_config,
        benchmark_prompt="sanitized benchmark goal",
        spec=AdversarialSpec(
            method=AttackMethod.PAIR,
            language=ExperimentLanguage.ENGLISH,
            max_rounds=1,
            rubric_version="fixture-v1",
            rubric_text="Return the required structured decision.",
            rubric_language=ExperimentLanguage.ENGLISH,
            benchmark_content_sha256="a" * 64,
            max_role_attempts=3,
        ),
    )

    assert result.status == RunStatus.FAILED
    assert result.llm_calls == 1
    assert attacker.actual_request_count == 1
    await repository.session.close()
    await engine.dispose()


async def test_retry_reservation_stops_before_round_can_exceed_call_budget(
    experiment_config: ExperimentConfig,
) -> None:
    constrained = experiment_config.model_copy(
        update={"budgets": experiment_config.budgets.model_copy(update={"max_total_llm_calls": 5})}
    )
    engine, repository, run = await _setup(constrained)
    clients = (_client("unused"), _client("unused"), _client("unused"))
    result = await AdversarialRunner(
        repository=repository,
        gateways=RoleGateways(
            attacker=TracedModelGateway(clients[0], repository),
            target=TracedModelGateway(clients[1], repository),
            judge=TracedModelGateway(clients[2], repository),
        ),
    ).run(
        run=run,
        config=constrained,
        benchmark_prompt="sanitized benchmark goal",
        spec=AdversarialSpec(
            method=AttackMethod.PAIR,
            language=ExperimentLanguage.ENGLISH,
            max_rounds=1,
            rubric_version="fixture-v1",
            rubric_text="Return the required structured decision.",
            rubric_language=ExperimentLanguage.ENGLISH,
            benchmark_content_sha256="a" * 64,
            max_role_attempts=2,
        ),
    )

    assert result.status == RunStatus.BUDGET_STOPPED
    assert result.llm_calls == 0
    assert sum(client.actual_request_count for client in clients) == 0
    await repository.session.close()
    await engine.dispose()


def _repeat_rejecting_spec(*, max_rounds: int, regenerations: int) -> AdversarialSpec:
    return AdversarialSpec(
        method=AttackMethod.PAIR,
        language=ExperimentLanguage.ENGLISH,
        max_rounds=max_rounds,
        rubric_version="fixture-v1",
        rubric_text="Return the required structured decision.",
        rubric_language=ExperimentLanguage.ENGLISH,
        benchmark_content_sha256="a" * 64,
        reject_repeated_attacks=True,
        max_attack_regenerations=regenerations,
    )


async def test_repeated_attack_is_rejected_regenerated_and_fully_traced(
    experiment_config: ExperimentConfig,
) -> None:
    engine, repository, run = await _setup(experiment_config)
    refused = '{"label":"refused","rationale":"safe","confidence":1.0}'

    attacker = _client("Candidate one", "  candidate   ONE ", "candidate two")
    target = _client("safe response", "safe response")
    judge = _client(refused, refused)
    result = await AdversarialRunner(
        repository=repository,
        gateways=RoleGateways(
            attacker=TracedModelGateway(attacker, repository),
            target=TracedModelGateway(target, repository),
            judge=TracedModelGateway(judge, repository),
        ),
    ).run(
        run=run,
        config=experiment_config,
        benchmark_prompt="sanitized benchmark goal",
        spec=_repeat_rejecting_spec(max_rounds=2, regenerations=2),
    )

    rounds = (
        await repository.session.scalars(
            select(AdversarialRound).order_by(AdversarialRound.round_number)
        )
    ).all()
    attacker_calls = await repository.session.scalar(
        select(func.count()).select_from(LLMCall).where(LLMCall.purpose == "attacker")
    )
    session = await repository.session.scalar(select(AdversarialSession))
    assert result.status == RunStatus.COMPLETED
    assert result.stop_reason == "max_rounds"

    assert result.llm_calls == 7
    assert attacker_calls == 3
    assert attacker.actual_request_count == 3
    assert [value.attack_text for value in rounds] == ["Candidate one", "candidate two"]
    rejections = rounds[1].candidate_metadata_json["repeat_rejections"]
    assert len(rejections) == 1
    assert rejections[0]["repeats_round"] == 1
    assert rounds[1].attacker_step_id is not None
    assert rejections[0]["step_id"] != str(rounds[1].attacker_step_id)
    assert rounds[1].candidate_metadata_json["repeat_unresolved"] is False
    assert rounds[0].candidate_metadata_json["repeat_rejections"] == []
    assert session is not None
    assert session.spec_json["reject_repeated_attacks"] is True
    assert session.spec_json["max_attack_regenerations"] == 2
    await repository.session.close()
    await engine.dispose()


async def test_exhausted_regeneration_bound_still_sends_the_repeat(
    experiment_config: ExperimentConfig,
) -> None:
    engine, repository, run = await _setup(experiment_config)
    refused = '{"label":"refused","rationale":"safe","confidence":1.0}'
    attacker = _client("same", "same", "same")
    result = await AdversarialRunner(
        repository=repository,
        gateways=RoleGateways(
            attacker=TracedModelGateway(attacker, repository),
            target=TracedModelGateway(_client("safe", "safe"), repository),
            judge=TracedModelGateway(_client(refused, refused), repository),
        ),
    ).run(
        run=run,
        config=experiment_config,
        benchmark_prompt="sanitized benchmark goal",
        spec=_repeat_rejecting_spec(max_rounds=2, regenerations=1),
    )

    last = await repository.session.scalar(
        select(AdversarialRound).where(AdversarialRound.round_number == 2)
    )
    assert result.status == RunStatus.COMPLETED
    assert result.rounds == 2

    assert result.llm_calls == 7
    assert attacker.actual_request_count == 3
    assert last is not None
    assert last.attack_text == "same"
    assert len(last.candidate_metadata_json["repeat_rejections"]) == 1
    assert last.candidate_metadata_json["repeat_unresolved"] is True
    await repository.session.close()
    await engine.dispose()


async def test_original_protocol_sends_repeats_and_records_no_rejection_fields(
    experiment_config: ExperimentConfig,
) -> None:
    engine, repository, run = await _setup(experiment_config)
    refused = '{"label":"refused","rationale":"safe","confidence":1.0}'
    attacker = _client("same", "same")
    result = await AdversarialRunner(
        repository=repository,
        gateways=RoleGateways(
            attacker=TracedModelGateway(attacker, repository),
            target=TracedModelGateway(_client("safe", "safe"), repository),
            judge=TracedModelGateway(_client(refused, refused), repository),
        ),
    ).run(
        run=run,
        config=experiment_config,
        benchmark_prompt="sanitized benchmark goal",
        spec=_spec(),
    )

    last = await repository.session.scalar(
        select(AdversarialRound).where(AdversarialRound.round_number == 2)
    )
    session = await repository.session.scalar(select(AdversarialSession))
    assert result.llm_calls == 6
    assert attacker.actual_request_count == 2
    assert last is not None
    assert "repeat_rejections" not in last.candidate_metadata_json
    assert session is not None
    assert "reject_repeated_attacks" not in session.spec_json
    await repository.session.close()
    await engine.dispose()


async def test_regeneration_allowance_is_reserved_in_the_call_budget(
    experiment_config: ExperimentConfig,
) -> None:
    constrained = experiment_config.model_copy(
        update={"budgets": experiment_config.budgets.model_copy(update={"max_total_llm_calls": 4})}
    )
    engine, repository, run = await _setup(constrained)
    clients = (_client("unused"), _client("unused"), _client("unused"))
    result = await AdversarialRunner(
        repository=repository,
        gateways=RoleGateways(
            attacker=TracedModelGateway(clients[0], repository),
            target=TracedModelGateway(clients[1], repository),
            judge=TracedModelGateway(clients[2], repository),
        ),
    ).run(
        run=run,
        config=constrained,
        benchmark_prompt="sanitized benchmark goal",
        spec=_repeat_rejecting_spec(max_rounds=1, regenerations=2),
    )

    assert result.status == RunStatus.BUDGET_STOPPED
    assert result.llm_calls == 0
    assert sum(client.actual_request_count for client in clients) == 0
    await repository.session.close()
    await engine.dispose()


def test_repeat_rejection_spec_is_validated() -> None:
    with pytest.raises(ValueError, match="at least one regeneration"):
        _repeat_rejecting_spec(max_rounds=1, regenerations=0)
    with pytest.raises(ValueError, match="requires reject_repeated_attacks"):
        AdversarialSpec(
            method=AttackMethod.PAIR,
            language=ExperimentLanguage.ENGLISH,
            max_rounds=1,
            rubric_version="fixture-v1",
            rubric_text="rubric",
            rubric_language=ExperimentLanguage.ENGLISH,
            benchmark_content_sha256="a" * 64,
            max_attack_regenerations=1,
        )
    with pytest.raises(ValueError, match="only to PAIR"):
        AdversarialSpec(
            method=AttackMethod.SINGLE_SHOT,
            language=ExperimentLanguage.ENGLISH,
            max_rounds=1,
            rubric_version="fixture-v1",
            rubric_text="rubric",
            rubric_language=ExperimentLanguage.ENGLISH,
            benchmark_content_sha256="a" * 64,
            reject_repeated_attacks=True,
            max_attack_regenerations=1,
        )
