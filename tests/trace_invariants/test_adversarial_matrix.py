from __future__ import annotations

from sqlalchemy import func, select

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
from harness.agents.native.types import ModelResponse
from harness.config.schema import ExperimentConfig
from harness.experiments.adversarial_matrix import (
    FrozenModelProfile,
    MatrixCell,
    MatrixDesign,
    MatrixItem,
    build_matrix_plan,
)
from harness.persistence.database import create_engine, create_schema, session_factory
from harness.persistence.models import AdversarialSession, LLMCall
from harness.persistence.repository import TraceRepository


def _profile(key: str) -> FrozenModelProfile:
    return FrozenModelProfile(
        key=key,
        provider="mock",
        model=f"{key}-model",
        endpoint_profile=None,
        revision=f"{key}-revision",
    )


def _responses(*contents: str) -> list[ModelResponse]:
    return [ModelResponse(content=content, input_tokens=2, output_tokens=3) for content in contents]


def _client(profile: FrozenModelProfile, *contents: str) -> DeterministicMockClient:
    assert profile.model is not None
    return DeterministicMockClient(_responses(*contents), model=profile.model)


def _localized(language: ExperimentLanguage, english: str, ukrainian: str) -> str:
    return english if language == ExperimentLanguage.ENGLISH else ukrainian


def _judgement(language: ExperimentLanguage) -> str:

    rationale = _localized(language, "refused the request", "відхилено запит")
    return f'{{"label":"refused","rationale":"{rationale}","confidence":1.0}}'


def _baseline_inputs(cell: MatrixCell) -> BaselineInputs | None:
    if cell.method not in {AttackMethod.STATIC, AttackMethod.RANDOM_MUTATION}:
        return None
    return BaselineInputs(
        candidates=(
            _localized(cell.item.language, "static candidate one", "статичний запит один"),
            _localized(cell.item.language, "static candidate two", "статичний запит два"),
        ),
        source_version="mock-matrix-baselines-v1",
        language=cell.item.language,
    )


async def test_complete_mock_matrix_preserves_cell_and_trace_invariants(
    experiment_config: ExperimentConfig,
) -> None:
    models = (_profile("model-a"), _profile("model-b"))
    judge_profile = _profile("fixed-judge")
    plan = build_matrix_plan(
        MatrixDesign(
            attackers=models,
            targets=models,
            judge=judge_profile,
            items=(
                MatrixItem(
                    stable_id="mock-en",
                    content_sha256="1" * 64,
                    language=ExperimentLanguage.ENGLISH,
                    category="fixture",
                    pair_id="mock-pair",
                    reviewer_status="approved",
                ),
                MatrixItem(
                    stable_id="mock-uk",
                    content_sha256="2" * 64,
                    language=ExperimentLanguage.UKRAINIAN,
                    category="fixture",
                    pair_id="mock-pair",
                    reviewer_status="approved",
                ),
            ),
            methods=tuple(AttackMethod),
            seeds=(11, 22),
            max_rounds=2,
            max_role_attempts=1,
            rubric_version="mock-matrix-v1",
        )
    )
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    await create_schema(engine)
    session = session_factory(engine)()
    repository = TraceRepository(session)
    experiment = await repository.create_experiment(
        name="Complete mock matrix",
        research_question="Does every frozen cell retain exact traces?",
        hypothesis="Yes",
        created_by="pytest",
    )

    for cell in plan.cells:
        config = experiment_config.model_copy(
            update={
                "models": experiment_config.models.model_copy(
                    update={
                        "attacker": experiment_config.models.attacker.model_copy(
                            update={
                                "model": (
                                    cell.attacker.model
                                    if cell.attacker is not None
                                    else models[0].model
                                )
                            }
                        ),
                        "target": experiment_config.models.target.model_copy(
                            update={"model": cell.target.model}
                        ),
                        "judge": experiment_config.models.judge.model_copy(
                            update={"model": cell.judge.model}
                        ),
                    }
                ),
                "task": experiment_config.task.model_copy(
                    update={"benchmark_item_id": cell.item.stable_id}
                ),
            }
        )
        snapshot = await repository.get_or_create_config(
            name=f"mock-matrix-{cell.cell_id}",
            config=config,
        )
        run = await repository.create_run(
            experiment_id=experiment.id,
            config_id=snapshot.id,
            benchmark_item_id=cell.item.stable_id,
            random_seed=cell.seed,
            code_version="mock-matrix-v1",
        )
        attacker_gateway = (
            TracedModelGateway(
                _client(
                    cell.attacker,
                    _localized(cell.item.language, "attempt one", "спроба один"),
                    _localized(cell.item.language, "attempt two", "спроба два"),
                ),
                repository,
            )
            if cell.attacker is not None
            else None
        )
        result = await AdversarialRunner(
            repository=repository,
            gateways=RoleGateways(
                attacker=attacker_gateway,
                target=TracedModelGateway(
                    _client(
                        cell.target,
                        _localized(cell.item.language, "safe one", "безпечно один"),
                        _localized(cell.item.language, "safe two", "безпечно два"),
                    ),
                    repository,
                ),
                judge=TracedModelGateway(
                    _client(
                        cell.judge,
                        _judgement(cell.item.language),
                        _judgement(cell.item.language),
                    ),
                    repository,
                ),
            ),
        ).run(
            run=run,
            config=config,
            benchmark_prompt=_localized(
                cell.item.language,
                "sanitized fixture goal",
                "перевірена тестова ціль",
            ),
            spec=AdversarialSpec(
                method=cell.method,
                language=cell.item.language,
                max_rounds=cell.max_rounds,
                rubric_version=cell.rubric_version,
                rubric_text=_localized(
                    cell.item.language,
                    "Return a structured decision.",
                    "Поверни структуроване рішення.",
                ),
                rubric_language=cell.item.language,
                benchmark_content_sha256=cell.item.content_sha256,
                max_role_attempts=cell.max_role_attempts,
            ),
            baseline_inputs=_baseline_inputs(cell),
        )
        assert result.rounds == cell.max_rounds
        assert result.llm_calls == cell.max_llm_calls

    calls = (
        await session.scalars(
            select(LLMCall).where(LLMCall.run_id.in_(select(AdversarialSession.run_id)))
        )
    ).all()
    assert await session.scalar(select(func.count()).select_from(AdversarialSession)) == len(
        plan.cells
    )
    assert len(calls) == plan.max_llm_calls
    assert sum(call.purpose == "target" for call in calls) == plan.max_target_queries
    assert sum(call.purpose == "judge" for call in calls) == plan.max_target_queries
    assert all(
        call.request_parameters_json["experiment_language"] in {"en", "uk"} for call in calls
    )
    await session.close()
    await engine.dispose()
