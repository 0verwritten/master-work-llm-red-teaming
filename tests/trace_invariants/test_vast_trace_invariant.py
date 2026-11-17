from __future__ import annotations

import httpx
from sqlalchemy import select

from harness.agents.native.gateway import TracedModelGateway
from harness.agents.native.vast_provider import VastOpenAIClient, VastServingProfile
from harness.config.schema import ExperimentConfig
from harness.persistence.database import create_engine, create_schema, session_factory
from harness.persistence.models import CallStatus, LLMCall
from harness.persistence.repository import TraceRepository
from harness.spend.vast import OWNER_LABEL, VastInstance


def _profile() -> VastServingProfile:
    return VastServingProfile(
        name="vast-trace-fixture-v1",
        served_model="fixture/model",
        checkpoint_revision="a" * 40,
        config_sha256="b" * 64,
        chat_template_sha256="c" * 64,
        inference_engine="vllm",
        inference_engine_version="0.10.1",
        container_image="fixture/vllm@sha256:abc",
        container_image_sha256="d" * 64,
        cuda_version="12.8",
        gpu_name="RTX 4090",
        num_gpus=1,
        tensor_parallel_size=1,
        dtype="auto",
        max_model_len=8192,
        max_output_tokens=1024,
        request_timeout_seconds=90,
    )


def _instance() -> VastInstance:
    return VastInstance(
        id=321,
        actual_status="running",
        gpu_name="RTX 4090",
        num_gpus=1,
        image_uuid="fixture/vllm@sha256:abc",
        label=OWNER_LABEL,
        dph_total=0.31,
    )


async def test_vast_request_is_persisted_before_crossing_provider_boundary(
    experiment_config: ExperimentConfig,
) -> None:
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    await create_schema(engine)
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                json={
                    "id": "completion-1",
                    "model": "fixture/model",
                    "choices": [{"message": {"content": "fixture"}}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1},
                },
            )
        )
    )
    client = VastOpenAIClient(
        profile=_profile(),
        endpoint_url="http://127.0.0.1:18000",
        instance=_instance(),
        http_client=http,
        vllm_api_key="fixture-secret",
    )

    async with session_factory(engine)() as session:
        repository = TraceRepository(session)
        experiment = await repository.create_experiment(
            name="Vast trace invariant",
            research_question="Is every Vast request traced?",
            hypothesis="Yes",
            created_by="pytest",
        )
        snapshot = await repository.get_or_create_config(
            name="vast-trace-fixture", config=experiment_config
        )
        run = await repository.create_run(
            experiment_id=experiment.id,
            config_id=snapshot.id,
            benchmark_item_id="benchmark-001",
            random_seed=1,
            code_version="test",
        )
        await repository.start_run(run)
        step = await repository.create_step(
            run_id=run.id,
            sequence_number=1,
            step_type="agent",
            purpose="research",
        )
        await session.commit()

        response = await TracedModelGateway(client, repository).request(
            run_id=run.id,
            step_id=step.id,
            purpose="research",
            messages=[{"role": "user", "content": "fixture"}],
            tools=[],
            parameters={"temperature": 0},
        )

        calls = tuple((await session.scalars(select(LLMCall))).all())
        assert response.estimated_cost_usd == 0
        assert client.actual_request_count == 1
        assert len(calls) == client.actual_request_count
        assert calls[0].provider == "vast.ai"
        assert calls[0].endpoint_profile == client.endpoint_profile
        assert calls[0].status == CallStatus.SUCCEEDED
        assert calls[0].estimated_cost_usd == 0
        assert "fixture-secret" not in str(calls[0].response_json)

    await http.aclose()
    await engine.dispose()
