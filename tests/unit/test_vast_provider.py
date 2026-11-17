from __future__ import annotations

import json

import httpx
import pytest

from harness.agents.native.types import TransientModelError
from harness.agents.native.vast_provider import VastOpenAIClient, VastServingProfile
from harness.spend.vast import OWNER_LABEL, VastInstance


def _profile(**updates: object) -> VastServingProfile:
    values: dict[str, object] = {
        "name": "vast-fixture-v1",
        "served_model": "fixture/model",
        "checkpoint_revision": "a" * 40,
        "config_sha256": "b" * 64,
        "chat_template_sha256": "c" * 64,
        "inference_engine": "vllm",
        "inference_engine_version": "0.10.1",
        "container_image": "fixture/vllm@sha256:abc",
        "container_image_sha256": "d" * 64,
        "cuda_version": "12.8",
        "gpu_name": "RTX 4090",
        "num_gpus": 1,
        "tensor_parallel_size": 1,
        "dtype": "auto",
        "max_model_len": 8192,
        "max_output_tokens": 1024,
        "request_timeout_seconds": 90,
        "request_overrides": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    values.update(updates)
    return VastServingProfile.model_validate(values)


def _instance(**updates: object) -> VastInstance:
    values: dict[str, object] = {
        "id": 321,
        "actual_status": "running",
        "gpu_name": "RTX 4090",
        "num_gpus": 1,
        "image_uuid": "fixture/vllm@sha256:abc",
        "label": OWNER_LABEL,
        "dph_total": 0.31,
    }
    values.update(updates)
    return VastInstance.model_validate(values)


async def test_vast_client_maps_completion_and_keeps_rental_cost_separate() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "http://127.0.0.1:18000/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer fixture-secret"
        payload = json.loads(request.content)
        assert payload["model"] == "fixture/model"
        assert payload["seed"] == 17
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        assert "experiment_language" not in payload
        return httpx.Response(
            200,
            json={
                "id": "completion-1",
                "model": "fixture/model",
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "finish_experiment",
                                        "arguments": '{"outcome":"completed"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 4,
                    "prompt_tokens_details": {"cached_tokens": 3},
                    "completion_tokens_details": {"reasoning_tokens": 2},
                },
            },
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = VastOpenAIClient(
        profile=_profile(),
        endpoint_url="http://127.0.0.1:18000",
        instance=_instance(),
        http_client=http,
        vllm_api_key="fixture-secret",
    )
    response = await client.request(
        messages=[{"role": "user", "content": "fixture"}],
        tools=[{"type": "function", "function": {"name": "finish_experiment"}}],
        parameters={"experiment_language": "en", "seed": 17, "temperature": 0},
    )

    assert response.content == ""
    assert response.tool_calls[0].name == "finish_experiment"
    assert response.tool_calls[0].arguments == {"outcome": "completed"}
    assert response.input_tokens == 12
    assert response.output_tokens == 4
    assert response.cached_input_tokens == 3
    assert response.reasoning_tokens == 2
    assert response.estimated_cost_usd == 0
    assert response.raw["vast_instance_id"] == 321
    assert response.raw["transport"] == "vast_loopback_tunnel"
    assert client.actual_request_count == 1
    assert "fixture-secret" not in str(response.raw)
    await http.aclose()


@pytest.mark.parametrize("status_code", [408, 429, 502, 503, 504])
async def test_vast_client_classifies_transient_http_statuses(status_code: int) -> None:
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(status_code, json={"error": "fixture"})
        )
    )
    client = VastOpenAIClient(
        profile=_profile(),
        endpoint_url="http://127.0.0.1:18000",
        instance=_instance(),
        http_client=http,
    )

    with pytest.raises(TransientModelError, match=str(status_code)):
        await client.request(messages=[], tools=[], parameters={})
    assert client.actual_request_count == 1
    await http.aclose()


def test_vast_client_refuses_non_project_or_drifted_instances_before_requests() -> None:
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda _request: httpx.Response(500)))
    invalid = (
        (_instance(label=None), "not labelled"),
        (_instance(actual_status="exited"), "not running"),
        (_instance(gpu_name="A100"), "GPU identity"),
        (_instance(image_uuid="different/image"), "image identity"),
    )

    for instance, message in invalid:
        with pytest.raises(ValueError, match=message):
            VastOpenAIClient(
                profile=_profile(),
                endpoint_url="http://127.0.0.1:18000",
                instance=instance,
                http_client=http,
            )


@pytest.mark.parametrize(
    "endpoint_url",
    [
        "https://example.com:8000",
        "http://203.0.113.1:8000",
        "http://user:pass@127.0.0.1:8000",
        "http://127.0.0.1:8000/v1",
        "http://127.0.0.1",
    ],
)
def test_vast_client_requires_a_loopback_tunnel(endpoint_url: str) -> None:
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda _request: httpx.Response(500)))
    with pytest.raises(ValueError, match="loopback HTTP"):
        VastOpenAIClient(
            profile=_profile(),
            endpoint_url=endpoint_url,
            instance=_instance(),
            http_client=http,
        )


async def test_vast_client_rejects_model_identity_drift() -> None:
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                json={
                    "id": "completion-1",
                    "model": "different/model",
                    "choices": [{"message": {"content": "fixture"}}],
                },
            )
        )
    )
    client = VastOpenAIClient(
        profile=_profile(),
        endpoint_url="http://127.0.0.1:18000",
        instance=_instance(),
        http_client=http,
    )

    with pytest.raises(RuntimeError, match="identity mismatch"):
        await client.request(messages=[], tools=[], parameters={})
    await http.aclose()


async def test_vast_client_rejects_output_limit_drift_without_a_request() -> None:
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda _request: httpx.Response(500)))
    client = VastOpenAIClient(
        profile=_profile(),
        endpoint_url="http://127.0.0.1:18000",
        instance=_instance(),
        http_client=http,
    )

    with pytest.raises(ValueError, match="max_tokens"):
        await client.request(messages=[], tools=[], parameters={"max_tokens": 1025})
    assert client.actual_request_count == 0
    await http.aclose()


def test_vast_profiles_are_content_addressed() -> None:
    original = _profile()
    changed_gpu = _profile(gpu_name="A100")
    changed_engine = _profile(inference_engine_version="0.10.2")

    assert original.profile_id != changed_gpu.profile_id
    assert original.profile_id != changed_engine.profile_id
    assert original.public_manifest()["container_image_sha256"] == "d" * 64


@pytest.mark.parametrize(
    "updates",
    [
        {"server_env": {"HF_TOKEN": "secret"}},
        {"request_overrides": {"api_key": "secret"}},
        {"server_args": ("--api-key", "secret")},
        {"server_args": ("--api-key=secret",)},
    ],
)
def test_vast_profile_rejects_secrets_from_public_manifest(
    updates: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match=r"secret|api-key"):
        _profile(**updates)
