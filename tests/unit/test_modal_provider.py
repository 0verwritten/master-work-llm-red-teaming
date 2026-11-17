from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from harness.agents.native.modal_provider import (
    GEMMA4_31B_CANDIDATE,
    GPT_OSS_20B_JUDGE_CANDIDATE,
    MISTRAL_SMALL_32_24B_CANDIDATE,
    QWEN36_27B_FP8_CANDIDATE,
    ModalOpenAIClient,
)
from harness.agents.native.types import TransientModelError


async def test_modal_client_maps_authenticated_completion_without_tracing_secrets() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Modal-Key"] == "wk-fixture"
        assert request.headers["Modal-Secret"] == "ws-fixture"
        payload = __import__("json").loads(request.content)
        assert payload["model"] == QWEN36_27B_FP8_CANDIDATE.served_model
        assert payload["seed"] == 123
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        assert "experiment_language" not in payload
        return httpx.Response(
            200,
            json={
                "id": "fixture-completion",
                "model": QWEN36_27B_FP8_CANDIDATE.served_model,
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "fixture answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 4,
                    "prompt_tokens_details": None,
                    "completion_tokens_details": {"reasoning_tokens": 1},
                },
            },
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ModalOpenAIClient(
        profile=QWEN36_27B_FP8_CANDIDATE,
        endpoint_url="https://workspace--pinned-model.eu-west.modal.direct",
        proxy_token_id="wk-fixture",
        proxy_token_secret="ws-fixture",
        http_client=http,
    )
    response = await client.request(
        messages=[{"role": "user", "content": "fixture"}],
        tools=[],
        parameters={
            "experiment_language": "en",
            "role_attempt": 1,
            "seed": 123,
            "temperature": 0,
        },
    )

    assert response.content == "fixture answer"
    assert response.input_tokens == 12
    assert response.output_tokens == 4
    assert response.cached_input_tokens == 0
    assert response.reasoning_tokens == 1
    assert response.raw["profile_id"] == QWEN36_27B_FP8_CANDIDATE.profile_id
    assert client.actual_request_count == 1
    assert "wk-fixture" not in str(response.raw)
    assert "ws-fixture" not in str(response.raw)
    await http.aclose()


@pytest.mark.parametrize("status_code", [408, 429, 502, 503, 504])
async def test_modal_client_classifies_only_documented_transient_http_statuses(
    status_code: int,
) -> None:
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(status_code, json={"error": "fixture"})
    )
    http = httpx.AsyncClient(transport=transport)
    client = ModalOpenAIClient(
        profile=QWEN36_27B_FP8_CANDIDATE,
        endpoint_url="https://workspace--pinned-model.eu-west.modal.direct",
        proxy_token_id="wk-fixture",
        proxy_token_secret="ws-fixture",
        http_client=http,
    )

    with pytest.raises(TransientModelError, match=str(status_code)):
        await client.request(
            messages=[{"role": "user", "content": "fixture"}],
            tools=[],
            parameters={},
        )
    assert client.actual_request_count == 1
    await http.aclose()


async def test_modal_client_rejects_served_model_identity_drift() -> None:
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            200,
            json={
                "id": "fixture-completion",
                "model": "unexpected/model",
                "choices": [{"message": {"content": "fixture"}, "finish_reason": "stop"}],
            },
        )
    )
    http = httpx.AsyncClient(transport=transport)
    client = ModalOpenAIClient(
        profile=QWEN36_27B_FP8_CANDIDATE,
        endpoint_url="https://workspace--pinned-model.eu-west.modal.direct",
        proxy_token_id="wk-fixture",
        proxy_token_secret="ws-fixture",
        http_client=http,
    )

    with pytest.raises(RuntimeError, match="identity mismatch"):
        await client.request(
            messages=[{"role": "user", "content": "fixture"}],
            tools=[],
            parameters={},
        )
    await http.aclose()


def test_modal_client_rejects_public_or_credential_bearing_origins() -> None:
    transport = httpx.MockTransport(lambda _request: httpx.Response(500))
    for endpoint_url in (
        "http://workspace--model.modal.direct",
        "https://user:pass@workspace--model.modal.direct",
        "https://example.com",
        "https://workspace--model.modal.direct/v1",
    ):
        with pytest.raises(ValueError, match="HTTPS Modal origin"):
            ModalOpenAIClient(
                profile=QWEN36_27B_FP8_CANDIDATE,
                endpoint_url=endpoint_url,
                proxy_token_id="wk-fixture",
                proxy_token_secret="ws-fixture",
                http_client=httpx.AsyncClient(transport=transport),
            )


def test_specialized_candidate_profiles_are_distinct_and_content_addressed() -> None:
    candidates = (
        GEMMA4_31B_CANDIDATE,
        MISTRAL_SMALL_32_24B_CANDIDATE,
        GPT_OSS_20B_JUDGE_CANDIDATE,
    )

    assert len({profile.profile_id for profile in candidates}) == 3
    assert all(profile.checkpoint_revision for profile in candidates)
    assert all(profile.min_containers == 1 for profile in candidates)
    assert all(profile.max_containers == 1 for profile in candidates)
    assert all(profile.server_args for profile in candidates)
    assert GPT_OSS_20B_JUDGE_CANDIDATE.request_overrides["reasoning_effort"] == "low"


async def test_router_error_object_is_transient_not_permanent(tmp_path: Path) -> None:

    from harness.agents.native.modal_curl_provider import ModalCurlClient

    stub = tmp_path / "fake-modal"
    stub.write_text(
        '#!/bin/sh\ncat > /dev/null\nprintf \'{"error": "no upstreams available"}\'\n',
        encoding="utf-8",
    )
    stub.chmod(0o755)
    client = ModalCurlClient(
        profile=QWEN36_27B_FP8_CANDIDATE,
        endpoint_url="https://fixture.modal.run",
        binary=str(stub),
    )
    with pytest.raises(TransientModelError, match="transient error"):
        await client.request(messages=[], tools=[], parameters={})
    assert client.actual_request_count == 1
