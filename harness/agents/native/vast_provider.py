from __future__ import annotations

import hashlib
import ipaddress
import json
import time
from typing import Any
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from harness.agents.native.types import ModelResponse, ToolRequest, TransientModelError
from harness.spend.vast import OWNER_LABEL, VastInstance


class VastServingProfile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    served_model: str = Field(min_length=1)
    checkpoint_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    chat_template_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    inference_engine: str = Field(min_length=1)
    inference_engine_version: str = Field(min_length=1)
    container_image: str = Field(min_length=1)
    container_image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cuda_version: str = Field(min_length=1)
    gpu_name: str = Field(min_length=1)
    num_gpus: int = Field(ge=1)
    tensor_parallel_size: int = Field(ge=1)
    dtype: str = Field(min_length=1)
    max_model_len: int = Field(ge=1)
    max_output_tokens: int = Field(ge=1)
    request_timeout_seconds: float = Field(gt=0)
    request_overrides: dict[str, Any] = Field(default_factory=dict)
    server_args: tuple[str, ...] = ()
    server_env: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_public_manifest(self) -> VastServingProfile:
        if self.tensor_parallel_size > self.num_gpus:
            raise ValueError("tensor_parallel_size cannot exceed num_gpus")
        sensitive_markers = (
            "API_KEY",
            "AUTH_TOKEN",
            "ACCESS_TOKEN",
            "HF_TOKEN",
            "PASSWORD",
            "SECRET",
            "CREDENTIAL",
        )
        public_keys = (*self.server_env, *self.request_overrides)
        if any(marker in key.upper() for key in public_keys for marker in sensitive_markers):
            raise ValueError("serving profile public manifest cannot contain secret settings")
        if any(
            argument == "--api-key" or argument.startswith("--api-key=")
            for argument in self.server_args
        ):
            raise ValueError("serving profile server_args cannot contain --api-key")
        return self

    def public_manifest(self) -> dict[str, Any]:

        return self.model_dump(mode="json")

    @property
    def manifest_sha256(self) -> str:
        canonical = json.dumps(
            self.public_manifest(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()

    @property
    def profile_id(self) -> str:
        return f"{self.name}@{self.manifest_sha256}"


class _UsageDetails(BaseModel):
    model_config = ConfigDict(extra="ignore")

    cached_tokens: int = 0
    reasoning_tokens: int = 0


class _Usage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_tokens_details: _UsageDetails | None = None
    completion_tokens_details: _UsageDetails | None = None


class _FunctionCall(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1)
    arguments: str


class _CompletionToolCall(BaseModel):
    model_config = ConfigDict(extra="ignore")

    function: _FunctionCall


class _Message(BaseModel):
    model_config = ConfigDict(extra="ignore")

    content: str | None = None
    tool_calls: list[_CompletionToolCall] = Field(default_factory=list)


class _Choice(BaseModel):
    model_config = ConfigDict(extra="ignore")

    message: _Message
    finish_reason: str | None = None


class _ChatCompletion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    model: str
    choices: list[_Choice] = Field(min_length=1)
    usage: _Usage = _Usage()
    system_fingerprint: str | None = None


class VastOpenAIClient:
    provider = "vast.ai"
    endpoint_profile: str | None

    def __init__(
        self,
        *,
        profile: VastServingProfile,
        endpoint_url: str,
        instance: VastInstance,
        http_client: httpx.AsyncClient,
        vllm_api_key: str | None = None,
    ) -> None:
        _validate_instance(instance, profile)
        self.profile = profile
        self.model = profile.served_model
        self.endpoint_profile = profile.profile_id
        self.actual_request_count = 0
        self.instance_id = instance.id
        self._endpoint_url = _validate_endpoint_url(endpoint_url)
        self._headers = {"Authorization": f"Bearer {vllm_api_key}"} if vllm_api_key else {}
        self._http = http_client

    async def request(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        parameters: dict[str, Any],
    ) -> ModelResponse:
        max_tokens = parameters.get("max_tokens")
        if max_tokens is not None and (
            not isinstance(max_tokens, int)
            or isinstance(max_tokens, bool)
            or max_tokens < 1
            or max_tokens > self.profile.max_output_tokens
        ):
            raise ValueError(
                f"max_tokens must be an integer from 1 to {self.profile.max_output_tokens}"
            )

        payload: dict[str, Any] = {
            "max_tokens": self.profile.max_output_tokens,
            "messages": messages,
            "model": self.profile.served_model,
            "stream": False,
            **self.profile.request_overrides,
        }
        for name in ("max_tokens", "seed", "temperature", "top_p"):
            if parameters.get(name) is not None:
                payload[name] = parameters[name]
        if tools:
            payload["tools"] = tools

        self.actual_request_count += 1
        started = time.monotonic()
        try:
            response = await self._http.post(
                f"{self._endpoint_url}/v1/chat/completions",
                json=payload,
                headers=self._headers,
                timeout=self.profile.request_timeout_seconds,
            )
        except httpx.TransportError as exc:
            raise TransientModelError(type(exc).__name__) from exc
        if response.status_code in {408, 429, 502, 503, 504}:
            raise TransientModelError(f"Vast endpoint HTTP {response.status_code}")
        if response.is_error:
            raise RuntimeError(f"Vast endpoint HTTP {response.status_code}")

        completion = _ChatCompletion.model_validate(response.json())
        if completion.model != self.profile.served_model:
            raise RuntimeError(
                "Vast endpoint model identity mismatch: "
                f"expected {self.profile.served_model}, got {completion.model}"
            )
        choice = completion.choices[0]
        tool_requests = _parse_tool_requests(choice.message.tool_calls)
        if not choice.message.content and not tool_requests:
            raise RuntimeError("Vast endpoint returned neither assistant content nor tool calls")

        elapsed = time.monotonic() - started
        prompt_details = completion.usage.prompt_tokens_details or _UsageDetails()
        completion_details = completion.usage.completion_tokens_details or _UsageDetails()
        return ModelResponse(
            content=choice.message.content or "",
            tool_calls=tool_requests,
            raw={
                "finish_reason": choice.finish_reason,
                "id": completion.id,
                "model": completion.model,
                "profile_id": self.profile.profile_id,
                "system_fingerprint": completion.system_fingerprint,
                "transport": "vast_loopback_tunnel",
                "vast_instance_id": self.instance_id,
                "provider_latency_seconds": elapsed,
            },
            input_tokens=completion.usage.prompt_tokens,
            output_tokens=completion.usage.completion_tokens,
            cached_input_tokens=prompt_details.cached_tokens,
            reasoning_tokens=completion_details.reasoning_tokens,
            estimated_cost_usd=0.0,
        )


def _validate_instance(instance: VastInstance, profile: VastServingProfile) -> None:
    if not instance.is_ours:
        raise ValueError(
            f"Vast instance {instance.id} is not labelled {OWNER_LABEL!r}; refusing to use it"
        )
    if not instance.is_running:
        raise ValueError(f"Vast instance {instance.id} is not running")
    if instance.gpu_name != profile.gpu_name or instance.num_gpus != profile.num_gpus:
        raise ValueError(
            f"Vast instance {instance.id} GPU identity does not match the serving profile"
        )
    if instance.image_uuid != profile.container_image:
        raise ValueError(
            f"Vast instance {instance.id} image identity does not match the serving profile"
        )


def _validate_endpoint_url(value: str) -> str:
    parsed = urlparse(value)
    try:
        host = parsed.hostname
        port = parsed.port
        is_loopback = host == "localhost" or (
            host is not None and ipaddress.ip_address(host).is_loopback
        )
    except ValueError:
        is_loopback = False
        port = None
    if (
        parsed.scheme not in {"http", "https"}
        or not is_loopback
        or port is None
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError(
            "endpoint_url must be a loopback HTTP(S) origin with an explicit tunnel port"
        )
    return value.rstrip("/")


def _parse_tool_requests(calls: list[_CompletionToolCall]) -> tuple[ToolRequest, ...]:
    requests: list[ToolRequest] = []
    for call in calls:
        try:
            arguments = json.loads(call.function.arguments)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Vast endpoint returned invalid arguments for tool {call.function.name}"
            ) from exc
        if not isinstance(arguments, dict):
            raise RuntimeError(
                f"Vast endpoint returned non-object arguments for tool {call.function.name}"
            )
        requests.append(ToolRequest(name=call.function.name, arguments=arguments))
    return tuple(requests)
