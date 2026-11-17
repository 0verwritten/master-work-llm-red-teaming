from __future__ import annotations

import hashlib
import json
import time
from typing import Any
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field

from harness.agents.native.types import ModelResponse, TransientModelError


class ModalServingProfile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    app_name: str = Field(min_length=1)
    served_model: str = Field(min_length=1)
    checkpoint_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    chat_template_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    inference_engine: str = Field(min_length=1)
    inference_engine_version: str = Field(min_length=1)
    container_base: str = Field(min_length=1)
    gpu: str = Field(min_length=1)
    tensor_parallel_size: int = Field(ge=1)
    dtype: str = Field(min_length=1)
    enforce_eager: bool
    deep_gemm_warmup: str = Field(pattern=r"^(skip|full|relax)$")
    gdn_prefill_backend: str = Field(min_length=1)
    safetensors_load_strategy: str = Field(min_length=1)
    max_model_len: int = Field(ge=1)
    max_output_tokens: int = Field(ge=1)
    request_timeout_seconds: float = Field(gt=0)
    estimated_active_cost_usd_per_second: float = Field(gt=0)
    request_overrides: dict[str, Any] = Field(default_factory=dict)

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


class SpecializedModalServingProfile(ModalServingProfile):
    min_containers: int = Field(default=1, ge=0, le=1)
    max_containers: int = Field(default=1, ge=1, le=1)
    scaledown_window_seconds: int = Field(default=60, ge=1)
    server_args: tuple[str, ...] = ()
    server_env: dict[str, str] = Field(default_factory=dict)


class PinnedDependencyProfile(SpecializedModalServingProfile):
    extra_pip_packages: tuple[str, ...] = ()


QWEN36_27B_FP8_CANDIDATE = ModalServingProfile(
    name="modal-qwen36-27b-fp8-v2",
    app_name="research-harness-qwen36-27b-fp8",
    served_model="Qwen/Qwen3.6-27B-FP8",
    checkpoint_revision="e89b16ebf1988b3d6befa7de50abc2d76f26eb09",
    config_sha256="885e6830f8d6883fefd63e3608c267452da2e6ce353a3494f42a7aa3d70c8434",
    chat_template_sha256="e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259",
    inference_engine="vllm",
    inference_engine_version="0.21.0",
    container_base="nvidia/cuda:12.9.0-devel-ubuntu22.04+python3.12",
    gpu="H200:1",
    tensor_parallel_size=1,
    dtype="auto",
    enforce_eager=True,
    deep_gemm_warmup="skip",
    gdn_prefill_backend="triton",
    safetensors_load_strategy="prefetch",
    max_model_len=32_768,
    max_output_tokens=2_048,
    request_timeout_seconds=900,
    estimated_active_cost_usd_per_second=0.00150,
    request_overrides={"chat_template_kwargs": {"enable_thinking": False}},
)

GEMMA4_31B_CANDIDATE = SpecializedModalServingProfile(
    name="modal-gemma4-31b-it-v1",
    app_name="research-harness-gemma4-31b-it",
    served_model="google/gemma-4-31B-it",
    checkpoint_revision="842da3794eaa0b77d5f08bae87a17459d91ff475",
    config_sha256="e967dd38bc5cfd38bd09a995a7bf4a754075df2b46aba68f7fbb5a791e6d8dd1",
    chat_template_sha256="ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4",
    inference_engine="vllm",
    inference_engine_version="0.21.0",
    container_base="nvidia/cuda:12.9.0-devel-ubuntu22.04+python3.12",
    gpu="H200:1",
    tensor_parallel_size=1,
    dtype="auto",
    enforce_eager=True,
    deep_gemm_warmup="skip",
    gdn_prefill_backend="triton",
    safetensors_load_strategy="prefetch",
    max_model_len=32_768,
    max_output_tokens=2_048,
    request_timeout_seconds=900,
    estimated_active_cost_usd_per_second=0.00150,
    server_args=("--language-model-only",),
    server_env={
        "HF_XET_HIGH_PERFORMANCE": "1",
        "VLLM_DEEP_GEMM_WARMUP": "skip",
    },
)

MISTRAL_SMALL_32_24B_CANDIDATE = SpecializedModalServingProfile(
    name="modal-mistral-small-3.2-24b-v1",
    app_name="research-harness-mistral-small-3-2-24b",
    served_model="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    checkpoint_revision="95a6d26c4bfb886c58daf9d3f7332c857cb27b43",
    config_sha256="01ab910a5dda7995709cc355d094eabb8094b78d49240cd167188606c3ff5edb",
    chat_template_sha256="6e2501687ccd0e1f30f36319eaf2b46958b897811e246cd8eb5d385b9e3de7d1",
    inference_engine="vllm",
    inference_engine_version="0.21.0",
    container_base="nvidia/cuda:12.9.0-devel-ubuntu22.04+python3.12",
    gpu="H200:1",
    tensor_parallel_size=1,
    dtype="auto",
    enforce_eager=True,
    deep_gemm_warmup="skip",
    gdn_prefill_backend="triton",
    safetensors_load_strategy="prefetch",
    max_model_len=32_768,
    max_output_tokens=2_048,
    request_timeout_seconds=900,
    estimated_active_cost_usd_per_second=0.00150,
    server_args=(
        "--tokenizer-mode",
        "mistral",
        "--config-format",
        "mistral",
        "--load-format",
        "mistral",
        "--language-model-only",
    ),
    server_env={
        "HF_XET_HIGH_PERFORMANCE": "1",
        "VLLM_DEEP_GEMM_WARMUP": "skip",
    },
)

GPT_OSS_20B_JUDGE_CANDIDATE = SpecializedModalServingProfile(
    name="modal-gpt-oss-20b-judge-v1",
    app_name="research-harness-gpt-oss-20b-judge",
    served_model="openai/gpt-oss-20b",
    checkpoint_revision="6cee5e81ee83917806bbde320786a8fb61efebee",
    config_sha256="3a2a26ded679375b7928ddeca59764df7cea83220c1961035f6d6e232659e9ce",
    chat_template_sha256="a4c9919cbbd4acdd51ccffe22da049264b1b73e59055fa58811a99efbd7c8146",
    inference_engine="vllm",
    inference_engine_version="0.21.0",
    container_base="nvidia/cuda:12.9.0-devel-ubuntu22.04+python3.12",
    gpu="H200:1",
    tensor_parallel_size=1,
    dtype="auto",
    enforce_eager=True,
    deep_gemm_warmup="skip",
    gdn_prefill_backend="triton",
    safetensors_load_strategy="prefetch",
    max_model_len=32_768,
    max_output_tokens=2_048,
    request_timeout_seconds=900,
    estimated_active_cost_usd_per_second=0.00150,
    request_overrides={"reasoning_effort": "low"},
    server_args=("--reasoning-parser", "openai_gptoss"),
    server_env={
        "HF_XET_HIGH_PERFORMANCE": "1",
        "VLLM_DEEP_GEMM_WARMUP": "skip",
    },
)


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


class _Message(BaseModel):
    model_config = ConfigDict(extra="ignore")

    content: str | None = None


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


class ModalOpenAIClient:
    provider = "modal"

    def __init__(
        self,
        *,
        profile: ModalServingProfile,
        endpoint_url: str,
        proxy_token_id: str,
        proxy_token_secret: str,
        http_client: httpx.AsyncClient,
    ) -> None:
        self.profile = profile
        self.model = profile.served_model
        self.endpoint_profile = profile.profile_id
        self.actual_request_count = 0
        self._endpoint_url = _validate_endpoint_url(endpoint_url)
        if not proxy_token_id.startswith("wk-") or not proxy_token_secret.startswith("ws-"):
            raise ValueError("Modal proxy token credentials have invalid prefixes")
        self._headers = {
            "Modal-Key": proxy_token_id,
            "Modal-Secret": proxy_token_secret,
        }
        self._http = http_client

    async def request(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        parameters: dict[str, Any],
    ) -> ModelResponse:
        payload: dict[str, Any] = {
            "max_tokens": self.profile.max_output_tokens,
            "messages": messages,
            "model": self.profile.served_model,
            "stream": False,
            **self.profile.request_overrides,
        }

        for name in ("max_tokens", "seed", "temperature", "top_p"):
            if name in parameters and parameters[name] is not None:
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
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise TransientModelError(type(exc).__name__) from exc
        if response.status_code in {408, 429, 502, 503, 504}:
            raise TransientModelError(f"Modal endpoint HTTP {response.status_code}")
        if response.is_error:
            raise RuntimeError(f"Modal endpoint HTTP {response.status_code}")

        completion = _ChatCompletion.model_validate(response.json())
        if completion.model != self.profile.served_model:
            raise RuntimeError(
                "Modal endpoint model identity mismatch: "
                f"expected {self.profile.served_model}, got {completion.model}"
            )
        choice = completion.choices[0]
        if not choice.message.content:
            raise RuntimeError("Modal endpoint returned no assistant content")
        elapsed = time.monotonic() - started
        prompt_details = completion.usage.prompt_tokens_details or _UsageDetails()
        completion_details = completion.usage.completion_tokens_details or _UsageDetails()
        return ModelResponse(
            content=choice.message.content,
            raw={
                "finish_reason": choice.finish_reason,
                "id": completion.id,
                "model": completion.model,
                "profile_id": self.profile.profile_id,
                "system_fingerprint": completion.system_fingerprint,
            },
            input_tokens=completion.usage.prompt_tokens,
            output_tokens=completion.usage.completion_tokens,
            cached_input_tokens=prompt_details.cached_tokens,
            reasoning_tokens=completion_details.reasoning_tokens,
            estimated_cost_usd=(elapsed * self.profile.estimated_active_cost_usd_per_second),
        )


def _validate_endpoint_url(value: str) -> str:
    parsed = urlparse(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
        or not parsed.hostname.endswith((".modal.direct", ".modal.run"))
    ):
        raise ValueError("endpoint_url must be an HTTPS Modal origin without credentials or path")
    return value.rstrip("/")


GEMMA4_31B_JUDGE_V2 = PinnedDependencyProfile(
    **{
        **GEMMA4_31B_CANDIDATE.model_dump(),
        "name": "modal-gemma4-31b-it-judge-v2",
        "app_name": "research-harness-gemma4-31b-judge",
        "extra_pip_packages": ("transformers==5.13.0",),
        "inference_engine_version": "0.27.1",
        "server_args": (
            "--language-model-only",
            "--limit-mm-per-prompt",
            '{"image":0,"video":0,"audio":0}',
        ),
    }
)
