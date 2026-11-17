from __future__ import annotations

import base64
import hashlib
import json
import random
import time
from dataclasses import dataclass
from typing import Any, Literal
from typing import Protocol as TypingProtocol

import httpx

from harness.agents.native.modal_provider import (
    GEMMA4_31B_CANDIDATE,
    ModalOpenAIClient,
    ModalServingProfile,
    SpecializedModalServingProfile,
)


class PreliminaryServingProfile(SpecializedModalServingProfile):
    extra_pip_packages: tuple[str, ...] = ()

    sampling_scope: Literal["request", "batch"] = "request"


GEMMA4_31B_VISION_CANDIDATE = PreliminaryServingProfile(
    **{
        **GEMMA4_31B_CANDIDATE.model_dump(),
        "name": "modal-gemma4-31b-it-vision-v1",
        "app_name": "research-harness-gemma4-31b-it-vision",
        "server_args": ("--limit-mm-per-prompt", '{"image":1,"video":0,"audio":0}'),
        "inference_engine_version": "0.27.1",
        "extra_pip_packages": ("transformers==5.15.0",),
    }
)


@dataclass(frozen=True)
class ChatRequest:
    messages: list[dict[str, Any]]
    seed: int
    temperature: float
    max_output_tokens: int

    @property
    def messages_sha256(self) -> str:

        return _canonical_sha256(self.messages)

    @property
    def request_sha256(self) -> str:

        return _canonical_sha256(
            {
                "max_output_tokens": self.max_output_tokens,
                "messages_sha256": self.messages_sha256,
                "seed": self.seed,
                "temperature": self.temperature,
            }
        )


@dataclass(frozen=True)
class ChatCompletion:
    content: str
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    estimated_cost_usd: float


class ChatProvider(TypingProtocol):
    provider_name: str
    profile: ModalServingProfile

    async def complete(self, request: ChatRequest) -> ChatCompletion: ...


def _canonical_sha256(payload: Any) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def image_content_part(png_bytes: bytes) -> dict[str, Any]:

    encoded = base64.b64encode(png_bytes).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}


class ModalFunctionChatProvider:
    provider_name = "modal"

    def __init__(self, *, profile: ModalServingProfile, app_name: str | None = None) -> None:
        import modal

        self.profile = profile
        self.actual_request_count = 0
        server_cls = modal.Cls.from_name(app_name or profile.app_name, "Server")
        self._server = server_cls()

    def _payload(self, request: ChatRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "max_tokens": min(request.max_output_tokens, self.profile.max_output_tokens),
            "messages": request.messages,
            "model": self.profile.served_model,
            "seed": request.seed,
            "stream": False,
            "temperature": request.temperature,
            "top_p": 1.0,
            **self.profile.request_overrides,
        }
        return payload

    async def complete(self, request: ChatRequest) -> ChatCompletion:
        self.actual_request_count += 1
        started = time.monotonic()
        raw = await self._server.chat.remote.aio(self._payload(request))
        elapsed = time.monotonic() - started

        served = raw.get("model")
        if served != self.profile.served_model:
            raise RuntimeError(
                f"Modal served model identity mismatch: expected "
                f"{self.profile.served_model}, got {served}"
            )
        choice = raw["choices"][0]
        content = choice["message"].get("content") or ""
        if not content:
            content = ""
        usage = raw.get("usage") or {}
        return ChatCompletion(
            content=content,
            finish_reason=choice.get("finish_reason"),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            latency_seconds=elapsed,
            estimated_cost_usd=elapsed * self.profile.estimated_active_cost_usd_per_second,
        )


class ModalChatProvider:
    provider_name = "modal"

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
        self._client = ModalOpenAIClient(
            profile=profile,
            endpoint_url=endpoint_url,
            proxy_token_id=proxy_token_id,
            proxy_token_secret=proxy_token_secret,
            http_client=http_client,
        )

    @property
    def actual_request_count(self) -> int:
        return self._client.actual_request_count

    async def complete(self, request: ChatRequest) -> ChatCompletion:
        started = time.monotonic()
        profile_limit = self.profile.max_output_tokens
        response = await self._client.request(
            messages=request.messages,
            tools=[],
            parameters={
                "max_tokens": min(request.max_output_tokens, profile_limit),
                "seed": request.seed,
                "temperature": request.temperature,
                "top_p": 1.0,
            },
        )
        return ChatCompletion(
            content=response.content,
            finish_reason=str(response.raw.get("finish_reason") or ""),
            prompt_tokens=response.input_tokens,
            completion_tokens=response.output_tokens,
            latency_seconds=time.monotonic() - started,
            estimated_cost_usd=response.estimated_cost_usd,
        )


class MockChatProvider:
    provider_name = "mock"

    def __init__(self, *, profile: ModalServingProfile, comply_rate: float = 0.25) -> None:
        self.profile = profile
        self.comply_rate = comply_rate
        self.actual_request_count = 0

    async def complete(self, request: ChatRequest) -> ChatCompletion:
        self.actual_request_count += 1
        digest = request.messages_sha256
        rng = random.Random(f"{digest}:{request.seed}")

        if _is_judge_request(request.messages):
            verdict = rng.choices(
                ["REFUSED", "PARTIAL", "COMPLIED"], weights=[0.6, 0.25, 0.15], k=1
            )[0]
            reason = f"mock judgement for request {digest[:12]}"
            content = json.dumps({"verdict": verdict, "reason": reason})
        elif rng.random() < self.comply_rate:
            content = f"Sure, here is the information you asked for. [mock {digest[:12]}]"
        else:
            content = f"I'm sorry, I can't help with that request. [mock {digest[:12]}]"

        return ChatCompletion(
            content=content,
            finish_reason="stop",
            prompt_tokens=len(digest),
            completion_tokens=len(content) // 4,
            latency_seconds=0.0,
            estimated_cost_usd=0.0,
        )


def _is_judge_request(messages: list[dict[str, Any]]) -> bool:
    return any(message.get("role") == "system" for message in messages)
