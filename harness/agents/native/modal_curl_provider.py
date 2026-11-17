from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from harness.agents.native.modal_provider import ModalServingProfile, _ChatCompletion, _UsageDetails
from harness.agents.native.types import ModelResponse, TransientModelError

MODAL_BIN = ".venv/bin/modal"
TRANSIENT_MARKERS = (
    "timed out",
    "connection reset",
    "temporarily unavailable",
    "no upstreams available",
    "502",
    "503",
    "504",
)


class ModalCurlClient:
    provider = "modal"
    endpoint_profile: str | None

    def __init__(
        self,
        *,
        profile: ModalServingProfile,
        endpoint_url: str,
        binary: str = MODAL_BIN,
        request_timeout_seconds: float | None = None,
    ) -> None:
        self.profile = profile
        self.model = profile.served_model
        self.endpoint_profile = profile.profile_id
        self.actual_request_count = 0
        self._endpoint_url = endpoint_url.rstrip("/")
        self._binary = binary

        self._timeout_seconds = request_timeout_seconds or profile.request_timeout_seconds

    def _payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
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
        return payload

    async def request(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        parameters: dict[str, Any],
    ) -> ModelResponse:
        payload = self._payload(messages, tools, parameters)
        args = [
            self._binary,
            "curl",
            "-sS",
            "-X",
            "POST",
            "-H",
            "Content-Type: application/json",
            "--data-binary",
            "@-",
            f"{self._endpoint_url}/v1/chat/completions",
        ]

        self.actual_request_count += 1
        started = time.monotonic()
        process = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(json.dumps(payload, ensure_ascii=False).encode()),
                timeout=self._timeout_seconds,
            )
        except TimeoutError as exc:
            process.kill()
            raise TransientModelError("modal curl timed out") from exc

        if process.returncode != 0:
            detail = stderr.decode(errors="replace").strip()
            if any(marker in detail.lower() for marker in TRANSIENT_MARKERS):
                raise TransientModelError(f"modal curl transient failure: {detail[:200]}")
            raise RuntimeError(f"modal curl failed ({process.returncode}): {detail[:400]}")

        body = stdout.decode(errors="replace").strip()
        if not body:
            raise TransientModelError("modal endpoint returned an empty body")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            if any(marker in body.lower() for marker in TRANSIENT_MARKERS):
                raise TransientModelError(f"modal endpoint transient body: {body[:200]}") from exc
            raise RuntimeError(f"modal curl returned non-JSON: {body[:400]}") from exc

        if isinstance(parsed, dict) and "choices" not in parsed:
            detail = json.dumps(parsed, ensure_ascii=False)[:200]
            if any(marker in detail.lower() for marker in TRANSIENT_MARKERS):
                raise TransientModelError(f"modal endpoint transient error: {detail}")
            raise RuntimeError(f"modal endpoint returned an error object: {detail}")
        completion = _ChatCompletion.model_validate(parsed)
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
                "transport": "modal_curl",
            },
            input_tokens=completion.usage.prompt_tokens,
            output_tokens=completion.usage.completion_tokens,
            cached_input_tokens=prompt_details.cached_tokens,
            reasoning_tokens=completion_details.reasoning_tokens,
            estimated_cost_usd=elapsed * self.profile.estimated_active_cost_usd_per_second,
        )
