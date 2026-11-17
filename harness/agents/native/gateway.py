from __future__ import annotations

import time
import uuid
from typing import Any

from harness.agents.native.types import ModelClient, ModelResponse
from harness.persistence.repository import TraceRepository


class TracedModelGateway:
    def __init__(self, client: ModelClient, repository: TraceRepository) -> None:
        self.client = client
        self.repository = repository

    async def request(
        self,
        *,
        run_id: uuid.UUID,
        step_id: uuid.UUID,
        purpose: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        parameters: dict[str, Any],
    ) -> ModelResponse:
        call = await self.repository.begin_llm_call(
            run_id=run_id,
            step_id=step_id,
            purpose=purpose,
            provider=self.client.provider,
            model=self.client.model,
            endpoint_profile=self.client.endpoint_profile,
            messages=messages,
            tools=tools,
            parameters=parameters,
        )

        await self.repository.session.commit()
        started = time.monotonic()
        try:
            response = await self.client.request(
                messages=messages,
                tools=tools,
                parameters=parameters,
            )
        except Exception as exc:
            latency_ms = int((time.monotonic() - started) * 1000)
            await self.repository.fail_llm_call(
                call,
                error={"type": type(exc).__name__, "message": str(exc)},
                latency_ms=latency_ms,
            )
            await self.repository.session.commit()
            raise
        latency_ms = int((time.monotonic() - started) * 1000)
        await self.repository.succeed_llm_call(
            call,
            response={"content": response.content, "raw": response.raw},
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cached_input_tokens=response.cached_input_tokens,
            reasoning_tokens=response.reasoning_tokens,
            latency_ms=latency_ms,
            estimated_cost_usd=response.estimated_cost_usd,
        )
        await self.repository.session.commit()
        return response
