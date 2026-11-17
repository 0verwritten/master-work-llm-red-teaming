from __future__ import annotations

from typing import Any

from harness.agents.native.types import ModelResponse, ToolRequest


class DeterministicMockClient:
    def __init__(
        self,
        responses: list[ModelResponse | Exception] | None = None,
        *,
        model: str = "deterministic-v1",
    ) -> None:
        self.provider = "mock"
        self.model = model
        self.endpoint_profile: str | None = None
        self.actual_request_count = 0
        self._responses = responses or [
            ModelResponse(
                content="Record the test finding.",
                tool_calls=(
                    ToolRequest(
                        name="note_finding",
                        arguments={
                            "claim": "The trace boundary records every mock request.",
                            "confidence": 1.0,
                        },
                    ),
                ),
                raw={"id": "mock-1"},
                input_tokens=20,
                output_tokens=10,
            ),
            ModelResponse(
                content="The experiment is complete.",
                tool_calls=(
                    ToolRequest(
                        name="finish_experiment",
                        arguments={
                            "summary": "Trace persistence invariant demonstrated.",
                            "outcome": "completed",
                        },
                    ),
                ),
                raw={"id": "mock-2"},
                input_tokens=35,
                output_tokens=12,
            ),
        ]

    async def request(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        parameters: dict[str, Any],
    ) -> ModelResponse:
        del messages, tools, parameters
        index = self.actual_request_count
        self.actual_request_count += 1
        if index >= len(self._responses):
            return ModelResponse(content="No completion tool was provided.", raw={"id": "mock-end"})
        result = self._responses[index]
        if isinstance(result, Exception):
            raise result
        return result
