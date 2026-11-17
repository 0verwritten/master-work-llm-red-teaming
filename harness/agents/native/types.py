from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class TransientModelError(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolRequest:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ModelResponse:
    content: str
    tool_calls: tuple[ToolRequest, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    estimated_cost_usd: float = 0.0


class ModelClient(Protocol):
    provider: str
    model: str
    endpoint_profile: str | None
    actual_request_count: int

    async def request(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        parameters: dict[str, Any],
    ) -> ModelResponse: ...
