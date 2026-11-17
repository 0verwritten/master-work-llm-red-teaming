from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from harness.agents.native.types import ToolRequest


@dataclass(frozen=True)
class ToolResult:
    name: str
    output: dict[str, Any]
    finished: bool = False


class ToolRegistry:
    definitions: tuple[dict[str, Any], ...] = (
        {
            "name": "note_finding",
            "description": "Record an evidence-backed finding.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["claim", "confidence"],
                "additionalProperties": False,
            },
        },
        {
            "name": "finish_experiment",
            "description": "Explicitly finish the experiment.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "outcome": {"type": "string", "enum": ["completed", "failed"]},
                },
                "required": ["summary", "outcome"],
                "additionalProperties": False,
            },
        },
    )

    async def execute(self, request: ToolRequest) -> ToolResult:
        if request.name == "note_finding":
            claim = request.arguments.get("claim")
            confidence = request.arguments.get("confidence")
            if not isinstance(claim, str) or not claim:
                raise ValueError("note_finding.claim must be a non-empty string")
            if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                raise ValueError("note_finding.confidence must be between 0 and 1")
            return ToolResult(
                name=request.name,
                output={"recorded": True, "claim": claim, "confidence": confidence},
            )
        if request.name == "finish_experiment":
            summary = request.arguments.get("summary")
            outcome = request.arguments.get("outcome")
            if not isinstance(summary, str) or not summary:
                raise ValueError("finish_experiment.summary must be a non-empty string")
            if outcome not in {"completed", "failed"}:
                raise ValueError("finish_experiment.outcome is invalid")
            return ToolResult(
                name=request.name,
                output={"accepted": True, "summary": summary, "outcome": outcome},
                finished=True,
            )
        raise ValueError(f"tool is not registered: {request.name}")
