from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal, Protocol

from pydantic import ValidationError

from harness.code_execution.schema import CodeExecutionRequest, CodeExecutionResult


@dataclass(frozen=True)
class BackendExecutionResult:
    status: Literal["succeeded", "failed", "timed_out", "platform_error"]
    exit_code: int | None
    stdout: bytes
    stderr: bytes
    artifacts: dict[str, bytes] = field(default_factory=dict)
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)
    error_summary: str | None = None


class SandboxBackend(Protocol):
    async def execute(self, request: CodeExecutionRequest) -> BackendExecutionResult: ...


class CodeExecutionGateway:
    def __init__(self, backend: SandboxBackend) -> None:
        self.backend = backend

    async def execute(
        self, request: CodeExecutionRequest | dict[str, object]
    ) -> CodeExecutionResult:
        execution_id = uuid.uuid4()
        started = time.monotonic()
        try:
            validated = (
                request
                if isinstance(request, CodeExecutionRequest)
                else CodeExecutionRequest.model_validate(request)
            )
        except ValidationError as exc:
            return CodeExecutionResult(
                execution_id=execution_id,
                status="policy_rejected",
                exit_code=None,
                stdout="",
                stderr="",
                duration_ms=int((time.monotonic() - started) * 1000),
                resource_metadata={"network_blocked": True},
                error_summary=str(exc),
            )

        try:
            backend_result = await asyncio.wait_for(
                self.backend.execute(validated),
                timeout=validated.timeout_seconds + 30,
            )
        except TimeoutError:
            return CodeExecutionResult(
                execution_id=execution_id,
                status="timed_out",
                exit_code=None,
                stdout="",
                stderr="",
                duration_ms=int((time.monotonic() - started) * 1000),
                resource_metadata={"network_blocked": True},
                error_summary="sandbox gateway timeout",
            )
        except Exception as exc:
            return CodeExecutionResult(
                execution_id=execution_id,
                status="platform_error",
                exit_code=None,
                stdout="",
                stderr="",
                duration_ms=int((time.monotonic() - started) * 1000),
                resource_metadata={"network_blocked": True},
                error_summary=f"{type(exc).__name__}: {exc}",
            )

        stdout, stdout_truncated = _bounded_decode(
            backend_result.stdout, validated.max_output_bytes
        )
        remaining = max(validated.max_output_bytes - len(stdout.encode()), 0)
        stderr, stderr_truncated = _bounded_decode(backend_result.stderr, remaining)
        artifact_bytes = sum(len(value) for value in backend_result.artifacts.values())
        output_exceeded = (
            stdout_truncated or stderr_truncated or artifact_bytes > validated.max_output_bytes
        )
        status = "output_limit_exceeded" if output_exceeded else backend_result.status
        artifacts = {} if artifact_bytes > validated.max_output_bytes else backend_result.artifacts
        return CodeExecutionResult(
            execution_id=execution_id,
            status=status,
            exit_code=backend_result.exit_code,
            stdout=stdout,
            stderr=stderr,
            artifacts=artifacts,
            duration_ms=int((time.monotonic() - started) * 1000),
            resource_metadata=backend_result.metadata,
            error_summary=backend_result.error_summary,
        )


def _bounded_decode(value: bytes, limit: int) -> tuple[str, bool]:
    truncated = len(value) > limit
    bounded = value[:limit]
    return bounded.decode(errors="replace"), truncated
