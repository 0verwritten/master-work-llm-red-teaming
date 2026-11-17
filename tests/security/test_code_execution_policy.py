from __future__ import annotations

import uuid

from harness.code_execution.gateway import BackendExecutionResult, CodeExecutionGateway
from harness.code_execution.schema import CodeExecutionRequest


class FakeBackend:
    def __init__(self, result: BackendExecutionResult) -> None:
        self.result = result
        self.calls = 0

    async def execute(self, request: CodeExecutionRequest) -> BackendExecutionResult:
        self.calls += 1
        return self.result


def valid_request() -> dict[str, object]:
    return {
        "run_id": str(uuid.uuid4()),
        "step_id": str(uuid.uuid4()),
        "files": {"main.py": "print('ok')"},
        "entrypoint": "main.py",
        "timeout_seconds": 10,
        "cpu": 1,
        "memory_mb": 256,
        "max_output_bytes": 100,
    }


async def test_path_traversal_is_rejected_before_backend() -> None:
    backend = FakeBackend(
        BackendExecutionResult(
            status="succeeded",
            exit_code=0,
            stdout=b"",
            stderr=b"",
        )
    )
    request = valid_request()
    request["files"] = {"../main.py": "print('escaped')"}
    request["entrypoint"] = "../main.py"

    result = await CodeExecutionGateway(backend).execute(request)

    assert result.status == "policy_rejected"
    assert backend.calls == 0


async def test_absolute_entrypoint_is_rejected() -> None:
    backend = FakeBackend(
        BackendExecutionResult(
            status="succeeded",
            exit_code=0,
            stdout=b"",
            stderr=b"",
        )
    )
    request = valid_request()
    request["entrypoint"] = "/etc/passwd"

    result = await CodeExecutionGateway(backend).execute(request)

    assert result.status == "policy_rejected"
    assert backend.calls == 0


async def test_output_is_bounded() -> None:
    backend = FakeBackend(
        BackendExecutionResult(
            status="succeeded",
            exit_code=0,
            stdout=b"x" * 101,
            stderr=b"",
            metadata={"network_blocked": True},
        )
    )

    result = await CodeExecutionGateway(backend).execute(valid_request())

    assert result.status == "output_limit_exceeded"
    assert len(result.stdout.encode()) == 100
    assert result.resource_metadata["network_blocked"] is True
