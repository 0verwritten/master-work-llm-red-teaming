from __future__ import annotations

import asyncio
from typing import Any, Literal

import modal

from harness.code_execution.gateway import BackendExecutionResult
from harness.code_execution.schema import CodeExecutionRequest


class ModalSandboxBackend:
    def __init__(self, *, app_name: str) -> None:
        self.app_name = app_name

    async def connectivity(self) -> dict[str, str | bool]:
        return await asyncio.to_thread(self._connectivity_sync)

    def _connectivity_sync(self) -> dict[str, str | bool]:
        app = modal.App.lookup(self.app_name, create_if_missing=True)
        sandbox: Any = None
        try:
            sandbox = modal.Sandbox.create(
                "true",
                app=app,
                timeout=30,
                cpu=0.125,
                memory=128,
                block_network=True,
                secrets=[],
                volumes={},
                network_file_systems={},
            )
            sandbox.wait()
            return {
                "connected": sandbox.returncode == 0,
                "app_name": self.app_name,
                "sandbox_id": sandbox.object_id,
            }
        finally:
            if sandbox is not None:
                sandbox.detach()

    async def execute(self, request: CodeExecutionRequest) -> BackendExecutionResult:
        return await asyncio.to_thread(self._execute_sync, request)

    def _execute_sync(self, request: CodeExecutionRequest) -> BackendExecutionResult:
        app = modal.App.lookup(self.app_name, create_if_missing=True)
        image = modal.Image.debian_slim(python_version="3.12")
        sandbox: Any = None
        try:
            sandbox = modal.Sandbox.create(
                "sleep",
                "infinity",
                app=app,
                image=image,
                timeout=request.timeout_seconds + 20,
                idle_timeout=request.timeout_seconds + 10,
                cpu=(request.cpu, request.cpu),
                memory=(request.memory_mb, request.memory_mb),
                block_network=True,
                secrets=[],
                volumes={},
                network_file_systems={},
                tags={"run_id": str(request.run_id), "step_id": str(request.step_id)},
            )
            for relative_path, content in request.files.items():
                sandbox.filesystem.write_text(content, f"/workspace/{relative_path}")

            process = sandbox.exec(
                "python",
                f"/workspace/{request.entrypoint}",
                timeout=request.timeout_seconds,
            )
            stdout = process.stdout.read()
            stderr = process.stderr.read()
            process.wait()

            artifacts: dict[str, bytes] = {}
            for relative_path in request.expected_artifacts:
                remote_path = f"/workspace/{relative_path}"
                info = sandbox.filesystem.stat(remote_path)
                if info.size > request.max_output_bytes:
                    raise RuntimeError(f"artifact is too large: {relative_path}")
                artifacts[relative_path] = sandbox.filesystem.read_bytes(remote_path)

            status: Literal["succeeded", "failed"] = (
                "succeeded" if process.returncode == 0 else "failed"
            )
            return BackendExecutionResult(
                status=status,
                exit_code=process.returncode,
                stdout=_to_bytes(stdout),
                stderr=_to_bytes(stderr),
                artifacts=artifacts,
                metadata={
                    "platform": "modal",
                    "app_name": self.app_name,
                    "sandbox_id": sandbox.object_id,
                    "network_blocked": True,
                    "cpu_limit": request.cpu,
                    "memory_limit_mb": request.memory_mb,
                    "timeout_seconds": request.timeout_seconds,
                    "image_profile": "modal-debian-slim-python-3.12",
                },
            )
        except modal.exception.SandboxTimeoutError as exc:
            return BackendExecutionResult(
                status="timed_out",
                exit_code=None,
                stdout=b"",
                stderr=b"",
                metadata={"platform": "modal", "network_blocked": True},
                error_summary=str(exc),
            )
        finally:
            if sandbox is not None:
                try:
                    sandbox.terminate()
                finally:
                    sandbox.detach()


def _to_bytes(value: str | bytes) -> bytes:
    return value if isinstance(value, bytes) else value.encode()
