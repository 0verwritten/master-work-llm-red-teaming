from __future__ import annotations

import uuid
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def validate_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError("path must be a normalized relative POSIX path without traversal")
    if "\\" in value or "\x00" in value:
        raise ValueError("path contains a forbidden character")
    return str(path)


class CodeExecutionRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: uuid.UUID
    step_id: uuid.UUID
    language: Literal["python"] = "python"
    files: dict[str, str]
    entrypoint: str
    timeout_seconds: int = Field(default=600, ge=1, le=3_600)
    cpu: float = Field(default=2, gt=0, le=16)

    memory_mb: int = Field(default=4_096, ge=256, le=131_072)
    expected_artifacts: tuple[str, ...] = ()
    max_files: int = Field(default=20, ge=1, le=100)
    max_input_bytes: int = Field(default=1_000_000, ge=1)
    max_output_bytes: int = Field(default=10_000_000, ge=1)

    @model_validator(mode="after")
    def validate_submission(self) -> CodeExecutionRequest:
        if not self.files:
            raise ValueError("at least one source file is required")
        if len(self.files) > self.max_files:
            raise ValueError("submitted file count exceeds max_files")
        normalized = {validate_relative_path(path) for path in self.files}
        if len(normalized) != len(self.files):
            raise ValueError("file paths are not unique after normalization")
        entrypoint = validate_relative_path(self.entrypoint)
        if entrypoint not in normalized:
            raise ValueError("entrypoint must identify a submitted file")
        artifacts = tuple(validate_relative_path(path) for path in self.expected_artifacts)
        if len(set(artifacts)) != len(artifacts):
            raise ValueError("expected_artifacts must be unique")
        total_bytes = sum(len(content.encode()) for content in self.files.values())
        if total_bytes > self.max_input_bytes:
            raise ValueError("submitted source exceeds max_input_bytes")
        return self


type ExecutionStatus = Literal[
    "succeeded",
    "failed",
    "timed_out",
    "policy_rejected",
    "platform_error",
    "output_limit_exceeded",
]


class CodeExecutionResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    execution_id: uuid.UUID
    status: ExecutionStatus
    exit_code: int | None
    stdout: str
    stderr: str
    artifacts: dict[str, bytes] = Field(default_factory=dict)
    duration_ms: int
    resource_metadata: dict[str, str | int | float | bool]
    error_summary: str | None = None
