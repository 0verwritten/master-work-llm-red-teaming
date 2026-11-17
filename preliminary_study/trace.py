from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from preliminary_study.protocol import Arm, canonical_sha256, text_sha256

VAULT_RUNS = Path(".research-inputs/preliminary-study-runs")

CallPurpose = Literal["target", "judge"]


class TraceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: int = Field(default=1, ge=1, le=1)
    call_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    purpose: CallPurpose

    item_id: str = Field(min_length=1)
    arm: Arm
    repetition: int = Field(ge=1)

    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    image_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    seed: int = Field(ge=0)
    temperature: float = Field(ge=0.0, le=2.0)
    max_output_tokens: int = Field(ge=1)

    sampling_scope: Literal["request", "batch"] = "request"

    judged_call_id: str | None = None
    judged_response_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    provider: str = Field(min_length=1)
    endpoint_profile: str = Field(min_length=1)
    served_model: str = Field(min_length=1)
    checkpoint_revision: str = Field(min_length=1)

    response_text: str
    response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    finish_reason: str | None = None
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    latency_seconds: float = Field(ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0)
    created_at: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_binding(self) -> Self:
        if self.response_sha256 != text_sha256(self.response_text):
            raise ValueError(f"{self.call_id}: response_sha256 does not match response_text")
        has_binding = self.judged_call_id is not None and self.judged_response_sha256 is not None
        if self.purpose == "judge" and not has_binding:
            raise ValueError(f"{self.call_id}: judge record must name the response it scored")
        if self.purpose == "target" and (
            self.judged_call_id is not None or self.judged_response_sha256 is not None
        ):
            raise ValueError(f"{self.call_id}: target record must not carry a judge binding")
        if self.purpose == "target" and self.arm is Arm.IMAGE and self.image_sha256 is None:
            raise ValueError(f"{self.call_id}: image-arm target record must record its image")
        if self.purpose == "target" and self.arm is not Arm.IMAGE and self.image_sha256 is not None:
            raise ValueError(f"{self.call_id}: text-arm target record must not carry an image")
        return self


def make_call_id(
    *,
    run_id: str,
    purpose: CallPurpose,
    item_id: str,
    arm: Arm,
    repetition: int,
) -> str:

    return canonical_sha256(
        {
            "arm": arm.value,
            "item_id": item_id,
            "purpose": purpose,
            "repetition": repetition,
            "run_id": run_id,
        }
    )[:32]


def utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


class TraceStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[TraceRecord]:
        if not self.path.exists():
            return []
        records: list[TraceRecord] = []
        for line_number, line in enumerate(self.path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(TraceRecord.model_validate_json(stripped))
            except ValueError as exc:
                raise ValueError(f"{self.path}:{line_number} is not a valid trace record") from exc
        return records

    def completed_call_ids(self) -> set[str]:
        return {record.call_id for record in self.load()}

    def append(self, record: TraceRecord) -> None:

        line = record.model_dump_json() + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())


class RunLockedError(RuntimeError):
    pass


class RunLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: int | None = None

    def __enter__(self) -> RunLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._handle = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RunLockedError(
                f"{self.path} exists: another run is writing here, or a previous run was "
                f"killed. Confirm no runner is alive, then remove the file to proceed."
            ) from exc
        os.write(self._handle, f"{os.getpid()} {utcnow_iso()}\n".encode())
        return self

    def __exit__(self, *exc: object) -> None:
        if self._handle is not None:
            os.close(self._handle)
            self._handle = None
        self.path.unlink(missing_ok=True)


def write_json_atomic(path: Path, payload: Any) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    )
    try:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        handle.close()
    Path(handle.name).replace(path)
