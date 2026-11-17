from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CHECKPOINT_VERSION = "batch-checkpoint-v1"
RETRYABLE_STOP_REASONS = frozenset({"role_request_error"})


@dataclass(frozen=True)
class CheckpointRecord:
    cell_id: str
    language: str
    category: str
    status: str
    judge_language_adherence: str | None = None
    label: str | None = None
    rationale_sha256: str | None = None
    estimated_cost_usd: float = 0.0
    stop_reason: str | None = None
    error: str | None = None
    recorded_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def as_json(self) -> dict[str, Any]:
        return {
            "checkpoint_version": CHECKPOINT_VERSION,
            "cell_id": self.cell_id,
            "language": self.language,
            "category": self.category,
            "status": self.status,
            "judge_language_adherence": self.judge_language_adherence,
            "label": self.label,
            "rationale_sha256": self.rationale_sha256,
            "estimated_cost_usd": self.estimated_cost_usd,
            "stop_reason": self.stop_reason,
            "error": self.error,
            "recorded_at": self.recorded_at,
        }


class CheckpointLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: CheckpointRecord) -> None:

        line = json.dumps(record.as_json(), ensure_ascii=False, sort_keys=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            self.path.chmod(0o600)
        except OSError:
            pass

    def records(self) -> list[CheckpointRecord]:

        if not self.path.exists():
            return []
        loaded: list[CheckpointRecord] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
                continue
            loaded.append(
                CheckpointRecord(
                    cell_id=str(payload["cell_id"]),
                    language=str(payload["language"]),
                    category=str(payload["category"]),
                    status=str(payload["status"]),
                    judge_language_adherence=payload.get("judge_language_adherence"),
                    label=payload.get("label"),
                    rationale_sha256=payload.get("rationale_sha256"),
                    estimated_cost_usd=float(payload.get("estimated_cost_usd") or 0.0),
                    stop_reason=payload.get("stop_reason"),
                    error=payload.get("error"),
                    recorded_at=str(payload.get("recorded_at", "")),
                )
            )
        return loaded

    def completed_cell_ids(self) -> set[str]:

        return {
            record.cell_id
            for record in self.records()
            if record.status == "completed"
            or (record.status == "stopped" and record.stop_reason not in RETRYABLE_STOP_REASONS)
        }

    def observed_cost_usd(self) -> float:
        return sum(record.estimated_cost_usd for record in self.records())

    def summary(self) -> dict[str, Any]:
        records = self.records()
        by_status: dict[str, int] = {}
        for record in records:
            by_status[record.status] = by_status.get(record.status, 0) + 1
        return {
            "path": str(self.path),
            "records": len(records),
            "by_status": dict(sorted(by_status.items())),
            "completed_cells": len(self.completed_cell_ids()),
            "observed_cost_usd": round(self.observed_cost_usd(), 6),
        }
