from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

LEDGER_RELATIVE_PATH = Path("docs/thesis/spend-ledger.jsonl")
ATTEMPT_LOG_RELATIVE_DIR = Path(".spend-logs")

SpendBasis = Literal["estimated", "observed", "reconciled"]
SpendKind = Literal["rental", "call_batch", "storage", "adjustment"]

MODAL_ESTIMATE_UNDERCOUNT_FACTOR = 2.2


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


class SpendEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    recorded_at: str = Field(default_factory=_utc_now)
    provider: str = Field(min_length=1)
    account: str = Field(
        min_length=1, description="Workspace, team, or account the charge lands on."
    )
    kind: SpendKind
    basis: SpendBasis
    amount_usd: float = Field(ge=0.0)
    reference: str = Field(min_length=1, description="Instance id, manifest hash, or run id.")
    purpose: str = Field(min_length=1)
    evidence: str = Field(
        min_length=1, description="Document, ADR, or command that substantiates the amount."
    )
    supersedes: str | None = Field(
        default=None,
        description=(
            "Reference of an earlier entry this one replaces, for a figure restated "
            "cumulatively or corrected after reconciliation. The superseded entry stays "
            "in the file and drops out of the totals."
        ),
    )
    detail: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _reconciled_needs_evidence(self) -> SpendEntry:
        if self.basis == "reconciled" and len(self.evidence.strip()) < 8:
            raise ValueError("a reconciled amount must name the evidence that settled it")
        return self

    @property
    def entry_sha256(self) -> str:
        return hashlib.sha256(_canonical(self.model_dump(mode="json")).encode()).hexdigest()


class SpendTotals(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    estimated_usd: float = 0.0
    observed_usd: float = 0.0
    reconciled_usd: float = 0.0
    entries: int = 0
    superseded_entries: int = 0

    @property
    def best_available_usd(self) -> float:

        if self.reconciled_usd > 0.0:
            return self.reconciled_usd
        if self.observed_usd > 0.0:
            return self.observed_usd
        return self.estimated_usd


class SpendLedger:
    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def at_repository_root(cls, root: Path) -> SpendLedger:
        return cls(root / LEDGER_RELATIVE_PATH)

    def append(self, entry: SpendEntry) -> SpendEntry:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = _canonical(entry.model_dump(mode="json"))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return entry

    def entries(self) -> list[SpendEntry]:
        if not self.path.exists():
            return []
        rows: list[SpendEntry] = []
        for number, raw in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                rows.append(SpendEntry.model_validate_json(stripped))
            except ValueError as error:
                raise ValueError(
                    f"{self.path}:{number} is not a valid spend entry: {error}"
                ) from error
        return rows

    @staticmethod
    def _live(rows: Sequence[SpendEntry]) -> list[SpendEntry]:

        superseded = {row.supersedes for row in rows if row.supersedes}
        return [row for row in rows if row.reference not in superseded]

    def totals(self, entries: Sequence[SpendEntry] | None = None) -> SpendTotals:
        given = list(entries) if entries is not None else self.entries()
        rows = self._live(given)
        return SpendTotals(
            superseded_entries=len(given) - len(rows),
            estimated_usd=round(sum(r.amount_usd for r in rows if r.basis == "estimated"), 6),
            observed_usd=round(sum(r.amount_usd for r in rows if r.basis == "observed"), 6),
            reconciled_usd=round(sum(r.amount_usd for r in rows if r.basis == "reconciled"), 6),
            entries=len(rows),
        )

    def has_reference(self, reference: str) -> bool:
        return any(entry.reference == reference for entry in self.entries())

    def totals_by_account(self) -> dict[str, SpendTotals]:
        grouped: dict[str, list[SpendEntry]] = {}
        for entry in self._live(self.entries()):
            grouped.setdefault(f"{entry.provider}/{entry.account}", []).append(entry)
        return {key: self.totals(rows) for key, rows in sorted(grouped.items())}


class AttemptRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    recorded_at: str = Field(default_factory=_utc_now)
    run_id: str = Field(min_length=1)
    attempt_index: int = Field(ge=1)
    provider: str = Field(min_length=1)
    account: str = Field(min_length=1)
    role: str = Field(min_length=1, description="attacker, target, judge, or a named batch role.")
    cell: str = Field(min_length=1)
    outcome: Literal["success", "transient_error", "request_error", "parse_error", "skipped"]
    elapsed_seconds: float = Field(ge=0.0)
    estimated_cost_usd: float = Field(ge=0.0)
    llm_call_id: str | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _failed_attempts_are_free_unless_told_otherwise(self) -> AttemptRecord:
        if self.outcome == "skipped" and self.estimated_cost_usd > 0.0:
            raise ValueError(
                "a skipped attempt never reached the provider and cannot cost anything"
            )
        return self


class AttemptLog:
    def __init__(self, directory: Path, run_id: str) -> None:
        self.directory = directory
        self.run_id = run_id
        self.path = directory / f"{run_id}.jsonl"

    @classmethod
    def at_repository_root(cls, root: Path, run_id: str) -> AttemptLog:
        return cls(root / ATTEMPT_LOG_RELATIVE_DIR, run_id)

    def append(self, record: AttemptRecord) -> AttemptRecord:
        if record.run_id != self.run_id:
            raise ValueError(f"record belongs to run {record.run_id}, not {self.run_id}")
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        line = _canonical(record.model_dump(mode="json"))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.path.chmod(0o600)
        return record

    def records(self) -> Iterator[AttemptRecord]:
        if not self.path.exists():
            return
        for raw in self.path.read_text(encoding="utf-8").splitlines():
            stripped = raw.strip()
            if stripped:
                yield AttemptRecord.model_validate_json(stripped)

    def summary(self) -> dict[str, Any]:
        records = list(self.records())
        by_outcome: dict[str, int] = {}
        for record in records:
            by_outcome[record.outcome] = by_outcome.get(record.outcome, 0) + 1
        return {
            "run_id": self.run_id,
            "attempts": len(records),
            "by_outcome": dict(sorted(by_outcome.items())),
            "estimated_cost_usd": round(sum(r.estimated_cost_usd for r in records), 6),
            "elapsed_seconds": round(sum(r.elapsed_seconds for r in records), 3),
        }

    def roll_up(self, *, purpose: str, evidence: str) -> SpendEntry:

        records = list(self.records())
        if not records:
            raise ValueError(f"run {self.run_id} has no recorded attempts to roll up")
        return SpendEntry(
            provider=records[0].provider,
            account=records[0].account,
            kind="call_batch",
            basis="estimated",
            amount_usd=round(sum(r.estimated_cost_usd for r in records), 6),
            reference=self.run_id,
            purpose=purpose,
            evidence=evidence,
            detail=self.summary(),
        )


def write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    )
    try:
        handle.write(_canonical(payload) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        Path(handle.name).replace(path)
    except BaseException:
        handle.close()
        Path(handle.name).unlink(missing_ok=True)
        raise
