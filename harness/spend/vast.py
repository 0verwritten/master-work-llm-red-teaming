from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from harness.spend.ledger import SpendEntry

VAST_CLI = "vastai"
SECONDS_PER_HOUR = 3600.0

OWNER_LABEL = "vladyslav.moiseienko.mknssh"

IDLE_ALERT_HOURS = 1.0


def launch_label_arguments() -> list[str]:

    return ["--label", OWNER_LABEL]


class VastCliUnavailableError(RuntimeError):
    pass


class VastInstance(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    id: int
    actual_status: str | None = None
    cur_state: str | None = None
    gpu_name: str | None = None
    num_gpus: int = 1
    dph_total: float = Field(default=0.0, ge=0.0, description="Total dollars per hour, all GPUs.")
    image_uuid: str | None = None
    label: str | None = None
    duration: float | None = Field(default=None, description="Seconds since the rental started.")
    disk_space: float = 0.0
    storage_cost: float = Field(default=0.0, ge=0.0, description="Storage dollars per month.")

    @property
    def is_running(self) -> bool:
        return self.actual_status == "running" or self.cur_state == "running"

    @property
    def is_ours(self) -> bool:

        return (self.label or "").strip() == OWNER_LABEL

    @property
    def rented_hours(self) -> float:
        return (self.duration or 0.0) / SECONDS_PER_HOUR

    @property
    def accrued_usd(self) -> float:

        return round(self.dph_total * self.rented_hours, 6)

    @property
    def daily_burn_usd(self) -> float:
        return round(self.dph_total * 24.0, 4)

    @property
    def needs_a_decision(self) -> bool:

        return self.is_ours and self.is_running and self.rented_hours >= IDLE_ALERT_HOURS

    def to_ledger_entry(self, *, account: str, purpose: str, evidence: str) -> SpendEntry:
        if not self.is_ours:
            raise ValueError(
                f"instance {self.id} is not labelled {OWNER_LABEL!r}; it belongs to another "
                "member of the shared team account and its cost is not this project's to record"
            )
        return SpendEntry(
            provider="vast.ai",
            account=account,
            kind="rental",
            basis="observed",
            amount_usd=self.accrued_usd,
            reference=f"instance:{self.id}",
            purpose=purpose,
            evidence=evidence,
            detail={
                "gpu_name": self.gpu_name,
                "num_gpus": self.num_gpus,
                "dph_total": self.dph_total,
                "rented_hours": round(self.rented_hours, 4),
                "image_uuid": self.image_uuid,
                "still_running": self.is_running,
                "disk_space_gb": round(self.disk_space, 2),
                "storage_cost_usd_per_month": self.storage_cost,
            },
        )


class VastAccount(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    id: int
    email: str | None = None
    credit: float = 0.0
    balance: float = 0.0
    billing_creditonly: int = 0
    is_team: bool = False

    @property
    def account_label(self) -> str:
        return f"vast:{self.id}"

    @property
    def credit_only(self) -> bool:

        return bool(self.billing_creditonly)

    @property
    def spendable_usd(self) -> float:
        return round(self.credit + self.balance, 6)


def _run_vast(arguments: list[str], *, timeout: float = 90.0) -> str:
    if shutil.which(VAST_CLI) is None:
        raise VastCliUnavailableError(f"{VAST_CLI} is not on PATH")

    completed = subprocess.run(
        [VAST_CLI, *arguments],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise VastCliUnavailableError(
            f"{VAST_CLI} {' '.join(arguments)} failed with {completed.returncode}: "
            f"{completed.stderr.strip()[:400]}"
        )
    return completed.stdout


def parse_instances(raw: str) -> list[VastInstance]:
    payload: Any = json.loads(raw)
    if isinstance(payload, dict):
        payload = payload.get("instances", [])
    if not isinstance(payload, list):
        raise ValueError("expected a list of instances")
    return [VastInstance.model_validate(item) for item in payload]


def parse_account(raw: str) -> VastAccount:
    return VastAccount.model_validate(json.loads(raw))


def fetch_instances() -> list[VastInstance]:
    return parse_instances(_run_vast(["show", "instances", "--raw"]))


def fetch_account() -> VastAccount:
    return parse_account(_run_vast(["show", "user", "--raw"]))


def status_report() -> dict[str, Any]:

    account = fetch_account()
    instances = fetch_instances()
    ours = [instance for instance in instances if instance.is_ours]
    theirs = [instance for instance in instances if not instance.is_ours]
    our_running = [instance for instance in ours if instance.is_running]
    their_running = [instance for instance in theirs if instance.is_running]
    our_burn = round(sum(instance.dph_total for instance in our_running), 6)
    return {
        "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "account": account.account_label,
        "shared_team_account": account.is_team,
        "owner_label": OWNER_LABEL,
        "credit_only": account.credit_only,
        "spendable_usd": account.spendable_usd,
        "ours": {
            "instances": len(ours),
            "running": len(our_running),
            "burn_usd_per_hour": our_burn,
            "burn_usd_per_day": round(our_burn * 24.0, 4),
            "accrued_usd": round(sum(instance.accrued_usd for instance in our_running), 6),
            "hours_of_credit_remaining": (
                round(account.spendable_usd / our_burn, 2) if our_burn > 0 else None
            ),
            "needs_a_decision": [
                {
                    "id": instance.id,
                    "gpu": instance.gpu_name,
                    "usd_per_hour": instance.dph_total,
                    "rented_hours": round(instance.rented_hours, 2),
                    "accrued_usd": instance.accrued_usd,
                    "usd_per_day": instance.daily_burn_usd,
                    "image": instance.image_uuid,
                }
                for instance in ours
                if instance.needs_a_decision
            ],
        },
        "others_on_the_shared_account": {
            "instances": len(theirs),
            "running": len(their_running),
            "burn_usd_per_hour": round(sum(i.dph_total for i in their_running), 6),
            "note": (
                "Listed for context only. Not ours to stop or destroy, and not charged "
                "to this project's ledger."
            ),
        },
    }
