from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from harness.spend.ledger import MODAL_ESTIMATE_UNDERCOUNT_FACTOR, SpendEntry

MODAL_CLI = "modal"


class ModalCliUnavailableError(RuntimeError):
    pass


class ModalApp(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    app_id: str = Field(min_length=1)
    name: str | None = None
    state: str | None = None
    tasks: int = 0
    created_at: str | None = None

    @property
    def is_deployed(self) -> bool:

        return (self.state or "").lower() in {"deployed", "ephemeral", "running"}

    @property
    def needs_a_decision(self) -> bool:

        return self.is_deployed and self.tasks > 0


def _run_modal(arguments: list[str], *, timeout: float = 90.0) -> str:
    if shutil.which(MODAL_CLI) is None:
        raise ModalCliUnavailableError(f"{MODAL_CLI} is not on PATH")

    completed = subprocess.run(
        [MODAL_CLI, *arguments],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[:400]
        raise ModalCliUnavailableError(f"{MODAL_CLI} {' '.join(arguments)} failed: {detail}")
    return completed.stdout


def parse_apps(raw: str) -> list[ModalApp]:
    payload: Any = json.loads(raw)
    if isinstance(payload, dict):
        payload = payload.get("apps", [])
    if not isinstance(payload, list):
        raise ValueError("expected a list of apps")
    return [ModalApp.model_validate(item) for item in payload]


def fetch_apps() -> list[ModalApp]:
    return parse_apps(_run_modal(["app", "list", "--json"]))


def estimated_entry(
    *,
    account: str,
    amount_usd: float,
    reference: str,
    purpose: str,
    evidence: str,
    calls: int | None = None,
) -> SpendEntry:

    detail: dict[str, Any] = {
        "cost_model": "request elapsed seconds x GPU rate",
        "excludes": ["image build", "checkpoint download", "warm-up", "idle scaledown"],
        "measured_undercount_factor": MODAL_ESTIMATE_UNDERCOUNT_FACTOR,
        "implied_billing_usd": round(amount_usd * MODAL_ESTIMATE_UNDERCOUNT_FACTOR, 6),
    }
    if calls is not None:
        detail["calls"] = calls
    return SpendEntry(
        provider="modal",
        account=account,
        kind="call_batch",
        basis="estimated",
        amount_usd=round(amount_usd, 6),
        reference=reference,
        purpose=purpose,
        evidence=evidence,
        detail=detail,
    )


def status_report() -> dict[str, Any]:

    apps = fetch_apps()
    deployed = [app for app in apps if app.is_deployed]
    return {
        "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "apps_total": len(apps),
        "apps_deployed": len(deployed),
        "live_tasks": sum(app.tasks for app in deployed),
        "needs_a_decision": [
            {"app_id": app.app_id, "name": app.name, "state": app.state, "tasks": app.tasks}
            for app in apps
            if app.needs_a_decision
        ],
        "note": (
            "Modal bills container-seconds. A deployed app with live tasks keeps costing "
            "between batches; stop it and confirm zero containers. The harness cost model "
            "is a floor and undercounts real billing by roughly "
            f"{MODAL_ESTIMATE_UNDERCOUNT_FACTOR}x."
        ),
    }
