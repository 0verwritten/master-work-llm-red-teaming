from __future__ import annotations

import json
import re
from typing import Any

from harness.experiments.modal_cli import ModalCli, clean_output
from harness.experiments.startup_health import inspect_logs

_CREDIT_MARKERS = ("exceeded its spend limit", "spend limit", "insufficient credit")


def diagnose(app_name: str, *, cli: ModalCli | None = None) -> dict[str, Any]:

    cli = cli or ModalCli()
    report: dict[str, Any] = {"app": app_name}

    try:
        listing = clean_output(cli.list_apps())
        row = next((ln for ln in listing.splitlines() if app_name in ln), "")
        report["app_row"] = " ".join(row.split()) or "not listed"
        report["still_deployed"] = "deployed" in row
    except RuntimeError as exc:
        report["app_row"] = f"unavailable: {exc}"[:200]

    try:
        logs = cli._run("app", "logs", app_name, timeout=180)
        health = inspect_logs(logs)
        report["logs_healthy"] = health.healthy
        report["logs_reason"] = health.reason
        plain = clean_output(logs)

        report["restarted_and_recovered"] = "Application startup complete" in plain[-4000:]
        report["log_excerpt"] = plain[-600:].strip()
    except RuntimeError as exc:
        report["logs"] = f"unavailable: {exc}"[:200]

    try:
        billing = clean_output(cli.billing_summary())
        report["credit_exhausted"] = any(m in billing.lower() for m in _CREDIT_MARKERS)
        numbers = re.findall(r"-?\d+\.\d{2,}", billing)
        report["billing_figures"] = numbers[:6]
    except RuntimeError as exc:
        report["billing"] = f"unavailable: {exc}"[:200]

    report["likely_cause"] = _classify(report)
    return report


def _classify(report: dict[str, Any]) -> str:
    if report.get("credit_exhausted"):
        return "workspace spend limit reached; refresh the window before retrying"
    if report.get("logs_healthy") is False:
        return f"container failed to start ({report.get('logs_reason')}); fix before retrying"
    if report.get("restarted_and_recovered"):
        return "container restarted and came back; retry is safe and should succeed"
    if report.get("still_deployed") is False:
        return "app is no longer deployed; redeploy before retrying"
    return "no single cause identified; inspect the log excerpt"


def render(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, ensure_ascii=False)
