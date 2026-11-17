"""Destroy allow-listed project Vast rentals when runner PIDs exit or at a deadline.

Run this as an independent user service after starting the batch runners. Every
instance is re-fetched and required to carry the exact project owner label
before spend is recorded or destruction is attempted.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from harness.spend.ledger import SpendLedger
from harness.spend.vast import OWNER_LABEL, VastInstance, fetch_account, fetch_instances


@dataclass(frozen=True)
class Arm:
    name: str
    pid: int
    instance_ids: tuple[int, ...]


def parse_arm(raw: str) -> Arm:
    try:
        name, pid_raw, ids_raw = raw.split(":", maxsplit=2)
        ids = tuple(int(value) for value in ids_raw.split(","))
        arm = Arm(name=name, pid=int(pid_raw), instance_ids=ids)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "arm must be NAME:PID:INSTANCE_ID[,INSTANCE_ID]"
        ) from error
    if not arm.name or arm.pid < 1 or not arm.instance_ids or any(value < 1 for value in ids):
        raise argparse.ArgumentTypeError("arm name, PID, and instance IDs must be positive")
    return arm


def pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _owned_instances(ids: tuple[int, ...]) -> list[VastInstance]:
    wanted = set(ids)
    found = {instance.id: instance for instance in fetch_instances() if instance.id in wanted}
    missing = wanted - set(found)
    if missing:
        print(json.dumps({"event": "instances_already_absent", "ids": sorted(missing)}), flush=True)
    wrong = [instance.id for instance in found.values() if instance.label != OWNER_LABEL]
    if wrong:
        raise RuntimeError(
            f"refusing to touch instances without exact owner label {OWNER_LABEL!r}: {wrong}"
        )
    return list(found.values())


DEFAULT_PURPOSE = "full HarmBench 510 {arm} continuation"
DEFAULT_EVIDENCE = "docs/adr/0035-freeze-full-harmbench-510-mixed-review-run.md"


def finish_arm(
    arm: Arm,
    *,
    repository_root: Path,
    reason: str,
    purpose: str = DEFAULT_PURPOSE,
    evidence: str = DEFAULT_EVIDENCE,
) -> None:
    instances = _owned_instances(arm.instance_ids)
    account = fetch_account()
    ledger = SpendLedger.at_repository_root(repository_root)
    recorded = {entry.reference for entry in ledger.entries()}
    for instance in instances:
        reference = f"instance:{instance.id}"
        if reference not in recorded:
            ledger.append(
                instance.to_ledger_entry(
                    account=account.account_label,
                    purpose=purpose.format(arm=arm.name),
                    evidence=evidence,
                )
            )
        subprocess.run(
            ["vastai", "destroy", "instance", str(instance.id), "--yes"],
            check=True,
            timeout=90,
        )
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if not _owned_instances(arm.instance_ids):
            print(
                json.dumps(
                    {
                        "event": "arm_destroyed",
                        "arm": arm.name,
                        "reason": reason,
                        "ids": arm.instance_ids,
                    }
                ),
                flush=True,
            )
            return
        time.sleep(5)
    raise TimeoutError(f"instances for arm {arm.name} remained visible after destroy")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", type=parse_arm, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--poll-seconds", type=float, default=20)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--purpose", default=DEFAULT_PURPOSE, help="ledger purpose; {arm} is the arm name"
    )
    parser.add_argument("--evidence", default=DEFAULT_EVIDENCE, help="ledger evidence path")
    args = parser.parse_args()
    if args.deadline_epoch <= time.time() or args.poll_seconds < 1:
        parser.error("deadline must be in the future and poll interval must be positive")
    ids = [instance_id for arm in args.arm for instance_id in arm.instance_ids]
    if len(ids) != len(set(ids)):
        parser.error("an instance ID may belong to only one arm")
    _owned_instances(tuple(ids))
    pending = {arm.name: arm for arm in args.arm}
    print(
        json.dumps(
            {
                "event": "watchdog_started",
                "arms": sorted(pending),
                "instance_ids": sorted(ids),
                "deadline_epoch": args.deadline_epoch,
            }
        ),
        flush=True,
    )
    while pending:
        expired = time.time() >= args.deadline_epoch
        for name, arm in list(pending.items()):
            if expired or not pid_running(arm.pid):
                finish_arm(
                    arm,
                    repository_root=args.repository_root.resolve(),
                    reason="deadline" if expired else "runner_exit",
                    purpose=args.purpose,
                    evidence=args.evidence,
                )
                del pending[name]
        if pending:
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
