from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from preliminary_study.instruments import parse_verdict
from preliminary_study.protocol import Arm, text_sha256
from preliminary_study.trace import VAULT_RUNS, TraceRecord, TraceStore


@dataclass(frozen=True)
class Gate:
    name: str
    blocking: bool
    passed: bool
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "blocking": self.blocking,
            "passed": self.passed,
            "detail": self.detail,
        }


class ValidationError(RuntimeError):
    pass


def validate_records(
    records: list[TraceRecord], *, expected_cells: int | None = None
) -> list[Gate]:
    targets = {record.call_id: record for record in records if record.purpose == "target"}
    judges = [record for record in records if record.purpose == "judge"]
    gates: list[Gate] = []

    mismatches: list[str] = []
    orphans: list[str] = []
    for judge in judges:
        target = targets.get(judge.judged_call_id or "")
        if target is None:
            orphans.append(judge.call_id)
            continue
        if judge.judged_response_sha256 != target.response_sha256:
            mismatches.append(judge.call_id)

        if text_sha256(target.response_text) != target.response_sha256:
            mismatches.append(judge.call_id)
    gates.append(
        Gate(
            name="judge_binding_integrity",
            blocking=True,
            passed=not mismatches and not orphans,
            detail=(
                f"{len(judges)} judge records; {len(orphans)} orphaned, "
                f"{len(mismatches)} digest mismatches"
            ),
        )
    )

    judged_ids = Counter(judge.judged_call_id for judge in judges)
    duplicated = [call_id for call_id, count in judged_ids.items() if count > 1]
    unjudged = sorted(set(targets) - set(judged_ids))
    gates.append(
        Gate(
            name="judge_target_bijection",
            blocking=True,
            passed=not duplicated and not unjudged,
            detail=(
                f"{len(targets)} target records; {len(unjudged)} unjudged, "
                f"{len(duplicated)} judged more than once"
            ),
        )
    )

    wrong_shape = [
        record.call_id
        for record in targets.values()
        if (record.arm is Arm.IMAGE) != (record.image_sha256 is not None)
    ]
    gates.append(
        Gate(
            name="arm_request_shape",
            blocking=True,
            passed=not wrong_shape,
            detail=f"{len(wrong_shape)} target records whose arm and image presence disagree",
        )
    )

    cells = Counter((record.item_id, record.arm, record.repetition) for record in targets.values())
    repeated_cells = [cell for cell, count in cells.items() if count > 1]
    complete = expected_cells is None or len(cells) == expected_cells
    gates.append(
        Gate(
            name="design_completeness",
            blocking=True,
            passed=not repeated_cells and complete,
            detail=(
                f"{len(cells)} distinct design cells"
                + (f" of {expected_cells} expected" if expected_cells is not None else "")
                + f"; {len(repeated_cells)} duplicated"
            ),
        )
    )

    per_arm_counter: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for judge in judges:
        verdict, _ = parse_verdict(judge.response_text)
        per_arm_counter[judge.arm.value][verdict] += 1
    per_arm_distribution = {arm: dict(counts) for arm, counts in per_arm_counter.items()}
    distinct_distributions = len(
        {json.dumps(counts, sort_keys=True) for counts in per_arm_distribution.values()}
    )
    gates.append(
        Gate(
            name="arm_verdict_distributions_differ",
            blocking=False,
            passed=distinct_distributions == len(per_arm_distribution),
            detail=json.dumps(per_arm_distribution, sort_keys=True),
        )
    )

    rationale_counts = Counter(text_sha256(judge.response_text) for judge in judges)
    duplicate_rationales = sum(count - 1 for count in rationale_counts.values() if count > 1)
    share = duplicate_rationales / len(judges) if judges else 0.0
    gates.append(
        Gate(
            name="judge_rationale_duplication",
            blocking=False,
            passed=share < 0.20,
            detail=f"{duplicate_rationales}/{len(judges)} duplicated rationales ({share:.1%})",
        )
    )

    unparsed = sum(1 for judge in judges if parse_verdict(judge.response_text)[0] == "UNPARSED")
    unparsed_share = unparsed / len(judges) if judges else 0.0
    gates.append(
        Gate(
            name="judge_parse_rate",
            blocking=False,
            passed=unparsed_share < 0.05,
            detail=f"{unparsed}/{len(judges)} replies unparseable ({unparsed_share:.1%})",
        )
    )

    return gates


def validate_run(run_dir: Path) -> list[Gate]:
    store = TraceStore(run_dir / "trace.jsonl")
    manifest_path = run_dir / "manifest.json"
    expected_cells: int | None = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        expected_cells = int(manifest["planned_target_calls"])
    return validate_records(store.load(), expected_cells=expected_cells)


def require_valid(gates: list[Gate]) -> None:
    failed = [gate for gate in gates if gate.blocking and not gate.passed]
    if failed:
        summary = "; ".join(f"{gate.name}: {gate.detail}" for gate in failed)
        raise ValidationError(f"blocking gate(s) failed -> {summary}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate one preliminary-study run")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--root", type=Path, default=VAULT_RUNS)
    args = parser.parse_args()

    gates = validate_run(args.root / args.run_id)
    width = max(len(gate.name) for gate in gates)
    for gate in gates:
        mark = "PASS" if gate.passed else ("FAIL" if gate.blocking else "WARN")
        print(f"[{mark}] {gate.name:<{width}}  {gate.detail}")
    require_valid(gates)


if __name__ == "__main__":
    main()
