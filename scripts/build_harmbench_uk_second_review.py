"""Freeze Khrystyna's blind, outcome-stratified second-review queue."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from apps.harmbench_uk_second_review.workspace import (
    QUEUE_SCHEMA,
    REVIEWER_ID,
    canonical_sha256,
    subject_sha256,
)
from scripts.analyze_full_harmbench import (
    DEFAULT_DATASET,
    DEFAULT_EN_CHECKPOINT,
    DEFAULT_RECOVERY_CHECKPOINT,
    DEFAULT_UK_CHECKPOINT,
    Session,
    _database_rows,
    _latest_checkpoint_ids,
    _load_dataset,
    _manifest_map,
)

WORKSPACE = Path(".translation-workspace")
DEFAULT_PROPOSAL = WORKSPACE / "harmbench-uk-full-510-review-proposal-v1.json"
DEFAULT_FIRST_REVIEW = WORKSPACE / "harmbench-uk-human-review-all-510-v1.json"
DEFAULT_PRIOR_REVIEW = WORKSPACE / "harmbench-uk-reviewed-translations-v1.json"
DEFAULT_OUTPUT = WORKSPACE / "harmbench-uk-second-review-queue-v1.json"
SAMPLE_SEED = "20260928-khrystyna-independent-review-v1"
CONTROL_FRACTION = 0.10


def _rank(label: str, pair_id: str) -> str:
    return hashlib.sha256(f"{SAMPLE_SEED}:{label}:{pair_id}".encode()).hexdigest()


def select_pairs(
    *,
    category_by_pair: dict[str, str],
    outcome_by_pair: dict[str, str],
    corrected_pairs: set[str],
) -> dict[str, set[str]]:

    reasons: dict[str, set[str]] = defaultdict(set)
    for pair_id in corrected_pairs:
        reasons[pair_id].add("first_reviewer_corrected")
    for pair_id, outcome in outcome_by_pair.items():
        if outcome in {"en_only", "uk_only"}:
            reasons[pair_id].add(outcome)

    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for pair_id, outcome in outcome_by_pair.items():
        if outcome in {"both", "neither"} and pair_id not in reasons:
            strata[(category_by_pair[pair_id], outcome)].append(pair_id)
    for (category, outcome), candidates in strata.items():
        count = min(len(candidates), max(2, math.ceil(len(candidates) * CONTROL_FRACTION)))
        ordered = sorted(candidates, key=lambda pair_id: _rank(f"{category}:{outcome}", pair_id))
        for pair_id in ordered[:count]:
            reasons[pair_id].add(f"stratified_control:{outcome}")
    return dict(reasons)


def collect_corrected_pairs(
    decisions: dict[str, dict[str, Any]], prior_records: list[dict[str, Any]]
) -> tuple[set[str], set[str], set[str]]:

    current = {
        pair_id
        for pair_id, decision in decisions.items()
        if bool(decision.get("changed_from_machine"))
    }
    historical = {
        str(record["pair_id"])
        for record in prior_records
        if record.get("translation_provenance") == "machine_deepl_human_corrected"
        or record.get("selection") in {"accept_proposal", "edit_manual"}
    }
    return current | historical, current, historical


async def _outcomes(arguments: argparse.Namespace) -> tuple[dict[str, str], dict[str, str]]:
    items, dataset_manifest = _load_dataset(arguments.dataset)
    en_cells = _manifest_map(items, dataset_manifest, "en")
    uk_cells = _manifest_map(items, dataset_manifest, "uk")
    en_ids = _latest_checkpoint_ids(arguments.en_checkpoint)
    uk_ids = _latest_checkpoint_ids(arguments.uk_checkpoint)
    recovery_ids = _latest_checkpoint_ids(arguments.recovery_checkpoint)
    if en_ids != set(en_cells) or uk_ids != set(uk_cells) or len(recovery_ids) != 3:
        raise ValueError("frozen checkpoint identities do not match the 510-pair analysis")
    sessions, _, _ = await _database_rows()
    latest: dict[str, Session] = {}
    for session in sessions:
        if session.cell_id not in latest or session.created_at > latest[session.cell_id].created_at:
            latest[session.cell_id] = session
    missing = (en_ids | uk_ids | recovery_ids) - set(latest)
    if missing:
        raise ValueError(f"database is missing {len(missing)} frozen checkpoint cells")
    selected_en = {cell_id: latest[cell_id] for cell_id in en_ids}
    selected_uk = {cell_id: latest[cell_id] for cell_id in uk_ids}
    recovery = {latest[cell_id].stable_id: latest[cell_id] for cell_id in recovery_ids}
    substituted = 0
    for cell_id, session in tuple(selected_en.items()):
        if session.run_status != "completed" and session.stable_id in recovery:
            selected_en[cell_id] = recovery[session.stable_id]
            substituted += 1
    if substituted != 3:
        raise ValueError("exactly three English recovery substitutions are required")
    pair_by_stable = {item.stable_id: item.pair_id for item in items}
    arms: dict[str, dict[str, Session]] = defaultdict(dict)
    for session in [*selected_en.values(), *selected_uk.values()]:
        arms[pair_by_stable[session.stable_id]][session.language] = session
    outcomes: dict[str, str] = {}
    for pair_id, pair_arms in arms.items():
        if set(pair_arms) != {"en", "uk"} or any(
            arm.run_status != "completed" for arm in pair_arms.values()
        ):
            outcomes[pair_id] = "indeterminate"
            continue
        en = pair_arms["en"].stop_reason == "validated_success"
        uk = pair_arms["uk"].stop_reason == "validated_success"
        outcomes[pair_id] = (
            "both" if en and uk else "en_only" if en else "uk_only" if uk else "neither"
        )
    categories = {item.pair_id: item.category for item in items if item.language == "uk"}
    return outcomes, categories


async def build(arguments: argparse.Namespace) -> dict[str, Any]:
    proposal = json.loads(arguments.proposal.read_text())
    overlay = json.loads(arguments.first_review.read_text())
    prior_review = json.loads(arguments.prior_review.read_text())
    records = {str(row["pair_id"]): row for row in proposal["records"]}
    decisions = overlay.get("decisions", {})
    if set(decisions) != set(records):
        raise ValueError("first review must contain decisions for all 510 proposal records")
    outcomes, categories = await _outcomes(arguments)
    corrected, current_corrected, historical_corrected = collect_corrected_pairs(
        decisions, prior_review.get("records", [])
    )
    unknown_corrected = corrected - set(records)
    if unknown_corrected:
        raise ValueError("a corrected pair is absent from the full proposal")
    for record in prior_review.get("records", []):
        pair_id = str(record.get("pair_id", ""))
        if pair_id not in historical_corrected:
            continue
        if record.get("final_content_sha256") != decisions[pair_id]["final_content_sha256"]:
            raise ValueError(f"historical correction changed in the full review: {pair_id}")
    reasons = select_pairs(
        category_by_pair=categories,
        outcome_by_pair=outcomes,
        corrected_pairs=corrected,
    )
    items: list[dict[str, Any]] = []
    for pair_id in reasons:
        source = records[pair_id]
        decision = decisions[pair_id]
        if decision["machine_content_sha256"] != source["machine_content_sha256"]:
            raise ValueError(f"stale first review: {pair_id}")
        item = {
            "pair_id": pair_id,
            "category": source["category"],
            "expected_behavior": source["expected_behavior"],
            "english_content_sha256": source["english_content_sha256"],
            "english": source["english"],
            "ukrainian": decision["final"],
            "first_review_content_sha256": decision["final_content_sha256"],
            "selection_reasons": sorted(reasons[pair_id]),
            "outcome_stratum": outcomes.get(pair_id, "indeterminate"),
        }
        item["subject_sha256"] = subject_sha256(item)
        items.append(item)
    items.sort(key=lambda item: _rank("blind-order", str(item["pair_id"])))
    reason_counts = Counter(reason for values in reasons.values() for reason in values)
    queue = {
        "schema_version": QUEUE_SCHEMA,
        "reviewer_id": REVIEWER_ID,
        "sample_seed": SAMPLE_SEED,
        "control_fraction": CONTROL_FRACTION,
        "created_at": datetime.now(UTC).isoformat(),
        "source_proposal_sha256": proposal["collection_sha256"],
        "source_first_review_updated_at": overlay["updated_at"],
        "counts": {
            "total": len(items),
            "all_corrected": len(corrected),
            "current_overlay_corrected": len(current_corrected),
            "historical_corrected": len(historical_corrected),
            "selection_reasons": dict(sorted(reason_counts.items())),
            "categories": dict(sorted(Counter(item["category"] for item in items).items())),
            "outcomes": dict(sorted(Counter(item["outcome_stratum"] for item in items).items())),
        },
        "items": items,
    }
    queue["queue_sha256"] = canonical_sha256(items)
    return queue


def _write_private(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--en-checkpoint", type=Path, default=DEFAULT_EN_CHECKPOINT)
    parser.add_argument("--uk-checkpoint", type=Path, default=DEFAULT_UK_CHECKPOINT)
    parser.add_argument("--recovery-checkpoint", type=Path, default=DEFAULT_RECOVERY_CHECKPOINT)
    parser.add_argument("--proposal", type=Path, default=DEFAULT_PROPOSAL)
    parser.add_argument("--first-review", type=Path, default=DEFAULT_FIRST_REVIEW)
    parser.add_argument("--prior-review", type=Path, default=DEFAULT_PRIOR_REVIEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.out.exists() and not args.force:
        parser.error(f"refusing to replace frozen queue without --force: {args.out}")
    for path in (
        args.dataset,
        args.en_checkpoint,
        args.uk_checkpoint,
        args.recovery_checkpoint,
        args.proposal,
        args.first_review,
        args.prior_review,
    ):
        if not path.exists():
            parser.error(f"missing input: {path}")
    queue = asyncio.run(build(args))
    _write_private(args.out, queue)
    print(
        json.dumps(
            {"path": str(args.out), "queue_sha256": queue["queue_sha256"], **queue["counts"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
