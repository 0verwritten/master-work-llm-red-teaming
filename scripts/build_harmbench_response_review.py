"""Build a blind, private human-label queue from the frozen 510-pair trace.

No provider is contacted.  Every ``partial`` and ``complied`` response is
included, together with its bilingual same-round counterpart when available and
seeded category/language/round-stratified refused controls.  Automated labels,
selection reasons, session identities, and outcome strata remain private queue
metadata and are never returned by the review UI.
"""

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

from sqlalchemy import text

from apps.harmbench_response_review.workspace import (
    QUEUE_SCHEMA,
    canonical_sha256,
    subject_sha256,
)
from harness.experiments.graded_rubric import GRADED_RUBRIC_VERSION, rubric_for
from harness.persistence.database import create_engine
from harness.settings import Settings
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
DEFAULT_OUTPUT = WORKSPACE / "harmbench-response-review-queue-v1.json"
SAMPLE_SEED = "20260928-human-response-validity-v1"
REFUSED_CONTROL_FRACTION = 0.05


def _rank(item: dict[str, Any], purpose: str) -> str:
    key = f"{SAMPLE_SEED}:{purpose}:{item['annotation_id']}"
    return hashlib.sha256(key.encode()).hexdigest()


def select_items(records: list[dict[str, Any]]) -> dict[str, set[str]]:

    reasons: dict[str, set[str]] = defaultdict(set)
    paired: dict[tuple[str, int, str], str] = {}
    for item in records:
        paired[(str(item["pair_id"]), int(item["round_number"]), str(item["language"]))] = str(
            item["annotation_id"]
        )

    for item in records:
        label = str(item["automated_label"])
        if label not in {"partial", "complied"}:
            continue
        identifier = str(item["annotation_id"])
        reasons[identifier].add(f"all_automated_{label}")
        other_language = "uk" if item["language"] == "en" else "en"
        counterpart = paired.get((str(item["pair_id"]), int(item["round_number"]), other_language))
        if counterpart is not None:
            reasons[counterpart].add("bilingual_same_round_counterpart")

    strata: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        identifier = str(item["annotation_id"])
        if item["automated_label"] == "refused" and identifier not in reasons:
            strata[
                (str(item["language"]), str(item["category"]), int(item["round_number"]))
            ].append(item)
    for stratum, candidates in strata.items():
        count = min(len(candidates), max(2, math.ceil(len(candidates) * REFUSED_CONTROL_FRACTION)))
        for item in sorted(candidates, key=lambda value: _rank(value, repr(stratum)))[:count]:
            reasons[str(item["annotation_id"])].add("stratified_refused_control")
    return dict(reasons)


def _selected_sessions(
    sessions: list[Session], en_ids: set[str], uk_ids: set[str], recovery_ids: set[str]
) -> list[Session]:
    latest: dict[str, Session] = {}
    for session in sessions:
        if session.cell_id not in latest or session.created_at > latest[session.cell_id].created_at:
            latest[session.cell_id] = session
    missing = (en_ids | uk_ids | recovery_ids) - set(latest)
    if missing:
        raise RuntimeError(f"database is missing {len(missing)} frozen checkpoint cells")
    selected_en = {cell_id: latest[cell_id] for cell_id in en_ids}
    selected_uk = {cell_id: latest[cell_id] for cell_id in uk_ids}
    recoveries = {latest[cell_id].stable_id: latest[cell_id] for cell_id in recovery_ids}
    substitutions = 0
    for cell_id, session in tuple(selected_en.items()):
        if session.run_status != "completed" and session.stable_id in recoveries:
            selected_en[cell_id] = recoveries[session.stable_id]
            substitutions += 1
    if substitutions != 3:
        raise RuntimeError("exactly three English recovery substitutions are required")
    return [*selected_en.values(), *selected_uk.values()]


async def build(arguments: argparse.Namespace) -> dict[str, Any]:
    items, manifest = _load_dataset(arguments.dataset)
    en_cells = _manifest_map(items, manifest, "en")
    uk_cells = _manifest_map(items, manifest, "uk")
    en_ids = _latest_checkpoint_ids(arguments.en_checkpoint)
    uk_ids = _latest_checkpoint_ids(arguments.uk_checkpoint)
    recovery_ids = _latest_checkpoint_ids(arguments.recovery_checkpoint)
    if en_ids != set(en_cells) or uk_ids != set(uk_cells) or len(recovery_ids) != 3:
        raise RuntimeError("frozen manifest and checkpoint identities do not match")
    sessions, _, _ = await _database_rows()
    selected = _selected_sessions(sessions, en_ids, uk_ids, recovery_ids)
    selected_ids = {session.id for session in selected}
    by_session = {session.id: session for session in selected}
    by_stable = {item.stable_id: item for item in items}

    engine = create_engine(Settings().database_url)
    try:
        async with engine.connect() as connection:
            rows = (
                await connection.execute(
                    text(
                        """
                        select ar.id, ar.session_id, ar.round_number, ar.target_response,
                               e.label
                        from adversarial_rounds ar
                        join evaluations e on e.round_id = ar.id
                        join adversarial_sessions s on s.id = ar.session_id
                        where s.spec_json->>'cell_id' is not null
                        order by ar.session_id, ar.round_number
                        """
                    )
                )
            ).all()
    finally:
        await engine.dispose()

    records: list[dict[str, Any]] = []
    for row in rows:
        session_id = str(row.session_id)
        if session_id not in selected_ids or str(row.label) not in {
            "refused",
            "partial",
            "complied",
        }:
            continue
        session = by_session[session_id]
        item = by_stable[session.stable_id]
        annotation_id = hashlib.sha256(
            f"{session.cell_id}:{row.id}:{row.round_number}".encode()
        ).hexdigest()[:24]
        record = {
            "annotation_id": annotation_id,
            "pair_id": item.pair_id,
            "language": session.language,
            "category": item.category,
            "round_number": int(row.round_number),
            "requested_behavior": item.prompt_text,
            "rubric": rubric_for(session.language),
            "target_response": str(row.target_response),
            "automated_label": str(row.label),
            "session_id": session.id,
            "round_id": str(row.id),
        }
        record["subject_sha256"] = subject_sha256(record)
        records.append(record)
    if len(records) != 2943:
        raise RuntimeError(f"expected 2,943 normally labelled frozen rounds, found {len(records)}")

    reasons = select_items(records)
    queue_items: list[dict[str, Any]] = []
    for record in records:
        identifier = str(record["annotation_id"])
        if identifier not in reasons:
            continue
        queue_items.append({**record, "selection_reasons": sorted(reasons[identifier])})
    queue_items.sort(key=lambda item: _rank(item, "blind_order"))
    queue = {
        "schema_version": QUEUE_SCHEMA,
        "rubric_version": GRADED_RUBRIC_VERSION,
        "sample_seed": SAMPLE_SEED,
        "refused_control_fraction": REFUSED_CONTROL_FRACTION,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_sha256": manifest["dataset_sha256"],
        "counts": {
            "total": len(queue_items),
            "language": dict(sorted(Counter(item["language"] for item in queue_items).items())),
            "automated_label": dict(
                sorted(Counter(item["automated_label"] for item in queue_items).items())
            ),
            "category": dict(sorted(Counter(item["category"] for item in queue_items).items())),
            "selection_reason": dict(
                sorted(
                    Counter(
                        reason for item in queue_items for reason in item["selection_reasons"]
                    ).items()
                )
            ),
        },
        "items": queue_items,
    }
    queue["queue_sha256"] = canonical_sha256(queue_items)
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
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.out.exists() and not args.force:
        parser.error(f"refusing to replace frozen queue without --force: {args.out}")
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
