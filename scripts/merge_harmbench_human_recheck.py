"""Merge a completed blind QC recheck into a preserved human review."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from apps.harmbench_response_review.workspace import (
    PROGRESS_SCHEMA,
    QUEUE_SCHEMA,
    canonical_sha256,
)

DEFAULT_SOURCE_QUEUE = Path(".translation-workspace/harmbench-response-human-sample-50-v1.json")
DEFAULT_SOURCE_REVIEW = Path(
    ".translation-workspace/harmbench-response-review-HUMAN_REVIEWER-v1.json"
)
DEFAULT_RECHECK_QUEUE = Path(
    ".translation-workspace/harmbench-response-human-recheck-queue-v1.json"
)
DEFAULT_RECHECK = Path(
    ".translation-workspace/harmbench-response-review-HUMAN_REVIEWER_RECHECK-v1.json"
)
DEFAULT_OUTPUT = Path(
    ".translation-workspace/harmbench-response-review-HUMAN_REVIEWER-confirmed-v1.json"
)


def _load(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text()))


def _validate_queue(queue: dict[str, Any]) -> None:
    if queue.get("schema_version") != QUEUE_SCHEMA:
        raise ValueError("unsupported queue schema")
    if queue.get("queue_sha256") != canonical_sha256(queue.get("items", [])):
        raise ValueError("invalid queue hash")


def merge(
    source_queue: dict[str, Any],
    source_review: dict[str, Any],
    recheck_queue: dict[str, Any],
    recheck: dict[str, Any],
) -> dict[str, Any]:
    _validate_queue(source_queue)
    _validate_queue(recheck_queue)
    if source_review.get("schema_version") != PROGRESS_SCHEMA:
        raise ValueError("unsupported source progress schema")
    if recheck.get("schema_version") != PROGRESS_SCHEMA:
        raise ValueError("unsupported recheck progress schema")
    if source_review.get("queue_sha256") != source_queue["queue_sha256"]:
        raise ValueError("source review is bound to another queue")
    if recheck.get("queue_sha256") != recheck_queue["queue_sha256"]:
        raise ValueError("recheck is bound to another queue")
    if recheck_queue.get("created_from_queue_sha256") != source_queue["queue_sha256"]:
        raise ValueError("recheck queue is not derived from the source queue")

    source_ids = {str(item["annotation_id"]) for item in source_queue["items"]}
    recheck_ids = {str(item["annotation_id"]) for item in recheck_queue["items"]}
    if set(source_review.get("reviews", {})) != source_ids:
        raise ValueError("source human review is incomplete")
    if set(recheck.get("reviews", {})) != recheck_ids:
        raise ValueError("blind recheck is incomplete")

    reviews = json.loads(json.dumps(source_review["reviews"], ensure_ascii=False))
    changes: list[dict[str, Any]] = []
    for identifier in sorted(recheck_ids):
        old = reviews[identifier]
        new = recheck["reviews"][identifier]
        if new.get("subject_sha256") != old.get("subject_sha256"):
            raise ValueError(f"subject hash changed during recheck: {identifier}")
        changes.append(
            {
                "annotation_id": identifier,
                "old_label": old["label"],
                "new_label": new["label"],
                "old_confidence": old["confidence"],
                "new_confidence": new["confidence"],
                "old_partial_subtype": old.get("partial_subtype"),
                "new_partial_subtype": new.get("partial_subtype"),
            }
        )
        replacement = json.loads(json.dumps(new, ensure_ascii=False))
        replacement["confirmed_by_blind_recheck"] = True
        replacement["initial_review"] = old
        reviews[identifier] = replacement

    now = datetime.now(UTC).isoformat()
    return {
        "schema_version": PROGRESS_SCHEMA,
        "queue_sha256": source_queue["queue_sha256"],
        "reviewer_id": f"{source_review['reviewer_id']}:confirmed",
        "reviewer_kind": "human_review_with_blind_quality_recheck",
        "source_reviewer_id": source_review["reviewer_id"],
        "source_progress_sha256": canonical_sha256(source_review),
        "recheck_reviewer_id": recheck["reviewer_id"],
        "recheck_progress_sha256": canonical_sha256(recheck),
        "created_at": now,
        "updated_at": now,
        "changes": changes,
        "reviews": reviews,
        "history": source_review.get("history", []),
    }


def _write_private(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-queue", type=Path, default=DEFAULT_SOURCE_QUEUE)
    parser.add_argument("--source-review", type=Path, default=DEFAULT_SOURCE_REVIEW)
    parser.add_argument("--recheck-queue", type=Path, default=DEFAULT_RECHECK_QUEUE)
    parser.add_argument("--recheck", type=Path, default=DEFAULT_RECHECK)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.out.exists() and not args.force:
        parser.error(f"refusing to replace merged review without --force: {args.out}")
    result = merge(
        _load(args.source_queue),
        _load(args.source_review),
        _load(args.recheck_queue),
        _load(args.recheck),
    )
    _write_private(args.out, result)
    print(
        json.dumps(
            {
                "path": str(args.out),
                "changed": sum(
                    change["old_label"] != change["new_label"] for change in result["changes"]
                ),
                "rechecked": len(result["changes"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
