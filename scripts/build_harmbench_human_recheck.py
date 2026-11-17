"""Build a blind recheck queue from private human-review quality notes."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from apps.harmbench_response_review.workspace import QUEUE_SCHEMA, canonical_sha256

DEFAULT_QUEUE = Path(".translation-workspace/harmbench-response-human-sample-50-v1.json")
DEFAULT_NOTES = Path(".translation-workspace/harmbench-response-human-quality-notes-v1.json")
DEFAULT_OUTPUT = Path(".translation-workspace/harmbench-response-human-recheck-queue-v1.json")


def _load(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text()))


def build(queue: dict[str, Any], notes: dict[str, Any]) -> dict[str, Any]:
    if queue.get("schema_version") != QUEUE_SCHEMA:
        raise ValueError("unsupported source queue schema")
    if queue.get("queue_sha256") != canonical_sha256(queue.get("items", [])):
        raise ValueError("invalid source queue hash")
    if notes.get("queue_sha256") != queue["queue_sha256"]:
        raise ValueError("quality notes are bound to another queue")
    note_by_id = {str(note["annotation_id"]): note for note in notes.get("notes", [])}
    source_by_id = {str(item["annotation_id"]): item for item in queue["items"]}
    if not note_by_id or not set(note_by_id) <= set(source_by_id):
        raise ValueError("quality notes contain no valid subjects")

    items = []
    for identifier in sorted(note_by_id):
        item = source_by_id[identifier]
        note = note_by_id[identifier]
        items.append(
            {
                **item,
                "quality_control_reason": note["concern"],
                "previous_human_label": note["current_human_label"],
                "suggested_recheck": note["suggested_recheck"],
            }
        )
    result = {
        "schema_version": QUEUE_SCHEMA,
        "rubric_version": queue["rubric_version"],
        "sample_design": "blind_human_quality_control_recheck",
        "created_at": datetime.now(UTC).isoformat(),
        "created_from_queue_sha256": queue["queue_sha256"],
        "items": items,
    }
    result["queue_sha256"] = canonical_sha256(items)
    return result


def _write_private(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--notes", type=Path, default=DEFAULT_NOTES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.out.exists() and not args.force:
        parser.error(f"refusing to replace recheck queue without --force: {args.out}")
    result = build(_load(args.queue), _load(args.notes))
    _write_private(args.out, result)
    print(
        json.dumps(
            {
                "path": str(args.out),
                "queue_sha256": result["queue_sha256"],
                "total": len(result["items"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
