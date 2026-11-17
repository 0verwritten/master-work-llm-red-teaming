from __future__ import annotations

import hashlib
import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

QUEUE_SCHEMA = "harmbench-response-review-queue-v1"
PROGRESS_SCHEMA = "harmbench-response-review-progress-v1"
LABELS = ("refused", "partial", "complied")
CONFIDENCE = ("low", "medium", "high")
PARTIAL_SUBTYPES = (
    "usable_but_incomplete",
    "hedged_but_enabling",
    "began_then_stopped",
    "ambiguous_boundary",
    "other",
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


def subject_sha256(item: dict[str, Any]) -> str:
    return canonical_sha256(
        {
            "language": item["language"],
            "requested_behavior": item["requested_behavior"],
            "rubric": item["rubric"],
            "target_response": item["target_response"],
        }
    )


class ResponseReviewWorkspace:
    def __init__(self, queue_path: Path, progress_path: Path, reviewer_id: str) -> None:
        self.queue_path = queue_path
        self.progress_path = progress_path
        self.reviewer_id = reviewer_id.strip()
        if not self.reviewer_id:
            raise ValueError("reviewer ID is required")
        self._lock = threading.RLock()
        self._queue = cast(dict[str, Any], json.loads(queue_path.read_text()))
        self._validate_queue()
        self._by_id = {str(item["annotation_id"]): item for item in self._queue["items"]}
        self._state = self._load_or_initialize()

    def _validate_queue(self) -> None:
        if self._queue.get("schema_version") != QUEUE_SCHEMA:
            raise ValueError("unsupported response-review queue schema")
        items = self._queue.get("items")
        if not isinstance(items, list) or not items:
            raise ValueError("response-review queue is empty")
        identifiers = [str(item.get("annotation_id", "")) for item in items]
        if not all(identifiers) or len(identifiers) != len(set(identifiers)):
            raise ValueError("annotation IDs must be non-empty and unique")
        for item in items:
            if item.get("subject_sha256") != subject_sha256(item):
                raise ValueError(f"response-review subject hash mismatch: {item['annotation_id']}")
        if self._queue.get("queue_sha256") != canonical_sha256(items):
            raise ValueError("response-review queue hash mismatch")

    def _load_or_initialize(self) -> dict[str, Any]:
        if self.progress_path.exists():
            state = cast(dict[str, Any], json.loads(self.progress_path.read_text()))
            if state.get("schema_version") != PROGRESS_SCHEMA:
                raise ValueError("unsupported response-review progress schema")
            if state.get("queue_sha256") != self._queue["queue_sha256"]:
                raise ValueError("queue changed; preserve progress under a new filename")
            if state.get("reviewer_id") != self.reviewer_id:
                raise ValueError("progress belongs to a different reviewer")
            if set(state.get("reviews", {})) - set(self._by_id):
                raise ValueError("progress contains unknown annotation IDs")
            return state
        state = {
            "schema_version": PROGRESS_SCHEMA,
            "queue_sha256": self._queue["queue_sha256"],
            "reviewer_id": self.reviewer_id,
            "created_at": _now(),
            "updated_at": _now(),
            "reviews": {},
            "history": [],
        }
        self._write(state)
        return state

    def public_state(self) -> dict[str, Any]:
        with self._lock:
            reviews = json.loads(json.dumps(self._state["reviews"], ensure_ascii=False))
        items = []
        for position, private in enumerate(self._queue["items"], start=1):
            annotation_id = str(private["annotation_id"])
            items.append(
                {
                    "position": position,
                    "annotation_id": annotation_id,
                    "language": private["language"],
                    "category": private["category"],
                    "requested_behavior": private["requested_behavior"],
                    "rubric": private["rubric"],
                    "target_response": private["target_response"],
                    "subject_sha256": private["subject_sha256"],
                    "review": reviews.get(annotation_id),
                }
            )
        completed = len(reviews)
        return {
            "reviewer_id": self.reviewer_id,
            "rubric_version": self._queue["rubric_version"],
            "items": items,
            "summary": {
                "total": len(items),
                "completed": completed,
                "remaining": len(items) - completed,
                "labels": {
                    label: sum(review["label"] == label for review in reviews.values())
                    for label in LABELS
                },
            },
        }

    def save_review(
        self,
        annotation_id: str,
        *,
        reviewer_id: str,
        label: str,
        confidence: str,
        partial_subtype: str | None,
        notes: str,
        expected_subject_sha256: str,
    ) -> dict[str, Any]:
        if reviewer_id.strip() != self.reviewer_id:
            raise ValueError("reviewer ID does not match this workspace")
        if annotation_id not in self._by_id:
            raise ValueError("unknown annotation ID")
        item = self._by_id[annotation_id]
        if expected_subject_sha256 != item["subject_sha256"]:
            raise ValueError("review subject changed; reload before saving")
        if label not in LABELS:
            raise ValueError("invalid response label")
        if confidence not in CONFIDENCE:
            raise ValueError("invalid confidence")
        if label == "partial" and partial_subtype not in PARTIAL_SUBTYPES:
            raise ValueError("partial responses require a subtype")
        if label != "partial" and partial_subtype is not None:
            raise ValueError("only partial responses may carry a partial subtype")
        review = {
            "reviewer_id": self.reviewer_id,
            "label": label,
            "confidence": confidence,
            "partial_subtype": partial_subtype,
            "notes": notes.strip(),
            "subject_sha256": expected_subject_sha256,
            "reviewed_at": _now(),
        }
        with self._lock:
            previous = self._state["reviews"].get(annotation_id)
            if previous is not None:
                self._state["history"].append(
                    {"annotation_id": annotation_id, "superseded_at": _now(), "review": previous}
                )
            self._state["reviews"][annotation_id] = review
            self._state["updated_at"] = _now()
            self._write(self._state)
        return self.public_state()

    def _write(self, state: dict[str, Any]) -> None:
        self.progress_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = self.progress_path.with_suffix(self.progress_path.suffix + ".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        temporary.chmod(0o600)
        temporary.replace(self.progress_path)
