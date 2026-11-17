from __future__ import annotations

import hashlib
import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from apps.harmbench_uk_review.workspace import REVIEW_DIMENSIONS, VERDICTS

PROGRESS_SCHEMA = "harmbench-uk-second-review-progress-v1"
QUEUE_SCHEMA = "harmbench-uk-second-review-queue-v1"
REVIEWER_ID = "khrystyna"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def subject_sha256(item: dict[str, Any]) -> str:
    return canonical_sha256(
        {
            "english_content_sha256": item["english_content_sha256"],
            "ukrainian": item["ukrainian"],
        }
    )


class SecondReviewWorkspace:
    def __init__(self, queue_path: Path, progress_path: Path) -> None:
        self.queue_path = queue_path
        self.progress_path = progress_path
        self._lock = threading.RLock()
        self._queue = cast(dict[str, Any], json.loads(queue_path.read_text()))
        self._validate_queue()
        self._by_pair = {str(item["pair_id"]): item for item in self._queue["items"]}
        self._state = self._load_or_initialize()

    def _validate_queue(self) -> None:
        if self._queue.get("schema_version") != QUEUE_SCHEMA:
            raise ValueError("unsupported second-review queue schema")
        items = self._queue.get("items")
        if not isinstance(items, list) or not items:
            raise ValueError("second-review queue is empty")
        pair_ids = [str(item.get("pair_id", "")) for item in items]
        if not all(pair_ids) or len(pair_ids) != len(set(pair_ids)):
            raise ValueError("second-review queue pair IDs must be non-empty and unique")
        for item in items:
            if item.get("subject_sha256") != subject_sha256(item):
                raise ValueError(f"second-review subject hash mismatch: {item['pair_id']}")
        declared = self._queue.get("queue_sha256")
        actual = canonical_sha256(items)
        if declared != actual:
            raise ValueError("second-review queue collection hash mismatch")

    def _load_or_initialize(self) -> dict[str, Any]:
        if self.progress_path.exists():
            state = cast(dict[str, Any], json.loads(self.progress_path.read_text()))
            if state.get("schema_version") != PROGRESS_SCHEMA:
                raise ValueError("unsupported second-review progress schema")
            if state.get("queue_sha256") != self._queue["queue_sha256"]:
                raise ValueError(
                    "second-review queue changed; preserve progress in a new workspace"
                )
            state.setdefault("history", [])
            state.setdefault("reviews", {})
            unknown = set(state["reviews"]) - set(self._by_pair)
            if unknown:
                raise ValueError("second-review progress contains unknown pair IDs")
            self._write(state)
            return state
        state = {
            "schema_version": PROGRESS_SCHEMA,
            "queue_sha256": self._queue["queue_sha256"],
            "reviewer_id": REVIEWER_ID,
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
            pair_id = str(private["pair_id"])
            items.append(
                {
                    "position": position,
                    "pair_id": pair_id,
                    "category": private["category"],
                    "expected_behavior": private["expected_behavior"],
                    "english": private["english"],
                    "ukrainian": private["ukrainian"],
                    "subject_sha256": private["subject_sha256"],
                    "review": reviews.get(pair_id),
                }
            )
        accepted = sum(review["verdict"] == "accept" for review in reviews.values())
        needs_fix = sum(review["verdict"] == "needs_fix" for review in reviews.values())
        return {
            "reviewer_id": REVIEWER_ID,
            "items": items,
            "summary": {
                "total": len(items),
                "completed": len(reviews),
                "remaining": len(items) - len(reviews),
                "accepted": accepted,
                "needs_fix": needs_fix,
            },
        }

    def save_review(
        self,
        pair_id: str,
        *,
        reviewer_id: str,
        verdict: str,
        dimensions: dict[str, str],
        notes: str,
        suggested_correction: dict[str, str | None] | None,
        expected_subject_sha256: str,
    ) -> dict[str, Any]:
        if reviewer_id.strip().casefold() != REVIEWER_ID:
            raise ValueError(f"this queue is assigned to {REVIEWER_ID}")
        if verdict not in {"accept", "needs_fix"}:
            raise ValueError("verdict must be accept or needs_fix")
        if set(dimensions) != set(REVIEW_DIMENSIONS):
            raise ValueError("all five review dimensions are required")
        if any(value not in VERDICTS for value in dimensions.values()):
            raise ValueError("dimensions must be pass, fail, or not_judgeable")
        item = self._by_pair.get(pair_id)
        if item is None:
            raise KeyError(f"unknown pair ID: {pair_id}")
        if expected_subject_sha256 != item["subject_sha256"]:
            raise ValueError("the translation changed; reload before reviewing")
        all_pass = all(value == "pass" for value in dimensions.values())
        if verdict == "accept" and not all_pass:
            raise ValueError("an accepted translation must pass all five dimensions")
        if verdict == "needs_fix" and all_pass:
            raise ValueError("mark at least one dimension fail or not judgeable")
        clean_notes = notes.strip()
        if verdict == "needs_fix" and not clean_notes:
            raise ValueError("a rejected translation requires a rationale")
        correction = None
        if verdict == "needs_fix":
            if suggested_correction is None:
                raise ValueError("a rejected translation requires a suggested correction")
            correction = self._normalize_correction(item, suggested_correction)
            if canonical_sha256(correction) == canonical_sha256(item["ukrainian"]):
                raise ValueError("the suggested correction must change the Ukrainian text")
        review = {
            "subject_sha256": item["subject_sha256"],
            "verdict": verdict,
            "dimensions": dimensions,
            "notes": clean_notes,
            "suggested_correction": correction,
            "reviewer_id": REVIEWER_ID,
            "reviewed_at": _now(),
        }
        with self._lock:
            previous = self._state["reviews"].get(pair_id)
            if previous is not None:
                self._state["history"].append(
                    {
                        "event": "review_superseded",
                        "pair_id": pair_id,
                        "value": previous,
                        "at": _now(),
                    }
                )
            self._state["reviews"][pair_id] = review
            self._save()
        return self.public_state()

    @staticmethod
    def _normalize_correction(
        item: dict[str, Any], correction: dict[str, str | None]
    ) -> dict[str, str | None]:
        prompt = str(correction.get("prompt_text") or "").strip()
        rubric = str(correction.get("rubric_text") or "").strip()
        if not prompt or not rubric:
            raise ValueError("corrected Ukrainian prompt and rubric are required")
        if not any("Ѐ" <= char <= "ӿ" for char in prompt):
            raise ValueError("corrected Ukrainian prompt must contain Cyrillic text")
        requires_context = item["english"]["context_text"] is not None
        raw_context = correction.get("context_text")
        context = str(raw_context).strip() if raw_context is not None else None
        if requires_context and not context:
            raise ValueError("corrected Ukrainian context is required")
        if not requires_context and context:
            raise ValueError("this item has no context to correct")
        return {
            "prompt_text": prompt,
            "context_text": context if requires_context else None,
            "rubric_text": rubric,
        }

    def _save(self) -> None:
        self._state["updated_at"] = _now()
        self._write(self._state)

    def _write(self, value: dict[str, Any]) -> None:
        self.progress_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = self.progress_path.with_suffix(self.progress_path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        temporary.chmod(0o600)
        temporary.replace(self.progress_path)
