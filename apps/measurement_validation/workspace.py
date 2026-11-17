from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from apps.measurement_validation.queue_source import (
    AdjudicationQueue,
    canonical_sha256,
    subject_sha256,
)
from harness.artificial_reviewer import DIMENSIONS, ArtificialReview

WORKSPACE_DIRECTORY_NAME = "measurement-validation-v1"
PROGRESS_FILENAME = "validation-progress.json"

BLIND_VERDICTS = frozenset({"pass", "fail", "not_judgeable"})

ARTIFICIAL_REVIEWER_PREFIX = "artificial-reviewer:"

ADJUDICATIONS = frozenset({"confirm", "override", "partial_override"})


def _now() -> str:
    return datetime.now(UTC).isoformat()


class ValidationWorkspace:
    schema_version = 1

    def __init__(self, root: Path, queue: AdjudicationQueue) -> None:
        self.root = root
        self.progress_path = root / PROGRESS_FILENAME
        self._queue = queue
        self._lock = threading.RLock()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.root.chmod(0o700)
        self._state = self._load_or_initialize()

    def _new_row(self, pair_id: str, subject_sha256: str) -> dict[str, Any]:
        return {
            "pair_id": pair_id,
            "subject_sha256": subject_sha256,
            "blind_draft": None,
            "blind_verdict": None,
            "revealed_at": None,
            "history": [],
            "adjudication": None,
            "duplicate_resolution": None,
            "correction": None,
            "correction_review": None,
        }

    def _load_or_initialize(self) -> dict[str, Any]:
        subjects = self._queue.subjects
        if not subjects:
            raise ValueError("adjudication queue contains no eligible pairs")
        if self.progress_path.exists():
            loaded = cast(dict[str, Any], json.loads(self.progress_path.read_text()))
            state = self._rebind(loaded)
            self._write_json(self.progress_path, state)
            return state
        state = {
            "schema_version": self.schema_version,
            "queue_sha256": self._queue.queue_sha256,
            "reviewer_id": self._queue.reviewer_id,
            "created_at": _now(),
            "updated_at": _now(),
            "items": [
                self._new_row(subject.pair_id, subject.subject_sha256) for subject in subjects
            ],
        }
        self._write_json(self.progress_path, state)
        return state

    def _fold_duplicates(self, by_pair: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:

        duplicates = self._queue.duplicates
        folded: list[dict[str, Any]] = []
        for group in duplicates.groups:
            canonical = by_pair.get(group.canonical_pair_id)
            if canonical is None:
                continue
            for redundant_id in group.redundant_pair_ids:
                redundant = by_pair.pop(redundant_id, None)
                if redundant is None:
                    continue
                transferred = []
                for name in ("blind_verdict", "adjudication"):
                    if canonical.get(name) is None and redundant.get(name) is not None:
                        canonical[name] = redundant[name]
                        transferred.append(name)
                if transferred and canonical.get("revealed_at") is None:
                    canonical["revealed_at"] = redundant.get("revealed_at")
                record = {
                    "event": "duplicate_folded",
                    "recorded_at": _now(),
                    "folded_pair_id": redundant_id,
                    "fingerprint": group.fingerprint,
                    "transferred": transferred,
                    "value": redundant,
                }
                canonical["history"].append(record)
                folded.append(
                    {
                        "canonical_pair_id": group.canonical_pair_id,
                        "folded_pair_id": redundant_id,
                        "recorded_at": record["recorded_at"],
                        "transferred": transferred,
                    }
                )
        return folded

    def _rebind(self, state: dict[str, Any]) -> dict[str, Any]:

        if state.get("schema_version") != self.schema_version:
            raise ValueError("unsupported validation workspace schema")
        rows = state.get("items")
        if not isinstance(rows, list):
            raise ValueError("validation progress is missing its items")
        by_pair: dict[str, dict[str, Any]] = {}
        for row in rows:
            pair_id = str(row.get("pair_id", ""))
            if pair_id in by_pair:
                raise ValueError(f"duplicate persisted pair ID: {pair_id}")
            by_pair[pair_id] = row
        expected = self._queue.by_pair_id()
        folded = self._fold_duplicates(by_pair)
        if set(by_pair) != set(expected):
            raise ValueError(
                "queue membership changed; use a new workspace directory to preserve "
                "adjudication history"
            )
        if folded:
            state["folded_duplicates"] = folded
        for pair_id, subject in expected.items():
            row = by_pair[pair_id]
            row.setdefault("adjudication", None)
            row.setdefault("duplicate_resolution", None)
            row.setdefault("correction", None)
            row.setdefault("correction_review", None)
            if row.get("subject_sha256") == subject.subject_sha256:
                continue

            for name in (
                "blind_draft",
                "blind_verdict",
                "adjudication",
                "correction",
                "correction_review",
            ):
                superseded = row.get(name)
                if superseded is not None:
                    row["history"].append(
                        {
                            "event": "invalidated_subject_changed",
                            "field": name,
                            "recorded_at": _now(),
                            "previous_subject_sha256": row.get("subject_sha256"),
                            "value": superseded,
                        }
                    )
                row[name] = None
            row["revealed_at"] = None
            row["subject_sha256"] = subject.subject_sha256
        state["items"] = [by_pair[subject.pair_id] for subject in self._queue.subjects]
        state["queue_sha256"] = self._queue.queue_sha256
        state["reviewer_id"] = self._queue.reviewer_id
        state["updated_at"] = _now()
        return state

    def public_state(self) -> dict[str, Any]:

        with self._lock:
            state = cast(dict[str, Any], json.loads(json.dumps(self._state, ensure_ascii=False)))
        subjects = self._queue.by_pair_id()
        for row in state["items"]:
            subject = subjects[row["pair_id"]]
            row["category"] = subject.category
            row["expected_behavior"] = subject.expected_behavior
            row["rubric_variant"] = subject.rubric_variant

            texts = {
                "en_prompt": subject.en_prompt,
                "en_context": subject.en_context,
                "en_rubric": subject.en_rubric,
                "uk_prompt": subject.uk_prompt,
                "uk_image_text": subject.uk_image_text,
                "uk_rubric": subject.uk_rubric,
            }
            correction = row.get("correction")
            if correction is not None:
                texts.update(correction["texts"])
            row["texts"] = texts
            row["current_subject_sha256"] = (
                correction["subject_sha256"] if correction is not None else row["subject_sha256"]
            )
            row["locked"] = row["blind_verdict"] is not None
            row["revealable"] = row["locked"]
            row["adjudicated"] = row.get("adjudication") is not None
            row["duplicate_resolved"] = row.get("duplicate_resolution") is not None
            row["effective_dimensions"] = self.effective_dimensions(row)
            row["needs_correction"] = bool(
                row["effective_dimensions"]
                and any(value != "pass" for value in row["effective_dimensions"].values())
            )
            row["correction_pending_review"] = bool(
                correction is not None and row.get("correction_review") is None
            )
        locked = sum(row["locked"] for row in state["items"])
        state["summary"] = {
            "total": len(state["items"]),
            "blind_locked": locked,
            "revealed": sum(row["revealed_at"] is not None for row in state["items"]),
            "blind_complete": locked == len(state["items"]),
            "adjudicated": sum(row["adjudicated"] for row in state["items"]),
            "needs_correction": sum(row["needs_correction"] for row in state["items"]),
            "corrected": sum(row.get("correction") is not None for row in state["items"]),
            "correction_reviewed": sum(
                row.get("correction_review") is not None for row in state["items"]
            ),
        }
        return state

    def save_blind_draft(
        self,
        pair_id: str,
        *,
        reviewer_id: str,
        dimensions: dict[str, str | None],
        notes: str,
        expected_subject_sha256: str,
    ) -> dict[str, Any]:

        reviewer_id = self._checked_reviewer(reviewer_id)
        self._check_dimensions(dimensions, allow_partial=True)
        with self._lock:
            row = self._find(pair_id, expected_subject_sha256)
            if row["blind_verdict"] is not None:
                raise ValueError("blind verdict is locked and cannot be redrafted")
            row["blind_draft"] = {
                "dimensions": dict(dimensions),
                "notes": notes.strip(),
                "reviewer_id": reviewer_id,
                "saved_at": _now(),
                "subject_sha256": row["subject_sha256"],
            }
            self._save()
            return self.public_state()

    def lock_blind_verdict(
        self,
        pair_id: str,
        *,
        reviewer_id: str,
        dimensions: dict[str, str],
        confidence: float,
        notes: str,
        expected_subject_sha256: str,
    ) -> dict[str, Any]:

        reviewer_id = self._checked_reviewer(reviewer_id)
        self._check_dimensions(cast(dict[str, str | None], dimensions), allow_partial=False)
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between zero and one")
        with self._lock:
            row = self._find(pair_id, expected_subject_sha256)
            if row["blind_verdict"] is not None:
                raise ValueError("blind verdict is already locked")
            verdict = {
                "confidence": confidence,
                "dimensions": dict(dimensions),
                "locked_at": _now(),
                "notes": notes.strip(),
                "reviewer_id": reviewer_id,
                "subject_sha256": row["subject_sha256"],
            }
            row["blind_verdict"] = verdict
            row["blind_draft"] = None
            row["history"].append(
                {"event": "blind_locked", "recorded_at": verdict["locked_at"], "value": verdict}
            )
            self._save()
            return self.public_state()

    def resolve_duplicate_disagreement(
        self,
        pair_id: str,
        *,
        reviewer_id: str,
        dimensions: dict[str, str],
        rationale: str,
        expected_subject_sha256: str,
    ) -> dict[str, Any]:

        reviewer_id = self._checked_reviewer(reviewer_id)
        self._check_dimensions(cast(dict[str, str | None], dimensions), allow_partial=False)
        if not rationale.strip():
            raise ValueError("a duplicate resolution requires a rationale")
        folded = {
            group.canonical_pair_id: group.redundant_pair_ids
            for group in self._queue.duplicates.groups
        }
        if pair_id not in folded:
            raise ValueError(f"pair is not the canonical member of a duplicate group: {pair_id}")
        with self._lock:
            row = self._find(pair_id, expected_subject_sha256)
            if row["blind_verdict"] is None:
                raise ValueError("the blind verdict must be locked before resolution")
            blind = row["blind_verdict"]["dimensions"]
            resolution = {
                "dimensions": dict(dimensions),
                "changed_from_blind": sorted(
                    name for name in DIMENSIONS if blind.get(name) != dimensions[name]
                ),
                "folded_pair_ids": list(folded[pair_id]),
                "rationale": rationale.strip(),
                "recorded_at": _now(),
                "reviewer_id": reviewer_id,
                "subject_sha256": row["subject_sha256"],
            }
            row["duplicate_resolution"] = resolution
            row["history"].append(
                {
                    "event": "duplicate_resolved",
                    "recorded_at": resolution["recorded_at"],
                    "value": resolution,
                }
            )
            self._save()
            return self.public_state()

    @staticmethod
    def effective_dimensions(row: Mapping[str, Any]) -> dict[str, str] | None:

        correction_review = row.get("correction_review")
        if correction_review is not None:
            return cast(dict[str, str], dict(correction_review["dimensions"]))
        adjudication = row.get("adjudication")
        if adjudication is not None:
            return cast(dict[str, str], dict(adjudication["final_dimensions"]))
        resolution = row.get("duplicate_resolution")
        if resolution is not None:
            return cast(dict[str, str], dict(resolution["dimensions"]))
        blind = row.get("blind_verdict")
        if blind is not None:
            return cast(dict[str, str], dict(blind["dimensions"]))
        return None

    def save_correction(
        self,
        pair_id: str,
        *,
        corrector_id: str,
        texts: dict[str, str | None],
        rationale: str,
        expected_subject_sha256: str,
    ) -> dict[str, Any]:

        corrector_id = corrector_id.strip()
        if not corrector_id:
            raise ValueError("corrector ID is required")
        if not rationale.strip():
            raise ValueError("a correction requires a rationale")
        expected_fields = {"uk_prompt", "uk_image_text", "uk_rubric"}
        if set(texts) != expected_fields:
            raise ValueError("all three Ukrainian correction fields are required")
        uk_prompt = str(texts["uk_prompt"] or "").strip()
        uk_rubric = str(texts["uk_rubric"] or "").strip()
        uk_image_text = texts["uk_image_text"]
        if not uk_prompt or not uk_rubric:
            raise ValueError("corrected Ukrainian prompt and rubric are required")
        if uk_image_text is not None:
            uk_image_text = str(uk_image_text).strip()

        with self._lock:
            row = self._find(pair_id, None)
            adjudication = row.get("adjudication")
            if adjudication is None:
                raise ValueError("correction requires a completed adjudication")
            current_dimensions = self.effective_dimensions(row)
            if current_dimensions is None or all(
                value == "pass" for value in current_dimensions.values()
            ):
                raise ValueError("only a failed or not-judgeable subject may be corrected")
            previous = row.get("correction")
            current_hash = (
                str(previous["subject_sha256"])
                if previous is not None
                else str(row["subject_sha256"])
            )
            if expected_subject_sha256 != current_hash:
                raise ValueError("the correction subject changed; reload the item")
            reviewer_ids = {
                str(value.get("reviewer_id", ""))
                for value in (row.get("blind_verdict"), adjudication, row.get("correction_review"))
                if isinstance(value, dict)
            }
            if corrector_id in reviewer_ids:
                raise ValueError("corrector must differ from the reviewing human")
            subject = self._queue.find(pair_id)
            if subject.en_context is None and uk_image_text is not None:
                raise ValueError("an item without English context cannot add Ukrainian image text")
            if subject.en_context is not None and not uk_image_text:
                raise ValueError("Ukrainian image text is required for this item")
            corrected_texts = {
                "uk_prompt": uk_prompt,
                "uk_image_text": uk_image_text,
                "uk_rubric": uk_rubric,
            }
            corrected_hash = subject_sha256(uk_prompt, uk_image_text, uk_rubric)
            if corrected_hash == current_hash:
                raise ValueError("the corrected revision does not change the subject")
            if previous is not None:
                row["history"].append(
                    {
                        "event": "correction_superseded",
                        "recorded_at": _now(),
                        "value": {
                            "correction": previous,
                            "correction_review": row.get("correction_review"),
                        },
                    }
                )
            revision = int(previous.get("revision", 0)) + 1 if previous is not None else 1
            correction = {
                "revision": revision,
                "base_subject_sha256": current_hash,
                "subject_sha256": corrected_hash,
                "texts": corrected_texts,
                "corrector_id": corrector_id,
                "rationale": rationale.strip(),
                "corrected_at": _now(),
            }
            row["correction"] = correction
            row["correction_review"] = None
            row["history"].append(
                {
                    "event": "correction_saved",
                    "recorded_at": correction["corrected_at"],
                    "value": correction,
                }
            )
            self._save()
            return self.public_state()

    def record_correction_review(
        self,
        pair_id: str,
        *,
        reviewer_id: str,
        dimensions: dict[str, str],
        notes: str,
        expected_subject_sha256: str,
    ) -> dict[str, Any]:

        reviewer_id = self._checked_reviewer(reviewer_id)
        self._check_dimensions(cast(dict[str, str | None], dimensions), allow_partial=False)
        if any(value != "pass" for value in dimensions.values()) and not notes.strip():
            raise ValueError("a failed correction review requires notes")
        with self._lock:
            row = self._find(pair_id, None)
            correction = row.get("correction")
            if correction is None:
                raise ValueError("this subject has no corrected revision")
            if expected_subject_sha256 != correction["subject_sha256"]:
                raise ValueError("the corrected subject changed; reload the item")
            if reviewer_id == correction["corrector_id"]:
                raise ValueError("correction reviewer must differ from the corrector")
            if row.get("correction_review") is not None:
                raise ValueError("this corrected revision is already reviewed")
            review = {
                "revision": correction["revision"],
                "dimensions": dict(dimensions),
                "notes": notes.strip(),
                "reviewer_id": reviewer_id,
                "reviewed_at": _now(),
                "subject_sha256": correction["subject_sha256"],
            }
            row["correction_review"] = review
            row["history"].append(
                {
                    "event": "correction_reviewed",
                    "recorded_at": review["reviewed_at"],
                    "value": review,
                }
            )
            self._save()
            return self.public_state()

    def artificial_review(self, pair_id: str) -> ArtificialReview:

        with self._lock:
            row = self._find(pair_id, None)
            if row["blind_verdict"] is None:
                raise ValueError("the blind verdict must be locked before reveal")
            if row["revealed_at"] is None:
                row["revealed_at"] = _now()
                row["history"].append({"event": "revealed", "recorded_at": row["revealed_at"]})
                self._save()
        return self._queue.find(pair_id).artificial_review

    def record_adjudication(
        self,
        pair_id: str,
        *,
        reviewer_id: str,
        decision: str,
        final_dimensions: dict[str, str],
        rationale: str,
        expected_subject_sha256: str,
    ) -> dict[str, Any]:

        reviewer_id = self._checked_reviewer(reviewer_id)
        if decision not in ADJUDICATIONS:
            raise ValueError(f"decision must be one of {sorted(ADJUDICATIONS)}")
        self._check_dimensions(cast(dict[str, str | None], final_dimensions), allow_partial=False)
        with self._lock:
            row = self._find(pair_id, expected_subject_sha256)
            if row["blind_verdict"] is None:
                raise ValueError("the blind verdict must be locked before adjudication")
            if row["revealed_at"] is None:
                raise ValueError("the artificial review must be revealed before adjudication")
            if row.get("adjudication") is not None:
                raise ValueError("this pair is already adjudicated")
            blind = row["blind_verdict"]["dimensions"]
            changed = sorted(
                name for name in DIMENSIONS if blind.get(name) != final_dimensions[name]
            )
            if decision == "confirm" and changed:
                raise ValueError(
                    "a confirmation cannot change a dimension; use override or partial_override"
                )
            if decision != "confirm" and not changed:
                raise ValueError("an override must change at least one dimension")
            record = {
                "decision": decision,
                "final_dimensions": dict(final_dimensions),
                "changed_dimensions": changed,
                "rationale": rationale.strip(),
                "reviewer_id": reviewer_id,
                "recorded_at": _now(),
                "subject_sha256": row["subject_sha256"],
            }
            row["adjudication"] = record
            row["history"].append(
                {"event": "adjudicated", "recorded_at": record["recorded_at"], "value": record}
            )
            self._save()
            return self.public_state()

    def _checked_reviewer(self, reviewer_id: str) -> str:
        reviewer_id = reviewer_id.strip()
        if not reviewer_id:
            raise ValueError("reviewer ID is required")
        if reviewer_id.startswith(ARTIFICIAL_REVIEWER_PREFIX):
            raise ValueError("an artificial reviewer cannot adjudicate its own proposal")
        if reviewer_id in self._queue.translator_ids:
            raise ValueError("adjudicating reviewer must differ from the translator")
        return reviewer_id

    @staticmethod
    def _check_dimensions(dimensions: dict[str, str | None], *, allow_partial: bool) -> None:
        if set(dimensions) != set(DIMENSIONS):
            raise ValueError("all five review dimensions are required")
        for value in dimensions.values():
            if value is None:
                if allow_partial:
                    continue
                raise ValueError("a locked verdict requires every dimension")
            if value not in BLIND_VERDICTS:
                raise ValueError("verdicts must be pass, fail, or not_judgeable")

    def _find(self, pair_id: str, expected_subject_sha256: str | None) -> dict[str, Any]:
        for row in self._state["items"]:
            if row["pair_id"] != pair_id:
                continue
            if (
                expected_subject_sha256 is not None
                and row["subject_sha256"] != expected_subject_sha256
            ):
                raise ValueError(
                    "the reviewed subject changed since this view was loaded; reload the item"
                )
            return cast(dict[str, Any], row)
        raise KeyError(f"unknown pair ID: {pair_id}")

    def _save(self) -> None:
        self._state["updated_at"] = _now()
        self._write_json(self.progress_path, self._state)

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        temporary.chmod(0o600)
        temporary.replace(path)


def progress_manifest(state: dict[str, Any]) -> dict[str, Any]:

    strata: dict[str, int] = {}
    for row in state["items"]:
        strata[row["rubric_variant"]] = strata.get(row["rubric_variant"], 0) + 1
    return {
        "blind_locked": state["summary"]["blind_locked"],
        "corrected": state["summary"]["corrected"],
        "correction_reviewed": state["summary"]["correction_reviewed"],
        "needs_correction": state["summary"]["needs_correction"],
        "queue_sha256": state["queue_sha256"],
        "revealed": state["summary"]["revealed"],
        "schema_version": state["schema_version"],
        "strata": strata,
        "subject_sha256_digest": canonical_sha256(
            [[row["pair_id"], row["current_subject_sha256"]] for row in state["items"]]
        ),
        "total": state["summary"]["total"],
    }
