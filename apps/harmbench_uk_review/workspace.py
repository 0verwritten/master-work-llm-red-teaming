from __future__ import annotations

import hashlib
import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

REVIEW_DIMENSIONS = (
    "semantic_equivalence",
    "naturalness",
    "category_match",
    "rubric_match",
    "expected_behavior_match",
)
VERDICTS = ("pass", "fail", "not_judgeable")
SELECTIONS = ("accept_deepl", "accept_proposal", "edit_manual")
OVERLAY_SCHEMA = "harmbench-uk-human-review-v2"
LEGACY_SCHEMA = "harmbench-uk-human-review-v1"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def final_content_sha256(fields: dict[str, str | None]) -> str:
    return _canonical_sha256(
        {
            "prompt_text": fields["prompt_text"],
            "context_text": fields["context_text"],
            "rubric_text": fields["rubric_text"],
        }
    )


def _has_cyrillic(text: str) -> bool:
    return any("Ѐ" <= char <= "ӿ" for char in text)


class ReviewWorkspace:
    def __init__(
        self,
        proposal_path: Path,
        overlay_path: Path,
        *,
        legacy_overlay_path: Path | None = None,
        reviewed_seed_path: Path | None = None,
        reviewed_output_path: Path | None = None,
    ) -> None:
        self.proposal_path = proposal_path
        self.overlay_path = overlay_path
        self.legacy_overlay_path = legacy_overlay_path
        self.reviewed_seed_path = reviewed_seed_path
        self.reviewed_path = reviewed_output_path or overlay_path.with_name(
            "harmbench-uk-reviewed-translations-v1.json"
        )
        self._lock = threading.RLock()
        self.proposal = json.loads(proposal_path.read_text())
        self._records = {record["pair_id"]: record for record in self.proposal["records"]}
        self._order = [record["pair_id"] for record in self.proposal["records"]]
        overlay_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._overlay = self._load_or_initialize()

    def _load_or_initialize(self) -> dict[str, Any]:
        if not self.overlay_path.exists():
            if self.reviewed_seed_path is not None and self.reviewed_seed_path.exists():
                reviewed = cast(dict[str, Any], json.loads(self.reviewed_seed_path.read_text()))
                overlay = self._seed_from_reviewed_artifact(reviewed)
                self._write(overlay)
                return overlay
            if self.legacy_overlay_path is not None and self.legacy_overlay_path.exists():
                legacy = cast(dict[str, Any], json.loads(self.legacy_overlay_path.read_text()))
                if legacy.get("schema_version") == LEGACY_SCHEMA:
                    overlay = self._migrate_legacy(legacy)
                    self._write(overlay)
                    return overlay
            overlay = {
                "schema_version": OVERLAY_SCHEMA,
                "subset_plan_sha256": self.proposal["subset_plan_sha256"],
                "proposal_collection_sha256": self.proposal["collection_sha256"],
                "reviewer_id": "vlad",
                "created_at": _now(),
                "updated_at": _now(),
                "decisions": {},
                "history": [],
            }
            self._write(overlay)
            return overlay
        loaded = cast(dict[str, Any], json.loads(self.overlay_path.read_text()))
        if loaded.get("schema_version") == LEGACY_SCHEMA:
            loaded = self._migrate_legacy(loaded)
        if loaded.get("schema_version") != OVERLAY_SCHEMA:
            raise ValueError("unsupported overlay schema")
        loaded.setdefault("history", [])
        loaded.setdefault("decisions", {})
        loaded.setdefault("rubric_reviews", {})
        self._invalidate_stale(loaded)
        self._write(loaded)
        return loaded

    def _seed_from_reviewed_artifact(self, reviewed: dict[str, Any]) -> dict[str, Any]:

        if reviewed.get("schema_version") != "harmbench-uk-reviewed-translations-v1":
            raise ValueError("unsupported reviewed seed schema")
        decisions: dict[str, Any] = {}
        for entry in reviewed.get("records", []):
            pair_id = str(entry.get("pair_id", ""))
            record = self._records.get(pair_id)
            if record is None:
                continue
            if entry.get("english_content_sha256") != record["english_content_sha256"]:
                raise ValueError(f"reviewed seed English identity mismatch: {pair_id}")
            final = {
                "prompt_text": entry.get("uk_prompt"),
                "context_text": entry.get("uk_context"),
                "rubric_text": entry.get("uk_rubric"),
            }
            if final_content_sha256(final) != final_content_sha256(record["machine_deepl"]):
                raise ValueError(f"reviewed seed Ukrainian content mismatch: {pair_id}")
            dimensions = entry.get("dimensions")
            if not isinstance(dimensions, dict) or set(dimensions) != set(REVIEW_DIMENSIONS):
                raise ValueError(f"reviewed seed dimensions are incomplete: {pair_id}")
            decisions[pair_id] = self._decision_row(
                record,
                reviewer_id=str(entry.get("reviewer_id", "prior-human-review")),
                selection="accept_deepl",
                final=final,
                dimensions={str(key): str(value) for key, value in dimensions.items()},
                notes=(
                    "Carried forward from the exact prior 120-item human review. "
                    + str(entry.get("review_notes", ""))
                ).strip(),
                decided_at=str(entry.get("decided_at", _now())),
            )
        return {
            "schema_version": OVERLAY_SCHEMA,
            "subset_plan_sha256": self.proposal["subset_plan_sha256"],
            "proposal_collection_sha256": self.proposal["collection_sha256"],
            "reviewer_id": "vlad",
            "created_at": _now(),
            "updated_at": _now(),
            "decisions": decisions,
            "history": [
                {
                    "seeded_from": str(self.reviewed_seed_path),
                    "decision_count": len(decisions),
                    "at": _now(),
                }
            ],
            "rubric_reviews": {},
        }

    def _migrate_legacy(self, legacy: dict[str, Any]) -> dict[str, Any]:

        decisions: dict[str, Any] = {}
        for entry in legacy.get("decisions", []):
            pair_id = entry["pair_id"]
            record = self._records.get(pair_id)
            if record is None:
                continue
            if entry.get("machine_content_sha256") != record["machine_content_sha256"]:
                continue
            machine = record["machine_deepl"]
            final = {
                "prompt_text": machine["prompt_text"],
                "context_text": machine["context_text"],
                "rubric_text": machine["rubric_text"],
            }
            for correction in entry.get("corrections", []):
                field = correction["field"]
                current = final[field]
                old, new = correction["old"], correction["new"]

                if current is not None and old in current:
                    final[field] = current.replace(old, new)
                else:
                    final[field] = new
            if entry["disposition"] == "accept_unchanged":
                selection = "accept_deepl"
            elif final_content_sha256(final) == final_content_sha256(record["proposed"]):
                selection = "accept_proposal"
                final = dict(record["proposed"])
            else:
                selection = "edit_manual"
            decisions[pair_id] = self._decision_row(
                record,
                reviewer_id=legacy.get("reviewer_id", "vlad"),
                selection=selection,
                final=final,
                dimensions=dict(entry["dimensions"]),
                notes=entry.get("notes", ""),
                decided_at=legacy.get("reviewed_at", _now()),
            )
        return {
            "schema_version": OVERLAY_SCHEMA,
            "subset_plan_sha256": self.proposal["subset_plan_sha256"],
            "proposal_collection_sha256": self.proposal["collection_sha256"],
            "reviewer_id": legacy.get("reviewer_id", "vlad"),
            "created_at": legacy.get("reviewed_at", _now()),
            "updated_at": _now(),
            "decisions": decisions,
            "history": [{"migrated_from": LEGACY_SCHEMA, "at": _now()}],
            "rubric_reviews": {},
        }

    def _invalidate_stale(self, overlay: dict[str, Any]) -> None:
        for pair_id in list(overlay["decisions"]):
            decision = overlay["decisions"][pair_id]
            record = self._records.get(pair_id)
            if (
                record is None
                or decision["machine_content_sha256"] != record["machine_content_sha256"]
                or decision["proposal_content_sha256"] != record["proposal_content_sha256"]
            ):
                overlay["history"].append(
                    {"invalidated": decision, "reason": "bound content changed", "at": _now()}
                )
                del overlay["decisions"][pair_id]

    def _decision_row(
        self,
        record: dict[str, Any],
        *,
        reviewer_id: str,
        selection: str,
        final: dict[str, str | None],
        dimensions: dict[str, str],
        notes: str,
        decided_at: str,
    ) -> dict[str, Any]:
        machine = record["machine_deepl"]
        changed = final_content_sha256(final) != final_content_sha256(
            {
                "prompt_text": machine["prompt_text"],
                "context_text": machine["context_text"],
                "rubric_text": machine["rubric_text"],
            }
        )
        return {
            "machine_content_sha256": record["machine_content_sha256"],
            "proposal_content_sha256": record["proposal_content_sha256"],
            "selection": selection,
            "final": final,
            "final_content_sha256": final_content_sha256(final),
            "changed_from_machine": changed,
            "dimensions": dimensions,
            "notes": notes,
            "reviewer_id": reviewer_id,
            "decided_at": decided_at,
        }

    def public_state(self) -> dict[str, Any]:
        with self._lock:
            decisions = json.loads(json.dumps(self._overlay["decisions"], ensure_ascii=False))
            rubric_reviews = json.loads(
                json.dumps(self._overlay.get("rubric_reviews", {}), ensure_ascii=False)
            )
        completed = len(decisions)
        rubrics: dict[str, dict[str, Any]] = {}
        for record in self.proposal["records"]:
            proposed = record["proposed"]
            rubric = {
                "english": record["english"]["rubric_text"],
                "ukrainian": proposed["rubric_text"],
            }
            rubric_id = _canonical_sha256(rubric)
            rubrics.setdefault(rubric_id, {"rubric_id": rubric_id, **rubric})
        return {
            "reviewer_id": self._overlay["reviewer_id"],
            "count": self.proposal["count"],
            "records": self.proposal["records"],
            "decisions": decisions,
            "rubrics": list(rubrics.values()),
            "rubric_reviews": rubric_reviews,
            "summary": {
                "total": self.proposal["count"],
                "completed": completed,
                "remaining": self.proposal["count"] - completed,
                "ready": completed == self.proposal["count"],
                "reviewed_path": (str(self.reviewed_path) if self.reviewed_path.exists() else None),
            },
        }

    def approve_rubric(self, rubric_id: str, *, reviewer_id: str, rationale: str) -> dict[str, Any]:

        reviewer_id = reviewer_id.strip()
        if not reviewer_id:
            raise ValueError("reviewer ID is required")
        state = self.public_state()
        if rubric_id not in {rubric["rubric_id"] for rubric in state["rubrics"]}:
            raise KeyError("unknown rubric pair")
        with self._lock:
            self._overlay.setdefault("rubric_reviews", {})[rubric_id] = {
                "verdict": "pass",
                "reviewer_id": reviewer_id,
                "rationale": rationale.strip() or "Approved as prompt-independent shared rubric.",
                "decided_at": _now(),
            }
            self._save()
        return self.public_state()

    def save_decision(
        self,
        pair_id: str,
        *,
        reviewer_id: str,
        selection: str,
        dimensions: dict[str, str],
        notes: str,
        manual_fields: dict[str, str | None] | None,
        expected_machine_sha256: str,
        expected_proposal_sha256: str,
    ) -> dict[str, Any]:
        reviewer_id = reviewer_id.strip()
        if not reviewer_id:
            raise ValueError("reviewer ID is required")
        if selection not in SELECTIONS:
            raise ValueError("invalid version selection")
        if set(dimensions) != set(REVIEW_DIMENSIONS):
            raise ValueError("all five review dimensions are required")
        if any(value not in VERDICTS for value in dimensions.values()):
            raise ValueError("dimensions must be pass, fail, or not_judgeable")
        with self._lock:
            record = self._records.get(pair_id)
            if record is None:
                raise KeyError(f"unknown pair ID: {pair_id}")
            if expected_machine_sha256 != record["machine_content_sha256"]:
                raise ValueError("stale decision: machine content changed; reload")
            if expected_proposal_sha256 != record["proposal_content_sha256"]:
                raise ValueError("stale decision: proposal content changed; reload")
            machine = record["machine_deepl"]
            if selection == "accept_deepl":
                final = {
                    "prompt_text": machine["prompt_text"],
                    "context_text": machine["context_text"],
                    "rubric_text": machine["rubric_text"],
                }
            elif selection == "accept_proposal":
                final = dict(record["proposed"])
            else:
                if manual_fields is None:
                    raise ValueError("manual selection requires edited fields")
                final = self._normalize_manual(record, manual_fields)
            previous = self._overlay["decisions"].get(pair_id)
            if previous is not None:
                self._overlay["history"].append({"superseded": previous, "at": _now()})
            self._overlay["decisions"][pair_id] = self._decision_row(
                record,
                reviewer_id=reviewer_id,
                selection=selection,
                final=final,
                dimensions=dimensions,
                notes=notes.strip(),
                decided_at=_now(),
            )
            self.reviewed_path.unlink(missing_ok=True)
            self._save()
            return self.public_state()

    def _normalize_manual(
        self, record: dict[str, Any], manual_fields: dict[str, str | None]
    ) -> dict[str, str | None]:
        machine = record["machine_deepl"]
        prompt = str(manual_fields.get("prompt_text", "")).strip()
        rubric = str(manual_fields.get("rubric_text", "")).strip()
        if not prompt:
            raise ValueError("Ukrainian prompt is required")
        if not _has_cyrillic(prompt):
            raise ValueError("Ukrainian prompt must contain Cyrillic text")
        if not rubric:
            raise ValueError("Ukrainian rubric is required")
        context_required = machine["context_text"] is not None
        raw_context = manual_fields.get("context_text")
        context = str(raw_context) if raw_context is not None else None
        if context_required and not (context and context.strip()):
            raise ValueError("this item requires context text")
        if not context_required and context is not None and context.strip():
            raise ValueError("this item has no context text")
        return {
            "prompt_text": prompt,
            "context_text": context if context_required else None,
            "rubric_text": rubric,
        }

    def finalize(self) -> Path:
        with self._lock:
            decisions = self._overlay["decisions"]
            missing = [pair_id for pair_id in self._order if pair_id not in decisions]
            if missing:
                raise ValueError(f"{len(missing)} items still lack a decision")
            records: list[dict[str, Any]] = []
            verified = corrected = failed = not_judgeable = 0
            for pair_id in self._order:
                record = self._records[pair_id]
                decision = decisions[pair_id]
                if (
                    decision["machine_content_sha256"] != record["machine_content_sha256"]
                    or decision["proposal_content_sha256"] != record["proposal_content_sha256"]
                ):
                    raise ValueError(f"stale decision for {pair_id}; reload and re-review")
                dims = decision["dimensions"]
                all_pass = all(dims[name] == "pass" for name in REVIEW_DIMENSIONS)
                if any(dims[name] == "fail" for name in REVIEW_DIMENSIONS):
                    failed += 1
                if any(dims[name] == "not_judgeable" for name in REVIEW_DIMENSIONS):
                    not_judgeable += 1
                if decision["changed_from_machine"]:
                    provenance = "machine_deepl_human_corrected"
                    corrected += 1
                else:
                    provenance = "machine_deepl_human_verified"
                    verified += 1
                reviewer_status = "human_verified" if all_pass else "human_reviewed_unresolved"
                records.append(
                    {
                        "pair_id": pair_id,
                        "english_content_sha256": record["english_content_sha256"],
                        "machine_content_sha256": record["machine_content_sha256"],
                        "uk_prompt": decision["final"]["prompt_text"],
                        "uk_context": decision["final"]["context_text"],
                        "uk_rubric": decision["final"]["rubric_text"],
                        "final_content_sha256": decision["final_content_sha256"],
                        "selection": decision["selection"],
                        "translation_provenance": provenance,
                        "reviewer_status": reviewer_status,
                        "reviewer_id": decision["reviewer_id"],
                        "dimensions": dims,
                        "review_notes": decision["notes"],
                        "decided_at": decision["decided_at"],
                    }
                )
            collection_sha256 = _canonical_sha256(
                [
                    {
                        "pair_id": row["pair_id"],
                        "final_content_sha256": row["final_content_sha256"],
                        "reviewer_status": row["reviewer_status"],
                    }
                    for row in records
                ]
            )
            artifact = {
                "schema_version": "harmbench-uk-reviewed-translations-v1",
                "subset_plan_sha256": self._overlay["subset_plan_sha256"],
                "proposal_collection_sha256": self.proposal["collection_sha256"],
                "dataset_sha256": self.proposal["dataset_sha256"],
                "finalized_at": _now(),
                "counts": {
                    "total": len(records),
                    "unchanged_verified": verified,
                    "corrected": corrected,
                    "human_verified": verified + corrected - failed - not_judgeable,
                    "with_failed_dimension": failed,
                    "with_not_judgeable_dimension": not_judgeable,
                },
                "records": records,
                "collection_sha256": collection_sha256,
            }
            self._write_json(self.reviewed_path, artifact)
            return self.reviewed_path

    def _save(self) -> None:
        self._overlay["updated_at"] = _now()
        self._write(self._overlay)

    def _write(self, overlay: dict[str, Any]) -> None:
        self._write_json(self.overlay_path, overlay)

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        temporary.chmod(0o600)
        temporary.replace(path)
