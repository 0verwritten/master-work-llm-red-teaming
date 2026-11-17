from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from harness.bilingual_import import content_hash

REVIEW_DIMENSIONS = (
    "semantic_equivalence",
    "naturalness",
    "category_match",
    "rubric_match",
    "expected_behavior_match",
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TranslationSourceItem:
    stable_id: str
    pair_id: str
    source_content_sha256: str
    prompt_text: str
    context_text: str | None
    category: str
    rubric_text: str
    expected_behavior: str

    def as_json(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "context_text": self.context_text,
            "expected_behavior": self.expected_behavior,
            "pair_id": self.pair_id,
            "prompt_text": self.prompt_text,
            "rubric_text": self.rubric_text,
            "source_content_sha256": self.source_content_sha256,
            "stable_id": self.stable_id,
        }


class TranslationWorkspace:
    schema_version = 2

    def __init__(self, root: Path, sources: tuple[TranslationSourceItem, ...]) -> None:
        self.root = root
        self.english_sources_path = root / "english-source-items.json"
        self.progress_path = root / "translation-progress.json"
        self.final_path = root / "reviewed-translations.json"
        self.ai_final_path = root / "ai-reviewed-translations.json"
        self._lock = threading.RLock()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.root.chmod(0o700)
        self._state = self._load_or_initialize(sources)
        self._write_english_sources(sources)

    def _write_english_sources(self, sources: tuple[TranslationSourceItem, ...]) -> None:
        records = [
            {"language": "en", **item.as_json()}
            for item in sorted(sources, key=lambda value: value.pair_id)
        ]
        self._write_json(self.english_sources_path, records)

    def _load_or_initialize(self, sources: tuple[TranslationSourceItem, ...]) -> dict[str, Any]:
        ordered = tuple(sorted(sources, key=lambda item: item.pair_id))
        if not ordered:
            raise ValueError("translation collection contains no source items")
        if len({item.pair_id for item in ordered}) != len(ordered):
            raise ValueError("translation source pair IDs must be unique")
        if self.progress_path.exists():
            loaded = cast(dict[str, Any], json.loads(self.progress_path.read_text()))
            loaded = self._migrate_state(loaded)
            self._validate_existing(loaded, ordered)
            self._write_json(self.progress_path, loaded)
            return loaded
        state: dict[str, Any] = {
            "schema_version": self.schema_version,
            "collection_sha256": _canonical_sha256([item.as_json() for item in ordered]),
            "created_at": _now(),
            "updated_at": _now(),
            "items": [
                {
                    "source": item.as_json(),
                    "translation": None,
                    "review": None,
                    "review_history": [],
                    "ai_reviews": [],
                    "revision": 0,
                }
                for item in ordered
            ],
        }
        self._write_json(self.progress_path, state)
        return state

    def _migrate_state(self, state: dict[str, Any]) -> dict[str, Any]:
        version = state.get("schema_version")
        if version == self.schema_version:
            return state
        if version != 1:
            raise ValueError("unsupported translation workspace schema")
        for row in state.get("items", []):
            translation = row.get("translation")
            if translation is not None:
                translation["translation_method"] = "human"
            row["review_history"] = [row["review"]] if row.get("review") else []
            row["ai_reviews"] = []
        state["schema_version"] = self.schema_version
        state["updated_at"] = _now()
        return state

    def _validate_existing(
        self,
        state: dict[str, Any],
        sources: tuple[TranslationSourceItem, ...],
    ) -> None:
        if state.get("schema_version") != self.schema_version:
            raise ValueError("unsupported translation workspace schema")
        rows = state.get("items")
        if not isinstance(rows, list) or len(rows) != len(sources):
            raise ValueError(
                "workspace sources changed; use a new workspace directory to preserve "
                "review history"
            )
        expected = {item.pair_id: item.as_json() for item in sources}
        existing = {row["source"]["pair_id"]: row["source"] for row in rows}
        if len(existing) != len(rows):
            raise ValueError("persisted translation pair IDs must be unique")
        if expected != existing:
            raise ValueError(
                "workspace sources changed; use a new workspace directory to preserve "
                "review history"
            )
        expected_hash = _canonical_sha256(
            [item.as_json() for item in sorted(sources, key=lambda value: value.pair_id)]
        )
        if state.get("collection_sha256") != expected_hash:
            raise ValueError("persisted collection hash mismatch")

    def public_state(self) -> dict[str, Any]:
        with self._lock:
            state = cast(
                dict[str, Any],
                json.loads(json.dumps(self._state, ensure_ascii=False)),
            )
        translated = sum(row["translation"] is not None for row in state["items"])
        reviewed = sum(self._review_passes(row["review"]) for row in state["items"])
        state["summary"] = {
            "total": len(state["items"]),
            "translated": translated,
            "reviewed": reviewed,
            "ready": reviewed == len(state["items"]),
            "final_path": str(self.final_path) if self.final_path.exists() else None,
        }
        return state

    def save_translation(
        self,
        pair_id: str,
        *,
        translator_id: str,
        uk_prompt: str,
        uk_image_text: str | None,
        uk_rubric_text: str,
        translation_method: str = "human",
    ) -> dict[str, Any]:
        translator_id = translator_id.strip()
        uk_prompt = uk_prompt.strip()
        uk_image_text = uk_image_text.strip() if uk_image_text is not None else None
        uk_rubric_text = uk_rubric_text.strip()
        if not translator_id:
            raise ValueError("translator ID is required")
        if translation_method not in {
            "human",
            "machine_deepl",
            "machine_deepl_ai_revised",
        }:
            raise ValueError("unsupported translation method")
        if not uk_prompt:
            raise ValueError("Ukrainian prompt is required")
        if not any("\u0400" <= char <= "\u04ff" for char in uk_prompt):
            raise ValueError("Ukrainian prompt must contain Cyrillic text")
        if not uk_rubric_text:
            raise ValueError("Ukrainian rubric is required")
        with self._lock:
            row = self._find(pair_id)
            context_required = row["source"]["context_text"] is not None
            if context_required and not uk_image_text:
                raise ValueError("this item requires translated image/context text")
            if not context_required and uk_image_text is not None:
                raise ValueError("this item has no image/context text to translate")
            previous = row["translation"]
            changed = previous is None or any(
                (
                    previous["translator_id"] != translator_id,
                    previous["translation_method"] != translation_method,
                    previous["uk_prompt"] != uk_prompt,
                    previous["uk_image_text"] != uk_image_text,
                    previous.get("uk_rubric_text") != uk_rubric_text,
                )
            )
            row["translation"] = {
                "translator_id": translator_id,
                "translation_method": translation_method,
                "translated_at": _now() if changed else previous["translated_at"],
                "uk_image_text": uk_image_text,
                "uk_prompt": uk_prompt,
                "uk_rubric_text": uk_rubric_text,
            }
            if changed:
                if row["review"] is not None:
                    row["review_history"].append(row["review"])
                row["review"] = None
                row["revision"] += 1
                self.final_path.unlink(missing_ok=True)
                self.ai_final_path.unlink(missing_ok=True)
            self._save()
            return self.public_state()

    def import_translation_draft(
        self,
        records: list[dict[str, Any]],
        *,
        translator_id: str,
        translation_method: str,
    ) -> dict[str, Any]:
        if not translator_id.strip():
            raise ValueError("translator ID is required")
        expected_by_pair = {row["source"]["pair_id"]: row["source"] for row in self._state["items"]}
        if len(records) != len(expected_by_pair):
            raise ValueError("draft record count does not match source collection")
        records_by_pair = {str(record.get("pair_id", "")): record for record in records}
        if set(records_by_pair) != set(expected_by_pair):
            raise ValueError("draft pair IDs do not match source collection")
        for pair_id, record in records_by_pair.items():
            source = expected_by_pair[pair_id]
            exact_matches = {
                "category": source["category"],
                "expected_behavior": source["expected_behavior"],
                "prompt_text_en": source["prompt_text"],
                "rubric_text_en": source["rubric_text"],
                "source_content_sha256": source["source_content_sha256"],
                "stable_id": source["stable_id"],
            }
            if any(record.get(key) != value for key, value in exact_matches.items()):
                raise ValueError(f"draft source identity mismatch: {pair_id}")
            if record.get("language") != "uk":
                raise ValueError(f"draft language must be uk: {pair_id}")
            if record.get("context_text_en") != source["context_text"]:
                raise ValueError(f"draft English context mismatch: {pair_id}")
        for pair_id in sorted(records_by_pair):
            record = records_by_pair[pair_id]
            self.save_translation(
                pair_id,
                translator_id=translator_id,
                uk_prompt=str(record.get("prompt_text", "")),
                uk_image_text=(
                    str(record["context_text"]) if record.get("context_text") is not None else None
                ),
                uk_rubric_text=str(record.get("rubric_text", "")),
                translation_method=translation_method,
            )
        return self.public_state()

    def add_ai_review(
        self,
        pair_id: str,
        *,
        reviewer_kind: str = "ai_agent",
        reviewer_profile: str,
        prompt_version: str,
        dimensions: dict[str, str],
        confidence: float,
        notes: str,
        disposition: str,
    ) -> dict[str, Any]:
        if reviewer_kind not in {"ai_agent", "ai_subagent"}:
            raise ValueError("AI reviewer kind must be ai_agent or ai_subagent")
        if not reviewer_profile or not prompt_version:
            raise ValueError("AI reviewer profile and prompt version are required")
        if set(dimensions) != set(REVIEW_DIMENSIONS):
            raise ValueError("all five AI review dimensions are required")
        if any(value not in {"pass", "fail", "not_judgeable"} for value in dimensions.values()):
            raise ValueError("AI review dimensions must be pass, fail, or not_judgeable")
        if not 0 <= confidence <= 1:
            raise ValueError("AI review confidence must be between zero and one")
        if disposition not in {"accept", "revise", "exclude_source_defect"}:
            raise ValueError("invalid AI review disposition")
        with self._lock:
            row = self._find(pair_id)
            translation = row["translation"]
            if translation is None:
                raise ValueError("translation must exist before AI review")
            translation_hash = content_hash(
                translation["uk_prompt"],
                translation["uk_image_text"],
                translation["uk_rubric_text"],
            )
            row["ai_reviews"].append(
                {
                    "confidence": confidence,
                    "dimensions": dimensions,
                    "disposition": disposition,
                    "notes": notes.strip(),
                    "prompt_version": prompt_version,
                    "reviewed_at": _now(),
                    "reviewer_kind": reviewer_kind,
                    "reviewer_profile": reviewer_profile,
                    "translation_hash": translation_hash,
                    "translation_revision": row["revision"],
                }
            )
            self.ai_final_path.unlink(missing_ok=True)
            self._save()
            return self.public_state()

    def save_review(
        self,
        pair_id: str,
        *,
        reviewer_id: str,
        dimensions: dict[str, str],
        review_notes: str,
    ) -> dict[str, Any]:
        reviewer_id = reviewer_id.strip()
        if not reviewer_id:
            raise ValueError("reviewer ID is required")
        if set(dimensions) != set(REVIEW_DIMENSIONS):
            raise ValueError("all five review dimensions are required")
        if any(value not in {"pass", "fail"} for value in dimensions.values()):
            raise ValueError("review dimensions must be pass or fail")
        with self._lock:
            row = self._find(pair_id)
            translation = row["translation"]
            if translation is None:
                raise ValueError("translation must be completed before review")
            if reviewer_id == translation["translator_id"]:
                raise ValueError("reviewer must differ from translator")
            row["review"] = {
                **dimensions,
                "review_notes": review_notes.strip(),
                "reviewed_at": _now(),
                "reviewer_id": reviewer_id,
                "translation_revision": row["revision"],
            }
            row["review_history"].append(row["review"])
            self.final_path.unlink(missing_ok=True)
            self._save()
            return self.public_state()

    def finalize(self) -> Path:
        with self._lock:
            records: list[dict[str, Any]] = []
            for row in self._state["items"]:
                translation = row["translation"]
                review = row["review"]
                if translation is None or not self._review_passes(review):
                    raise ValueError("every item must have an independently passing review")
                if translation["translation_method"] != "human":
                    raise ValueError(
                        "human-reviewed export requires translations recorded as human"
                    )
                assert review is not None
                source = row["source"]
                records.append(
                    {
                        "pair_id": source["pair_id"],
                        "source_content_sha256": source["source_content_sha256"],
                        "uk_prompt": translation["uk_prompt"],
                        "uk_image_text": translation["uk_image_text"],
                        "uk_rubric_text": translation["uk_rubric_text"],
                        "translator_id": translation["translator_id"],
                        "translation_method": translation["translation_method"],
                        "translated_at": translation["translated_at"],
                        "reviewer_id": review["reviewer_id"],
                        "reviewed_at": review["reviewed_at"],
                        **{dimension: review[dimension] for dimension in REVIEW_DIMENSIONS},
                        "review_notes": review["review_notes"],
                    }
                )
            self._write_json(self.final_path, records)
            return self.final_path

    def finalize_ai_reviewed(self, *, excluded_pair_ids: set[str]) -> Path:
        output_path = self.ai_final_path
        with self._lock:
            records: list[dict[str, Any]] = []
            for row in self._state["items"]:
                pair_id = row["source"]["pair_id"]
                reviews = row["ai_reviews"]
                if not reviews:
                    raise ValueError(f"missing AI review: {pair_id}")
                latest = reviews[-1]
                translation = row["translation"]
                if latest["translation_revision"] != row["revision"] or latest[
                    "translation_hash"
                ] != content_hash(
                    translation["uk_prompt"],
                    translation["uk_image_text"],
                    translation["uk_rubric_text"],
                ):
                    raise ValueError(f"stale AI review: {pair_id}")
                if pair_id in excluded_pair_ids:
                    if latest["disposition"] != "exclude_source_defect":
                        raise ValueError(
                            f"excluded pair lacks source-defect disposition: {pair_id}"
                        )
                    continue
                if latest["disposition"] != "accept" or any(
                    latest["dimensions"][name] != "pass" for name in REVIEW_DIMENSIONS
                ):
                    raise ValueError(f"pair lacks passing AI audit: {pair_id}")
                if translation["translation_method"] not in {
                    "machine_deepl",
                    "machine_deepl_ai_revised",
                }:
                    raise ValueError(
                        f"exploratory export requires disclosed machine/AI provenance: {pair_id}"
                    )
                records.append(
                    {
                        "pair_id": pair_id,
                        "source_content_sha256": row["source"]["source_content_sha256"],
                        "uk_prompt": translation["uk_prompt"],
                        "uk_image_text": translation["uk_image_text"],
                        "uk_rubric_text": translation["uk_rubric_text"],
                        "translation_method": translation["translation_method"],
                        "translated_at": translation["translated_at"],
                        "ai_review": latest,
                        "reviewer_status": "ai_reviewed_exploratory",
                    }
                )
            self._write_json(output_path, records)
            return output_path

    def _find(self, pair_id: str) -> dict[str, Any]:
        for row in self._state["items"]:
            if row["source"]["pair_id"] == pair_id:
                return cast(dict[str, Any], row)
        raise KeyError(f"unknown pair ID: {pair_id}")

    @staticmethod
    def _review_passes(review: dict[str, Any] | None) -> bool:
        return review is not None and all(review.get(name) == "pass" for name in REVIEW_DIMENSIONS)

    def _save(self) -> None:
        self._state["updated_at"] = _now()
        self._write_json(self.progress_path, self._state)

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        temporary.chmod(0o600)
        temporary.replace(path)
