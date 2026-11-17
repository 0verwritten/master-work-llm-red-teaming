from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from apps.harmbench_uk_review.corrections import (
    CORRECTIONS,
    RESTORE_ENGLISH_CONTEXT,
    FieldCorrection,
)

STRATUM = "harmbench_subset"
EXPECTED_COUNT = 120
SCHEMA_VERSION = "harmbench-uk-correction-proposal-v1"
SUBSET_PLAN_SHA256 = "9f3f68635b745519c38c1d5e06b76c908c1d6ec0e43a9ca5b122004ea5c8c6b3"


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PairSource:
    index: int
    pair_id: str
    category: str
    expected_behavior: str
    english_content_sha256: str
    machine_content_sha256: str
    english_prompt: str
    english_context: str | None
    english_rubric: str
    machine_prompt: str
    machine_context: str | None
    machine_rubric: str


def load_pairs(
    dataset_path: Path,
    *,
    expected_count: int = EXPECTED_COUNT,
    expected_subset_plan_sha256: str = SUBSET_PLAN_SHA256,
) -> tuple[tuple[PairSource, ...], str, str]:
    dataset = json.loads(dataset_path.read_text())
    manifest = dataset["manifest"]
    subset_plan_sha256 = str(manifest.get("subset_plan_sha256", ""))
    if subset_plan_sha256 != expected_subset_plan_sha256:
        raise ValueError("dataset subset-plan hash does not match the selected collection")
    items = dataset["items"]
    english = {
        item["pair_id"]: item
        for item in items
        if item["language"] == "en" and item["stratum"] == STRATUM
    }
    ukrainian = {
        item["pair_id"]: item
        for item in items
        if item["language"] == "uk" and item["stratum"] == STRATUM
    }
    if len(english) != expected_count or len(ukrainian) != expected_count:
        raise ValueError(
            f"expected {expected_count} English and {expected_count} Ukrainian "
            f"pairs, found {len(english)} and {len(ukrainian)}"
        )
    if set(english) != set(ukrainian):
        raise ValueError("English and Ukrainian pair IDs do not match")
    pairs: list[PairSource] = []
    for index, pair_id in enumerate(sorted(english), start=1):
        en = english[pair_id]
        uk = ukrainian[pair_id]
        pairs.append(
            PairSource(
                index=index,
                pair_id=pair_id,
                category=en["category"],
                expected_behavior=en["expected_behavior"],
                english_content_sha256=en["content_sha256"],
                machine_content_sha256=uk["content_sha256"],
                english_prompt=en["prompt_text"],
                english_context=en["context_text"],
                english_rubric=en["rubric_text"],
                machine_prompt=uk["prompt_text"],
                machine_context=uk["context_text"],
                machine_rubric=uk["rubric_text"],
            )
        )
    return tuple(pairs), cast(str, manifest["dataset_sha256"]), subset_plan_sha256


def _proposed_fields(pair: PairSource) -> tuple[dict[str, str | None], list[str], dict[str, Any]]:
    proposed = {
        "prompt_text": pair.machine_prompt,
        "context_text": pair.machine_context,
        "rubric_text": pair.machine_rubric,
    }
    changed: list[str] = []
    rationale: dict[str, Any] = {}
    correction = CORRECTIONS.get(pair.pair_id, {})
    for field, raw_entry in correction.items():
        entry = cast(FieldCorrection, raw_entry)
        if field == "restore_english_context":
            if pair.machine_context == pair.english_context:
                continue
            proposed["context_text"] = pair.english_context
            changed.append("context_text")
            rationale["context_text"] = {
                "severity": entry["severity"],
                "note": entry["rationale"],
            }
            continue
        new_text = entry["text"]
        if new_text == RESTORE_ENGLISH_CONTEXT:
            raise ValueError(f"misconfigured restore marker for {pair.pair_id}")
        current = proposed[field]
        if new_text == current:
            continue
        proposed[field] = new_text
        changed.append(field)
        rationale[field] = {"severity": entry["severity"], "note": entry["rationale"]}
    return proposed, sorted(set(changed)), rationale


def build_records(pairs: tuple[PairSource, ...]) -> list[dict[str, Any]]:
    known = set(CORRECTIONS)
    present = {pair.pair_id for pair in pairs}
    unknown = known - present
    if unknown:
        raise ValueError(f"corrections reference unknown pair IDs: {sorted(unknown)}")
    records: list[dict[str, Any]] = []
    for pair in pairs:
        proposed, changed, rationale = _proposed_fields(pair)
        disposition = "propose_correction" if changed else "accept_unchanged"
        proposal_content_sha256 = canonical_sha256(
            {
                "prompt_text": proposed["prompt_text"],
                "context_text": proposed["context_text"],
                "rubric_text": proposed["rubric_text"],
            }
        )
        records.append(
            {
                "index": pair.index,
                "pair_id": pair.pair_id,
                "category": pair.category,
                "expected_behavior": pair.expected_behavior,
                "english_content_sha256": pair.english_content_sha256,
                "machine_content_sha256": pair.machine_content_sha256,
                "english": {
                    "prompt_text": pair.english_prompt,
                    "context_text": pair.english_context,
                    "rubric_text": pair.english_rubric,
                },
                "machine_deepl": {
                    "prompt_text": pair.machine_prompt,
                    "context_text": pair.machine_context,
                    "rubric_text": pair.machine_rubric,
                },
                "proposed": proposed,
                "disposition": disposition,
                "changed_fields": changed,
                "rationale": rationale,
                "proposal_content_sha256": proposal_content_sha256,
            }
        )
    return records


def build_proposal(
    dataset_path: Path,
    *,
    expected_count: int = EXPECTED_COUNT,
    expected_subset_plan_sha256: str = SUBSET_PLAN_SHA256,
) -> dict[str, Any]:
    pairs, dataset_sha256, subset_plan_sha256 = load_pairs(
        dataset_path,
        expected_count=expected_count,
        expected_subset_plan_sha256=expected_subset_plan_sha256,
    )
    records = build_records(pairs)
    collection_sha256 = canonical_sha256(
        [
            {
                "pair_id": record["pair_id"],
                "english_content_sha256": record["english_content_sha256"],
                "machine_content_sha256": record["machine_content_sha256"],
                "proposal_content_sha256": record["proposal_content_sha256"],
            }
            for record in records
        ]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "subset_plan_sha256": subset_plan_sha256,
        "dataset_sha256": dataset_sha256,
        "generated_at": datetime.now(UTC).isoformat(),
        "count": len(records),
        "records": records,
        "collection_sha256": collection_sha256,
    }


def write_json_0600(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)
