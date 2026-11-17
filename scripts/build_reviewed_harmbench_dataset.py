"""Apply the finalized Ukrainian HarmBench review to a new dataset artifact."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from apps.harmbench_uk_review.workspace import final_content_sha256
from harness.bilingual_dataset import BilingualDataset, DatasetItem

WORKSPACE = Path(".translation-workspace")
DEFAULT_BASE = WORKSPACE / "bilingual-dataset-v1.json"
DEFAULT_REVIEW = WORKSPACE / "harmbench-uk-reviewed-translations-v1.json"
DEFAULT_OUTPUT = WORKSPACE / "bilingual-dataset-v1-harmbench-reviewed.json"
DATASET_VERSION = "thesis-bilingual-v1-harmbench-human-reviewed"


def _item(row: dict[str, Any]) -> DatasetItem:
    return DatasetItem(
        stable_id=str(row["stable_id"]),
        pair_id=str(row["pair_id"]),
        language=row["language"],
        stratum=str(row["stratum"]),
        category=str(row["category"]),
        prompt_text=str(row["prompt_text"]),
        context_text=row["context_text"],
        rubric_text=str(row["rubric_text"]),
        expected_behavior=str(row["expected_behavior"]),
        reviewer_status=str(row["reviewer_status"]),
        translation_provenance=str(row["translation_provenance"]),
        license_id=str(row["license_id"]),
    )


def build_reviewed_dataset(base: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    base_manifest = base["manifest"]
    if review["subset_plan_sha256"] != base_manifest["subset_plan_sha256"]:
        raise ValueError("review and base dataset subset hashes differ")
    reviewed = {str(row["pair_id"]): row for row in review["records"]}
    if len(reviewed) != 120 or review["counts"]["human_verified"] != 120:
        raise ValueError("review artifact must contain 120 human-verified records")

    items: list[DatasetItem] = []
    seen: set[str] = set()
    english_by_pair = {
        str(row["pair_id"]): row
        for row in base["items"]
        if row["language"] == "en" and row["stratum"] == "harmbench_subset"
    }
    for row in base["items"]:
        if row["language"] != "uk" or row["stratum"] != "harmbench_subset":
            items.append(_item(row))
            continue
        pair_id = str(row["pair_id"])
        decision = reviewed.get(pair_id)
        if decision is None:
            raise ValueError(f"review is missing {pair_id}")
        if row["content_sha256"] != decision["machine_content_sha256"]:
            raise ValueError(f"machine content hash mismatch for {pair_id}")
        if english_by_pair[pair_id]["content_sha256"] != decision["english_content_sha256"]:
            raise ValueError(f"English content hash mismatch for {pair_id}")
        replacement = DatasetItem(
            stable_id=str(row["stable_id"]),
            pair_id=pair_id,
            language="uk",
            stratum=str(row["stratum"]),
            category=str(row["category"]),
            prompt_text=str(decision["uk_prompt"]),
            context_text=decision["uk_context"],
            rubric_text=str(decision["uk_rubric"]),
            expected_behavior=str(row["expected_behavior"]),
            reviewer_status="human_verified",
            translation_provenance=str(decision["translation_provenance"]),
            license_id=str(row["license_id"]),
        )
        if (
            final_content_sha256(
                {
                    "prompt_text": replacement.prompt_text,
                    "context_text": replacement.context_text,
                    "rubric_text": replacement.rubric_text,
                }
            )
            != decision["final_content_sha256"]
        ):
            raise ValueError(f"final content hash mismatch for {pair_id}")
        items.append(replacement)
        seen.add(pair_id)
    if seen != set(reviewed):
        raise ValueError("review contains pair IDs outside the base HarmBench subset")

    dataset = BilingualDataset(
        dataset_version=DATASET_VERSION,
        items=tuple(items),
        subset_plan_sha256=str(base_manifest["subset_plan_sha256"]),
        assumed_validation=True,
    )
    return {
        "manifest": {
            **dataset.manifest(),
            "harmbench_review_collection_sha256": review["collection_sha256"],
        },
        "items": [item.as_json() for item in dataset.items],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_reviewed_dataset(
        json.loads(args.base.read_text(encoding="utf-8")),
        json.loads(args.review.read_text(encoding="utf-8")),
    )
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.chmod(args.output, 0o600)
    print(json.dumps(payload["manifest"], indent=2))


if __name__ == "__main__":
    main()
