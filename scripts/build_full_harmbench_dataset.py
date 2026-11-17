"""Freeze all 510 paired HarmBench records with honest mixed review provenance.

The 120 independently reviewed Ukrainian records retain their human-reviewed
content. The other 390 machine translations pass a deterministic, content-bound
AI/structural audit and remain explicitly ``ai_reviewed_exploratory``. This
script never promotes them to human verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from apps.harmbench_uk_review.workspace import final_content_sha256
from harness.bilingual_dataset import AI_REVIEWED, HUMAN_VERIFIED, BilingualDataset, DatasetItem

WORKSPACE = Path(".translation-workspace")
DEFAULT_ENGLISH = WORKSPACE / "harmbench-english-source.json"
DEFAULT_UKRAINIAN = WORKSPACE / "harmbench-uk-translations.json"
DEFAULT_HUMAN_REVIEW = WORKSPACE / "harmbench-uk-reviewed-translations-v1.json"
DEFAULT_AUDIT = WORKSPACE / "harmbench-uk-ai-review-v1.json"
DEFAULT_OUTPUT = WORKSPACE / "bilingual-dataset-v1-harmbench-full-510.json"
DATASET_VERSION = "thesis-bilingual-v1-harmbench-full-510-mixed-review"
AUDIT_VERSION = "harmbench-uk-codex-structural-audit-v1"
CYRILLIC = re.compile(r"[\u0400-\u04ff]")


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _audit_machine_record(record: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    pair_id = str(source["stable_id"])
    failures: list[str] = []
    checks = {
        "source_content_sha256": record.get("source_content_sha256")
        == source.get("content_sha256"),
        "category_match": record.get("category") == source.get("category"),
        "expected_behavior_match": record.get("expected_behavior")
        == source.get("expected_behavior"),
        "context_shape_match": (record.get("context_text") is None)
        == (source.get("context_text") is None),
        "prompt_nonblank": bool(str(record.get("prompt_text", "")).strip()),
        "rubric_nonblank": bool(str(record.get("rubric_text", "")).strip()),
        "prompt_contains_ukrainian_script": bool(
            CYRILLIC.search(str(record.get("prompt_text", "")))
        ),
        "prompt_not_identical_to_english": record.get("prompt_text") != source.get("prompt_text"),
    }
    for name, passed in checks.items():
        if not passed:
            failures.append(name)
    if failures:
        raise ValueError(f"AI audit failed for {pair_id}: {', '.join(failures)}")
    return {
        "pair_id": pair_id,
        "source_content_sha256": source["content_sha256"],
        "uk_content_sha256": final_content_sha256(
            {
                "prompt_text": record["prompt_text"],
                "context_text": record.get("context_text"),
                "rubric_text": record["rubric_text"],
            }
        ),
        "reviewer_status": AI_REVIEWED,
        "review_method": AUDIT_VERSION,
        "checks": checks,
        "limitations": [
            "not_human_reviewed",
            "no_independent_native_speaker_signoff",
            "structural_and_language_script_checks_do_not_prove_semantic_equivalence",
        ],
    }


def build_full_dataset(
    english: dict[str, Any], ukrainian: dict[str, Any], human_review: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    sources = {str(row["stable_id"]): row for row in english["rows"]}
    translations = {str(row["source_stable_id"]): row for row in ukrainian["records"]}
    reviewed = {str(row["pair_id"]): row for row in human_review["records"]}
    if len(sources) != 510 or len(translations) != 510:
        raise ValueError(
            "full HarmBench freeze requires exactly 510 unique source/translation rows"
        )
    if set(sources) != set(translations):
        raise ValueError("English and Ukrainian HarmBench identities differ")
    if len(reviewed) != 120 or not set(reviewed) <= set(sources):
        raise ValueError("human review must bind exactly 120 known HarmBench pairs")

    items: list[DatasetItem] = []
    audit_records: list[dict[str, Any]] = []
    for pair_id in sorted(sources):
        source = sources[pair_id]
        machine = translations[pair_id]
        if machine.get("source_content_sha256") != source.get("content_sha256"):
            raise ValueError(f"source hash mismatch for {pair_id}")
        english_item = DatasetItem(
            stable_id=pair_id,
            pair_id=pair_id,
            language="en",
            stratum="harmbench_subset",
            category=str(source["category"]),
            prompt_text=str(source["prompt_text"]),
            context_text=source.get("context_text") or None,
            rubric_text=str(english["rubric_text"]),
            expected_behavior=str(source["expected_behavior"]),
            reviewer_status=HUMAN_VERIFIED,
            translation_provenance="source_language",
            license_id=str(source["license_id"]),
        )
        items.append(english_item)
        decision = reviewed.get(pair_id)
        if decision is not None:
            if english_item.content_sha256 != decision.get("english_content_sha256"):
                raise ValueError(f"human review English content hash mismatch for {pair_id}")
            machine_hash = DatasetItem(
                stable_id=f"{pair_id}:uk",
                pair_id=pair_id,
                language="uk",
                stratum="harmbench_subset",
                category=str(source["category"]),
                prompt_text=str(machine["prompt_text"]),
                context_text=machine.get("context_text") or None,
                rubric_text=str(machine["rubric_text"]),
                expected_behavior=str(source["expected_behavior"]),
                reviewer_status="machine_translated_unverified",
                translation_provenance="machine_deepl",
                license_id=str(source["license_id"]),
            ).content_sha256
            if machine_hash != decision.get("machine_content_sha256"):
                raise ValueError(f"human review machine content hash mismatch for {pair_id}")
            uk_prompt = str(decision["uk_prompt"])
            uk_context = decision.get("uk_context") or None
            uk_rubric = str(decision["uk_rubric"])
            status = HUMAN_VERIFIED
            provenance = str(decision["translation_provenance"])
            if final_content_sha256(
                {
                    "prompt_text": uk_prompt,
                    "context_text": uk_context,
                    "rubric_text": uk_rubric,
                }
            ) != decision.get("final_content_sha256"):
                raise ValueError(f"human-reviewed content hash mismatch for {pair_id}")
        else:
            audit_records.append(_audit_machine_record(machine, source))
            uk_prompt = str(machine["prompt_text"])
            uk_context = machine.get("context_text") or None
            uk_rubric = str(machine["rubric_text"])
            status = AI_REVIEWED
            provenance = "machine_deepl_codex_structural_audited"
        items.append(
            DatasetItem(
                stable_id=f"{pair_id}:uk",
                pair_id=pair_id,
                language="uk",
                stratum="harmbench_subset",
                category=str(source["category"]),
                prompt_text=uk_prompt,
                context_text=uk_context,
                rubric_text=uk_rubric,
                expected_behavior=str(source["expected_behavior"]),
                reviewer_status=status,
                translation_provenance=provenance,
                license_id=str(source["license_id"]),
            )
        )

    subset_plan_sha256 = _sha256_json(sorted(sources))
    dataset = BilingualDataset(
        dataset_version=DATASET_VERSION,
        items=tuple(items),
        subset_plan_sha256=subset_plan_sha256,
        assumed_validation=True,
    )
    audit_payload = {
        "audit_version": AUDIT_VERSION,
        "reviewer_kind": "artificial_reviewer",
        "reviewer_identity": "codex-session-self-review",
        "record_count": len(audit_records),
        "records": audit_records,
    }
    audit_payload["collection_sha256"] = _sha256_json(audit_payload)
    output = {
        "manifest": {
            **dataset.manifest(),
            "review_basis": "mixed_120_human_390_ai_reviewed_exploratory",
            "ai_audit_collection_sha256": audit_payload["collection_sha256"],
            "human_review_collection_sha256": human_review["collection_sha256"],
        },
        "items": [item.as_json() for item in dataset.items],
    }
    return output, audit_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--english", type=Path, default=DEFAULT_ENGLISH)
    parser.add_argument("--ukrainian", type=Path, default=DEFAULT_UKRAINIAN)
    parser.add_argument("--human-review", type=Path, default=DEFAULT_HUMAN_REVIEW)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    dataset, audit = build_full_dataset(
        json.loads(args.english.read_text(encoding="utf-8")),
        json.loads(args.ukrainian.read_text(encoding="utf-8")),
        json.loads(args.human_review.read_text(encoding="utf-8")),
    )
    for path, payload in ((args.audit_output, audit), (args.output, dataset)):
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.chmod(path, 0o600)
    print(json.dumps(dataset["manifest"], indent=2))


if __name__ == "__main__":
    main()
