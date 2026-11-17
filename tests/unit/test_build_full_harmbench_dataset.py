from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_full_harmbench_dataset import build_full_dataset

WORKSPACE = Path(".translation-workspace")
ENGLISH = WORKSPACE / "harmbench-english-source.json"
UKRAINIAN = WORKSPACE / "harmbench-uk-translations.json"
HUMAN_REVIEW = WORKSPACE / "harmbench-uk-reviewed-translations-v1.json"


@pytest.mark.skipif(
    not ENGLISH.exists() or not UKRAINIAN.exists() or not HUMAN_REVIEW.exists(),
    reason="private artifacts absent",
)
def test_full_dataset_preserves_human_review_and_quarantines_ai_review() -> None:
    dataset, audit = build_full_dataset(
        json.loads(ENGLISH.read_text()),
        json.loads(UKRAINIAN.read_text()),
        json.loads(HUMAN_REVIEW.read_text()),
    )
    uk = [row for row in dataset["items"] if row["language"] == "uk"]
    assert len(dataset["items"]) == 1020
    assert len(uk) == 510
    assert sum(row["reviewer_status"] == "human_verified" for row in uk) == 120
    assert sum(row["reviewer_status"] == "ai_reviewed_exploratory" for row in uk) == 390
    assert audit["record_count"] == 390
    assert all(all(record["checks"].values()) for record in audit["records"])
    assert dataset["manifest"]["ai_audit_collection_sha256"] == audit["collection_sha256"]
