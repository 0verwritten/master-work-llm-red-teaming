from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_reviewed_harmbench_dataset import build_reviewed_dataset

WORKSPACE = Path(".translation-workspace")
BASE = WORKSPACE / "bilingual-dataset-v1.json"
REVIEW = WORKSPACE / "harmbench-uk-reviewed-translations-v1.json"


@pytest.mark.skipif(not BASE.exists() or not REVIEW.exists(), reason="private artifacts absent")
def test_real_review_builds_120_verified_harmbench_rows() -> None:
    payload = build_reviewed_dataset(json.loads(BASE.read_text()), json.loads(REVIEW.read_text()))
    selected = [
        row
        for row in payload["items"]
        if row["language"] == "uk" and row["stratum"] == "harmbench_subset"
    ]
    assert len(selected) == 120
    assert {row["reviewer_status"] for row in selected} == {"human_verified"}
    assert payload["manifest"]["ukrainian_human_verified_count"] == 120
    assert payload["manifest"]["dataset_sha256"] == (
        "276d21fa96842830d84ace87d9e47628d31f4936bc20b8020ad51fa75d6936ef"
    )
