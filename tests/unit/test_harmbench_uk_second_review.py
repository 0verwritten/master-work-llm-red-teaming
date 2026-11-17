from __future__ import annotations

import json
from pathlib import Path

import pytest

from apps.harmbench_uk_review.workspace import REVIEW_DIMENSIONS
from apps.harmbench_uk_second_review import main as server_main
from apps.harmbench_uk_second_review.workspace import (
    QUEUE_SCHEMA,
    REVIEWER_ID,
    SecondReviewWorkspace,
    canonical_sha256,
    subject_sha256,
)
from scripts.build_harmbench_uk_second_review import collect_corrected_pairs, select_pairs

ALL_PASS = {dimension: "pass" for dimension in REVIEW_DIMENSIONS}


def _item(pair_id: str = "harmbench:a") -> dict[str, object]:
    item: dict[str, object] = {
        "pair_id": pair_id,
        "category": "illegal",
        "expected_behavior": "refuse_or_safe_redirect",
        "english_content_sha256": canonical_sha256(["en", pair_id]),
        "english": {
            "prompt_text": "English prompt",
            "context_text": None,
            "rubric_text": "English rubric",
        },
        "ukrainian": {
            "prompt_text": "Український запит",
            "context_text": None,
            "rubric_text": "Українська рубрика",
        },
        "first_review_content_sha256": canonical_sha256(["uk", pair_id]),
        "selection_reasons": ["en_only", "first_reviewer_corrected"],
        "outcome_stratum": "en_only",
    }
    item["subject_sha256"] = subject_sha256(item)
    return item


def _workspace(tmp_path: Path) -> SecondReviewWorkspace:
    items = [_item()]
    queue = {
        "schema_version": QUEUE_SCHEMA,
        "reviewer_id": REVIEWER_ID,
        "items": items,
        "queue_sha256": canonical_sha256(items),
    }
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(json.dumps(queue, ensure_ascii=False))
    return SecondReviewWorkspace(queue_path, tmp_path / "progress.json")


def test_selection_includes_every_correction_and_discordant_pair() -> None:
    categories = {
        "corrected": "cat-a",
        "en": "cat-a",
        "uk": "cat-b",
        **{f"control-{index}": "cat-b" for index in range(20)},
    }
    outcomes = {
        "corrected": "indeterminate",
        "en": "en_only",
        "uk": "uk_only",
        **{f"control-{index}": "neither" for index in range(20)},
    }

    selected = select_pairs(
        category_by_pair=categories,
        outcome_by_pair=outcomes,
        corrected_pairs={"corrected"},
    )

    assert {"corrected", "en", "uk"} <= set(selected)
    assert selected["corrected"] == {"first_reviewer_corrected"}
    assert len([pair for pair in selected if pair.startswith("control-")]) == 2


def test_corrections_include_historical_edits_already_folded_into_dataset() -> None:
    corrected, current, historical = collect_corrected_pairs(
        {
            "new": {"changed_from_machine": True},
            "historical": {"changed_from_machine": False},
            "plain": {"changed_from_machine": False},
        },
        [
            {
                "pair_id": "historical",
                "selection": "accept_proposal",
                "translation_provenance": "machine_deepl_human_corrected",
            },
            {"pair_id": "plain", "selection": "accept_deepl"},
        ],
    )

    assert corrected == {"new", "historical"}
    assert current == {"new"}
    assert historical == {"historical"}


def test_public_state_keeps_first_review_and_outcomes_blind(tmp_path: Path) -> None:
    state = _workspace(tmp_path).public_state()
    serialized = json.dumps(state)

    assert state["reviewer_id"] == "khrystyna"
    assert "selection_reasons" not in serialized
    assert "outcome_stratum" not in serialized
    assert "first_review_content_sha256" not in serialized
    assert "en_only" not in serialized


def test_khrystyna_can_accept_and_progress_resumes(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    item = workspace.public_state()["items"][0]
    state = workspace.save_review(
        item["pair_id"],
        reviewer_id="Khrystyna",
        verdict="accept",
        dimensions=ALL_PASS,
        notes="",
        suggested_correction=None,
        expected_subject_sha256=item["subject_sha256"],
    )

    assert state["summary"]["accepted"] == 1
    resumed = SecondReviewWorkspace(workspace.queue_path, workspace.progress_path)
    assert resumed.public_state()["items"][0]["review"]["reviewer_id"] == "khrystyna"


def test_rejection_requires_distinct_correction_and_nonpass_dimension(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    item = workspace.public_state()["items"][0]
    dimensions = {**ALL_PASS, "naturalness": "fail"}
    correction = {
        **item["ukrainian"],
        "prompt_text": "Виправлений український запит",
    }

    state = workspace.save_review(
        item["pair_id"],
        reviewer_id="khrystyna",
        verdict="needs_fix",
        dimensions=dimensions,
        notes="Unnatural wording.",
        suggested_correction=correction,
        expected_subject_sha256=item["subject_sha256"],
    )

    assert state["summary"]["needs_fix"] == 1
    assert state["items"][0]["review"]["suggested_correction"] == correction


def test_other_reviewer_cannot_write_khrystynas_queue(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    item = workspace.public_state()["items"][0]

    with pytest.raises(ValueError, match="assigned to khrystyna"):
        workspace.save_review(
            item["pair_id"],
            reviewer_id="vlad",
            verdict="accept",
            dimensions=ALL_PASS,
            notes="",
            suggested_correction=None,
            expected_subject_sha256=item["subject_sha256"],
        )


def test_page_exposes_case_insensitive_review_shortcuts() -> None:
    page = server_main._load_page("secret-token").decode()

    assert "Khrystyna · independent bilingual review" in page
    assert "const k=e.key.toLowerCase()" in page
    assert 'if(k==="y")accept()' in page
    assert 'else if(k==="n")openReject()' in page
