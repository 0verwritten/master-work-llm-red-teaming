from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from apps.harmbench_response_review.workspace import (
    PROGRESS_SCHEMA,
    QUEUE_SCHEMA,
    ResponseReviewWorkspace,
    canonical_sha256,
    subject_sha256,
)
from scripts.adjudicate_harmbench_human_review import adjudicate
from scripts.analyze_harmbench_response_review import analyse_single_review
from scripts.build_harmbench_human_sample import select_sample
from scripts.build_harmbench_response_review import select_items
from scripts.merge_harmbench_human_recheck import merge


def _item(
    identifier: str, language: str, label: str, category: str = "illegal"
) -> dict[str, object]:
    item: dict[str, object] = {
        "annotation_id": identifier,
        "pair_id": identifier.split("-")[0],
        "language": language,
        "category": category,
        "round_number": 1,
        "requested_behavior": "Requested behaviour",
        "rubric": "Three-level rubric",
        "target_response": f"Response {identifier}",
        "automated_label": label,
        "session_id": f"session-{identifier}",
        "round_id": f"round-{identifier}",
    }
    item["subject_sha256"] = subject_sha256(item)
    return item


def _workspace(tmp_path: Path) -> ResponseReviewWorkspace:
    items = [_item("pair-en", "en", "partial")]
    items[0]["reviewer_c_label"] = "complied"
    items[0]["reviewer_c_confidence"] = "medium"
    items[0]["audit_flags"] = ["reviewer_c_disagrees_with_automated_judge"]
    items[0]["tricky_case"] = True
    queue = {
        "schema_version": QUEUE_SCHEMA,
        "rubric_version": "graded-three-level-v1",
        "items": items,
        "queue_sha256": canonical_sha256(items),
    }
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(json.dumps(queue))
    return ResponseReviewWorkspace(queue_path, tmp_path / "review.json", "reviewer-a")


def test_selection_keeps_all_minority_labels_and_counterpart() -> None:
    records = [
        _item("pair-en", "en", "partial"),
        _item("pair-uk", "uk", "refused"),
        _item("other-en", "en", "complied"),
        _item("other-uk", "uk", "refused"),
    ]
    selected = select_items(records)
    assert "all_automated_partial" in selected["pair-en"]
    assert "bilingual_same_round_counterpart" in selected["pair-uk"]
    assert "all_automated_complied" in selected["other-en"]
    assert "bilingual_same_round_counterpart" in selected["other-uk"]


def test_public_state_hides_automated_label_and_sampling_reason(tmp_path: Path) -> None:
    state = _workspace(tmp_path).public_state()
    serialized = json.dumps(state)
    assert "automated_label" not in serialized
    assert "selection_reasons" not in serialized
    assert "reviewer_c_label" not in serialized
    assert "audit_flags" not in serialized
    assert "tricky_case" not in serialized
    assert "session_id" not in serialized
    assert "round_id" not in serialized


def test_partial_requires_subtype_and_review_is_hash_bound(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    item = workspace.public_state()["items"][0]
    with pytest.raises(ValueError, match="require a subtype"):
        workspace.save_review(
            item["annotation_id"],
            reviewer_id="reviewer-a",
            label="partial",
            confidence="high",
            partial_subtype=None,
            notes="",
            expected_subject_sha256=item["subject_sha256"],
        )
    state = workspace.save_review(
        item["annotation_id"],
        reviewer_id="reviewer-a",
        label="partial",
        confidence="high",
        partial_subtype="usable_but_incomplete",
        notes="Some usable detail remains.",
        expected_subject_sha256=item["subject_sha256"],
    )
    assert state["summary"]["completed"] == 1


def test_progress_cannot_be_reopened_as_another_reviewer(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    with pytest.raises(ValueError, match="different reviewer"):
        ResponseReviewWorkspace(workspace.queue_path, workspace.progress_path, "reviewer-b")


def test_targeted_human_sample_covers_cells_and_keeps_all_uk_partials() -> None:
    categories = [
        "chemical_biological",
        "copyright",
        "cybercrime_intrusion",
        "harassment_bullying",
        "harmful",
        "illegal",
        "misinformation_disinformation",
    ]
    labels = ("refused", "partial", "complied")
    records: list[dict[str, object]] = []
    reviews: dict[str, dict[str, object]] = {}
    for language in ("en", "uk"):
        for category_index, category in enumerate(categories):
            for label_index, label in enumerate(labels):
                identifier = f"{language}-{category_index}-{label_index}"
                item = _item(identifier, language, label, category)
                records.append(item)
                reviews[identifier] = {
                    "label": labels[(label_index + category_index + 1) % len(labels)],
                    "confidence": "medium" if label_index == 1 else "high",
                    "partial_subtype": None,
                }

    selected = select_sample(records, reviews, size=24)
    cells = {(item["language"], item["category"]) for item in selected}
    assert cells == {(language, category) for language in ("en", "uk") for category in categories}
    uk_partials = {
        item["annotation_id"]
        for item in records
        if item["language"] == "uk" and item["automated_label"] == "partial"
    }
    assert uk_partials <= {item["annotation_id"] for item in selected}
    assert len(selected) == 24


def test_single_human_review_analysis_is_supported() -> None:
    item = _item("one-en", "en", "partial")
    item["reviewer_c_label"] = "complied"
    queue = {
        "queue_sha256": canonical_sha256([item]),
        "sample_design": "targeted_error_audit_not_probability_sample",
        "items": [item],
    }
    progress: dict[str, Any] = {
        "reviewer_id": "human-a",
        "reviews": {
            "one-en": {
                "label": "partial",
                "partial_subtype": "usable_but_incomplete",
            }
        },
    }
    result = analyse_single_review(queue, progress)
    assert result["status"] == "complete"
    assert result["automated_judge_vs_human"]["all"]["n"] == 1
    assert result["artificial_reviewer_c_vs_human"]["all"]["n"] == 1


def test_conditional_adjudication_preserves_source_and_requires_other_refusal() -> None:
    first = _item("first-en", "en", "refused")
    first["reviewer_c_label"] = "partial"
    second = _item("second-en", "en", "partial")
    second["reviewer_c_label"] = "complied"
    items = [first, second]
    queue = {
        "schema_version": QUEUE_SCHEMA,
        "queue_sha256": canonical_sha256(items),
        "items": items,
    }
    progress: dict[str, Any] = {
        "schema_version": PROGRESS_SCHEMA,
        "queue_sha256": queue["queue_sha256"],
        "reviewer_id": "human-a",
        "reviews": {
            identifier: {
                "label": "partial",
                "partial_subtype": "ambiguous_boundary",
            }
            for identifier in ("first-en", "second-en")
        },
    }
    result = adjudicate(queue, progress)
    assert progress["reviews"]["first-en"]["label"] == "partial"
    assert result["reviews"]["first-en"]["label"] == "refused"
    assert result["reviews"]["second-en"]["label"] == "partial"
    assert len(result["changes"]) == 1


def test_blind_recheck_merge_preserves_initial_decision() -> None:
    item = _item("one-en", "en", "partial")
    source_queue = {
        "schema_version": QUEUE_SCHEMA,
        "queue_sha256": canonical_sha256([item]),
        "items": [item],
    }
    recheck_item = {**item, "previous_human_label": "partial"}
    recheck_queue = {
        "schema_version": QUEUE_SCHEMA,
        "created_from_queue_sha256": source_queue["queue_sha256"],
        "queue_sha256": canonical_sha256([recheck_item]),
        "items": [recheck_item],
    }
    initial = {
        "reviewer_id": "human-a",
        "label": "partial",
        "confidence": "medium",
        "partial_subtype": "ambiguous_boundary",
        "subject_sha256": item["subject_sha256"],
    }
    confirmed = {
        "reviewer_id": "human-a-recheck",
        "label": "refused",
        "confidence": "high",
        "partial_subtype": None,
        "subject_sha256": item["subject_sha256"],
    }
    source_review = {
        "schema_version": PROGRESS_SCHEMA,
        "queue_sha256": source_queue["queue_sha256"],
        "reviewer_id": "human-a",
        "reviews": {"one-en": initial},
    }
    recheck_review = {
        "schema_version": PROGRESS_SCHEMA,
        "queue_sha256": recheck_queue["queue_sha256"],
        "reviewer_id": "human-a-recheck",
        "reviews": {"one-en": confirmed},
    }
    result = merge(source_queue, source_review, recheck_queue, recheck_review)
    assert result["reviews"]["one-en"]["label"] == "refused"
    assert result["reviews"]["one-en"]["initial_review"]["label"] == "partial"
    assert result["changes"][0]["old_label"] == "partial"
