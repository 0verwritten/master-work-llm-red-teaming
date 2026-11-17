from __future__ import annotations

import json
from pathlib import Path

import pytest

from apps.translation_review.workspace import (
    REVIEW_DIMENSIONS,
    TranslationSourceItem,
    TranslationWorkspace,
)
from harness.bilingual_import import TranslationRecord


def _source(index: int, *, context: bool = False) -> TranslationSourceItem:
    return TranslationSourceItem(
        stable_id=f"source-{index}",
        pair_id=f"pair-{index}",
        source_content_sha256=f"{index:064x}",
        prompt_text=f"English fixture {index}",
        context_text=f"Context fixture {index}" if context else None,
        category="fixture",
        rubric_text="Fixture rubric",
        expected_behavior="refuse_or_safe_redirect",
    )


def _passes() -> dict[str, str]:
    return dict.fromkeys(REVIEW_DIMENSIONS, "pass")


def test_workspace_autosaves_arbitrary_collection_and_finalizes_import_records(
    tmp_path: Path,
) -> None:
    workspace = TranslationWorkspace(tmp_path / "private", (_source(1), _source(2)))

    english_sources = json.loads(workspace.english_sources_path.read_text())
    assert len(english_sources) == 2
    assert english_sources[0]["language"] == "en"
    assert english_sources[0]["prompt_text"] == "English fixture 1"
    assert workspace.english_sources_path.stat().st_mode & 0o777 == 0o600

    workspace.save_translation(
        "pair-1",
        translator_id="translator-a",
        uk_prompt="Український тест один",
        uk_image_text=None,
        uk_rubric_text="Українська рубрика один",
    )
    workspace.save_translation(
        "pair-2",
        translator_id="translator-a",
        uk_prompt="Український тест два",
        uk_image_text=None,
        uk_rubric_text="Українська рубрика два",
    )
    for pair_id in ("pair-1", "pair-2"):
        workspace.save_review(
            pair_id,
            reviewer_id="reviewer-b",
            dimensions=_passes(),
            review_notes="",
        )

    assert workspace.public_state()["summary"] == {
        "final_path": None,
        "ready": True,
        "reviewed": 2,
        "total": 2,
        "translated": 2,
    }
    final_path = workspace.finalize()
    records = json.loads(final_path.read_text())
    assert len(records) == 2
    assert all(TranslationRecord.model_validate(record) for record in records)
    assert final_path.stat().st_mode & 0o777 == 0o600


def test_translation_edit_invalidates_review_and_source_changes_require_new_workspace(
    tmp_path: Path,
) -> None:
    root = tmp_path / "private"
    workspace = TranslationWorkspace(root, (_source(1),))
    workspace.save_translation(
        "pair-1",
        translator_id="translator-a",
        uk_prompt="Початковий український тест",
        uk_image_text=None,
        uk_rubric_text="Початкова українська рубрика",
    )
    with pytest.raises(ValueError, match="differ"):
        workspace.save_review(
            "pair-1",
            reviewer_id="translator-a",
            dimensions=_passes(),
            review_notes="",
        )
    workspace.save_review(
        "pair-1",
        reviewer_id="reviewer-b",
        dimensions=_passes(),
        review_notes="approved",
    )
    workspace.finalize()

    workspace.save_translation(
        "pair-1",
        translator_id="translator-a",
        uk_prompt="Виправлений український тест",
        uk_image_text=None,
        uk_rubric_text="Виправлена українська рубрика",
    )
    state = workspace.public_state()
    assert state["items"][0]["review"] is None
    assert state["summary"]["ready"] is False
    assert not workspace.final_path.exists()

    with pytest.raises(ValueError, match="sources changed"):
        TranslationWorkspace(root, (_source(1), _source(2)))


def test_context_translation_presence_is_enforced(tmp_path: Path) -> None:
    workspace = TranslationWorkspace(tmp_path / "private", (_source(1, context=True),))

    with pytest.raises(ValueError, match="requires"):
        workspace.save_translation(
            "pair-1",
            translator_id="translator-a",
            uk_prompt="Український тест",
            uk_image_text=None,
            uk_rubric_text="Українська рубрика",
        )


def test_import_translation_draft_validates_source_echoes(tmp_path: Path) -> None:
    source = _source(1)
    workspace = TranslationWorkspace(tmp_path / "private", (source,))
    draft = [
        {
            "stable_id": source.stable_id,
            "pair_id": source.pair_id,
            "source_content_sha256": source.source_content_sha256,
            "language": "uk",
            "category": source.category,
            "expected_behavior": source.expected_behavior,
            "prompt_text": "Український тест",
            "prompt_text_en": source.prompt_text,
            "context_text": None,
            "context_text_en": None,
            "rubric_text": "Українська рубрика",
            "rubric_text_en": source.rubric_text,
        }
    ]

    state = workspace.import_translation_draft(
        draft,
        translator_id="deepl-private-draft",
        translation_method="machine_deepl",
    )

    assert state["summary"]["translated"] == 1
    assert state["items"][0]["translation"]["uk_rubric_text"] == "Українська рубрика"
    assert state["items"][0]["translation"]["translation_method"] == "machine_deepl"


def test_human_and_ai_exports_enforce_distinct_provenance(tmp_path: Path) -> None:
    workspace = TranslationWorkspace(tmp_path / "private", (_source(1),))
    workspace.save_translation(
        "pair-1",
        translator_id="deepl-codex-revised",
        uk_prompt="Український тест",
        uk_image_text=None,
        uk_rubric_text="Українська рубрика",
        translation_method="machine_deepl_ai_revised",
    )
    workspace.save_review(
        "pair-1",
        reviewer_id="reviewer-b",
        dimensions=_passes(),
        review_notes="fixture only",
    )
    workspace.add_ai_review(
        "pair-1",
        reviewer_profile="codex-fixture",
        prompt_version="fixture-v1",
        dimensions=_passes(),
        confidence=0.9,
        notes="fixture",
        disposition="accept",
    )

    with pytest.raises(ValueError, match="recorded as human"):
        workspace.finalize()
    path = workspace.finalize_ai_reviewed(excluded_pair_ids=set())
    [record] = json.loads(path.read_text())
    assert record["translation_method"] == "machine_deepl_ai_revised"
    assert record["reviewer_status"] == "ai_reviewed_exploratory"

    workspace.save_translation(
        "pair-1",
        translator_id="deepl-codex-revised",
        uk_prompt="Змінений український тест",
        uk_image_text=None,
        uk_rubric_text="Українська рубрика",
        translation_method="machine_deepl_ai_revised",
    )
    assert not path.exists()
    with pytest.raises(ValueError, match="stale AI review"):
        workspace.finalize_ai_reviewed(excluded_pair_ids=set())
