from __future__ import annotations

import json
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from apps.measurement_validation.queue_source import (
    SOURCE_DEFECT_EXCLUSIONS,
    AdjudicationQueue,
    load_queue,
    subject_sha256,
)
from apps.measurement_validation.workspace import (
    ValidationWorkspace,
    progress_manifest,
)
from harness.artificial_reviewer import CLAUDE_UK_NATIVE_REVIEW_V1, DIMENSIONS

TRANSLATOR_ID = "deepl-private-draft"
REVIEWER_ID = "uk-native-01"

PAIRS = (
    ("legacy-pair:demo:1", "refuse_or_safe_redirect"),
    ("legacy-pair:demo:2", "comply_safely"),
)


def _translation(pair_id: str) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "source_content_sha256": f"{abs(hash(pair_id)) % 10**8:064d}",
        "uk_prompt": f"тест {pair_id}",
        "uk_image_text": None,
        "uk_rubric_text": f"рубрика {pair_id}",
        "translation_method": "machine_deepl_ai_revised",
        "translated_at": "2026-08-14T00:00:00+00:00",
        "reviewer_status": "ai_reviewed_exploratory",
    }


def _review(translation: dict[str, Any], *, blocking: bool) -> dict[str, Any]:
    dimensions = dict.fromkeys(DIMENSIONS, "pass")
    dimensions["rubric_match"] = "fail"
    defects = [
        {
            "dimension": "rubric_match",
            "severity": "blocking",
            "scope": "collection_wide",
            "note": "the Ukrainian rubric threshold is weaker than the English one",
            "correction_ref": "rubric#variant",
        }
    ]
    if blocking:
        dimensions["semantic_equivalence"] = "fail"
        defects.append(
            {
                "dimension": "semantic_equivalence",
                "severity": "blocking",
                "scope": "item",
                "note": "the Ukrainian adds a speaker attribute absent from the source",
                "correction_ref": "item#semantic_equivalence",
            }
        )
    return {
        "reviewer": CLAUDE_UK_NATIVE_REVIEW_V1.model_dump(mode="json"),
        "reviewed_at": datetime(2026, 8, 16, tzinfo=UTC).isoformat(),
        "subject_sha256": subject_sha256(
            translation["uk_prompt"],
            translation["uk_image_text"],
            translation["uk_rubric_text"],
        ),
        "dimensions": dimensions,
        "disposition": "revise" if blocking else "accept_with_notes",
        "defects": defects,
        "observations": [],
        "status": "ai_adjudication_pending_human_signoff",
        "human_signoff": {
            "reviewer_id": None,
            "reviewed_at": None,
            "decision": None,
            "dimensions": dict.fromkeys(DIMENSIONS),
            "notes": "",
        },
    }


@pytest.fixture
def private_root(tmp_path: Path) -> Path:
    root = tmp_path / ".translation-workspace"
    root.mkdir(mode=0o700)
    translations = [_translation(pair_id) for pair_id, _ in PAIRS]
    (root / "ai-reviewed-translations.json").write_text(
        json.dumps(translations, ensure_ascii=False), encoding="utf-8"
    )
    (root / "translation-progress.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "items": [{"translation": {"translator_id": TRANSLATOR_ID}} for _ in translations],
            }
        ),
        encoding="utf-8",
    )
    (root / "native-review-queue.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "generated_at": "2026-08-16T00:00:00+00:00",
                "reviewer": CLAUDE_UK_NATIVE_REVIEW_V1.model_dump(mode="json"),
                "human_signoff_complete": False,
                "note": "artificial-reviewer output",
                "items": [
                    {
                        "pair_id": translation["pair_id"],
                        "source_content_sha256": translation["source_content_sha256"],
                        "category": "demo",
                        "expected_behavior": behavior,
                        "prior_ai_review": {"disposition": "accept"},
                        "artificial_review": _review(translation, blocking=index == 1),
                        "suggested_uk_text": {"rubric#variant": "виправлений текст"},
                    }
                    for index, (translation, (_, behavior)) in enumerate(
                        zip(translations, PAIRS, strict=True)
                    )
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return root


@pytest.fixture
def queue(private_root: Path) -> AdjudicationQueue:
    return load_queue(private_root, expected_count=len(PAIRS))


def _mutate_queue(root: Path, mutate: Any) -> None:
    path = root / "native-review-queue.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _full(verdict: str = "pass") -> dict[str, str]:
    return dict.fromkeys(DIMENSIONS, verdict)


def test_load_queue_binds_every_record_to_its_translation(queue: AdjudicationQueue) -> None:
    assert [subject.pair_id for subject in queue.subjects] == [pair for pair, _ in PAIRS]
    assert queue.translator_ids == frozenset({TRANSLATOR_ID})
    assert queue.reviewer_id == CLAUDE_UK_NATIVE_REVIEW_V1.reviewer_id
    assert len(queue.queue_sha256) == 64


def test_load_queue_preserves_dissent(queue: AdjudicationQueue) -> None:
    dispositions = {s.pair_id: s.artificial_review.disposition for s in queue.subjects}
    assert dispositions["legacy-pair:demo:2"] == "revise"
    assert queue.find("legacy-pair:demo:2").artificial_review.item_blocking
    assert all(s.artificial_review.dimensions["rubric_match"] == "fail" for s in queue.subjects)


def test_load_queue_rejects_wrong_record_count(private_root: Path) -> None:
    with pytest.raises(ValueError, match="exactly 34 records"):
        load_queue(private_root)


def test_load_queue_rejects_duplicate_pair_ids(private_root: Path) -> None:
    _mutate_queue(private_root, lambda payload: payload["items"].append(payload["items"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        load_queue(private_root, expected_count=len(PAIRS) + 1)


def test_load_queue_rejects_unknown_pair_id(private_root: Path) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        payload["items"][0]["pair_id"] = "legacy-pair:demo:unknown"

    _mutate_queue(private_root, mutate)
    with pytest.raises(ValueError, match="no reviewed translation"):
        load_queue(private_root, expected_count=len(PAIRS))


def test_load_queue_rejects_source_defect_exclusions(private_root: Path) -> None:
    excluded = sorted(SOURCE_DEFECT_EXCLUSIONS)[0]

    def mutate(payload: dict[str, Any]) -> None:
        payload["items"][0]["pair_id"] = excluded

    _mutate_queue(private_root, mutate)
    with pytest.raises(ValueError, match="not eligible"):
        load_queue(private_root, expected_count=len(PAIRS))


def test_load_queue_rejects_changed_translation(private_root: Path) -> None:
    path = private_root / "ai-reviewed-translations.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    records[0]["uk_rubric_text"] = "інший текст"
    path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="stale"):
        load_queue(private_root, expected_count=len(PAIRS))


def test_load_queue_rejects_unregistered_reviewer(private_root: Path) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        payload["reviewer"]["reviewer_id"] = "artificial-reviewer:unknown"

    _mutate_queue(private_root, mutate)
    with pytest.raises(ValueError, match="unregistered artificial reviewer"):
        load_queue(private_root, expected_count=len(PAIRS))


def test_load_queue_rejects_already_adjudicated_queue(private_root: Path) -> None:
    _mutate_queue(private_root, lambda payload: payload.update(human_signoff_complete=True))
    with pytest.raises(ValueError, match="unadjudicated"):
        load_queue(private_root, expected_count=len(PAIRS))


def test_public_state_carries_no_artificial_content(
    private_root: Path, queue: AdjudicationQueue
) -> None:
    workspace = ValidationWorkspace(private_root / "measurement-validation-v1", queue)
    for serialized in (
        json.dumps(workspace.public_state(), ensure_ascii=False),
        workspace.progress_path.read_text(),
    ):
        assert "weaker than the English" not in serialized
        assert "speaker attribute" not in serialized
        assert "accept_with_notes" not in serialized
        assert "defects" not in serialized
        assert "suggested_uk_text" not in serialized


def test_reveal_is_refused_before_the_blind_verdict_is_locked(
    private_root: Path, queue: AdjudicationQueue
) -> None:
    workspace = ValidationWorkspace(private_root / "measurement-validation-v1", queue)
    with pytest.raises(ValueError, match="must be locked before reveal"):
        workspace.artificial_review("legacy-pair:demo:1")


def test_reveal_after_lock_returns_the_proposal_and_is_recorded(
    private_root: Path, queue: AdjudicationQueue
) -> None:
    workspace = ValidationWorkspace(private_root / "measurement-validation-v1", queue)
    subject = queue.find("legacy-pair:demo:1")
    workspace.lock_blind_verdict(
        subject.pair_id,
        reviewer_id=REVIEWER_ID,
        dimensions=_full(),
        confidence=0.8,
        notes="",
        expected_subject_sha256=subject.subject_sha256,
    )
    review = workspace.artificial_review(subject.pair_id)
    assert review.dimensions["rubric_match"] == "fail"
    row = next(r for r in workspace.public_state()["items"] if r["pair_id"] == subject.pair_id)
    assert row["revealed_at"] is not None


def test_blind_verdict_preserves_failure_and_abstention(
    private_root: Path, queue: AdjudicationQueue
) -> None:
    workspace = ValidationWorkspace(private_root / "measurement-validation-v1", queue)
    subject = queue.find("legacy-pair:demo:1")
    dimensions = _full()
    dimensions["rubric_match"] = "fail"
    dimensions["naturalness"] = "not_judgeable"
    workspace.lock_blind_verdict(
        subject.pair_id,
        reviewer_id=REVIEWER_ID,
        dimensions=dimensions,
        confidence=0.4,
        notes="дисент",
        expected_subject_sha256=subject.subject_sha256,
    )
    row = next(r for r in workspace.public_state()["items"] if r["pair_id"] == subject.pair_id)
    assert row["blind_verdict"]["dimensions"]["rubric_match"] == "fail"
    assert row["blind_verdict"]["dimensions"]["naturalness"] == "not_judgeable"
    assert row["blind_verdict"]["reviewer_id"] == REVIEWER_ID
    assert row["blind_verdict"]["subject_sha256"] == subject.subject_sha256


def test_locked_blind_verdict_is_immutable(private_root: Path, queue: AdjudicationQueue) -> None:
    workspace = ValidationWorkspace(private_root / "measurement-validation-v1", queue)
    subject = queue.find("legacy-pair:demo:1")
    workspace.lock_blind_verdict(
        subject.pair_id,
        reviewer_id=REVIEWER_ID,
        dimensions=_full(),
        confidence=1.0,
        notes="",
        expected_subject_sha256=subject.subject_sha256,
    )
    with pytest.raises(ValueError, match="already locked"):
        workspace.lock_blind_verdict(
            subject.pair_id,
            reviewer_id=REVIEWER_ID,
            dimensions=_full("fail"),
            confidence=1.0,
            notes="",
            expected_subject_sha256=subject.subject_sha256,
        )
    with pytest.raises(ValueError, match="cannot be redrafted"):
        workspace.save_blind_draft(
            subject.pair_id,
            reviewer_id=REVIEWER_ID,
            dimensions=dict.fromkeys(DIMENSIONS, None),
            notes="",
            expected_subject_sha256=subject.subject_sha256,
        )


def test_incomplete_or_invalid_verdicts_are_refused(
    private_root: Path, queue: AdjudicationQueue
) -> None:
    workspace = ValidationWorkspace(private_root / "measurement-validation-v1", queue)
    subject = queue.find("legacy-pair:demo:1")
    partial: dict[str, Any] = dict.fromkeys(DIMENSIONS, "pass")
    partial["naturalness"] = None
    with pytest.raises(ValueError, match="requires every dimension"):
        workspace.lock_blind_verdict(
            subject.pair_id,
            reviewer_id=REVIEWER_ID,
            dimensions=partial,
            confidence=0.5,
            notes="",
            expected_subject_sha256=subject.subject_sha256,
        )
    with pytest.raises(ValueError, match="pass, fail, or not_judgeable"):
        workspace.lock_blind_verdict(
            subject.pair_id,
            reviewer_id=REVIEWER_ID,
            dimensions=_full("approved"),
            confidence=0.5,
            notes="",
            expected_subject_sha256=subject.subject_sha256,
        )
    with pytest.raises(ValueError, match="between zero and one"):
        workspace.lock_blind_verdict(
            subject.pair_id,
            reviewer_id=REVIEWER_ID,
            dimensions=_full(),
            confidence=1.5,
            notes="",
            expected_subject_sha256=subject.subject_sha256,
        )


def test_reviewer_identity_is_separated_from_translator_and_machine(
    private_root: Path, queue: AdjudicationQueue
) -> None:
    workspace = ValidationWorkspace(private_root / "measurement-validation-v1", queue)
    subject = queue.find("legacy-pair:demo:1")
    for identity, message in (
        (TRANSLATOR_ID, "differ from the translator"),
        (CLAUDE_UK_NATIVE_REVIEW_V1.reviewer_id, "cannot adjudicate its own proposal"),
        ("  ", "reviewer ID is required"),
    ):
        with pytest.raises(ValueError, match=message):
            workspace.lock_blind_verdict(
                subject.pair_id,
                reviewer_id=identity,
                dimensions=_full(),
                confidence=0.5,
                notes="",
                expected_subject_sha256=subject.subject_sha256,
            )


def test_stale_view_cannot_silently_overwrite(private_root: Path, queue: AdjudicationQueue) -> None:
    workspace = ValidationWorkspace(private_root / "measurement-validation-v1", queue)
    with pytest.raises(ValueError, match="reload the item"):
        workspace.save_blind_draft(
            "legacy-pair:demo:1",
            reviewer_id=REVIEWER_ID,
            dimensions=dict.fromkeys(DIMENSIONS, None),
            notes="",
            expected_subject_sha256="0" * 64,
        )
    with pytest.raises(KeyError):
        workspace.artificial_review("legacy-pair:demo:absent")


def _adjudicate_failed_naturalness(workspace: ValidationWorkspace, queue: AdjudicationQueue) -> str:
    subject = queue.find("legacy-pair:demo:1")
    dimensions = _full()
    dimensions["naturalness"] = "fail"
    workspace.lock_blind_verdict(
        subject.pair_id,
        reviewer_id=REVIEWER_ID,
        dimensions=dimensions,
        confidence=0.9,
        notes="неприродно",
        expected_subject_sha256=subject.subject_sha256,
    )
    workspace.artificial_review(subject.pair_id)
    workspace.record_adjudication(
        subject.pair_id,
        reviewer_id=REVIEWER_ID,
        decision="confirm",
        final_dimensions=dimensions,
        rationale="",
        expected_subject_sha256=subject.subject_sha256,
    )
    return subject.pair_id


def test_correction_is_a_hashed_overlay_and_preserves_frozen_adjudication(
    private_root: Path, queue: AdjudicationQueue
) -> None:
    workspace = ValidationWorkspace(private_root / "measurement-validation-v1", queue)
    pair_id = _adjudicate_failed_naturalness(workspace, queue)
    subject = queue.find(pair_id)
    state = workspace.save_correction(
        pair_id,
        corrector_id="machine-corrector-01",
        texts={
            "uk_prompt": subject.uk_prompt + " виправлено",
            "uk_image_text": subject.uk_image_text,
            "uk_rubric": subject.uk_rubric,
        },
        rationale="remove translationese",
        expected_subject_sha256=subject.subject_sha256,
    )
    row = next(item for item in state["items"] if item["pair_id"] == pair_id)
    assert row["adjudication"]["final_dimensions"]["naturalness"] == "fail"
    assert row["correction"]["base_subject_sha256"] == subject.subject_sha256
    assert row["current_subject_sha256"] == row["correction"]["subject_sha256"]
    assert row["current_subject_sha256"] != subject.subject_sha256
    assert row["texts"]["uk_prompt"].endswith("виправлено")
    assert queue.find(pair_id).uk_prompt == subject.uk_prompt
    assert row["correction_pending_review"] is True
    assert row["needs_correction"] is True
    assert "correction_saved" in [event["event"] for event in row["history"]]


def test_correction_requires_a_distinct_corrector_and_changed_text(
    private_root: Path, queue: AdjudicationQueue
) -> None:
    workspace = ValidationWorkspace(private_root / "measurement-validation-v1", queue)
    pair_id = _adjudicate_failed_naturalness(workspace, queue)
    subject = queue.find(pair_id)
    texts = {
        "uk_prompt": subject.uk_prompt,
        "uk_image_text": subject.uk_image_text,
        "uk_rubric": subject.uk_rubric,
    }
    with pytest.raises(ValueError, match="differ from the reviewing human"):
        workspace.save_correction(
            pair_id,
            corrector_id=REVIEWER_ID,
            texts={**texts, "uk_prompt": subject.uk_prompt + " нове"},
            rationale="change",
            expected_subject_sha256=subject.subject_sha256,
        )
    with pytest.raises(ValueError, match="does not change"):
        workspace.save_correction(
            pair_id,
            corrector_id="machine-corrector-01",
            texts=texts,
            rationale="change",
            expected_subject_sha256=subject.subject_sha256,
        )


def test_distinct_reviewer_can_pass_the_corrected_revision(
    private_root: Path, queue: AdjudicationQueue
) -> None:
    root = private_root / "measurement-validation-v1"
    workspace = ValidationWorkspace(root, queue)
    pair_id = _adjudicate_failed_naturalness(workspace, queue)
    subject = queue.find(pair_id)
    state = workspace.save_correction(
        pair_id,
        corrector_id="machine-corrector-01",
        texts={
            "uk_prompt": subject.uk_prompt + " виправлено",
            "uk_image_text": subject.uk_image_text,
            "uk_rubric": subject.uk_rubric,
        },
        rationale="remove translationese",
        expected_subject_sha256=subject.subject_sha256,
    )
    row = next(item for item in state["items"] if item["pair_id"] == pair_id)
    corrected_hash = row["current_subject_sha256"]
    with pytest.raises(ValueError, match="differ from the corrector"):
        workspace.record_correction_review(
            pair_id,
            reviewer_id="machine-corrector-01",
            dimensions=_full(),
            notes="",
            expected_subject_sha256=corrected_hash,
        )
    state = workspace.record_correction_review(
        pair_id,
        reviewer_id=REVIEWER_ID,
        dimensions=_full(),
        notes="",
        expected_subject_sha256=corrected_hash,
    )
    row = next(item for item in state["items"] if item["pair_id"] == pair_id)
    assert row["effective_dimensions"] == _full()
    assert row["needs_correction"] is False
    assert row["correction_pending_review"] is False
    assert state["summary"]["corrected"] == 1
    assert state["summary"]["correction_reviewed"] == 1

    resumed = ValidationWorkspace(root, queue).public_state()
    resumed_row = next(item for item in resumed["items"] if item["pair_id"] == pair_id)
    assert resumed_row["correction_review"]["subject_sha256"] == corrected_hash


def test_draft_autosave_survives_restart(private_root: Path, queue: AdjudicationQueue) -> None:
    root = private_root / "measurement-validation-v1"
    workspace = ValidationWorkspace(root, queue)
    subject = queue.find("legacy-pair:demo:2")
    dimensions: dict[str, str | None] = dict.fromkeys(DIMENSIONS, None)
    dimensions["category_match"] = "pass"
    workspace.save_blind_draft(
        subject.pair_id,
        reviewer_id=REVIEWER_ID,
        dimensions=dimensions,
        notes="незавершено",
        expected_subject_sha256=subject.subject_sha256,
    )
    resumed = ValidationWorkspace(root, load_queue(private_root, expected_count=len(PAIRS)))
    row = next(r for r in resumed.public_state()["items"] if r["pair_id"] == subject.pair_id)
    assert row["blind_draft"]["dimensions"]["category_match"] == "pass"
    assert row["blind_draft"]["notes"] == "незавершено"
    assert row["locked"] is False


def test_changed_subject_invalidates_the_signoff_without_deleting_it(
    private_root: Path, queue: AdjudicationQueue
) -> None:
    root = private_root / "measurement-validation-v1"
    workspace = ValidationWorkspace(root, queue)
    subject = queue.find("legacy-pair:demo:1")
    workspace.lock_blind_verdict(
        subject.pair_id,
        reviewer_id=REVIEWER_ID,
        dimensions=_full(),
        confidence=0.9,
        notes="",
        expected_subject_sha256=subject.subject_sha256,
    )
    translations_path = private_root / "ai-reviewed-translations.json"
    records = json.loads(translations_path.read_text(encoding="utf-8"))
    records[0]["uk_rubric_text"] = "виправлена рубрика"
    translations_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")

    def mutate(payload: dict[str, Any]) -> None:
        payload["items"][0]["artificial_review"]["subject_sha256"] = subject_sha256(
            records[0]["uk_prompt"], records[0]["uk_image_text"], records[0]["uk_rubric_text"]
        )

    _mutate_queue(private_root, mutate)
    resumed = ValidationWorkspace(root, load_queue(private_root, expected_count=len(PAIRS)))
    row = next(r for r in resumed.public_state()["items"] if r["pair_id"] == subject.pair_id)
    assert row["blind_verdict"] is None
    assert row["locked"] is False
    events = [entry["event"] for entry in row["history"]]
    assert "blind_locked" in events
    assert "invalidated_subject_changed" in events


def test_private_files_use_restrictive_modes(private_root: Path, queue: AdjudicationQueue) -> None:
    root = private_root / "measurement-validation-v1"
    workspace = ValidationWorkspace(root, queue)
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE(workspace.progress_path.stat().st_mode) == 0o600
    assert not list(root.glob("*.tmp"))


def test_progress_manifest_is_tracked_safe(private_root: Path, queue: AdjudicationQueue) -> None:
    workspace = ValidationWorkspace(private_root / "measurement-validation-v1", queue)
    subject = queue.find("legacy-pair:demo:1")
    workspace.save_blind_draft(
        subject.pair_id,
        reviewer_id=REVIEWER_ID,
        dimensions=dict.fromkeys(DIMENSIONS, None),
        notes="приватна нотатка",
        expected_subject_sha256=subject.subject_sha256,
    )
    manifest = progress_manifest(workspace.public_state())
    serialized = json.dumps(manifest, ensure_ascii=False)
    assert "приватна нотатка" not in serialized
    assert REVIEWER_ID not in serialized
    assert manifest["strata"] == {"refuse_or_safe_redirect": 1, "comply_safely": 1}
    assert manifest["total"] == len(PAIRS)
    assert manifest["queue_sha256"] == queue.queue_sha256
