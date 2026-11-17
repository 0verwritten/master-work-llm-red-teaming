from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from harness.artificial_reviewer import (
    ARTIFICIAL_REVIEWER_STATUS,
    CLAUDE_UK_NATIVE_REVIEW_V1,
    DIMENSIONS,
    FORBIDDEN_STATUSES,
    ArtificialReview,
    ArtificialReviewer,
    Dimension,
    ReviewDefect,
    Verdict,
    get_reviewer,
)

SUBJECT = "a" * 64


def _dimensions(**overrides: Verdict) -> dict[Dimension, Verdict]:
    dimensions: dict[Dimension, Verdict] = dict.fromkeys(DIMENSIONS, "pass")
    dimensions.update(overrides)  # type: ignore[arg-type]
    return dimensions


def _review(**overrides: object) -> ArtificialReview:
    payload: dict[str, object] = {
        "reviewer": CLAUDE_UK_NATIVE_REVIEW_V1,
        "reviewed_at": datetime.now(UTC),
        "subject_sha256": SUBJECT,
        "dimensions": _dimensions(),
        "disposition": "accept",
    }
    payload.update(overrides)
    return ArtificialReview.model_validate(payload)


def test_reviewer_id_must_be_prefixed() -> None:
    with pytest.raises(ValidationError, match="artificial-reviewer prefix"):
        ArtificialReviewer(
            reviewer_id="maria-k",
            profile="p",
            model_identity="m",
            procedure_version="v1",
            competencies=("x",),
            limits=("y",),
        )


def test_registered_reviewer_declares_its_limits() -> None:
    reviewer = get_reviewer("artificial-reviewer:claude-uk-review-01")
    assert reviewer.reviewer_kind == "artificial_reviewer"
    assert any("not a human reviewer" in limit for limit in reviewer.limits)


def test_unregistered_reviewer_is_refused() -> None:
    with pytest.raises(ValueError, match="unregistered artificial reviewer"):
        get_reviewer("artificial-reviewer:nobody")


def test_status_cannot_be_a_human_approval() -> None:
    assert ARTIFICIAL_REVIEWER_STATUS not in FORBIDDEN_STATUSES
    with pytest.raises(ValidationError):
        _review(status="approved")


def test_signoff_starts_unresolved() -> None:
    review = _review()
    assert review.human_signoff.resolved is False
    assert set(review.human_signoff.dimensions) == set(DIMENSIONS)
    assert all(value is None for value in review.human_signoff.dimensions.values())


def test_every_dimension_must_be_judged() -> None:
    partial = _dimensions()
    del partial["naturalness"]
    with pytest.raises(ValidationError, match="all five dimensions"):
        _review(dimensions=partial)


def test_failed_dimension_requires_a_defect() -> None:
    with pytest.raises(ValidationError, match="every failed dimension"):
        _review(dimensions=_dimensions(naturalness="fail"), disposition="revise")


def test_defect_requires_a_failed_dimension() -> None:
    defect = ReviewDefect(dimension="naturalness", severity="minor", scope="item", note="n")
    with pytest.raises(ValidationError, match="every defect must correspond"):
        _review(defects=(defect,), disposition="accept_with_notes")


def test_blocking_item_defect_cannot_be_accepted() -> None:
    defect = ReviewDefect(
        dimension="semantic_equivalence", severity="blocking", scope="item", note="n"
    )
    with pytest.raises(ValidationError, match="blocking item defect"):
        _review(
            dimensions=_dimensions(semantic_equivalence="fail"),
            defects=(defect,),
            disposition="accept_with_notes",
        )


def test_reviewer_can_dissent() -> None:

    defect = ReviewDefect(
        dimension="rubric_match", severity="blocking", scope="collection_wide", note="n"
    )
    review = _review(
        dimensions=_dimensions(rubric_match="fail"),
        defects=(defect,),
        disposition="revise",
    )
    assert review.dimensions["rubric_match"] == "fail"
    assert review.item_blocking is False
