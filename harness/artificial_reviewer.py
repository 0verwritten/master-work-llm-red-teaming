from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

REVIEWER_KIND: Literal["artificial_reviewer"] = "artificial_reviewer"

Dimension = Literal[
    "semantic_equivalence",
    "naturalness",
    "category_match",
    "rubric_match",
    "expected_behavior_match",
]

DIMENSIONS: tuple[Dimension, ...] = (
    "semantic_equivalence",
    "naturalness",
    "category_match",
    "rubric_match",
    "expected_behavior_match",
)

Verdict = Literal["pass", "fail", "abstain"]
Severity = Literal["blocking", "minor"]
Scope = Literal["item", "collection_wide"]
Disposition = Literal["accept", "accept_with_notes", "revise", "reject", "abstain"]

ARTIFICIAL_REVIEWER_STATUS: Literal["ai_adjudication_pending_human_signoff"] = (
    "ai_adjudication_pending_human_signoff"
)

FORBIDDEN_STATUSES = frozenset({"approved", "human_reviewed", "validated"})


class ArtificialReviewer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    reviewer_id: str = Field(min_length=1)
    reviewer_kind: Literal["artificial_reviewer"] = REVIEWER_KIND
    profile: str = Field(min_length=1)
    model_identity: str = Field(min_length=1)
    procedure_version: str = Field(min_length=1)

    competencies: tuple[str, ...] = Field(min_length=1)

    limits: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def reviewer_id_must_not_impersonate_a_person(self) -> ArtificialReviewer:
        if not self.reviewer_id.startswith("artificial-reviewer:"):
            raise ValueError("artificial reviewer IDs must carry the artificial-reviewer prefix")
        return self


class ReviewDefect(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    dimension: Dimension
    severity: Severity
    scope: Scope

    note: str = Field(min_length=1)

    correction_ref: str | None = None


class HumanSignoff(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    reviewer_id: str | None = None
    reviewed_at: datetime | None = None
    decision: Literal["confirm", "override", "partial_override"] | None = None
    dimensions: dict[Dimension, Verdict | None] = Field(
        default_factory=lambda: dict.fromkeys(DIMENSIONS)
    )
    notes: str = ""

    @property
    def resolved(self) -> bool:
        return self.decision is not None


class ArtificialReview(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    reviewer: ArtificialReviewer
    reviewed_at: datetime

    subject_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dimensions: dict[Dimension, Verdict]
    disposition: Disposition
    defects: tuple[ReviewDefect, ...] = ()
    observations: tuple[str, ...] = ()
    status: Literal["ai_adjudication_pending_human_signoff"] = ARTIFICIAL_REVIEWER_STATUS
    human_signoff: HumanSignoff = Field(default_factory=HumanSignoff)

    @model_validator(mode="after")
    def dimensions_must_be_complete(self) -> ArtificialReview:
        if set(self.dimensions) != set(DIMENSIONS):
            raise ValueError("an artificial review must judge all five dimensions")
        return self

    @model_validator(mode="after")
    def failed_dimensions_need_defects(self) -> ArtificialReview:
        failed = {name for name, verdict in self.dimensions.items() if verdict == "fail"}
        explained = {defect.dimension for defect in self.defects}
        if failed - explained:
            raise ValueError("every failed dimension must carry at least one defect")
        if explained - failed:
            raise ValueError("every defect must correspond to a failed dimension")
        return self

    @model_validator(mode="after")
    def disposition_must_follow_defects(self) -> ArtificialReview:
        item_blocking = any(
            defect.severity == "blocking" and defect.scope == "item" for defect in self.defects
        )
        if item_blocking and self.disposition in {"accept", "accept_with_notes"}:
            raise ValueError("a blocking item defect cannot carry an accepting disposition")
        if self.disposition == "accept" and self.defects:
            raise ValueError("a plain accept disposition cannot carry defects")
        return self

    @property
    def item_blocking(self) -> bool:
        return any(
            defect.severity == "blocking" and defect.scope == "item" for defect in self.defects
        )


CLAUDE_UK_NATIVE_REVIEW_V1 = ArtificialReviewer(
    reviewer_id="artificial-reviewer:claude-uk-review-01",
    profile="claude-uk-native-review-v1",
    model_identity="claude-opus-5",
    procedure_version="uk-equivalence-adjudication-v1",
    competencies=(
        "English-to-Ukrainian semantic equivalence of benchmark prompts and rubrics",
        "Ukrainian naturalness and register",
        "rubric decision-boundary drift between paired languages",
        "category and expected-behaviour consistency with the English source",
    ),
    limits=(
        "not a human reviewer and not a native speaker",
        "cannot satisfy the independent-review requirement of the bilingual protocol",
        "cannot produce approved status or human translation provenance",
        "judgements are proposals that a human adjudicator must confirm or override",
        "no access to target-model outcomes at review time",
    ),
)

REGISTRY: dict[str, ArtificialReviewer] = {
    CLAUDE_UK_NATIVE_REVIEW_V1.reviewer_id: CLAUDE_UK_NATIVE_REVIEW_V1,
}


def get_reviewer(reviewer_id: str) -> ArtificialReviewer:

    try:
        return REGISTRY[reviewer_id]
    except KeyError:
        raise ValueError(f"unregistered artificial reviewer: {reviewer_id}") from None
