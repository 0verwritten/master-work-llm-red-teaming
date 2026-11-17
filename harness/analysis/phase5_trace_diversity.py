from __future__ import annotations

import hashlib
import itertools
import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict
from sqlalchemy import select

from harness.persistence.models import (
    AdversarialRound,
    AdversarialSession,
    Evaluation,
    Phase3ExecutionBatch,
    Phase3ExecutionCell,
    Phase3ExecutionCellStatus,
)
from harness.persistence.repository import TraceRepository

TOKENIZER_VERSION = "unicode-nfc-casefold-alnum-v1"
TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)


class AggregateModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Rate(AggregateModel):
    numerator: int
    denominator: int
    rate: float | None


class LexicalSummary(AggregateModel):
    observation_count: int
    exact_unique_count: int
    exact_duplicate_rate: Rate
    total_tokens: int
    vocabulary_size: int
    distinct_1: float | None
    distinct_2: float | None
    mean_pairwise_token_jaccard_distance: float | None


class RefinementPoint(AggregateModel):
    round_number: int
    observation_count: int
    exact_new_within_run: Rate
    mean_new_token_fraction_within_run: float | None
    mean_cumulative_unique_attacks: float
    mean_cumulative_unique_fraction: float
    adjacent_exact_repeat: Rate | None


class ExactCollapseSummary(AggregateModel):
    no_exact_collapse: int
    partial_exact_collapse: int
    complete_exact_collapse: int


class CoveragePoint(AggregateModel):
    through_round: int
    observations: int
    unique_attacks: int
    unique_responses: int
    fixed_judge_positive_rounds: int
    fixed_judge_positive_runs: int
    unique_fixed_judge_positive_responses: int
    fixed_judge_positive_categories: int


class TraceStratum(AggregateModel):
    run_count: int
    round_count: int
    category_count: int
    attacks: LexicalSummary
    responses: LexicalSummary
    exact_collapse: ExactCollapseSummary
    refinement: tuple[RefinementPoint, ...]
    coverage_curve: tuple[CoveragePoint, ...]


class Phase5TraceDiversityAggregate(AggregateModel):
    schema_version: Literal[1] = 1
    aggregate_only: Literal[True] = True
    raw_content_exported: Literal[False] = False
    zero_inference: Literal[True] = True
    tokenizer_version: str
    run_count: int
    round_count: int
    strata: dict[str, TraceStratum]
    semantic_vulnerability_coverage_estimable: Literal[False] = False
    semantic_vulnerability_coverage_reason: Literal[
        "embedding_profile_and_threshold_not_pinned"
    ] = "embedding_profile_and_threshold_not_pinned"
    vendi_score_estimable: Literal[False] = False
    vendi_score_reason: Literal["no_pinned_similarity_kernel"] = "no_pinned_similarity_kernel"
    inferential_outputs_estimable: Literal[False] = False
    inferential_outputs_reason: Literal["stopped_single_seed_all_pair_trace"] = (
        "stopped_single_seed_all_pair_trace"
    )
    comparative_method_outputs_estimable: Literal[False] = False
    comparative_method_outputs_reason: Literal["all_executed_runs_use_pair"] = (
        "all_executed_runs_use_pair"
    )


@dataclass(frozen=True)
class TraceObservation:
    run_id: UUID
    round_number: int
    language: Literal["en", "uk"]
    attacker: str
    category: str
    method: str
    attack_text: str
    target_response: str
    fixed_judge_label: Literal["refused", "partial", "complied"]


async def analyze_frozen_phase3_trace(
    *, repository: TraceRepository, batch: Phase3ExecutionBatch
) -> Phase5TraceDiversityAggregate:

    cells = tuple(
        (
            await repository.session.scalars(
                select(Phase3ExecutionCell).where(
                    Phase3ExecutionCell.batch_id == batch.id,
                    Phase3ExecutionCell.status == Phase3ExecutionCellStatus.TERMINAL,
                )
            )
        ).all()
    )
    if len(cells) != 7:
        raise RuntimeError("the frozen stopped trace requires exactly seven terminal cells")
    observations: list[TraceObservation] = []
    for cell in cells:
        if cell.run_id is None:
            raise RuntimeError("terminal Phase 3 cell lacks a source run")
        source_session = await repository.session.scalar(
            select(AdversarialSession).where(AdversarialSession.run_id == cell.run_id)
        )
        if source_session is None:
            raise RuntimeError("terminal Phase 3 cell lacks an adversarial session")
        rounds = tuple(
            (
                await repository.session.scalars(
                    select(AdversarialRound)
                    .where(AdversarialRound.session_id == source_session.id)
                    .order_by(AdversarialRound.round_number)
                )
            ).all()
        )
        if [round_.round_number for round_ in rounds] != [1, 2, 3]:
            raise RuntimeError("each frozen source run must contain rounds one through three")
        attacker = cell.cell_spec_json.get("attacker", {}).get("key")
        category = cell.cell_spec_json.get("item", {}).get("category")
        language = source_session.language
        if (
            not isinstance(attacker, str)
            or not isinstance(category, str)
            or language not in {"en", "uk"}
        ):
            raise RuntimeError("source trace factor identity is incomplete")
        for round_ in rounds:
            evaluation = await repository.session.scalar(
                select(Evaluation).where(Evaluation.round_id == round_.id)
            )
            if evaluation is None or evaluation.label not in {
                "refused",
                "partial",
                "complied",
            }:
                raise RuntimeError("source trace evaluation is incomplete")
            observations.append(
                TraceObservation(
                    run_id=source_session.run_id,
                    round_number=round_.round_number,
                    language=cast(Literal["en", "uk"], language),
                    attacker=attacker,
                    category=category,
                    method=source_session.method,
                    attack_text=round_.attack_text,
                    target_response=round_.target_response,
                    fixed_judge_label=cast(
                        Literal["refused", "partial", "complied"], evaluation.label
                    ),
                )
            )
    return aggregate_trace_diversity(tuple(observations))


def aggregate_trace_diversity(
    observations: tuple[TraceObservation, ...],
) -> Phase5TraceDiversityAggregate:

    _validate_frozen_trace(observations)
    strata: dict[str, tuple[TraceObservation, ...]] = {
        "overall": observations,
        "en": tuple(value for value in observations if value.language == "en"),
        "uk": tuple(value for value in observations if value.language == "uk"),
    }
    for attacker in sorted({value.attacker for value in observations}):
        strata[f"attacker:{attacker}"] = tuple(
            value for value in observations if value.attacker == attacker
        )
    return Phase5TraceDiversityAggregate(
        tokenizer_version=TOKENIZER_VERSION,
        run_count=len({value.run_id for value in observations}),
        round_count=len(observations),
        strata={name: _stratum(values) for name, values in strata.items()},
    )


def _validate_frozen_trace(observations: tuple[TraceObservation, ...]) -> None:
    if len(observations) != 21:
        raise ValueError("the frozen stopped trace contains exactly 21 rounds")
    by_run: dict[UUID, list[TraceObservation]] = defaultdict(list)
    for value in observations:
        by_run[value.run_id].append(value)
    if len(by_run) != 7:
        raise ValueError("the frozen stopped trace contains exactly seven runs")
    for values in by_run.values():
        if sorted(value.round_number for value in values) != [1, 2, 3]:
            raise ValueError("each run must contribute exactly rounds one through three")
        factors = {
            (value.language, value.attacker, value.category, value.method) for value in values
        }
        if len(factors) != 1:
            raise ValueError("source factors changed within a run")
    if {value.method for value in observations} != {"pair"}:
        raise ValueError("the frozen stopped trace contains only PAIR runs")


def _stratum(observations: tuple[TraceObservation, ...]) -> TraceStratum:
    return TraceStratum(
        run_count=len({value.run_id for value in observations}),
        round_count=len(observations),
        category_count=len({value.category for value in observations}),
        attacks=_lexical(tuple(value.attack_text for value in observations)),
        responses=_lexical(tuple(value.target_response for value in observations)),
        exact_collapse=_exact_collapse(observations),
        refinement=_refinement(observations),
        coverage_curve=_coverage_curve(observations),
    )


def _lexical(texts: tuple[str, ...]) -> LexicalSummary:
    token_sequences = tuple(_tokens(text) for text in texts)
    token_total = sum(len(tokens) for tokens in token_sequences)
    vocabulary = {token for tokens in token_sequences for token in tokens}
    bigrams = [pair for tokens in token_sequences for pair in itertools.pairwise(tokens)]
    hashes = {_text_sha256(text) for text in texts}
    distances = [
        _jaccard_distance(set(left), set(right))
        for left, right in itertools.combinations(token_sequences, 2)
    ]
    return LexicalSummary(
        observation_count=len(texts),
        exact_unique_count=len(hashes),
        exact_duplicate_rate=_rate(len(texts) - len(hashes), len(texts)),
        total_tokens=token_total,
        vocabulary_size=len(vocabulary),
        distinct_1=len(vocabulary) / token_total if token_total else None,
        distinct_2=len(set(bigrams)) / len(bigrams) if bigrams else None,
        mean_pairwise_token_jaccard_distance=(
            math.fsum(distances) / len(distances) if distances else None
        ),
    )


def _refinement(
    observations: tuple[TraceObservation, ...],
) -> tuple[RefinementPoint, ...]:
    by_run: dict[UUID, list[TraceObservation]] = defaultdict(list)
    for value in observations:
        by_run[value.run_id].append(value)
    exact_new: dict[int, list[bool]] = defaultdict(list)
    token_novelty: dict[int, list[float]] = defaultdict(list)
    cumulative_unique: dict[int, list[int]] = defaultdict(list)
    adjacent_repeat: dict[int, list[bool]] = defaultdict(list)
    for values in by_run.values():
        prior_hashes: set[str] = set()
        prior_tokens: set[str] = set()
        previous_digest: str | None = None
        for value in sorted(values, key=lambda item: item.round_number):
            digest = _text_sha256(value.attack_text)
            tokens = set(_tokens(value.attack_text))
            exact_new[value.round_number].append(digest not in prior_hashes)
            token_novelty[value.round_number].append(
                len(tokens - prior_tokens) / len(tokens) if tokens else 0.0
            )
            prior_hashes.add(digest)
            prior_tokens.update(tokens)
            cumulative_unique[value.round_number].append(len(prior_hashes))
            if previous_digest is not None:
                adjacent_repeat[value.round_number].append(digest == previous_digest)
            previous_digest = digest
    return tuple(
        RefinementPoint(
            round_number=round_number,
            observation_count=len(exact_new[round_number]),
            exact_new_within_run=_rate(sum(exact_new[round_number]), len(exact_new[round_number])),
            mean_new_token_fraction_within_run=(
                math.fsum(token_novelty[round_number]) / len(token_novelty[round_number])
            ),
            mean_cumulative_unique_attacks=(
                math.fsum(cumulative_unique[round_number]) / len(cumulative_unique[round_number])
            ),
            mean_cumulative_unique_fraction=(
                math.fsum(cumulative_unique[round_number])
                / (len(cumulative_unique[round_number]) * round_number)
            ),
            adjacent_exact_repeat=(
                _rate(
                    sum(adjacent_repeat[round_number]),
                    len(adjacent_repeat[round_number]),
                )
                if round_number > 1
                else None
            ),
        )
        for round_number in (1, 2, 3)
    )


def _exact_collapse(observations: tuple[TraceObservation, ...]) -> ExactCollapseSummary:
    by_run: dict[UUID, set[str]] = defaultdict(set)
    for value in observations:
        by_run[value.run_id].add(_text_sha256(value.attack_text))
    counts = {size: sum(len(hashes) == size for hashes in by_run.values()) for size in (1, 2, 3)}
    return ExactCollapseSummary(
        no_exact_collapse=counts[3],
        partial_exact_collapse=counts[2],
        complete_exact_collapse=counts[1],
    )


def _coverage_curve(
    observations: tuple[TraceObservation, ...],
) -> tuple[CoveragePoint, ...]:
    points: list[CoveragePoint] = []
    for through_round in (1, 2, 3):
        included = tuple(value for value in observations if value.round_number <= through_round)
        positives = tuple(value for value in included if value.fixed_judge_label == "complied")
        points.append(
            CoveragePoint(
                through_round=through_round,
                observations=len(included),
                unique_attacks=len({_text_sha256(value.attack_text) for value in included}),
                unique_responses=len({_text_sha256(value.target_response) for value in included}),
                fixed_judge_positive_rounds=len(positives),
                fixed_judge_positive_runs=len({value.run_id for value in positives}),
                unique_fixed_judge_positive_responses=len(
                    {_text_sha256(value.target_response) for value in positives}
                ),
                fixed_judge_positive_categories=len({value.category for value in positives}),
            )
        )
    return tuple(points)


def _tokens(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFC", text).casefold()
    return tuple(TOKEN_PATTERN.findall(normalized))


def _text_sha256(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    return hashlib.sha256(normalized.encode()).hexdigest()


def _jaccard_distance(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return 1 - len(left & right) / len(union)


def _rate(numerator: int, denominator: int) -> Rate:
    return Rate(
        numerator=numerator,
        denominator=denominator,
        rate=numerator / denominator if denominator else None,
    )
